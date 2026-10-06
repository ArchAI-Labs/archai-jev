//! Downloading a pinned file into the blob store: resume, retries, hash while downloading,
//! atomic rename. Trust is in the SHA-256 of the registry, not in the transport.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::blobs::BlobStore;
use super::config::HubConfig;
use super::events::{Cancel, Event, Observer};
use crate::error::{Error, Result};
use crate::models::failures::DownloadFailure;
use crate::models::files::{FileEntry, FileOrigin};
use crate::models::hash::Hasher;
use crate::models::incompat::Incompat;

/// What the network returned for one request.
pub struct Response {
    /// HTTP status after redirects.
    pub status: u16,
    /// First byte of the body in the whole file (from `Content-Range`), for a `206`.
    pub range_start: Option<u64>,
    /// The body.
    pub body: Box<dyn Read + Send>,
}

/// Something that can fetch a URL (the real one uses `ureq`; tests use a local server or a fake).
pub trait Transport: Send + Sync {
    /// GET `url`, optionally from byte `range_from`; the body must be fully read within `budget`.
    ///
    /// # Errors
    /// A description of a network-level failure (DNS, connection, TLS, timeout).
    fn get(
        &self,
        url: &str,
        range_from: Option<u64>,
        budget: Duration,
    ) -> std::result::Result<Response, String>;
}

fn parse_content_range(value: &str) -> Option<u64> {
    let rest = value.trim().strip_prefix("bytes ")?;
    rest.split('-').next()?.trim().parse().ok()
}

/// The real transport: HTTPS with the system certificates, proxies from the environment.
pub struct UreqTransport {
    connect_timeout: Duration,
    response_timeout: Duration,
    max_redirects: u32,
}

impl UreqTransport {
    /// A transport with the timeouts of `cfg`.
    pub fn new(cfg: &HubConfig) -> Self {
        UreqTransport {
            connect_timeout: cfg.connect_timeout,
            response_timeout: cfg.response_timeout,
            max_redirects: cfg.max_redirects,
        }
    }
}

impl Transport for UreqTransport {
    fn get(
        &self,
        url: &str,
        range_from: Option<u64>,
        budget: Duration,
    ) -> std::result::Result<Response, String> {
        use ureq::tls::{RootCerts, TlsConfig};
        let config = ureq::Agent::config_builder()
            .timeout_connect(Some(self.connect_timeout))
            .timeout_recv_response(Some(self.response_timeout))
            .timeout_recv_body(Some(budget))
            .max_redirects(self.max_redirects)
            .http_status_as_error(false)
            .tls_config(
                TlsConfig::builder()
                    .root_certs(RootCerts::PlatformVerifier)
                    .build(),
            )
            .build();
        let agent = ureq::Agent::new_with_config(config);
        let mut request = agent.get(url).header(
            "User-Agent",
            concat!("archai-jev/", env!("CARGO_PKG_VERSION")),
        );
        if let Some(from) = range_from {
            request = request.header("Range", format!("bytes={from}-"));
        }
        let response = request.call().map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        let range_start = response
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range);
        Ok(Response {
            status,
            range_start,
            body: Box::new(response.into_body().into_reader()),
        })
    }
}

fn io_failure(path: &std::path::Path, e: &std::io::Error) -> Error {
    Error::ModelDownload(DownloadFailure::Io {
        path: path.display().to_string(),
        cause: e.to_string(),
    })
}

fn mismatch(entry: &FileEntry, got: String, size: u64) -> Error {
    Error::IncompatibleModel(Incompat::HashMismatch {
        path: entry.path.clone(),
        expected: entry.sha256.clone(),
        got,
        size,
    })
}

/// What a failed write to the partial file means: the `.part` is deleted, and a full disk is
/// reported with how many more bytes are needed.
fn on_write_error(
    part: &std::path::Path,
    entry: &FileEntry,
    written: u64,
    e: &std::io::Error,
) -> Error {
    let _ = std::fs::remove_file(part);
    if e.kind() == ErrorKind::StorageFull {
        return Error::ModelDownload(DownloadFailure::DiskFull {
            file: entry.path.clone(),
            dir: part
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            need: entry.size.saturating_sub(written),
        });
    }
    io_failure(part, e)
}

#[cfg(all(test, feature = "testing"))]
pub(crate) fn on_write_error_for_tests(
    part: &std::path::Path,
    entry: &FileEntry,
    written: u64,
    e: &std::io::Error,
) -> Error {
    on_write_error(part, entry, written, e)
}

/// Outcome of reading one response body.
enum Body {
    /// All bytes written; the hash is of the whole `.part`.
    Done { hash: String, total: u64 },
    /// The connection failed or ended early after `progressed` bytes of progress.
    Interrupted { cause: String, progressed: bool },
}

struct Job<'a> {
    cfg: &'a HubConfig,
    entry: &'a FileEntry,
    part: PathBuf,
    observer: &'a dyn Observer,
    cancel: &'a dyn Cancel,
}

impl Job<'_> {
    fn read_body(&self, resp: Response, resume_from: u64) -> Result<Body> {
        let resuming =
            resume_from > 0 && resp.status == 206 && resp.range_start == Some(resume_from);
        let mut hasher = Hasher::new();
        let mut file: File;
        let mut written_before: u64 = 0;
        if resuming {
            let mut existing = File::open(&self.part).map_err(|e| io_failure(&self.part, &e))?;
            let mut buf = vec![0u8; 1 << 20];
            let mut left = resume_from;
            while left > 0 {
                let want = usize::try_from(left.min(buf.len() as u64)).unwrap_or(buf.len());
                let slice = buf.get_mut(..want).unwrap_or_default();
                existing
                    .read_exact(slice)
                    .map_err(|e| io_failure(&self.part, &e))?;
                hasher.update(slice);
                left -= want as u64;
            }
            file = OpenOptions::new()
                .append(true)
                .open(&self.part)
                .map_err(|e| io_failure(&self.part, &e))?;
            written_before = resume_from;
        } else {
            file = File::create(&self.part).map_err(|e| io_failure(&self.part, &e))?;
        }

        let mut body = resp.body;
        let mut buf = vec![0u8; 1 << 20];
        let mut written = written_before;
        let mut last_event = Instant::now();
        loop {
            if self.cancel.cancelled() {
                let _ = file.flush();
                return Err(Error::Cancelled);
            }
            let n = match body.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = file.flush();
                    return Ok(Body::Interrupted {
                        cause: e.to_string(),
                        progressed: written > written_before,
                    });
                }
            };
            let chunk = buf.get(..n).unwrap_or_default();
            if let Err(e) = file.write_all(chunk) {
                return Err(on_write_error(&self.part, self.entry, written, &e));
            }
            hasher.update(chunk);
            written += n as u64;
            if last_event.elapsed() >= self.cfg.progress_interval {
                last_event = Instant::now();
                self.observer.on_event(&Event::DownloadProgress {
                    file: self.entry.path.clone(),
                    done: written,
                    total: self.entry.size,
                });
            }
        }
        let _ = file.flush();
        let progressed = written > written_before;
        if written < self.entry.size {
            return Ok(Body::Interrupted {
                cause: format!(
                    "the connection ended after {written} of {} bytes",
                    self.entry.size
                ),
                progressed,
            });
        }
        let _ = file.sync_all();
        Ok(Body::Done {
            hash: hasher.finish_hex(),
            total: written,
        })
    }
}

fn pause(cfg: &HubConfig, failures: usize, cancel: &dyn Cancel) -> Result<()> {
    let idx = failures
        .saturating_sub(1)
        .min(cfg.backoff.len().saturating_sub(1));
    if let Some(d) = cfg.backoff.get(idx)
        && !d.is_zero()
    {
        std::thread::sleep(*d);
    }
    if cancel.cancelled() {
        return Err(Error::Cancelled);
    }
    Ok(())
}

/// A blob in the store.
pub struct Fetched {
    /// Its path (`blobs/<sha256>`).
    pub path: PathBuf,
    /// True if it was downloaded and hash-verified in this very call.
    pub verified_now: bool,
}

/// Make sure the blob for `entry` is in the store, downloading it from `origin` if needed.
///
/// A file already complete in the store is used as it is (its hash was verified when it was
/// written; the first-open verification re-checks it). In offline mode nothing is fetched.
///
/// # Errors
/// [`DownloadFailure`] for network, disk and offline problems; an `IncompatibleModel` hash
/// mismatch if the server keeps sending other bytes; [`Error::Cancelled`] on cancellation.
pub fn ensure_blob(
    cfg: &HubConfig,
    store: &BlobStore,
    entry: &FileEntry,
    origin: &FileOrigin,
    transport: &dyn Transport,
    observer: &dyn Observer,
    cancel: &dyn Cancel,
) -> Result<Fetched> {
    let final_path = store.path(&entry.sha256);
    if store.is_complete(&entry.sha256, entry.size) {
        observer.on_event(&Event::Cached {
            file: entry.path.clone(),
        });
        return Ok(Fetched {
            path: final_path,
            verified_now: false,
        });
    }
    if final_path.exists() {
        let size = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        if cfg.offline {
            return Err(Error::IncompatibleModel(Incompat::SizeMismatch {
                path: final_path.display().to_string(),
                expected: entry.size,
                got: size,
            }));
        }
        let _ = std::fs::remove_file(&final_path);
    }
    if cfg.offline {
        return Err(Error::ModelDownload(DownloadFailure::Offline {
            file: entry.path.clone(),
            size: entry.size,
            dir: store.dir().display().to_string(),
        }));
    }
    store
        .ensure_dir()
        .map_err(|e| io_failure(store.dir(), &e))?;
    let _guard = store
        .lock(&entry.sha256)
        .map_err(|e| io_failure(store.dir(), &e))?;
    if store.is_complete(&entry.sha256, entry.size) {
        observer.on_event(&Event::Cached {
            file: entry.path.clone(),
        });
        return Ok(Fetched {
            path: final_path,
            verified_now: false,
        });
    }

    let url = cfg.url(&origin.repo, &origin.revision, &entry.path);
    observer.on_event(&Event::DownloadStart {
        file: entry.path.clone(),
        size: entry.size,
        url: url.clone(),
        destination: final_path.display().to_string(),
    });
    let job = Job {
        cfg,
        entry,
        part: store.part_path(&entry.sha256),
        observer,
        cancel,
    };
    let network = |cause: String| {
        Error::ModelDownload(DownloadFailure::Network {
            file: entry.path.clone(),
            url: url.clone(),
            cause,
        })
    };

    let mut failures = 0usize;
    let mut restarted = false;
    loop {
        if cancel.cancelled() {
            return Err(Error::Cancelled);
        }
        let resume_from = match std::fs::metadata(&job.part) {
            Ok(m) if m.len() < entry.size => m.len(),
            Ok(_) => {
                let _ = std::fs::remove_file(&job.part);
                0
            }
            Err(_) => 0,
        };
        let remaining = entry.size.saturating_sub(resume_from);
        let budget = Duration::from_secs((remaining / cfg.min_bytes_per_second.max(1)).max(60));
        let range = (resume_from > 0).then_some(resume_from);

        let response = match transport.get(&url, range, budget) {
            Ok(r) => r,
            Err(cause) => {
                failures += 1;
                if failures >= cfg.attempts {
                    return Err(network(cause));
                }
                pause(cfg, failures, cancel)?;
                continue;
            }
        };
        match response.status {
            200 | 206 => {}
            404 => {
                return Err(Error::ModelDownload(DownloadFailure::NotFound {
                    file: entry.path.clone(),
                    url: url.clone(),
                    status: 404,
                }));
            }
            s @ (401 | 403) => {
                return Err(Error::ModelDownload(DownloadFailure::Refused {
                    file: entry.path.clone(),
                    url: url.clone(),
                    status: s,
                }));
            }
            s @ 500..=599 => {
                failures += 1;
                if failures >= cfg.attempts {
                    return Err(network(format!("HTTP {s}")));
                }
                pause(cfg, failures, cancel)?;
                continue;
            }
            s => return Err(network(format!("unexpected HTTP status {s}"))),
        }
        let ignored_range = range.is_some() && response.status == 200;
        let resumed = range.is_some() && !ignored_range;

        match job.read_body(response, if resumed { resume_from } else { 0 })? {
            Body::Interrupted { cause, progressed } => {
                failures = if progressed { 0 } else { failures + 1 };
                if failures >= cfg.attempts {
                    return Err(network(cause));
                }
                pause(cfg, failures.max(1), cancel)?;
            }
            Body::Done { hash, total, .. } => {
                if hash == entry.sha256 && total == entry.size {
                    std::fs::rename(&job.part, &final_path)
                        .map_err(|e| io_failure(&final_path, &e))?;
                    observer.on_event(&Event::DownloadDone {
                        file: entry.path.clone(),
                    });
                    return Ok(Fetched {
                        path: final_path,
                        verified_now: true,
                    });
                }
                let _ = std::fs::remove_file(&job.part);
                if resumed && !restarted {
                    restarted = true;
                    continue;
                }
                return Err(mismatch(entry, hash, total));
            }
        }
    }
}
