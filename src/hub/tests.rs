//! Download tests against a local HTTP server (feature `testing`): no real network.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::blobs::BlobStore;
use super::config::HubConfig;
use super::download::{Fetched, UreqTransport, ensure_blob};
use super::events::{Cancel, CancelFlag, Event, NeverCancel, NoObserver, Observer, Recorder};
use crate::error::{Error, Result};
use crate::models::failures::DownloadFailure;
use crate::models::files::{FileEntry, FileOrigin};
use crate::models::hash::sha256_hex;
use crate::models::incompat::Incompat;
use crate::models::testing::server::{Fault, TestServer};

const REPO: &str = "org/repo";
const REV: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn bytes(n: usize) -> Vec<u8> {
    let mut x: u64 = 0x1234_5678_9abc_def0;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

struct Rig {
    server: TestServer,
    _dir: tempfile::TempDir,
    cfg: HubConfig,
    store: BlobStore,
    entry: FileEntry,
    origin: FileOrigin,
    data: Vec<u8>,
}

fn rig(size: usize) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let server = TestServer::start();
    let data = bytes(size);
    server.add_file(&format!("/{REPO}/resolve/{REV}/model.bin"), data.clone());
    let mut cfg = HubConfig::new(dir.path().to_path_buf());
    cfg.endpoint = server.url();
    cfg.backoff = vec![Duration::ZERO];
    cfg.progress_interval = Duration::ZERO;
    let store = BlobStore::new(dir.path().join("v1").join("blobs"));
    let entry = FileEntry {
        path: "model.bin".to_string(),
        size: size as u64,
        sha256: sha256_hex(&data),
        origin: None,
    };
    let origin = FileOrigin {
        repo: REPO.to_string(),
        revision: REV.to_string(),
    };
    Rig {
        server,
        _dir: dir,
        cfg,
        store,
        entry,
        origin,
        data,
    }
}

impl Rig {
    fn fetch(&self, observer: &dyn Observer, cancel: &dyn Cancel) -> Result<Fetched> {
        let transport = UreqTransport::new(&self.cfg);
        ensure_blob(
            &self.cfg,
            &self.store,
            &self.entry,
            &self.origin,
            &transport,
            observer,
            cancel,
        )
    }
    fn get(&self) -> Result<Fetched> {
        self.fetch(&NoObserver, &NeverCancel)
    }
    fn part(&self) -> std::path::PathBuf {
        self.store.part_path(&self.entry.sha256)
    }
    fn final_path(&self) -> std::path::PathBuf {
        self.store.path(&self.entry.sha256)
    }
}

fn net_err(r: Result<Fetched>) -> String {
    match r {
        Err(Error::ModelDownload(DownloadFailure::Network { cause, .. })) => cause,
        other => panic!("expected a network error, got {:?}", other.map(|f| f.path)),
    }
}

const MIB: usize = 1 << 20;

#[test]
fn happy_path_then_cache_hit() {
    let r = rig(3 * MIB + 17);
    let got = r.get().unwrap();
    assert!(got.verified_now);
    assert_eq!(std::fs::read(&got.path).unwrap(), r.data);
    assert!(!r.part().exists());
    assert_eq!(r.server.resolve_requests().len(), 1);
    // the URL scheme is exactly /{repo}/resolve/{revision}/{path}
    assert_eq!(
        r.server.log()[0].path,
        format!("/{REPO}/resolve/{REV}/model.bin")
    );
    let again = r.get().unwrap();
    assert!(!again.verified_now);
    assert_eq!(r.server.log().len(), 1, "a cache hit makes no request");
}

#[test]
fn a_dropped_connection_leaves_only_a_part_and_the_next_call_resumes() {
    let mut r = rig(3 * MIB + 17);
    r.cfg.attempts = 1;
    let cut = MIB + 5;
    // the connection drops after `cut` bytes, then the server starts failing
    r.server.push_fault(Fault::DropAfter(cut));
    r.server.push_fault(Fault::Status(503));
    net_err(r.get());
    assert!(!r.final_path().exists(), "no file under the final name");
    assert_eq!(std::fs::metadata(r.part()).unwrap().len(), cut as u64);
    // next call: resumes with a Range request and transfers only the rest
    r.get().unwrap();
    let log = r.server.log();
    assert_eq!(log.last().unwrap().range_from, Some(cut as u64));
    assert_eq!(std::fs::read(r.final_path()).unwrap(), r.data);
    assert!(!r.part().exists());
}

#[test]
fn retries_within_one_call_keep_the_progress() {
    let mut r = rig(3 * MIB);
    r.cfg.attempts = 2;
    // two interruptions with progress in each do not use up the attempts
    r.server.push_fault(Fault::DropAfter(MIB));
    r.server.push_fault(Fault::DropAfter(MIB));
    r.get().unwrap();
    let ranges: Vec<Option<u64>> = r.server.log().iter().map(|l| l.range_from).collect();
    assert_eq!(ranges, vec![None, Some(MIB as u64), Some(2 * MIB as u64)]);
    assert_eq!(std::fs::read(r.final_path()).unwrap(), r.data);
}

#[test]
fn a_redirect_is_followed_and_every_attempt_restarts_from_resolve() {
    let mut r = rig(2 * MIB);
    r.server.redirect(true);
    r.cfg.attempts = 3;
    r.server.push_fault(Fault::DropAfter(MIB / 2));
    r.get().unwrap();
    let log = r.server.log();
    let resolves = log.iter().filter(|l| !l.path.starts_with("/cdn")).count();
    let cdn = log.iter().filter(|l| l.path.starts_with("/cdn")).count();
    assert_eq!((resolves, cdn), (2, 2), "{log:?}");
    assert_eq!(std::fs::read(r.final_path()).unwrap(), r.data);
}

#[test]
fn a_corrupt_part_restarts_from_zero_once_and_succeeds() {
    let r = rig(MIB + 3);
    r.store.ensure_dir().unwrap();
    std::fs::write(r.part(), vec![7u8; 1000]).unwrap();
    r.get().unwrap();
    let ranges: Vec<Option<u64>> = r.server.log().iter().map(|l| l.range_from).collect();
    assert_eq!(ranges, vec![Some(1000), None]);
    assert_eq!(std::fs::read(r.final_path()).unwrap(), r.data);
}

#[test]
fn a_server_that_always_sends_wrong_bytes_ends_in_a_hash_error() {
    let r = rig(MIB + 3);
    r.store.ensure_dir().unwrap();
    std::fs::write(r.part(), vec![7u8; 1000]).unwrap();
    r.server.push_fault(Fault::WrongBytes);
    r.server.push_fault(Fault::WrongBytes);
    let err = r.get().err().unwrap();
    let Error::IncompatibleModel(i @ Incompat::HashMismatch { .. }) = err else {
        panic!("{err:?}")
    };
    assert!(i.to_string().contains("delete it and retry"));
    assert_eq!(r.server.log().len(), 2, "exactly two attempts");
    assert!(!r.part().exists() && !r.final_path().exists());
    // without a part there is one attempt only
    let r = rig(MIB);
    r.server.push_fault(Fault::WrongBytes);
    assert!(matches!(
        r.get().err().unwrap(),
        Error::IncompatibleModel(Incompat::HashMismatch { .. })
    ));
    assert_eq!(r.server.log().len(), 1);
}

#[test]
fn http_errors_are_classified_and_not_retried_when_final() {
    for (status, kind) in [(404u16, "not found"), (401, "refused"), (403, "refused")] {
        let r = rig(1000);
        r.server.push_fault(Fault::Status(status));
        let err = r.get().err().unwrap();
        match (&err, kind) {
            (Error::ModelDownload(DownloadFailure::NotFound { .. }), "not found")
            | (Error::ModelDownload(DownloadFailure::Refused { .. }), "refused") => {}
            _ => panic!("{status}: {err:?}"),
        }
        assert_eq!(r.server.log().len(), 1, "{status} must not be retried");
    }
    // a URL the server does not know is a 404 too
    let mut r = rig(1000);
    r.origin.revision = "c".repeat(40);
    assert!(matches!(
        r.get().err().unwrap(),
        Error::ModelDownload(DownloadFailure::NotFound { .. })
    ));
}

#[test]
fn server_errors_are_retried_then_reported() {
    let mut r = rig(1000);
    r.cfg.attempts = 3;
    r.server.push_fault(Fault::Status(503));
    r.server.push_fault(Fault::Status(503));
    r.get().unwrap();
    assert_eq!(r.server.log().len(), 3);
    let r = rig(1000);
    for _ in 0..3 {
        r.server.push_fault(Fault::Status(500));
    }
    let cause = net_err(r.get());
    assert!(cause.contains("HTTP 500"), "{cause}");
    assert_eq!(r.server.log().len(), 3);
}

#[test]
fn a_refused_connection_is_a_network_error_with_the_cause() {
    let mut r = rig(1000);
    r.cfg.attempts = 2;
    let url = r.server.url();
    drop(std::mem::replace(&mut r.server, TestServer::start()));
    r.cfg.endpoint = url; // the old port is closed now
    let cause = net_err(r.get());
    assert!(!cause.is_empty());
}

#[test]
fn a_full_disk_is_reported_with_the_missing_bytes_and_cleans_up() {
    use std::io::{Error as IoError, ErrorKind};
    let dir = tempfile::tempdir().unwrap();
    let part = dir.path().join("x.part");
    std::fs::write(&part, b"partial").unwrap();
    let entry = FileEntry {
        path: "model.bin".into(),
        size: 1000,
        sha256: "0".repeat(64),
        origin: None,
    };
    let err = super::download::on_write_error_for_tests(
        &part,
        &entry,
        400,
        &IoError::from(ErrorKind::StorageFull),
    );
    let msg = err.to_string();
    assert!(
        msg.contains("not enough disk space") && msg.contains("600 more bytes needed"),
        "{msg}"
    );
    assert!(!part.exists());
    let other = super::download::on_write_error_for_tests(
        &part,
        &entry,
        0,
        &IoError::from(ErrorKind::PermissionDenied),
    );
    assert!(matches!(
        other,
        Error::ModelDownload(DownloadFailure::Io { .. })
    ));
}

#[test]
fn offline_mode_opens_no_socket() {
    let mut r = rig(1000);
    r.cfg.offline = true;
    // missing
    let err = r.get().err().unwrap();
    assert!(
        matches!(err, Error::ModelDownload(DownloadFailure::Offline { .. })),
        "{err:?}"
    );
    assert!(err.to_string().contains("offline mode"));
    // present and complete
    r.store.ensure_dir().unwrap();
    std::fs::write(r.final_path(), &r.data).unwrap();
    let got = r.get().unwrap();
    assert!(!got.verified_now);
    // present but altered (wrong size)
    std::fs::write(r.final_path(), &r.data[..500]).unwrap();
    assert!(matches!(
        r.get().err().unwrap(),
        Error::IncompatibleModel(Incompat::SizeMismatch { .. })
    ));
    assert_eq!(
        r.server.log().len(),
        0,
        "not one connection in offline mode"
    );
}

#[test]
fn two_threads_download_the_same_file_once() {
    let r = Arc::new(rig(3 * MIB));
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let (r, barrier) = (r.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                r.get().map(|f| f.path)
            })
        })
        .collect();
    let paths: Vec<_> = threads
        .into_iter()
        .map(|t| t.join().unwrap().unwrap())
        .collect();
    assert!(paths.iter().all(|p| *p == paths[0]));
    assert_eq!(r.server.resolve_requests().len(), 1, "{:?}", r.server.log());
    assert_eq!(std::fs::read(&paths[0]).unwrap(), r.data);
}

struct CancelAfter(AtomicUsize, usize);
impl Cancel for CancelAfter {
    fn cancelled(&self) -> bool {
        self.0.fetch_add(1, Ordering::SeqCst) + 1 >= self.1
    }
}

#[test]
fn cancelling_keeps_the_part_and_a_later_call_resumes() {
    let r = rig(3 * MIB + 1);
    let err = r
        .fetch(&NoObserver, &CancelAfter(AtomicUsize::new(0), 3))
        .err()
        .unwrap();
    assert!(matches!(err, Error::Cancelled), "{err:?}");
    let kept = std::fs::metadata(r.part()).unwrap().len();
    assert!(kept > 0 && kept < r.entry.size, "{kept}");
    assert!(!r.final_path().exists());
    r.get().unwrap();
    assert_eq!(r.server.log().last().unwrap().range_from, Some(kept));
    assert_eq!(std::fs::read(r.final_path()).unwrap(), r.data);
    // the flag version of cancel
    let flag = CancelFlag::default();
    flag.cancel();
    let r2 = rig(MIB);
    assert!(matches!(
        r2.fetch(&NoObserver, &flag).err().unwrap(),
        Error::Cancelled
    ));
}

#[test]
fn events_are_reported_and_progress_is_rate_limited() {
    let mut r = rig(3 * MIB);
    let rec = Recorder::default();
    r.cfg.progress_interval = Duration::ZERO;
    r.fetch(&rec, &NeverCancel).unwrap();
    let events = rec.events();
    assert!(
        matches!(&events[0], Event::DownloadStart { size, url, .. } if *size == 3 * MIB as u64 && url.contains("/resolve/"))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::DownloadProgress { .. }))
    );
    assert!(matches!(events.last().unwrap(), Event::DownloadDone { .. }));

    let mut r = rig(3 * MIB);
    r.cfg.progress_interval = Duration::from_secs(3600);
    let rec = Recorder::default();
    r.fetch(&rec, &NeverCancel).unwrap();
    assert!(
        !rec.events()
            .iter()
            .any(|e| matches!(e, Event::DownloadProgress { .. }))
    );
    // a second call is a cache hit and says so
    let rec = Recorder::default();
    r.fetch(&rec, &NeverCancel).unwrap();
    assert!(matches!(rec.events()[0], Event::Cached { .. }));
}
