//! A minimal HTTP/1.1 server on the loopback, written with the standard library only, to test
//! downloads without the real network: ranges, redirects, injected errors, request log.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// One request seen by the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Logged {
    pub path: String,
    pub range_from: Option<u64>,
}

/// A misbehaviour applied to the next matching request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// Answer with this status and an empty body.
    Status(u16),
    /// Send the headers and this many body bytes, then close the connection.
    DropAfter(usize),
    /// Send the right length of bytes but with every byte flipped.
    WrongBytes,
}

#[derive(Default)]
struct State {
    files: HashMap<String, Vec<u8>>,
    log: Vec<Logged>,
    faults: VecDeque<Fault>,
    redirect: bool,
    slow_ms: u64,
}

/// The running server; stops when dropped.
pub struct TestServer {
    port: u16,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl TestServer {
    pub fn start() -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let state = Arc::new(Mutex::new(State::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (st, sp) = (Arc::clone(&state), Arc::clone(&stop));
        let handle = std::thread::spawn(move || {
            for conn in listener.incoming() {
                if sp.load(Ordering::SeqCst) {
                    break;
                }
                if let Ok(stream) = conn {
                    let st = Arc::clone(&st);
                    std::thread::spawn(move || serve(stream, &st));
                }
            }
        });
        TestServer {
            port,
            state,
            stop,
            handle: Some(handle),
        }
    }

    /// `http://127.0.0.1:<port>`
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Serve `bytes` at the URL path `path` (for example `/org/repo/resolve/<rev>/file`).
    pub fn add_file(&self, path: &str, bytes: Vec<u8>) {
        self.state
            .lock()
            .unwrap()
            .files
            .insert(path.to_string(), bytes);
    }

    /// Queue a fault for the next request that reaches a file.
    pub fn push_fault(&self, fault: Fault) {
        self.state.lock().unwrap().faults.push_back(fault);
    }

    /// Answer `/…/resolve/…` with a redirect to a `/cdn/…` URL, like Hugging Face does.
    pub fn redirect(&self, on: bool) {
        self.state.lock().unwrap().redirect = on;
    }

    /// Serve bodies slowly: pause this many milliseconds after every 64 KiB.
    pub fn slow(&self, ms: u64) {
        self.state.lock().unwrap().slow_ms = ms;
    }

    /// Every request so far.
    pub fn log(&self) -> Vec<Logged> {
        self.state.lock().unwrap().log.clone()
    }

    /// Requests whose path contains `/resolve/` (the ones a client starts from).
    pub fn resolve_requests(&self) -> Vec<Logged> {
        self.log()
            .into_iter()
            .filter(|l| l.path.contains("/resolve/"))
            .collect()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn respond(stream: &mut TcpStream, status: &str, headers: &[(&str, String)], body: &[u8]) {
    let mut head = format!("HTTP/1.1 {status}\r\n");
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("Connection: close\r\n\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn serve(mut stream: TcpStream, state: &Mutex<State>) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();
    let mut range_from = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range:") {
            range_from = v
                .trim()
                .strip_prefix("bytes=")
                .and_then(|r| r.split('-').next())
                .and_then(|n| n.parse().ok());
        }
    }

    let (file, fault, redirect, slow_ms) = {
        let mut st = state.lock().unwrap();
        st.log.push(Logged {
            path: path.clone(),
            range_from,
        });
        let cdn = path.strip_prefix("/cdn");
        let key = cdn.map_or(path.clone(), str::to_string);
        let file = st.files.get(&key).cloned();
        let redirect = st.redirect && cdn.is_none() && file.is_some();
        let fault = if file.is_some() && !redirect {
            st.faults.pop_front()
        } else {
            None
        };
        (file, fault, redirect, st.slow_ms)
    };
    let Some(mut bytes) = file else {
        respond(
            &mut stream,
            "404 Not Found",
            &[("Content-Length", "0".to_string())],
            b"",
        );
        return;
    };
    if redirect {
        respond(
            &mut stream,
            "302 Found",
            &[
                ("Location", format!("/cdn{path}")),
                ("Content-Length", "0".to_string()),
            ],
            b"",
        );
        return;
    }
    match &fault {
        Some(Fault::Status(code)) => {
            respond(
                &mut stream,
                &format!("{code} Error"),
                &[("Content-Length", "0".to_string())],
                b"",
            );
            return;
        }
        Some(Fault::WrongBytes) => bytes.iter_mut().for_each(|b| *b ^= 0xFF),
        _ => {}
    }
    let total = bytes.len();
    let start = range_from
        .map_or(0, |r| usize::try_from(r).unwrap_or(total))
        .min(total);
    let body = bytes.get(start..).unwrap_or_default();
    let (status, mut headers) = if range_from.is_some() {
        (
            "206 Partial Content",
            vec![(
                "Content-Range",
                format!("bytes {start}-{}/{total}", total.saturating_sub(1)),
            )],
        )
    } else {
        ("200 OK", vec![])
    };
    headers.push(("Content-Length", body.len().to_string()));
    headers.push(("Accept-Ranges", "bytes".to_string()));
    if let Some(Fault::DropAfter(k)) = fault {
        respond(
            &mut stream,
            status,
            &headers,
            body.get(..k.min(body.len())).unwrap_or_default(),
        );
        return; // dropping the stream closes the connection before the promised length
    }
    if slow_ms == 0 {
        respond(&mut stream, status, &headers, body);
        return;
    }
    respond(&mut stream, status, &headers, b"");
    for piece in body.chunks(64 * 1024) {
        if stream.write_all(piece).is_err() {
            return;
        }
        let _ = stream.flush();
        std::thread::sleep(std::time::Duration::from_millis(slow_ms));
    }
}
