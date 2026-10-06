//! Progress and notices. The core only *emits* events; the Python binding turns them into
//! `logging` records, so the core never touches Python or any global logger.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Something worth telling the user.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A file is already in the cache.
    Cached { file: String },
    /// A download starts: what, how big, from where, to where.
    DownloadStart {
        file: String,
        size: u64,
        url: String,
        destination: String,
    },
    /// A download is under way (rate-limited).
    DownloadProgress { file: String, done: u64, total: u64 },
    /// A download finished and was verified.
    DownloadDone { file: String },
    /// The model is loaded: its license, shown once.
    License {
        name: String,
        spdx: String,
        url: String,
    },
    /// A warning (restrictive license, notice of the model, cache not writable...).
    Warning(String),
}

/// Receives events.
pub trait Observer: Send + Sync {
    /// Called for every event, possibly from the downloading thread.
    fn on_event(&self, event: &Event);
}

/// Ignores everything.
pub struct NoObserver;

impl Observer for NoObserver {
    fn on_event(&self, _event: &Event) {}
}

/// Keeps every event, for tests.
#[derive(Default)]
pub struct Recorder(Mutex<Vec<Event>>);

impl Recorder {
    /// A copy of the events seen so far.
    pub fn events(&self) -> Vec<Event> {
        self.0.lock().map(|v| v.clone()).unwrap_or_default()
    }
}

impl Observer for Recorder {
    fn on_event(&self, event: &Event) {
        if let Ok(mut v) = self.0.lock() {
            v.push(event.clone());
        }
    }
}

/// Asked between blocks of a download: should it stop? (`Ctrl-C` in Python.)
pub trait Cancel: Send + Sync {
    /// True to stop as soon as possible.
    fn cancelled(&self) -> bool;
}

/// Never cancels.
pub struct NeverCancel;

impl Cancel for NeverCancel {
    fn cancelled(&self) -> bool {
        false
    }
}

/// A flag another thread can set to cancel.
#[derive(Default)]
pub struct CancelFlag(AtomicBool);

impl CancelFlag {
    /// Ask for cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Cancel for CancelFlag {
    fn cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
