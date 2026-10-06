//! A fake model for tests (feature `testing`).
//!
//! Its numbers are a function of the **whole prefix** of the row (state + branch up to the
//! position read), so if a question ever saw tokens of another question, or the state changed, the
//! numbers would change and a test would notice. It counts what it is asked to do.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use super::{Forward, Output, OutputSpec, Session};
use crate::error::{Error, Result};

/// What the fake was asked to do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// Sessions started.
    pub sessions: usize,
    /// Rows run.
    pub rows: usize,
    /// Tokens decoded: state once per session plus every branch.
    pub decoded_tokens: usize,
    /// Output positions read.
    pub outputs: usize,
    /// The full row (state + branch) of every row, in order.
    pub seen_rows: Vec<Vec<u32>>,
    /// The output specs of every row, in order.
    pub seen_outputs: Vec<Vec<OutputSpec>>,
}

/// A fault to inject.
#[derive(Debug, Clone, Default)]
pub struct Faults {
    /// Fail the row with this global index (0-based, across sessions) with an engine error.
    pub fail_row: Option<usize>,
    /// Return NaN in the first value of every hidden state / logit of this global row.
    pub nan_row: Option<usize>,
    /// Pause this long in every row (to test that the GIL is released).
    pub latency: Duration,
}

/// The fake model.
pub struct FakeForward {
    dim: usize,
    lock: Mutex<()>,
    stats: Mutex<Stats>,
    faults: Mutex<Faults>,
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn hash_prefix(ids: &[u32]) -> u64 {
    ids.iter()
        .fold(0x1234_5678_9abc_def0u64, |h, &i| mix(h ^ u64::from(i)))
}

fn unit(h: u64) -> f32 {
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

impl FakeForward {
    /// A fake with hidden states of `dim` values.
    pub fn new(dim: usize) -> Self {
        FakeForward {
            dim,
            lock: Mutex::new(()),
            stats: Mutex::new(Stats::default()),
            faults: Mutex::new(Faults::default()),
        }
    }

    /// Replace the faults.
    pub fn set_faults(&self, faults: Faults) {
        *self.faults.lock().unwrap() = faults;
    }

    /// A copy of the counters.
    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }

    /// Forget the counters (not the faults).
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = Stats::default();
    }

    /// The hidden state this fake returns at the end of `prefix` (exposed so a test can compute
    /// the expected answer independently of the engine plumbing).
    pub fn hidden_of(&self, prefix: &[u32]) -> Vec<f32> {
        let h = hash_prefix(prefix);
        (0..self.dim).map(|j| unit(mix(h ^ j as u64))).collect()
    }

    /// The logit this fake returns for `id` at the end of `prefix`.
    pub fn logit_of(&self, prefix: &[u32], id: u32) -> f32 {
        unit(mix(hash_prefix(prefix) ^ u64::from(id) ^ 0xabcd)) * 4.0
    }
}

struct FakeSession<'a> {
    owner: &'a FakeForward,
    state: Vec<u32>,
    _guard: MutexGuard<'a, ()>,
}

impl Session for FakeSession<'_> {
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        let faults = self.owner.faults.lock().unwrap().clone();
        let global_row = {
            let mut s = self.owner.stats.lock().unwrap();
            let idx = s.rows;
            s.rows += 1;
            s.decoded_tokens += branch.len();
            s.outputs += outputs.len();
            let mut row = self.state.clone();
            row.extend_from_slice(branch);
            s.seen_rows.push(row);
            s.seen_outputs.push(outputs.to_vec());
            idx
        };
        if !faults.latency.is_zero() {
            std::thread::sleep(faults.latency);
        }
        if faults.fail_row == Some(global_row) {
            return Err(Error::Inference {
                detail: "injected fault".to_string(),
            });
        }
        let mut full = self.state.clone();
        full.extend_from_slice(branch);
        let base = self.state.len();
        let mut out = Vec::with_capacity(outputs.len());
        for spec in outputs {
            match spec {
                OutputSpec::Hidden { position } => {
                    let prefix = full.get(..=base + position).unwrap_or(&full);
                    let mut h = self.owner.hidden_of(prefix);
                    if faults.nan_row == Some(global_row) {
                        h[0] = f32::NAN;
                        return Err(Error::NonFiniteHidden {
                            question: String::new(),
                            position: *position,
                            value: f64::NAN,
                        });
                    }
                    out.push(Output::Hidden(h));
                }
                OutputSpec::Logits { position, ids } => {
                    let prefix = full.get(..=base + position).unwrap_or(&full);
                    out.push(Output::Logits(
                        ids.iter()
                            .map(|&id| self.owner.logit_of(prefix, id))
                            .collect(),
                    ));
                }
            }
        }
        Ok(out)
    }
}

impl Forward for FakeForward {
    fn begin<'a>(&'a self, state: &[u32]) -> Result<Box<dyn Session + 'a>> {
        let guard = self.lock.lock().unwrap();
        {
            let mut s = self.stats.lock().unwrap();
            s.sessions += 1;
            s.decoded_tokens += state.len();
        }
        Ok(Box::new(FakeSession {
            owner: self,
            state: state.to_vec(),
            _guard: guard,
        }))
    }
}
