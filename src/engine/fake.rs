//! A fake model for tests (feature `testing`).
//!
//! Its numbers are a function of the **whole prefix** of the row (state + branch up to the
//! position read), so if a question ever saw tokens of another question, or the state changed, the
//! numbers would change and a test would notice. It counts what it is asked to do.
//!
//! With [`FakeForward::partition_sensitive`] the numbers also depend on **how the row was split**
//! into decodes (as the real engine's do, by summation order): a test that says "the result is
//! bit-identical whatever the cache did" can then fail, instead of passing for lack of any
//! difference to see.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::{Forward, Output, OutputSpec, Session, Snapshot};
use crate::control::{Control, NoControl};
use crate::error::{Error, Result};
use crate::pool::{PoolStats, Slot, SlotPool};

/// What the fake was asked to do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// Sessions started.
    pub sessions: usize,
    /// Rows run.
    pub rows: usize,
    /// Tokens decoded: state once per session plus every branch (and every `extend`).
    pub decoded_tokens: usize,
    /// Tokens decoded after the prompt to score option keys of more than one token (spec 010).
    pub continuation_tokens: usize,
    /// Output positions read.
    pub outputs: usize,
    /// The full row (state + branch) of every row, in order.
    pub seen_rows: Vec<Vec<u32>>,
    /// The output specs of every row, in order.
    pub seen_outputs: Vec<Vec<OutputSpec>>,
    /// The sizes of the decodes of every row, shared part included, in order.
    pub seen_segments: Vec<Vec<usize>>,
    /// Snapshots taken.
    pub snapshots: usize,
    /// Snapshots restored.
    pub restores: usize,
    /// The most sessions that were alive together.
    pub max_concurrent: usize,
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
    /// Fail every `restore` with an engine error.
    pub fail_restore: bool,
}

/// The fake model.
pub struct FakeForward {
    dim: usize,
    vocab: u32,
    sensitive: bool,
    thread_sensitive: bool,
    /// A snapshot of more tokens than this restores to a session whose numbers differ from a
    /// cold decode (what a sliding attention window does to a saved prefix, spec 010b).
    lossy_over: Option<usize>,
    pool: SlotPool,
    in_flight: AtomicUsize,
    stats: Mutex<Stats>,
    faults: Mutex<Faults>,
}

/// What a snapshot of the fake holds: the shared ids and how they were decoded.
struct FakeState {
    ids: Vec<u32>,
    sizes: Vec<usize>,
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

/// A number that depends on the sizes of the decodes of the row.
fn partition_salt(sizes: &[usize]) -> u64 {
    sizes
        .iter()
        .fold(0x0f0f_f0f0_1234_4321u64, |h, &s| mix(h ^ s as u64))
}

impl FakeForward {
    /// A fake with hidden states of `dim` values.
    pub fn new(dim: usize) -> Self {
        FakeForward {
            dim,
            vocab: 64,
            sensitive: false,
            thread_sensitive: false,
            lossy_over: None,
            pool: SlotPool::new(1, 1),
            in_flight: AtomicUsize::new(0),
            stats: Mutex::new(Stats::default()),
            faults: Mutex::new(Faults::default()),
        }
    }

    /// The size of the vocabulary over which the log-probabilities of
    /// [`OutputSpec::KeyLogProbs`] are normalised (64 by default).
    #[must_use]
    pub fn with_vocab(mut self, vocab: u32) -> Self {
        self.vocab = vocab.max(1);
        self
    }

    /// Make the numbers depend on how each row was split into decodes.
    #[must_use]
    pub fn partition_sensitive(mut self) -> Self {
        self.sensitive = true;
        self
    }

    /// Make a restored snapshot of more than `tokens` tokens give other numbers than a cold
    /// decode of the same prefix, as a model with a sliding attention window does: a test that
    /// says "cold, warm and no cache agree" can then fail.
    #[must_use]
    pub fn lossy_snapshots_over(mut self, tokens: usize) -> Self {
        self.lossy_over = Some(tokens);
        self
    }

    /// Make the numbers depend on the threads of the slot that ran the request (as a real
    /// engine's would if its sums were split differently): a test that says "the result does not
    /// depend on the slot" can then fail.
    #[must_use]
    pub fn thread_sensitive(mut self) -> Self {
        self.thread_sensitive = true;
        self
    }

    /// Run up to `max_concurrency` requests together, with `thread_budget` threads in all
    /// (the rule of spec 008). Without this the fake is a pool of one, as before.
    #[must_use]
    pub fn with_pool(mut self, max_concurrency: usize, thread_budget: usize) -> Self {
        self.pool = SlotPool::new(max_concurrency, thread_budget);
        self
    }

    /// The pool's counters.
    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// The pool, to announce the demand of an `ask_many`.
    pub fn pool(&self) -> &SlotPool {
        &self.pool
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
    /// the expected answer independently of the engine plumbing). Not for a
    /// [`partition_sensitive`](Self::partition_sensitive) fake: see [`Self::hidden_split`].
    pub fn hidden_of(&self, prefix: &[u32]) -> Vec<f32> {
        self.hidden_with(prefix, 0)
    }

    /// The logit this fake returns for `id` at the end of `prefix`.
    pub fn logit_of(&self, prefix: &[u32], id: u32) -> f32 {
        self.logit_with(prefix, id, 0)
    }

    /// The hidden state a partition-sensitive fake returns at the end of `prefix` when the row
    /// was decoded in decodes of `sizes` tokens.
    pub fn hidden_split(&self, prefix: &[u32], sizes: &[usize]) -> Vec<f32> {
        self.hidden_with(prefix, self.salt(sizes))
    }

    /// The log-probability of the token `id` after `prefix`, normalised over the vocabulary of
    /// the fake (ids `0..vocab`), and `salt` the way the other numbers are.
    fn log_prob_with(&self, prefix: &[u32], id: u32, salt: u64) -> f64 {
        let logits: Vec<f64> = (0..self.vocab)
            .map(|t| f64::from(self.logit_with(prefix, t, salt)))
            .collect();
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_z = top + logits.iter().map(|v| (v - top).exp()).sum::<f64>().ln();
        f64::from(self.logit_with(prefix, id, salt)) - log_z
    }

    /// The sum of the log-probabilities of the tokens of `key` as the continuation of `prefix`
    /// (exposed so a test can compute the expected score by hand, token by token).
    pub fn key_log_prob_of(&self, prefix: &[u32], key: &[u32]) -> f64 {
        self.key_log_prob_with(prefix, key, 0)
    }

    fn key_log_prob_with(&self, prefix: &[u32], key: &[u32], salt: u64) -> f64 {
        let mut context = prefix.to_vec();
        let mut total = 0.0;
        for &token in key {
            total += self.log_prob_with(&context, token, salt);
            context.push(token);
        }
        total
    }

    fn salt(&self, sizes: &[usize]) -> u64 {
        if self.sensitive {
            partition_salt(sizes)
        } else {
            0
        }
    }

    fn hidden_with(&self, prefix: &[u32], salt: u64) -> Vec<f32> {
        let h = hash_prefix(prefix) ^ salt;
        (0..self.dim).map(|j| unit(mix(h ^ j as u64))).collect()
    }

    fn logit_with(&self, prefix: &[u32], id: u32, salt: u64) -> f32 {
        unit(mix(hash_prefix(prefix) ^ salt ^ u64::from(id) ^ 0xabcd)) * 4.0
    }
}

struct FakeSession<'a> {
    owner: &'a FakeForward,
    /// The shared ids and the sizes of the decodes that produced them.
    state: Vec<u32>,
    sizes: Vec<usize>,
    /// A lossy snapshot was restored: every number of this session is another one.
    degraded: bool,
    threads: usize,
    _slot: Slot<'a>,
}

impl Drop for FakeSession<'_> {
    fn drop(&mut self) {
        self.owner.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

impl FakeSession<'_> {
    fn run(&mut self, segments: &[&[u32]], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        let faults = self.owner.faults.lock().unwrap().clone();
        let mut full = self.state.clone();
        let mut sizes = self.sizes.clone();
        for s in segments {
            full.extend_from_slice(s);
            sizes.push(s.len());
        }
        let branch_len = full.len() - self.state.len();
        let global_row = {
            let mut s = self.owner.stats.lock().unwrap();
            let idx = s.rows;
            s.rows += 1;
            s.decoded_tokens += branch_len;
            s.outputs += outputs.len();
            s.seen_rows.push(full.clone());
            s.seen_outputs.push(outputs.to_vec());
            s.seen_segments.push(sizes.clone());
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
        let thread_salt = if self.owner.thread_sensitive {
            mix(self.threads as u64 ^ 0x7777)
        } else {
            0
        };
        let degraded_salt = if self.degraded { 0xdead_beef_u64 } else { 0 };
        let salt = self.owner.salt(&sizes) ^ thread_salt ^ degraded_salt;
        let base = self.state.len();
        let mut out = Vec::with_capacity(outputs.len());
        for spec in outputs {
            match spec {
                OutputSpec::Hidden { position } => {
                    let prefix = full.get(..=base + position).unwrap_or(&full);
                    let mut h = self.owner.hidden_with(prefix, salt);
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
                            .map(|&id| self.owner.logit_with(prefix, id, salt))
                            .collect(),
                    ));
                }
                OutputSpec::KeyLogProbs { position, keys } => {
                    if keys.iter().any(Vec::is_empty) {
                        return Err(Error::Inference {
                            detail: "a key without tokens cannot be scored".to_string(),
                        });
                    }
                    if base + position + 1 != full.len() {
                        return Err(Error::Inference {
                            detail: "keys are scored after the last token of the row".to_string(),
                        });
                    }
                    let prefix = full.get(..=base + position).unwrap_or(&full);
                    self.owner.stats.lock().unwrap().continuation_tokens +=
                        keys.iter().map(|k| k.len() - 1).sum::<usize>();
                    out.push(Output::KeyLogProbs(
                        keys.iter()
                            .map(|key| self.owner.key_log_prob_with(prefix, key, salt))
                            .collect(),
                    ));
                }
            }
        }
        Ok(out)
    }
}

impl Session for FakeSession<'_> {
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        self.run(&[branch], outputs)
    }

    fn shared_len(&self) -> usize {
        self.state.len()
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        if self.owner.faults.lock().unwrap().fail_restore {
            return Err(Error::Inference {
                detail: "injected fault: restore".to_string(),
            });
        }
        if !self.state.is_empty() {
            return Err(Error::Inference {
                detail: "a snapshot can only be restored on an empty session".to_string(),
            });
        }
        let saved = snapshot
            .downcast_ref::<FakeState>()
            .ok_or_else(|| Error::Inference {
                detail: "the snapshot was not made by this engine".to_string(),
            })?;
        self.state = saved.ids.clone();
        self.sizes = saved.sizes.clone();
        self.degraded = self.owner.lossy_over.is_some_and(|n| saved.ids.len() > n);
        self.owner.stats.lock().unwrap().restores += 1;
        Ok(())
    }

    fn extend(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        self.state.extend_from_slice(tokens);
        self.sizes.push(tokens.len());
        self.owner.stats.lock().unwrap().decoded_tokens += tokens.len();
        Ok(())
    }

    fn snapshot(&mut self) -> Result<Snapshot> {
        self.owner.stats.lock().unwrap().snapshots += 1;
        let tokens = self.state.len();
        Ok(Snapshot::new(
            tokens,
            tokens * 8,
            FakeState {
                ids: self.state.clone(),
                sizes: self.sizes.clone(),
            },
        ))
    }

    fn row_segments(&mut self, segments: &[&[u32]], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        self.run(segments, outputs)
    }
}

impl Forward for FakeForward {
    fn begin<'a>(&'a self, state: &[u32]) -> Result<Box<dyn Session + 'a>> {
        self.begin_with(state, &NoControl)
    }

    fn begin_empty<'a>(&'a self) -> Result<Box<dyn Session + 'a>> {
        self.begin_empty_with(&NoControl)
    }

    fn begin_with<'a>(
        &'a self,
        state: &[u32],
        control: &dyn Control,
    ) -> Result<Box<dyn Session + 'a>> {
        let mut session = self.begin_empty_with(control)?;
        session.extend(state)?;
        Ok(session)
    }

    fn begin_empty_with<'a>(&'a self, control: &dyn Control) -> Result<Box<dyn Session + 'a>> {
        let slot = self.pool.acquire(control)?;
        slot.note_context_created();
        let inside = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        {
            let mut s = self.stats.lock().unwrap();
            s.sessions += 1;
            s.max_concurrent = s.max_concurrent.max(inside);
        }
        Ok(Box::new(FakeSession {
            owner: self,
            state: Vec::new(),
            sizes: Vec::new(),
            degraded: false,
            threads: slot.threads(),
            _slot: slot,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> Vec<OutputSpec> {
        vec![OutputSpec::Hidden { position: 1 }]
    }

    fn hidden(out: Vec<Output>) -> Vec<f32> {
        match out.into_iter().next() {
            Some(Output::Hidden(h)) => h,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn begin_and_row_behave_as_before() {
        let f = FakeForward::new(4);
        let mut s = f.begin(&[1, 2, 3]).unwrap();
        let got = hidden(s.row(&[4, 5], &spec()).unwrap());
        assert_eq!(got, f.hidden_of(&[1, 2, 3, 4, 5][..=4]));
        let st = f.stats();
        assert_eq!((st.sessions, st.rows, st.decoded_tokens), (1, 1, 5));
        assert_eq!(st.seen_rows, vec![vec![1, 2, 3, 4, 5]]);
    }

    #[test]
    fn restore_then_extend_equals_extending_the_whole_prefix_when_the_split_is_the_same() {
        let f = FakeForward::new(4).partition_sensitive();
        // Cold: the prefix is decoded as [2, 2], then the tail.
        let mut cold = f.begin_empty().unwrap();
        cold.extend(&[1, 2]).unwrap();
        cold.extend(&[3, 4]).unwrap();
        let snap = cold.snapshot().unwrap();
        let cold_out = hidden(cold.row_segments(&[&[5, 6]], &spec()).unwrap());
        drop(cold);
        // Warm: the same prefix restored, the same tail.
        let mut warm = f.begin_empty().unwrap();
        warm.restore(&snap).unwrap();
        assert_eq!(warm.shared_len(), 4);
        let warm_out = hidden(warm.row_segments(&[&[5, 6]], &spec()).unwrap());
        drop(warm);
        assert_eq!(cold_out, warm_out);
        // Restoring decodes nothing: 4 + 2 for the cold session, 2 for the warm one.
        assert_eq!(f.stats().decoded_tokens, 4 + 2 + 2);
        assert_eq!((f.stats().snapshots, f.stats().restores), (1, 1));
        // The same ids split another way give other numbers: the fake sees the partition.
        let mut other = f.begin_empty().unwrap();
        other.extend(&[1, 2, 3, 4]).unwrap();
        let other_out = hidden(other.row_segments(&[&[5, 6]], &spec()).unwrap());
        assert_ne!(cold_out, other_out);
        // A fake that is not sensitive does not tell them apart.
        let plain = FakeForward::new(4);
        let a = {
            let mut s = plain.begin_empty().unwrap();
            s.extend(&[1, 2]).unwrap();
            s.extend(&[3, 4]).unwrap();
            hidden(s.row_segments(&[&[5, 6]], &spec()).unwrap())
        };
        let b = {
            let mut s = plain.begin_empty().unwrap();
            s.extend(&[1, 2, 3, 4]).unwrap();
            hidden(s.row_segments(&[&[5, 6]], &spec()).unwrap())
        };
        assert_eq!(a, b);
    }

    #[test]
    fn a_row_in_segments_is_the_split_of_its_tail_too() {
        let f = FakeForward::new(4).partition_sensitive();
        let mut s = f.begin_empty().unwrap();
        s.extend(&[1, 2]).unwrap();
        let whole = hidden(s.row_segments(&[&[3, 4, 5]], &spec()).unwrap());
        let split = hidden(s.row_segments(&[&[3], &[4, 5]], &spec()).unwrap());
        assert_ne!(whole, split);
        assert_eq!(f.stats().seen_segments, vec![vec![2, 3], vec![2, 1, 2]]);
        // Both rows saw the same ids.
        assert_eq!(f.stats().seen_rows[0], f.stats().seen_rows[1]);
    }

    #[test]
    fn restore_needs_an_empty_session_and_a_snapshot_of_this_engine() {
        let f = FakeForward::new(4);
        let mut s = f.begin_empty().unwrap();
        s.extend(&[1, 2]).unwrap();
        let snap = s.snapshot().unwrap();
        assert!(matches!(s.restore(&snap), Err(Error::Inference { .. })));
        drop(s);

        let mut empty = f.begin_empty().unwrap();
        let foreign = Snapshot::new(2, 16, "another engine");
        let err = empty.restore(&foreign).unwrap_err();
        assert!(err.to_string().contains("not made by this engine"), "{err}");
        // The session is still usable.
        empty.restore(&snap).unwrap();
        assert_eq!(empty.shared_len(), 2);
    }

    #[test]
    fn fail_restore_is_an_engine_error_and_the_session_stays_empty() {
        let f = FakeForward::new(4);
        let mut s = f.begin_empty().unwrap();
        s.extend(&[1, 2]).unwrap();
        let snap = s.snapshot().unwrap();
        drop(s);
        f.set_faults(Faults {
            fail_restore: true,
            ..Faults::default()
        });
        let mut s = f.begin_empty().unwrap();
        assert!(matches!(s.restore(&snap), Err(Error::Inference { .. })));
        assert_eq!(s.shared_len(), 0);
    }

    #[test]
    fn snapshot_reports_its_tokens_and_bytes() {
        let f = FakeForward::new(4);
        let mut s = f.begin_empty().unwrap();
        s.extend(&[1, 2, 3]).unwrap();
        let snap = s.snapshot().unwrap();
        assert_eq!((snap.tokens(), snap.bytes()), (3, 24));
        assert_eq!(snap.clone().tokens(), 3);
    }

    #[test]
    fn the_fake_records_only_ids_and_positions() {
        // No text can reach the engine: everything recorded is `u32` ids, `usize` sizes and specs.
        let f = FakeForward::new(2);
        let mut s = f.begin(&[1]).unwrap();
        s.row(&[2, 3], &spec()).unwrap();
        let st = f.stats();
        let _: &Vec<Vec<u32>> = &st.seen_rows;
        let _: &Vec<Vec<usize>> = &st.seen_segments;
    }
}
