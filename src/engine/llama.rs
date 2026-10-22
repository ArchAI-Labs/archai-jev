//! The real engine: llama.cpp through the safe wrapper `llama-cpp-2` (D22, spec 006b).
//!
//! This is the only file that touches the C++ engine. The pattern is the one proven in spikes S4
//! and S5: the state is decoded once on sequence 0; each question is decoded on sequence 1, a
//! copy of the state (`kv_cache_seq_cp`, which also copies the recurrent state of Gated DeltaNet
//! layers); numbers are read **by batch index**, never by "row of output" index (that returns
//! `Ok` with other values). A context is created per request and the model is shared. A context
//! is never used by two requests, and the requests that may run together are the slots of a
//! [`SlotPool`] (spec 008): one slot by default, so concurrent requests wait, like in 0.2.0.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use super::cpu::HostCpu;
use super::device::OffloadPlan;
use super::waves::plan_waves;
use super::{Forward, Output, OutputSpec, Session, Snapshot};
use crate::control::{Control, NoControl};
use crate::error::{Error, Result};
use crate::pool::{PoolStats, Slot, SlotPool};

mod device;
pub use device::{LlamaProbe, compiled_backends, cpu_plan, model_params, prepare_model};

fn inference(detail: impl Into<String>) -> Error {
    Error::Inference {
        detail: detail.into(),
    }
}

/// llama.cpp can be initialized once per process.
fn backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // before init: starting a GPU backend prints a banner that `void_logs` is too late for
            device::quiet_logs();
            LlamaBackend::init()
                .map(|mut b| {
                    b.void_logs();
                    b
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| inference(format!("cannot start llama.cpp: {e}")))
}

/// How to run the model.
#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    /// Threads for decoding.
    pub threads: i32,
    /// Most tokens in one row (state + branch): the `max_context` of the manifest.
    pub max_context: u32,
    /// Read hidden states (a pointer head) instead of logits (a letters head).
    pub hidden_states: bool,
    /// Where the model runs: the layers, the one device, and whether the scheduler may move
    /// operations to a GPU by itself (spec 011). Never left to the defaults of llama.cpp.
    pub plan: OffloadPlan,
}

/// A GGUF model loaded in llama.cpp.
pub struct LlamaForward {
    model: LlamaModel,
    config: LlamaConfig,
    pool: SlotPool,
    /// The device of the plan as the engine reports it, for the identity of a saved prefix.
    device_label: String,
    /// `engine_fingerprint`, kept: every snapshot carries it and only restores into the same.
    engine: Arc<str>,
    /// Longest option key the model scores (spec 010); `0` = the model reads no keys.
    key_tokens: u32,
    /// How many keys continue together in one decode (one llama.cpp sequence each); `1` = one
    /// key at a time, the reference path.
    key_wave: u32,
}

/// Keys that continue together by default. A decode of a few tokens costs about as much as a
/// decode of many (the weights are read once either way), so a request with eight keys of several
/// tokens pays two decodes instead of eight (spec 010, P2). Each slot of a wave is one more
/// sequence in every context, and a context is built for each request: the KV cache of the extra
/// sequences costs time even for keys of one token. Measured on the v2 (context 512, Q8_0): a wave
/// of 4 gives within 3 % of what 8 gives and costs half the allocation.
pub const KEY_WAVE: u32 = 4;

impl LlamaForward {
    /// Load the GGUF at `path` (already checked by model loading).
    ///
    /// # Errors
    /// [`Error::Inference`] if the CPU lacks AVX2/FMA/F16C/BMI2 (whatever the device), the plan names
    /// a device the engine does not have, or llama.cpp cannot load the file.
    pub fn load(path: &Path, config: LlamaConfig) -> Result<Self> {
        let params = prepare_model(&config.plan, &HostCpu)?;
        let device_label = device::device_label(&config.plan)?;
        let model = LlamaModel::load_from_file(backend()?, path, &params)
            .map_err(|e| inference(format!("cannot load {}: {e}", path.display())))?;
        let budget = usize::try_from(config.threads).unwrap_or(1);
        let mut forward = LlamaForward {
            model,
            config,
            pool: SlotPool::new(1, budget),
            device_label,
            engine: Arc::from(""),
            key_tokens: 0,
            key_wave: 1,
        };
        forward.engine = Arc::from(forward.engine_fingerprint());
        Ok(forward)
    }

    /// Let the model score option keys of up to `max_key_tokens` tokens
    /// ([`OutputSpec::KeyLogProbs`], spec 010). It costs a third sequence in each context, where
    /// the continuation of a key is decoded on a copy of the question.
    #[must_use]
    pub fn with_key_scoring(mut self, max_key_tokens: u32) -> Self {
        self.key_tokens = max_key_tokens;
        self.key_wave = KEY_WAVE;
        self.engine = Arc::from(self.engine_fingerprint());
        self
    }

    /// How many keys of a request continue in the same decode (at least 1). `1` is the reference
    /// path: one key at a time on one extra sequence, as before waves existed. Only before the
    /// first request: it changes the contexts that are created.
    #[must_use]
    pub fn with_key_wave(mut self, keys: u32) -> Self {
        self.key_wave = keys.max(1);
        self.engine = Arc::from(self.engine_fingerprint());
        self
    }

    /// Sequences of a context: the state, the question and one per key of a wave.
    fn sequences(&self) -> u32 {
        if self.key_tokens > 0 {
            2 + self.key_wave
        } else {
            2
        }
    }

    /// Let up to `k` requests run together (spec 008). The default is 1: requests wait their
    /// turn, as in 0.2.0. With more, the pool opens only when requests queue up (see
    /// [`crate::pool::policy`]).
    #[must_use]
    pub fn with_max_concurrency(mut self, k: usize) -> Self {
        let budget = usize::try_from(self.config.threads).unwrap_or(1);
        self.pool = SlotPool::new(k, budget);
        self
    }

    /// The slots of the model: counters, and the demand of an `ask_many`.
    pub fn pool(&self) -> &SlotPool {
        &self.pool
    }

    /// The counters of the pool.
    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Hidden size of the model.
    pub fn n_embd(&self) -> usize {
        usize::try_from(self.model.n_embd()).unwrap_or(0)
    }

    /// Vocabulary size of the model.
    pub fn n_vocab(&self) -> usize {
        usize::try_from(self.model.n_vocab()).unwrap_or(0)
    }

    /// The engine and the settings of the contexts it creates, as a string for the identity of a
    /// saved prefix: built from what is really set, so changing a parameter changes the key.
    pub fn engine_fingerprint(&self) -> String {
        fingerprint_text(
            &self.config,
            self.key_tokens,
            self.sequences(),
            &self.device_label,
        )
    }

    fn context_params(&self, threads: i32) -> Result<LlamaContextParams> {
        context_params_for(&self.config, self.key_tokens, self.sequences(), threads)
    }
}

/// The text of the identity of the engine: every setting that changes the numbers or where they
/// are computed, including the device (backend, name, description) and the offload plan. Pure,
/// so a test can show that each of them changes it.
fn fingerprint_text(
    config: &LlamaConfig,
    key_tokens: u32,
    rows: u32,
    device_label: &str,
) -> String {
    let plan = &config.plan;
    let device = format!(
        "device={device_label}; layers={:?}; op_offload={}",
        plan.layers, plan.op_offload
    );
    if key_tokens > 0 {
        return format!(
            "llama-cpp-2=0.1.157; n_ctx=({}+{key_tokens})x{rows}; n_batch={}; n_ubatch={}; n_seq_max={rows}; embeddings={}; kv=library default; {device}",
            config.max_context, config.max_context, config.max_context, config.hidden_states
        );
    }
    format!(
        "llama-cpp-2=0.1.157; n_ctx={}x2; n_batch={}; n_ubatch={}; n_seq_max=2; embeddings={}; kv=library default; {device}",
        config.max_context, config.max_context, config.max_context, config.hidden_states
    )
}

/// The parameters of every context of the model. `rows` is how many sequences a context holds:
/// the state, the question and, for a head that scores keys, one per key of a wave. Pure.
///
/// `op_offload` comes from the plan, never from the default (`true`).
fn context_params_for(
    config: &LlamaConfig,
    key_tokens: u32,
    rows: u32,
    threads: i32,
) -> Result<LlamaContextParams> {
    // sequence 0 holds the state, sequence 1 the question being run and, if the model scores
    // option keys, sequences 2.. the continuation of the keys of one wave
    let n_ctx = config
        .max_context
        .saturating_add(key_tokens)
        .checked_mul(rows)
        .and_then(NonZeroU32::new)
        .ok_or_else(|| inference("max_context is zero or too large"))?;
    let mut params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(config.max_context)
        .with_n_ubatch(config.max_context)
        .with_n_seq_max(rows)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_op_offload(config.plan.op_offload);
    if config.hidden_states {
        params = params
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::None);
    }
    Ok(params)
}

/// What a snapshot holds: the saved sequence and the identity of the engine that made it. A
/// snapshot only restores into a context with the same identity, so one made on a GPU never goes
/// into a context on the CPU (or on another GPU): the bytes would be those of another device.
struct Saved<T> {
    engine: Arc<str>,
    state: T,
}

/// The saved state inside `snapshot`, if it is a `T` made by an engine with the identity `engine`.
///
/// # Errors
/// [`Error::Inference`] for a snapshot of another type (another engine) or of another
/// configuration of this one (another device, other context settings).
fn open<'s, T: std::any::Any>(snapshot: &'s Snapshot, engine: &str) -> Result<&'s T> {
    let saved = snapshot
        .downcast_ref::<Saved<T>>()
        .ok_or_else(|| inference("the snapshot was not made by this engine"))?;
    if *saved.engine != *engine {
        return Err(inference(
            "the snapshot was made by another engine configuration (another device or other \
             context settings) and cannot be restored here",
        ));
    }
    Ok(&saved.state)
}

/// Decode `tokens` on sequence `seq` from position `start`; outputs only where `wanted` says.
fn decode(
    ctx: &mut LlamaContext<'_>,
    tokens: &[u32],
    start: usize,
    seq: i32,
    wanted: &[bool],
) -> Result<()> {
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    for (i, &t) in tokens.iter().enumerate() {
        let id = i32::try_from(t).map_err(|_| inference(format!("token id {t} is too large")))?;
        let pos = i32::try_from(start + i).map_err(|_| inference("position is too large"))?;
        let output = wanted.get(i).copied().unwrap_or(false);
        batch
            .add(LlamaToken(id), pos, &[seq], output)
            .map_err(|e| inference(format!("cannot build the batch: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| inference(format!("decoding failed: {e}")))
}

struct LlamaSession<'a> {
    ctx: LlamaContext<'a>,
    /// The engine fingerprint of the model this session belongs to.
    engine: Arc<str>,
    /// Tokens on sequence 0, the part every question starts from.
    shared_len: usize,
    hidden_states: bool,
    /// Longest option key this session scores (0 = none).
    key_tokens: u32,
    /// Keys that continue together in one decode (at least 1).
    key_wave: usize,
    /// Most tokens one decode takes (`n_batch` of the context).
    max_batch: usize,
    /// Last on purpose: the context is freed before the slot is given back.
    _slot: Slot<'a>,
}

fn first_non_finite(values: &[f32]) -> Option<f32> {
    values.iter().copied().find(|v| !v.is_finite())
}

/// What to read after the last decode of a row. Numbers are read **by batch index** (spike S4):
/// a position of the spec minus where the last decode's batch starts.
struct Reading<'s> {
    specs: &'s [OutputSpec],
    /// Where the last decode's batch starts, counted like the positions of the specs.
    last_start: usize,
    /// How many tokens the row has after the shared part (the last one is at `row_end - 1`).
    row_end: usize,
}

impl LlamaSession<'_> {
    fn read(&mut self, reading: &Reading<'_>) -> Result<Vec<Output>> {
        let mut out = Vec::with_capacity(reading.specs.len());
        for spec in reading.specs {
            match spec {
                OutputSpec::Hidden { position } => {
                    let i = batch_index(*position, reading.last_start)?;
                    let h = self.ctx.embeddings_ith(i).map_err(|e| {
                        inference(format!("cannot read the hidden state at {position}: {e}"))
                    })?;
                    if let Some(bad) = first_non_finite(h) {
                        return Err(Error::NonFiniteHidden {
                            question: String::new(),
                            position: *position,
                            value: f64::from(bad),
                        });
                    }
                    out.push(Output::Hidden(h.to_vec()));
                }
                OutputSpec::Logits { position, ids } => {
                    let i = batch_index(*position, reading.last_start)?;
                    let row = self.ctx.get_logits_ith(i);
                    let picked = super::pick_logits(row, ids, *position)?;
                    out.push(Output::Logits(picked));
                }
                OutputSpec::KeyLogProbs { position, keys } => {
                    out.push(Output::KeyLogProbs(
                        self.read_keys(*position, keys, reading)?,
                    ));
                }
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(inference(
                        "this kind of output is not supported by the engine",
                    ));
                }
            }
        }
        Ok(out)
    }

    /// The sum of the log-probabilities of each key as the continuation of the row (spec 010).
    ///
    /// The first token of every key is scored at the last token of the row, whose distribution
    /// the last decode left in the logits. A longer key goes on, token by token, on a copy of the
    /// row that is discarded; the keys of a wave go on together, each on its own copy (sequences
    /// 2..), in one decode. The keys never see each other, which gives the same numbers as the
    /// block-diagonal mask of the reference engine.
    fn read_keys(
        &mut self,
        position: usize,
        keys: &[Vec<u32>],
        reading: &Reading<'_>,
    ) -> Result<Vec<f64>> {
        if position + 1 != reading.row_end {
            return Err(inference("keys are scored after the last token of the row"));
        }
        let i = batch_index(position, reading.last_start)?;
        let firsts: Vec<u32> = keys
            .iter()
            .map(|k| {
                k.first()
                    .copied()
                    .ok_or_else(|| inference("a key without tokens cannot be scored"))
            })
            .collect::<Result<_>>()?;
        let mut sums = super::log_probs_of(self.ctx.get_logits_ith(i), &firsts, position)?;
        // positions count from the start of the sequence: the row ends at `last`
        let last = self.shared_len + position;
        // the keys that go on after their first token, in waves: each key of a wave continues on
        // a copy of the row (its own sequence, 2..) and the whole wave is one decode
        for (index, key) in keys.iter().enumerate() {
            if u32::try_from(key.len()).map_or(true, |n| n > self.key_tokens) && key.len() >= 2 {
                return Err(inference(format!(
                    "a key of {} tokens is longer than the {} this model was loaded to score (key {index})",
                    key.len(),
                    self.key_tokens
                )));
            }
        }
        for wave in plan_waves(keys, self.key_wave, self.max_batch) {
            let totals = self.continue_keys(&wave, keys, last)?;
            for (&index, total) in wave.iter().zip(totals) {
                if let Some(sum) = sums.get_mut(index) {
                    *sum += total;
                }
            }
        }
        Ok(sums)
    }

    /// Score the tokens after the first of each key of `wave`, all in one decode: key `w` of the
    /// wave goes on sequence `2 + w`, a copy of the row (sequence 1), from position `last + 1`.
    /// The keys never see each other (separate sequences), which gives the same numbers as the
    /// block-diagonal mask of the reference engine. The sequences are emptied before returning,
    /// whatever happens.
    fn continue_keys(
        &mut self,
        wave: &[usize],
        keys: &[Vec<u32>],
        last: usize,
    ) -> Result<Vec<f64>> {
        let seq_of = |w: usize| i32::try_from(2 + w).map_err(|_| inference("too many keys"));
        let mut copied = 0;
        let result = (|| {
            for w in 0..wave.len() {
                self.ctx
                    .kv_cache_seq_cp(1, seq_of(w)?, None, None)
                    .map_err(|e| inference(format!("cannot copy the question: {e}")))?;
                copied += 1;
            }
            let total: usize = wave
                .iter()
                .map(|&i| keys.get(i).map_or(0, |k| k.len() - 1))
                .sum();
            let mut batch = LlamaBatch::new(total.max(1), 1);
            // where each key's first fed token is in the batch
            let mut starts = Vec::with_capacity(wave.len());
            let mut at = 0usize;
            for (w, &index) in wave.iter().enumerate() {
                let key = keys.get(index).map_or(&[][..], Vec::as_slice);
                starts.push(at);
                for (t, &token) in key.iter().take(key.len() - 1).enumerate() {
                    let id = i32::try_from(token)
                        .map_err(|_| inference(format!("token id {token} is too large")))?;
                    let pos = i32::try_from(last + 1 + t)
                        .map_err(|_| inference("position is too large"))?;
                    batch
                        .add(LlamaToken(id), pos, &[seq_of(w)?], true)
                        .map_err(|e| inference(format!("cannot build the batch: {e}")))?;
                    at += 1;
                }
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| inference(format!("decoding failed: {e}")))?;
            let mut totals = Vec::with_capacity(wave.len());
            for (&index, &first) in wave.iter().zip(&starts) {
                let key = keys.get(index).map_or(&[][..], Vec::as_slice);
                let mut total = 0.0;
                for (t, &next) in key.iter().skip(1).enumerate() {
                    let at =
                        i32::try_from(first + t).map_err(|_| inference("a key is too long"))?;
                    let lp =
                        super::log_probs_of(self.ctx.get_logits_ith(at), &[next], last + 1 + t)?;
                    total += lp.first().copied().unwrap_or(f64::NAN);
                }
                totals.push(total);
            }
            Ok::<Vec<f64>, Error>(totals)
        })();
        let mut cleared: Result<()> = Ok(());
        for w in 0..copied {
            let done = seq_of(w).and_then(|seq| {
                self.ctx
                    .clear_kv_cache_seq(Some(u32::try_from(seq).unwrap_or(0)), None, None)
                    .map(|_| ())
                    .map_err(|e| inference(format!("cannot clear the key: {e}")))
            });
            if cleared.is_ok() {
                cleared = done;
            }
        }
        let totals = result?;
        cleared?;
        Ok(totals)
    }

    /// The position of each spec, checking the model was loaded for that kind of output.
    fn positions(&self, outputs: &[OutputSpec]) -> Result<Vec<usize>> {
        outputs
            .iter()
            .map(|spec| {
                let (position, wants_hidden) = match spec {
                    OutputSpec::Hidden { position } => (*position, true),
                    OutputSpec::Logits { position, .. } => (*position, false),
                    OutputSpec::KeyLogProbs { position, .. } => {
                        if self.key_tokens == 0 {
                            return Err(inference(
                                "this model was not loaded to score option keys",
                            ));
                        }
                        (*position, false)
                    }
                    #[allow(unreachable_patterns)]
                    _ => {
                        return Err(inference(
                            "this kind of output is not supported by the engine",
                        ));
                    }
                };
                if wants_hidden == self.hidden_states {
                    Ok(position)
                } else {
                    Err(inference(if wants_hidden {
                        "this model was loaded to read logits, not hidden states"
                    } else {
                        "this model was loaded to read hidden states, not logits"
                    }))
                }
            })
            .collect()
    }
}

fn batch_index(position: usize, last_start: usize) -> Result<i32> {
    position
        .checked_sub(last_start)
        .and_then(|i| i32::try_from(i).ok())
        .ok_or_else(|| inference("output position is outside the last segment"))
}

impl Session for LlamaSession<'_> {
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        self.row_segments(&[branch], outputs)
    }

    fn shared_len(&self) -> usize {
        self.shared_len
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        if self.shared_len != 0 {
            return Err(inference(
                "a snapshot can only be restored on an empty session",
            ));
        }
        let saved = open::<SeqState>(snapshot, &self.engine)?;
        self.ctx
            .state_seq_set(saved, 0)
            .map_err(|e| inference(format!("cannot restore the saved state: {e}")))?;
        self.shared_len = snapshot.tokens();
        Ok(())
    }

    fn extend(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut wanted = vec![false; tokens.len()];
        if let Some(last) = wanted.last_mut() {
            *last = true;
        }
        decode(&mut self.ctx, tokens, self.shared_len, 0, &wanted)?;
        self.shared_len += tokens.len();
        Ok(())
    }

    fn snapshot(&mut self) -> Result<Snapshot> {
        let saved = self
            .ctx
            .state_seq_get(0, LlamaStateSeqFlags::empty())
            .map_err(|e| inference(format!("cannot save the state: {e}")))?;
        let bytes = saved.byte_len();
        Ok(Snapshot::new(
            self.shared_len,
            bytes,
            Saved {
                engine: Arc::clone(&self.engine),
                state: saved,
            },
        ))
    }

    fn row_segments(&mut self, segments: &[&[u32]], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        let positions = self.positions(outputs)?;
        let total: usize = segments.iter().map(|s| s.len()).sum();
        let last = match segments.last() {
            Some(last) if !last.is_empty() => *last,
            _ => return Err(inference("a row needs a last segment with tokens")),
        };
        let last_start = total - last.len();
        // Outputs come only from the last decode: a position elsewhere is a fault of the caller.
        let mut wanted = vec![false; last.len()];
        for position in positions {
            match position
                .checked_sub(last_start)
                .and_then(|i| wanted.get_mut(i))
            {
                Some(slot) => *slot = true,
                None => {
                    return Err(inference(format!(
                        "output position {position} is outside the last segment ({last_start}..{total})"
                    )));
                }
            }
        }
        if self.shared_len > 0 {
            self.ctx
                .kv_cache_seq_cp(0, 1, None, None)
                .map_err(|e| inference(format!("cannot copy the state: {e}")))?;
        }
        let result = (|| {
            let mut start = self.shared_len;
            for (i, segment) in segments.iter().enumerate() {
                if segment.is_empty() {
                    continue;
                }
                if i + 1 == segments.len() {
                    decode(&mut self.ctx, segment, start, 1, &wanted)?;
                } else {
                    // The last token of a segment has an output, as with the shared part: a decode
                    // with no output at all is not something to rely on.
                    let mut only_last = vec![false; segment.len()];
                    if let Some(slot) = only_last.last_mut() {
                        *slot = true;
                    }
                    decode(&mut self.ctx, segment, start, 1, &only_last)?;
                }
                start += segment.len();
            }
            self.read(&Reading {
                specs: outputs,
                last_start,
                row_end: total,
            })
        })();
        // The copy is discarded whatever happened: the next question starts from the state alone.
        let cleared = self
            .ctx
            .clear_kv_cache_seq(Some(1), None, None)
            .map_err(|e| inference(format!("cannot clear the question: {e}")));
        let numbers = result?;
        cleared?;
        Ok(numbers)
    }
}

impl LlamaForward {
    fn start<'a>(&'a self, control: &dyn Control) -> Result<LlamaSession<'a>> {
        let slot = self.pool.acquire(control)?;
        let threads = i32::try_from(slot.threads())
            .unwrap_or(self.config.threads)
            .max(1);
        let ctx = self
            .model
            .new_context(backend()?, self.context_params(threads)?)
            .map_err(|e| inference(format!("cannot create the context: {e}")))?;
        slot.note_context_created();
        Ok(LlamaSession {
            ctx,
            engine: Arc::clone(&self.engine),
            shared_len: 0,
            hidden_states: self.config.hidden_states,
            key_tokens: self.key_tokens,
            key_wave: usize::try_from(self.key_wave.max(1)).unwrap_or(1),
            max_batch: usize::try_from(self.config.max_context).unwrap_or(usize::MAX),
            _slot: slot,
        })
    }
}

impl Forward for LlamaForward {
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
        let mut session = self.start(control)?;
        session.extend(state)?;
        Ok(Box::new(session))
    }

    fn begin_empty_with<'a>(&'a self, control: &dyn Control) -> Result<Box<dyn Session + 'a>> {
        Ok(Box::new(self.start(control)?))
    }
}

#[cfg(test)]
mod tests;
