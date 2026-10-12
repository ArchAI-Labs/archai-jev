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
use std::sync::OnceLock;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;

use super::cpu::{self, HostCpu};
use super::{Forward, Output, OutputSpec, Session, Snapshot};
use crate::control::{Control, NoControl};
use crate::error::{Error, Result};
use crate::pool::{PoolStats, Slot, SlotPool};

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
}

/// A GGUF model loaded in llama.cpp.
pub struct LlamaForward {
    model: LlamaModel,
    config: LlamaConfig,
    pool: SlotPool,
}

impl LlamaForward {
    /// Load the GGUF at `path` (already checked by model loading).
    ///
    /// # Errors
    /// [`Error::Inference`] if the CPU lacks AVX2/FMA/F16C/BMI2 or llama.cpp cannot load the file.
    pub fn load(path: &Path, config: LlamaConfig) -> Result<Self> {
        cpu::check(&HostCpu)?;
        let model = LlamaModel::load_from_file(backend()?, path, &LlamaModelParams::default())
            .map_err(|e| inference(format!("cannot load {}: {e}", path.display())))?;
        let budget = usize::try_from(config.threads).unwrap_or(1);
        Ok(LlamaForward {
            model,
            config,
            pool: SlotPool::new(1, budget),
        })
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
        format!(
            "llama-cpp-2=0.1.157; n_ctx={}x2; n_batch={}; n_ubatch={}; n_seq_max=2; embeddings={}; kv=library default",
            self.config.max_context,
            self.config.max_context,
            self.config.max_context,
            self.config.hidden_states
        )
    }

    fn context_params(&self, threads: i32) -> Result<LlamaContextParams> {
        let rows = 2u32; // sequence 0 holds the state, sequence 1 the question being run
        let n_ctx = self
            .config
            .max_context
            .checked_mul(rows)
            .and_then(NonZeroU32::new)
            .ok_or_else(|| inference("max_context is zero or too large"))?;
        let mut params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(self.config.max_context)
            .with_n_ubatch(self.config.max_context)
            .with_n_seq_max(rows)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        if self.config.hidden_states {
            params = params
                .with_embeddings(true)
                .with_pooling_type(LlamaPoolingType::None);
        }
        Ok(params)
    }
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
    /// Tokens on sequence 0, the part every question starts from.
    shared_len: usize,
    hidden_states: bool,
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
}

impl LlamaSession<'_> {
    fn read(&self, reading: &Reading<'_>) -> Result<Vec<Output>> {
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

    /// The position of each spec, checking the model was loaded for that kind of output.
    fn positions(&self, outputs: &[OutputSpec]) -> Result<Vec<usize>> {
        outputs
            .iter()
            .map(|spec| {
                let (position, wants_hidden) = match spec {
                    OutputSpec::Hidden { position } => (*position, true),
                    OutputSpec::Logits { position, .. } => (*position, false),
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
        let saved = snapshot
            .downcast_ref::<SeqState>()
            .ok_or_else(|| inference("the snapshot was not made by this engine"))?;
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
        Ok(Snapshot::new(self.shared_len, saved.byte_len(), saved))
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
            shared_len: 0,
            hidden_states: self.config.hidden_states,
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
