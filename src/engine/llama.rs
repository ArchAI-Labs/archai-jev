//! The real engine: llama.cpp through the safe wrapper `llama-cpp-2` (D22, spec 006b).
//!
//! This is the only file that touches the C++ engine. The pattern is the one proven in spikes S4
//! and S5: the state is decoded once on sequence 0; each question is decoded on sequence 1, a
//! copy of the state (`kv_cache_seq_cp`, which also copies the recurrent state of Gated DeltaNet
//! layers); numbers are read **by batch index**, never by "row of output" index (that returns
//! `Ok` with other values). A context is created per request and the model is shared, so the
//! one-context-at-a-time rule of llama.cpp holds by construction, and concurrent requests wait.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;

use super::cpu::{self, HostCpu};
use super::{Forward, Output, OutputSpec, Session};
use crate::error::{Error, Result};

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
    lock: Mutex<()>,
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
        Ok(LlamaForward {
            model,
            config,
            lock: Mutex::new(()),
        })
    }

    /// Hidden size of the model.
    pub fn n_embd(&self) -> usize {
        usize::try_from(self.model.n_embd()).unwrap_or(0)
    }

    /// Vocabulary size of the model.
    pub fn n_vocab(&self) -> usize {
        usize::try_from(self.model.n_vocab()).unwrap_or(0)
    }

    fn context_params(&self) -> Result<LlamaContextParams> {
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
            .with_n_threads(self.config.threads)
            .with_n_threads_batch(self.config.threads);
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
    state_len: usize,
    hidden_states: bool,
    _guard: MutexGuard<'a, ()>,
}

fn first_non_finite(values: &[f32]) -> Option<f32> {
    values.iter().copied().find(|v| !v.is_finite())
}

impl LlamaSession<'_> {
    fn read(&self, specs: &[OutputSpec]) -> Result<Vec<Output>> {
        let mut out = Vec::with_capacity(specs.len());
        for spec in specs {
            match spec {
                OutputSpec::Hidden { position } => {
                    let i =
                        i32::try_from(*position).map_err(|_| inference("position too large"))?;
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
                    let i =
                        i32::try_from(*position).map_err(|_| inference("position too large"))?;
                    let row = self.ctx.get_logits_ith(i);
                    if let Some(bad) = first_non_finite(row) {
                        return Err(Error::NonFiniteHidden {
                            question: String::new(),
                            position: *position,
                            value: f64::from(bad),
                        });
                    }
                    let mut picked = Vec::with_capacity(ids.len());
                    for &id in ids {
                        let v = row.get(id as usize).copied().ok_or_else(|| {
                            inference(format!("token id {id} is outside the vocabulary"))
                        })?;
                        picked.push(v);
                    }
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
}

impl Session for LlamaSession<'_> {
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>> {
        let mut wanted = vec![false; branch.len()];
        for spec in outputs {
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
            if wants_hidden != self.hidden_states {
                return Err(inference(if wants_hidden {
                    "this model was loaded to read logits, not hidden states"
                } else {
                    "this model was loaded to read hidden states, not logits"
                }));
            }
            match wanted.get_mut(position) {
                Some(slot) => *slot = true,
                None => {
                    return Err(inference(format!(
                        "output position {position} is outside the branch of {} tokens",
                        branch.len()
                    )));
                }
            }
        }
        if self.state_len > 0 {
            self.ctx
                .kv_cache_seq_cp(0, 1, None, None)
                .map_err(|e| inference(format!("cannot copy the state: {e}")))?;
        }
        let result = decode(&mut self.ctx, branch, self.state_len, 1, &wanted)
            .and_then(|()| self.read(outputs));
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

impl Forward for LlamaForward {
    fn begin<'a>(&'a self, state: &[u32]) -> Result<Box<dyn Session + 'a>> {
        let guard = self
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ctx = self
            .model
            .new_context(backend()?, self.context_params()?)
            .map_err(|e| inference(format!("cannot create the context: {e}")))?;
        if !state.is_empty() {
            let mut wanted = vec![false; state.len()];
            if let Some(last) = wanted.last_mut() {
                *last = true;
            }
            decode(&mut ctx, state, 0, 0, &wanted)?;
        }
        Ok(Box::new(LlamaSession {
            ctx,
            state_len: state.len(),
            hidden_states: self.config.hidden_states,
            _guard: guard,
        }))
    }
}
