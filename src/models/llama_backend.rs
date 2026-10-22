//! The production backend: llama.cpp, the template of the manifest and its head (spec 006b/006c).
//!
//! `load_model` has already checked the manifest, the files, the GGUF header and the tokenizer;
//! this puts the pieces together and returns the scorer (which answers requests) and the runner
//! (which runs the self-check vectors through the very same code).

use std::sync::Arc;

use super::backend::{Backend, Engine, ValidatedCheckpoint};
use super::families;
use super::head::HeadReader;
use super::incompat::Incompat;
use super::manifest::HeadSpec;
use crate::convert::headpt::HeadPtReader;
use crate::engine::device::OffloadPlan;
use crate::engine::llama::{LlamaConfig, LlamaForward};
use crate::error::{Error, Result};
use crate::heads::{AnswerLettersHead, Head, KeyLogProbHead, LettersHead, PointerHead};
use crate::model_scorer::ModelScorer;
use crate::prefix::{Fingerprint, FingerprintParts};
use crate::prompt::template::{TemplateInputs, instantiate};
use crate::prompt::{PromptTokenizer, Roles};

/// The llama.cpp backend.
pub struct LlamaBackend {
    threads: i32,
    prefix_cache_bytes: u64,
    max_concurrency: usize,
}

impl LlamaBackend {
    /// A backend that decodes with `threads` threads and does not reuse what the questions of a
    /// request share (the behaviour of 0.1.0).
    pub fn new(threads: i32) -> Self {
        LlamaBackend {
            threads,
            prefix_cache_bytes: 0,
            max_concurrency: 1,
        }
    }

    /// Reuse what the questions of a request share, using at most `bytes` of memory for the
    /// cache. `0` turns the reuse off.
    #[must_use]
    pub fn with_prefix_cache(mut self, bytes: u64) -> Self {
        self.prefix_cache_bytes = bytes;
        self
    }

    /// Let up to `k` requests run together (spec 008, D46). `1`, the default, is one request at a
    /// time with every thread, as in 0.2.0.
    #[must_use]
    pub fn with_max_concurrency(mut self, k: usize) -> Self {
        self.max_concurrency = k.max(1);
        self
    }
}

/// Threads to use when the user does not say: all the logical cores (llama.cpp scales to them,
/// spike S4; the default is re-measured with the benchmark of spec 006b).
pub fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .ok()
        .and_then(|n| i32::try_from(n.get()).ok())
        .unwrap_or(4)
}

impl Backend for LlamaBackend {
    fn load(&self, checkpoint: &ValidatedCheckpoint) -> Result<Engine> {
        let m = &checkpoint.manifest;
        let family = families::lookup(&m.family).ok_or_else(|| {
            Error::IncompatibleModel(Incompat::UnknownFamily {
                got: m.family.clone(),
                supported: families::keys(),
            })
        })?;

        let roles = Roles::new(
            m.tokenizer
                .special_tokens
                .iter()
                .map(|(role, token)| (role.clone(), token.id)),
        );
        let tokenizer = PromptTokenizer::from_file(&checkpoint.tokenizer_path, roles)?;

        let (head, hidden_states, max_options, max_key_tokens): (
            Box<dyn Head>,
            bool,
            usize,
            usize,
        ) = match &m.head {
            HeadSpec::Letters {
                choice_targets,
                yes_no,
                ..
            } => (
                Box::new(LettersHead::new(choice_targets, yes_no.as_ref())),
                false,
                choice_targets.len(),
                1,
            ),
            HeadSpec::KeyLogProb {
                max_options,
                max_key_tokens,
                ..
            } => {
                let options = usize::try_from(*max_options).unwrap_or(usize::MAX);
                let tokens = usize::try_from(*max_key_tokens).unwrap_or(usize::MAX);
                (
                    Box::new(KeyLogProbHead::new(options, tokens)),
                    false,
                    options,
                    tokens,
                )
            }
            HeadSpec::AnswerLetters {
                letters,
                max_options,
                ..
            } => {
                let options = usize::try_from(*max_options).unwrap_or(usize::MAX);
                (
                    Box::new(AnswerLettersHead::new(letters, options)),
                    false,
                    options,
                    1,
                )
            }
            HeadSpec::Pointer {
                d_model, proj_dim, ..
            } => {
                let path = checkpoint.head_path.as_ref().ok_or_else(|| {
                    Error::IncompatibleModel(Incompat::HeadWeights {
                        detail: "a pointer head needs its weights file".to_string(),
                    })
                })?;
                let tensors = HeadPtReader.read(path).map_err(Error::IncompatibleModel)?;
                (
                    Box::new(PointerHead::from_tensors(&tensors, *d_model, *proj_dim)?),
                    true,
                    255,
                    1,
                )
            }
        };
        let template = instantiate(
            &m.template.id,
            m.template.version,
            &TemplateInputs {
                tasks: m.tasks.as_deref(),
                max_options,
                max_key_tokens,
                control_tokens: &m.tokenizer.control_tokens,
            },
        )
        .map_err(Error::IncompatibleModel)?;

        let max_context = u32::try_from(m.max_context).map_err(|_| {
            Error::IncompatibleModel(Incompat::ManifestBadValue {
                path: "max_context".to_string(),
                detail: format!("{} does not fit in 32 bits", m.max_context),
            })
        })?;
        let forward = LlamaForward::load(
            &checkpoint.model_path,
            LlamaConfig {
                threads: self.threads,
                max_context,
                hidden_states,
                plan: OffloadPlan::for_device(&checkpoint.device),
            },
        )?
        .with_max_concurrency(self.max_concurrency);
        let forward = match &m.head {
            HeadSpec::KeyLogProb { max_key_tokens, .. } => {
                forward.with_key_scoring(u32::try_from(*max_key_tokens).unwrap_or(u32::MAX))
            }
            _ => forward,
        };

        // a family with a sliding attention window keeps no prefix longer than the window
        let params = family
            .read_params(&m.architecture)
            .map_err(Error::IncompatibleModel)?;
        let snapshot_limit = (family.snapshot_max_tokens)(&params);
        let engine_text = forward.engine_fingerprint();
        let fingerprint = Fingerprint::new(&FingerprintParts {
            model: &checkpoint.manifest_sha256,
            gguf_sha256: &checkpoint.model_sha256,
            family: &m.family,
            dtype: &checkpoint.dtype,
            template_id: &m.template.id,
            template_version: m.template.version,
            tokenizer_sha256: &m.tokenizer.file.sha256,
            engine: &engine_text,
        });
        let scorer = Arc::new(
            ModelScorer::new(
                m.name.clone(),
                template,
                tokenizer,
                Arc::new(forward),
                head,
                checkpoint.calibration.calibration,
                m.max_context,
                family
                    .question_types
                    .iter()
                    .map(|t| (*t).to_string())
                    .collect(),
            )
            .with_fingerprint(fingerprint)
            .with_snapshot_limit(snapshot_limit)
            .with_prefix_cache(self.prefix_cache_bytes),
        );
        Ok(Engine {
            scorer: scorer.clone(),
            runner: scorer,
        })
    }
}
