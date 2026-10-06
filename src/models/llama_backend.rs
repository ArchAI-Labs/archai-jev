//! The production backend: llama.cpp, the template of the manifest and its head (spec 006b/006c).
//!
//! `load_model` has already checked the manifest, the files, the GGUF header and the tokenizer;
//! this puts the pieces together and returns the scorer (which answers requests) and the runner
//! (which runs the self-check vectors through the very same code).

use std::sync::Arc;

use super::backend::{Backend, Engine, ValidatedCheckpoint};
use super::families;
use super::incompat::Incompat;
use super::manifest::HeadSpec;
use crate::engine::llama::{LlamaConfig, LlamaForward};
use crate::error::{Error, Result};
use crate::heads::{Head, LettersHead};
use crate::model_scorer::ModelScorer;
use crate::prompt::template::instantiate;
use crate::prompt::{PromptTokenizer, Roles};

/// The llama.cpp backend.
pub struct LlamaBackend {
    threads: i32,
}

impl LlamaBackend {
    /// A backend that decodes with `threads` threads.
    pub fn new(threads: i32) -> Self {
        LlamaBackend { threads }
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

        let (head, hidden_states, max_options): (Box<dyn Head>, bool, usize) = match &m.head {
            HeadSpec::Letters {
                choice_targets,
                yes_no,
                ..
            } => (
                Box::new(LettersHead::new(choice_targets, yes_no.as_ref())),
                false,
                choice_targets.len(),
            ),
            HeadSpec::Pointer { .. } => {
                return Err(Error::IncompatibleModel(Incompat::HeadWeights {
                    detail: "pointer heads need the reader of head weight files, which this build does not have yet (it arrives with the checkpoint converter)".to_string(),
                }));
            }
        };
        let template = instantiate(
            &m.template.id,
            m.template.version,
            m.tasks.as_deref(),
            max_options,
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
            },
        )?;

        let scorer = Arc::new(ModelScorer::new(
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
        ));
        Ok(Engine {
            scorer: scorer.clone(),
            runner: scorer,
        })
    }
}
