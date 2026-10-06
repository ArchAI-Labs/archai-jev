//! The interfaces 005 defines and the engine (006) and converter (018) implement.
//!
//! Each is a Rust `trait`: like an abstract class (or a Python `Protocol`), a list of methods
//! that any implementation must provide.

use std::path::PathBuf;
use std::sync::Arc;

use super::calibration_gate::Decision;
use super::manifest::Manifest;
use super::verify::VectorRunner;
use crate::error::Result;
use crate::scorer::Scorer;

/// A checkpoint that passed every check that does not need the engine.
#[derive(Debug, Clone)]
pub struct ValidatedCheckpoint {
    /// The manifest as read.
    pub manifest: Manifest,
    /// The variant chosen (`dtype`).
    pub dtype: String,
    /// The verified GGUF file.
    pub model_path: PathBuf,
    /// The verified `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// The head weights file, for pointer heads.
    pub head_path: Option<PathBuf>,
    /// The calibration decided from the manifest and the arguments.
    pub calibration: Decision,
}

/// What an engine returns after loading a checkpoint.
#[derive(Clone)]
pub struct Engine {
    /// Answers requests; applies the decided calibration.
    pub scorer: Arc<dyn Scorer>,
    /// Runs self-check vectors through the same path as real requests.
    pub runner: Arc<dyn VectorRunner>,
}

/// An inference engine: loads a validated checkpoint (006 implements it with llama.cpp).
pub trait Backend: Send + Sync {
    /// Load the weights and return the scorer and the self-check runner.
    ///
    /// # Errors
    /// Whatever the engine reports when it cannot load the model.
    fn load(&self, checkpoint: &ValidatedCheckpoint) -> Result<Engine>;
}
