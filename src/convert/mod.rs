//! Checkpoint conversion (spec 018, decision D40): turns the original files of a checkpoint
//! (safetensors weights, a LoRA adapter, `head.pt`) into a GGUF that llama.cpp loads, and reads
//! `head.pt` without ever running it.
//!
//! Pure Rust: nothing here depends on Python. Every anomaly is an `IncompatibleModelError`.

pub mod config;
pub mod gguf_out;
pub mod headpt;
pub mod layout;
pub mod lora;
pub mod numeric;
pub mod plan;
pub mod qwen35;
pub mod safetensors;
pub mod stream;
pub mod vocab;

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod layout_tests;
#[cfg(test)]
mod lora_tests;
#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod reject_tests;
#[cfg(test)]
mod robust_tests;
#[cfg(test)]
mod vocab_tests;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
#[cfg(any(test, feature = "testing"))]
pub mod testing_gguf;
#[cfg(any(test, feature = "testing"))]
pub mod testing_pt;

pub use qwen35::{Qwen35Converter, RECIPE};

use crate::json_strict::FieldProblem;

/// A short description of a field problem, for messages about files that are not manifests.
pub(crate) fn problem_text(p: &FieldProblem) -> String {
    match p {
        FieldProblem::Missing { path } => format!("the field {path} is missing"),
        FieldProblem::Unknown { path, field } => {
            format!("unknown field \"{field}\" in {path}")
        }
        FieldProblem::Bad { path, detail } => format!("{path} {detail}"),
    }
}
