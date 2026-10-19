//! Prompt construction (spec 006a): sanitization, tokenization, templates and the checks that
//! refuse a request before any forward pass. Pure Rust, no engine inside.

pub mod chatml;
#[cfg(test)]
mod chatml_cuts_tests;
#[cfg(test)]
mod chatml_tests;
pub mod di;
pub mod kev;
pub mod limits;
#[allow(clippy::module_inception)]
pub mod prompt;
pub mod sanitize;
#[cfg(test)]
mod sanitize_control_tests;
pub mod spark;
#[cfg(test)]
mod spark_tests;
pub mod spark_text;
pub mod tasks;
pub mod template;
pub mod tokenize;
pub mod unsupported;

pub use limits::{NoLimits, QuestionSupport, RequestLimits, kind_name};
pub use prompt::{Branch, Cut, CutKind, Prompt};
pub use sanitize::sanitize;
pub use template::{Template, TemplateInputs, lookup};
pub use tokenize::{PromptTokenizer, Roles};
pub use unsupported::Unsupported;
