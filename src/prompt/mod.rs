//! Prompt construction (spec 006a): sanitization, tokenization, templates and the checks that
//! refuse a request before any forward pass. Pure Rust, no engine inside.

pub mod chatml;
#[cfg(test)]
mod chatml_tests;
pub mod kev;
pub mod limits;
#[allow(clippy::module_inception)]
pub mod prompt;
pub mod sanitize;
pub mod tasks;
pub mod template;
pub mod tokenize;
pub mod unsupported;

pub use limits::{NoLimits, QuestionSupport, RequestLimits, kind_name};
pub use prompt::{Branch, Prompt};
pub use sanitize::sanitize;
pub use template::{Template, lookup};
pub use tokenize::{PromptTokenizer, Roles};
pub use unsupported::Unsupported;
