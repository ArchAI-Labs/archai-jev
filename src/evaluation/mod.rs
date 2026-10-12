//! Evaluating and calibrating a model on labelled data of the user (spec 017).
//!
//! Pure Rust (no PyO3): the binding in `python::evaluation` converts and releases the GIL.

pub mod bootstrap;
pub mod calibrate;
pub mod dataset;
pub mod evaluate;
pub mod fingerprint;
pub mod groups;
pub mod notes;
pub mod problem;
pub mod read;
pub mod recalibrated;
pub mod scopes;

#[cfg(test)]
mod calibrate_tests;
#[cfg(test)]
mod evaluate_tests;
#[cfg(test)]
pub(crate) mod testing;
