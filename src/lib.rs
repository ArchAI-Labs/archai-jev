//! Rust core of `archai_jev`: typed decisions with calibrated probabilities.
//!
//! Everything outside the `python` module is pure Rust and does not depend on PyO3.
#![forbid(unsafe_code)]

pub mod calibration;
pub mod error;
mod python;

pub use error::{Error, Result};
