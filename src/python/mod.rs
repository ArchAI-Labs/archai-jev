//! Thin PyO3 binding layer: type conversion, calls into the core, error mapping.

mod classes;
mod convert;
mod errors;
mod models;

use pyo3::prelude::*;

/// Python module `archai_jev._core`, implemented in Rust.
#[pymodule]
pub mod _core {
    #[pymodule_export]
    use super::classes::{
        PyFault, PyMockScorer, PyModel, PyQuestion, calibrated_softmax, make_question, render_state,
    };
    #[pymodule_export]
    use super::models::{list_models, load_model};
}
