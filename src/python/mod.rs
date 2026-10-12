//! Thin PyO3 binding layer: type conversion, calls into the core, error mapping.

mod calibration;
mod classes;
mod convert;
mod errors;
mod evaluation;
mod models;
mod rows;

use pyo3::prelude::*;

/// Python module `archai_jev._core`, implemented in Rust.
#[pymodule]
pub mod _core {
    #[pymodule_export]
    use super::calibration::{
        calibration_report, choice_confidence, choice_confidence_batch, fit_temperature_py,
        log_loss_from_logits, score_confidence, score_confidence_batch, score_value,
        score_value_batch, softmax_batch, yes_no_probability, yes_no_probability_batch,
    };
    #[pymodule_export]
    use super::classes::{
        PyCancelToken, PyFault, PyMockScorer, PyModel, PyQuestion, calibrated_softmax,
        make_question, render_state,
    };
    #[pymodule_export]
    use super::evaluation::{
        PyDataset, check_evaluate_options, dataset_from_items, dataset_from_jsonl, split,
    };
    #[pymodule_export]
    use super::models::{list_models, load_model};
}
