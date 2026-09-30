//! Thin PyO3 binding layer: type conversion, calls into the core, error mapping.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::Error;

impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        // Exhaustive on purpose: a new variant must get an explicit Python exception.
        match err {
            Error::Empty { .. } | Error::NonFinite { .. } | Error::InvalidTemperature(_) => {
                PyValueError::new_err(err.to_string())
            }
        }
    }
}

/// Python module `archai_jev._core`, implemented in Rust.
#[pymodule]
pub mod _core {
    use pyo3::prelude::*;

    /// Convert raw option scores (logits) into calibrated probabilities
    /// using a temperature-scaled softmax.
    ///
    /// A temperature > 1 flattens the distribution (less confident),
    /// a temperature < 1 sharpens it (more confident).
    #[pyfunction]
    #[pyo3(signature = (logits, temperature = 1.0))]
    fn calibrated_softmax(logits: Vec<f64>, temperature: f64) -> PyResult<Vec<f64>> {
        Ok(crate::calibration::calibrated_softmax(
            &logits,
            temperature,
        )?)
    }
}
