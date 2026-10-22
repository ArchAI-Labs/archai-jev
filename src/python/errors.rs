//! Error mapping: one place, exhaustive through [`Error::category`].
//!
//! The exception classes live in Python (`archai_jev._exceptions`) because several of them
//! inherit from both `JevError` and a builtin (`ValueError`, `RuntimeError`); they are looked
//! up when an error happens.

use pyo3::exceptions::{
    PyFileNotFoundError, PyKeyboardInterrupt, PyNotImplementedError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::error::{Category, Error};

fn exception_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyType>> {
    py.import("archai_jev._exceptions")?
        .getattr(name)?
        .cast_into::<PyType>()
        .map_err(PyErr::from)
}

fn raise(name: &str, message: String) -> PyErr {
    Python::attach(|py| match exception_class(py, name) {
        Ok(class) => PyErr::from_type(class, message),
        Err(import_error) => import_error,
    })
}

/// The private exception of a call stopped by its token (the cancellation of `aask`); the Python
/// layer turns it into `asyncio.CancelledError`.
pub fn private_cancelled(py: Python<'_>) -> PyErr {
    match exception_class(py, "_StoppedError") {
        Ok(class) => PyErr::from_type(class, "the call was cancelled"),
        Err(import_error) => import_error,
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        let message = err.to_string();
        match err.category() {
            Category::Value | Category::Argument => PyValueError::new_err(message),
            Category::PathNotFound => PyFileNotFoundError::new_err(message),
            Category::Cancelled => PyKeyboardInterrupt::new_err(message),
            Category::NotImplemented => PyNotImplementedError::new_err(message),
            Category::State => raise("InvalidStateError", message),
            Category::Question => raise("InvalidQuestionError", message),
            Category::Numerical => raise("NumericalError", message),
            Category::Calibration => raise("CalibrationError", message),
            Category::Dataset => raise("InvalidDatasetError", message),
            Category::Unsupported => raise("UnsupportedRequestError", message),
            Category::Inference => raise("InferenceError", message),
            Category::DeviceUnavailable => raise("DeviceUnavailableError", message),
            Category::Incompatible => raise("IncompatibleModelError", message),
            Category::Verification => raise("ModelVerificationError", message),
            Category::Download => raise("ModelDownloadError", message),
        }
    }
}
