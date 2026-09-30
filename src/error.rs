//! Error type shared by the whole core.

/// Every failure the core can report. Each variant maps to a typed Python exception
/// in the binding layer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A required sequence was empty.
    #[error("{name} must not be empty")]
    Empty { name: &'static str },

    /// A sequence contained NaN or infinity.
    #[error("{name} must contain only finite numbers")]
    NonFinite { name: &'static str },

    /// The softmax temperature was not a finite number greater than zero.
    #[error("temperature must be a finite number > 0, got {0}")]
    InvalidTemperature(f64),
}

/// Result alias used across the core.
pub type Result<T> = std::result::Result<T, Error>;
