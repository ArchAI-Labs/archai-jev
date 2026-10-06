//! The single point where a model plugs into the core.

use crate::error::{Error, Result};
use crate::schema::{Questions, State};

/// The temperature applied to raw logits and whether it is a declared or measured calibration
/// (D20: with `calibrated == false` every answer says so).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    temperature: f64,
    calibrated: bool,
}

impl Calibration {
    /// Build a calibration.
    ///
    /// # Errors
    /// [`Error::InvalidTemperature`] if `temperature` is not a finite number > 0.
    pub fn new(temperature: f64, calibrated: bool) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Error::InvalidTemperature(temperature));
        }
        Ok(Calibration {
            temperature,
            calibrated,
        })
    }

    /// The uncalibrated default: temperature 1.0, `calibrated = false`.
    pub fn uncalibrated() -> Self {
        Calibration {
            temperature: 1.0,
            calibrated: false,
        }
    }

    /// The temperature divided out of the logits.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Whether the temperature is a declared or measured calibration.
    pub fn calibrated(&self) -> bool {
        self.calibrated
    }
}

/// A source of raw logits: a real model, or the deterministic mock.
///
/// It receives the state and **all** the questions in one call (so a real engine can compute
/// the state once) and returns raw logits, before temperature and softmax: one list per
/// question, one value per option, in order. `Send + Sync` because one `Jev` is shared across
/// Python threads.
pub trait Scorer: Send + Sync {
    /// The temperature to apply and whether it is a real calibration.
    fn calibration(&self) -> Calibration;

    /// Raw logits for every option of every question.
    ///
    /// # Errors
    /// Whatever the model reports; the pipeline checks the shape and finiteness afterwards.
    fn score(&self, state: &State, questions: &Questions) -> Result<Vec<Vec<f64>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_rejects_bad_temperatures() {
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                Calibration::new(t, true),
                Err(Error::InvalidTemperature(_))
            ));
        }
        let c = Calibration::new(2.5, true).unwrap();
        assert_eq!((c.temperature(), c.calibrated()), (2.5, true));
        assert!(!Calibration::uncalibrated().calibrated());
    }
}
