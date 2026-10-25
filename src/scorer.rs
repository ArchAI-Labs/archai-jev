//! The single point where a model plugs into the core.

use crate::control::Control;
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

    /// Like [`score`](Self::score), but asks `control` between the rows of the request (and while
    /// it waits for the engine), so the call can be stopped without a partial result. A scorer
    /// that has no rows to wait between (the mock) keeps this default and ignores the control.
    ///
    /// # Errors
    /// [`Error::Cancelled`] when `control` says stop; otherwise as [`score`](Self::score).
    fn score_with(
        &self,
        state: &State,
        questions: &Questions,
        _control: &dyn Control,
    ) -> Result<Vec<Vec<f64>>> {
        self.score(state, questions)
    }

    /// What the memory of reused prefixes holds and has done; `None` for a scorer that has none.
    fn cache_info(&self) -> Option<CacheInfo> {
        None
    }

    /// Forget every saved prefix (the counters stay). Never changes a result.
    fn clear_cache(&self) {}

    /// Set the counters of the prefix memory back to zero.
    fn reset_cache_counters(&self) {}
}

/// The prefix memory of a scorer: what it may hold, what it holds, and what it did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheInfo {
    /// The most bytes it may hold.
    pub limit_bytes: u64,
    /// Bytes held now.
    pub bytes: u64,
    /// Saved prefixes held now.
    pub entries: u64,
    /// Requests that restored a saved prefix.
    pub hits: u64,
    /// Requests that looked for one and found none.
    pub misses: u64,
    /// Saved prefixes removed to make room.
    pub evictions: u64,
    /// Tokens not decoded because a prefix was shared or restored.
    pub tokens_reused: u64,
    /// Tokens the engine decoded.
    pub tokens_decoded: u64,
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
