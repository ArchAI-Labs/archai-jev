//! Turning raw option scores into calibrated probabilities.

pub mod batch;
pub mod fit;
pub mod formulas;
pub mod problem;
pub mod rows;

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod fit_tests;

use crate::error::{Error, Result};

/// The one rule for a temperature: a finite number greater than 0.
///
/// # Errors
/// `Error::InvalidTemperature` otherwise.
pub fn check_temperature(temperature: f64) -> Result<()> {
    if temperature.is_finite() && temperature > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidTemperature(temperature))
    }
}

/// Convert raw option scores (logits) into probabilities with a temperature-scaled softmax.
///
/// A temperature > 1 flattens the distribution (less confident), a temperature < 1
/// sharpens it (more confident).
///
/// # Errors
/// Fails if `logits` is empty or not finite, or if `temperature` is not a finite number > 0.
pub fn calibrated_softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if logits.is_empty() {
        return Err(Error::Empty { name: "logits" });
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite { name: "logits" });
    }
    check_temperature(temperature)?;

    Ok(softmax_of_valid(logits, temperature))
}

/// The softmax of logits already known to be finite, with a valid temperature.
pub(crate) fn softmax_of_valid(logits: &[f64], temperature: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(logits.len());
    softmax_into(logits, temperature, &mut out);
    out
}

/// [`softmax_of_valid`] into `out` (cleared first), so a loop over many rows allocates once.
pub(crate) fn softmax_into(logits: &[f64], temperature: f64, out: &mut Vec<f64>) {
    // Subtracting the max keeps exp() from overflowing without changing the result.
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    out.clear();
    out.extend(logits.iter().map(|x| ((x - max) / temperature).exp()));
    // The max element contributes exp(0) = 1, so the sum is always >= 1.
    let sum: f64 = out.iter().sum();
    for p in out.iter_mut() {
        *p /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tol, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn known_values() {
        let probs = calibrated_softmax(&[2.0, 1.0, 0.1], 1.0).unwrap();
        assert_close(&probs, &[0.659, 0.242, 0.099], 1e-3);
    }

    #[test]
    fn temperature_flattens() {
        let sharp = calibrated_softmax(&[2.0, 1.0, 0.1], 0.5).unwrap();
        let flat = calibrated_softmax(&[2.0, 1.0, 0.1], 2.0).unwrap();
        assert!(flat[0] < sharp[0]);
    }

    #[test]
    fn huge_logits_are_stable() {
        let probs = calibrated_softmax(&[1e300, 1e300], 1.0).unwrap();
        assert_close(&probs, &[0.5, 0.5], 1e-12);
    }

    #[test]
    fn single_logit_is_certain() {
        assert_eq!(calibrated_softmax(&[-7.0], 1.0).unwrap(), vec![1.0]);
    }

    #[test]
    fn rejects_invalid_input() {
        assert_eq!(
            calibrated_softmax(&[], 1.0),
            Err(Error::Empty { name: "logits" })
        );
        assert_eq!(
            calibrated_softmax(&[1.0, f64::NAN], 1.0),
            Err(Error::NonFinite { name: "logits" })
        );
        assert_eq!(
            calibrated_softmax(&[1.0, f64::INFINITY], 1.0),
            Err(Error::NonFinite { name: "logits" })
        );
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                calibrated_softmax(&[1.0, 2.0], t),
                Err(Error::InvalidTemperature(_))
            ));
        }
    }

    proptest! {
        #[test]
        fn is_a_probability_distribution(
            logits in prop::collection::vec(-1e6f64..1e6, 1..300),
            temperature in 1e-3f64..1e3,
        ) {
            let probs = calibrated_softmax(&logits, temperature).unwrap();
            prop_assert_eq!(probs.len(), logits.len());
            prop_assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
            prop_assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }

        #[test]
        fn preserves_argmax(
            logits in prop::collection::vec(-1e3f64..1e3, 1..50),
            temperature in 1e-2f64..1e2,
        ) {
            let probs = calibrated_softmax(&logits, temperature).unwrap();
            let argmax = |v: &[f64]| {
                v.iter()
                    .enumerate()
                    .fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
            };
            prop_assert_eq!(argmax(&probs), argmax(&logits));
        }
    }
}
