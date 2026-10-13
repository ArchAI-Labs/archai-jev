//! Calibration rules at load time (D20, spec 005 section 8). Decided from the manifest and the
//! arguments alone, so it runs before any download.

use super::incompat::Incompat;
use super::manifest::Variant;
use crate::error::{Error, Result};
use crate::scorer::Calibration;

/// Where the temperature in use comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationSource {
    /// Declared in the manifest.
    Manifest,
    /// Given by the user (`temperature=`).
    User,
    /// None: the user accepted uncalibrated probabilities.
    None,
}

impl CalibrationSource {
    /// The name shown in `ModelInfo`.
    pub fn name(self) -> &'static str {
        match self {
            CalibrationSource::Manifest => "manifest",
            CalibrationSource::User => "user",
            CalibrationSource::None => "none",
        }
    }
}

/// The calibration a loaded model will use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// Temperature and the `calibrated` flag of every answer.
    pub calibration: Calibration,
    /// Where it comes from.
    pub source: CalibrationSource,
    /// The temperature of the manifest (1.0 if undeclared): what the self-check uses.
    pub manifest_temperature: f64,
}

/// Decide the calibration of `variant` of model `name`.
///
/// `variants` are all the variants of the manifest: they only serve the message of a refusal
/// (naming a calibration declared for another variant, which does **not** transfer), and never
/// lend a temperature (spec 009, section 3).
///
/// # Errors
/// [`Error::InvalidTemperature`] for a bad user temperature; [`Incompat::Uncalibrated`] if the
/// variant has no declared calibration and the user gave neither a temperature nor consent.
pub fn decide(
    name: &str,
    variant: &Variant,
    variants: &[Variant],
    user_temperature: Option<f64>,
    allow_uncalibrated: bool,
) -> Result<Decision> {
    let declared = if variant.calibration.declared {
        variant.calibration.temperature
    } else {
        None
    };
    let manifest_temperature = declared.unwrap_or(1.0);
    if let Some(t) = user_temperature {
        return Ok(Decision {
            calibration: Calibration::new(t, true)?,
            source: CalibrationSource::User,
            manifest_temperature,
        });
    }
    if let Some(t) = declared {
        return Ok(Decision {
            calibration: Calibration::new(t, true)?,
            source: CalibrationSource::Manifest,
            manifest_temperature,
        });
    }
    if allow_uncalibrated {
        return Ok(Decision {
            calibration: Calibration::uncalibrated(),
            source: CalibrationSource::None,
            manifest_temperature,
        });
    }
    // Only for the message. `variant` is undeclared here (a declared one returned above), so
    // it cannot name itself.
    let other_declared = variants
        .iter()
        .filter(|v| v.calibration.declared)
        .find_map(|v| v.calibration.temperature.map(|t| (v.dtype.clone(), t)));
    Err(Error::IncompatibleModel(Incompat::Uncalibrated {
        name: name.to_string(),
        dtype: variant.dtype.clone(),
        other_declared,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::files::FileEntry;
    use crate::models::manifest::{CalibrationDecl, Source};
    use crate::models::tolerance::Reference;

    fn variant(declared: Option<f64>) -> Variant {
        variant_of("q8_0", declared)
    }

    fn variant_of(dtype: &str, declared: Option<f64>) -> Variant {
        Variant {
            dtype: dtype.into(),
            source: Source::Gguf {
                file: FileEntry {
                    path: "m.gguf".into(),
                    size: 1,
                    sha256: "0".repeat(64),
                    origin: None,
                },
            },
            calibration: CalibrationDecl {
                declared: declared.is_some(),
                temperature: declared,
                evidence: declared.map(|_| "test".to_string()),
            },
            vectors: vec![],
            reference: Reference::default(),
        }
    }

    #[test]
    fn declared_calibration_is_used() {
        let d = decide("m", &variant(Some(2.09)), &[], None, false).unwrap();
        assert_eq!(
            (
                d.calibration.temperature(),
                d.calibration.calibrated(),
                d.source
            ),
            (2.09, true, CalibrationSource::Manifest)
        );
        assert_eq!(d.manifest_temperature, 2.09);
    }

    #[test]
    fn declared_with_user_temperature_user_wins_and_allow_is_a_noop() {
        let d = decide("m", &variant(Some(2.09)), &[], Some(1.7), true).unwrap();
        assert_eq!(
            (d.calibration.temperature(), d.source),
            (1.7, CalibrationSource::User)
        );
        assert_eq!(d.manifest_temperature, 2.09);
        let d = decide("m", &variant(Some(2.09)), &[], None, true).unwrap();
        assert_eq!(d.source, CalibrationSource::Manifest);
    }

    #[test]
    fn undeclared_needs_a_temperature_or_consent() {
        let err = decide("m", &variant(None), &[], None, false).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("temperature=<value>") && msg.contains("allow_uncalibrated=True"),
            "{msg}"
        );
        let d = decide("m", &variant(None), &[], Some(1.7), false).unwrap();
        assert_eq!(
            (d.calibration.calibrated(), d.source),
            (true, CalibrationSource::User)
        );
        let d = decide("m", &variant(None), &[], None, true).unwrap();
        assert_eq!(
            (
                d.calibration.temperature(),
                d.calibration.calibrated(),
                d.source
            ),
            (1.0, false, CalibrationSource::None)
        );
    }

    #[test]
    fn bad_user_temperatures_are_value_errors() {
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                decide("m", &variant(None), &[], Some(t), false),
                Err(Error::InvalidTemperature(_))
            ));
        }
    }

    #[test]
    fn the_refusal_names_a_calibration_declared_for_another_variant() {
        // spec 009 AC-5: the message says the other temperature does not transfer
        let q4 = variant_of("q4_k_m", None);
        let q8 = variant_of("q8_0", Some(2.09));
        let err = decide("m", &q4, &[q8.clone(), q4.clone()], None, false).unwrap_err();
        let msg = err.to_string();
        for part in [
            "(q4_k_m)",
            "Jev.calibrate()",
            "temperature=<value>",
            "'q8_0'",
            "2.09",
            "does not transfer",
        ] {
            assert!(msg.contains(part), "{part:?} missing from {msg}");
        }
        // with nobody else declaring anything, the sentence is not there
        let err = decide("m", &q4, std::slice::from_ref(&q4), None, false).unwrap_err();
        assert!(!err.to_string().contains("does not transfer"), "{err}");
    }

    #[test]
    fn other_variants_never_lend_a_temperature() {
        // spec 009 AC-6: consent or silence, T is 1.0 and the answers say uncalibrated
        let q4 = variant_of("q4_k_m", None);
        let q8 = variant_of("q8_0", Some(2.09));
        let d = decide("m", &q4, &[q8, q4.clone()], None, true).unwrap();
        assert_eq!(
            (
                d.calibration.temperature(),
                d.calibration.calibrated(),
                d.source,
                d.manifest_temperature
            ),
            (1.0, false, CalibrationSource::None, 1.0)
        );
    }

    #[test]
    fn the_users_temperature_is_absolute_whatever_the_siblings_declare() {
        let q4 = variant_of("q4_k_m", None);
        let q8 = variant_of("q8_0", Some(2.09));
        let d = decide("m", &q4, &[q8, q4.clone()], Some(1.7), false).unwrap();
        assert_eq!(
            (
                d.calibration.temperature(),
                d.calibration.calibrated(),
                d.source
            ),
            (1.7, true, CalibrationSource::User)
        );
        // not 1.7 x 2.09 and not relative to anything
        assert_eq!(d.manifest_temperature, 1.0);
    }
}
