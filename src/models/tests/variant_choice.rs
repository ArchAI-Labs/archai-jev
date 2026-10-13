//! Which variant of the weights is loaded is the user's explicit choice and nothing else
//! (spec 009, AC-2): `dtype=None` is the default of the manifest, whatever else is asked of the
//! loader, and the library has no environment variable or setting that picks a variant.

use super::Harness;
use super::pipeline::with_declared_sibling;
use crate::error::Error;
use crate::models::incompat::Incompat;
use crate::models::testing::builder::Builder;
use crate::models::testing::jsonedit::s;

#[test]
fn dtype_none_is_the_manifest_default_whatever_else_is_asked() {
    // two variants: q8_0 (with its file) and bf16 (declared); the manifest's default is q8_0
    let fixture = with_declared_sibling(Builder::tiny()).build();
    for offline in [false, true] {
        for endpoint in [None, Some("https://mirror.invalid".to_string())] {
            for temperature in [None, Some(1.7), Some(0.5)] {
                for allow_uncalibrated in [false, true] {
                    // a refusal for lack of calibration is allowed; a different variant is not
                    let mut h = Harness::new();
                    h.offline = offline;
                    h.endpoint.clone_from(&endpoint);
                    let mut req = h.local(fixture.path());
                    req.temperature = temperature;
                    req.allow_uncalibrated = allow_uncalibrated;
                    match h.load(&req) {
                        Ok(model) => assert_eq!(model.data.dtype, "q8_0"),
                        Err(Error::IncompatibleModel(Incompat::Uncalibrated { dtype, .. })) => {
                            assert_eq!(dtype, "q8_0");
                        }
                        Err(other) => panic!("{offline} {endpoint:?} {temperature:?}: {other:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn only_the_manifest_moves_the_default_and_only_the_argument_overrides_it() {
    // the manifest says the default is bf16: dtype=None follows it (the bf16 variant shares the
    // q8_0 file here, so it is refused for its tensor types, which proves which one was chosen)
    let fixture = with_declared_sibling(Builder::tiny())
        .set("default_dtype", s("bf16"))
        .build();
    let h = Harness::new();
    let err = h.load(&h.local(fixture.path())).err().unwrap();
    assert!(
        matches!(&err, Error::IncompatibleModel(Incompat::TensorType { got, .. }) if got == "Q8_0"),
        "{err:?}"
    );
    // asking for q8_0 by name is the user's explicit choice and loads
    let mut req = h.local(fixture.path());
    req.selection.dtype = Some("q8_0".to_string());
    assert_eq!(h.load(&req).unwrap().data.dtype, "q8_0");
}

#[test]
fn no_setting_of_the_environment_picks_a_variant() {
    // The core reads no environment (D6); the Python binding reads it once. Whatever it reads,
    // none of it may be about the variant: no precision, quantisation or memory switch.
    let binding = include_str!("../../python/models.rs");
    let mut names = Vec::new();
    for marker in ["env(\"", "truthy(\""] {
        for part in binding.split(marker).skip(1) {
            if let Some((name, _)) = part.split_once('"') {
                names.push(name.to_string());
            }
        }
    }
    assert!(
        names.iter().any(|n| n == "ARCHAI_JEV_OFFLINE"),
        "the scan finds nothing: {names:?}"
    );
    for name in &names {
        for forbidden in ["DTYPE", "QUANT", "VARIANT", "PRECISION", "LOW_MEM", "FAST"] {
            assert!(
                !name.contains(forbidden),
                "{name} would let the environment pick a variant"
            );
        }
    }
}
