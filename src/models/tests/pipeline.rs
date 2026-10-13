//! The loading pipeline end to end with the fake engine: success, calibration, self-check,
//! verification records, stage order, agnosticism, conversion.

use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use super::Harness;
use crate::error::Error;
use crate::hub::events::Event;
use crate::models::failures::VerifyFailure;
use crate::models::incompat::Incompat;
use crate::models::registry::Registry;
use crate::models::resolve::{NameOrPath, Selection};
use crate::models::testing::builder::{Builder, COMMIT};
use crate::models::testing::fake::Perturb;
use crate::models::testing::jsonedit::{obj, s};
use crate::models::testing::{FakeBackend, FakeConverter};

fn registry_for(fixture_text: &str) -> Registry {
    let index =
        format!(r#"{{"default": "test/tiny@{COMMIT}", "current": {{"test/tiny": "{COMMIT}"}}}}"#);
    Registry::from_sources(&[fixture_text], &index).unwrap()
}

fn touch(path: &std::path::Path, delta_secs: u64) {
    let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(delta_secs))
        .unwrap();
}

#[test]
fn loads_a_valid_local_checkpoint() {
    let fixture = Builder::tiny().declared(2.09).build();
    let h = Harness::new();
    let model = h.load(&h.local(fixture.path())).unwrap();
    let d = &model.data;
    assert_eq!(
        (
            d.name.as_str(),
            d.revision.as_str(),
            d.family.as_str(),
            d.head.as_str(),
            d.template.as_str()
        ),
        (
            "local/tiny",
            "v1",
            "qwen2-letters",
            "letters",
            "chatml-letters-v1"
        )
    );
    assert_eq!(
        (
            d.dtype.as_str(),
            d.source,
            d.max_context,
            d.license.as_str()
        ),
        ("q8_0", "local", 512, "Apache-2.0")
    );
    assert_eq!(
        (d.temperature, d.calibrated, d.calibration_source),
        (2.09, true, "manifest")
    );
    assert_eq!(model.scorer.calibration().temperature(), 2.09);
    assert_eq!(
        (h.backend.loads(), h.backend.runs(), h.records()),
        (1, 1, 1)
    );
}

#[test]
fn second_open_runs_no_vectors_and_does_not_rehash() {
    let fixture = Builder::tiny().build();
    let h = Harness::new();
    h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(h.backend.runs(), 1);
    // Same size and modification time but different content: only the record can notice, and
    // the fast path deliberately does not read the file (documented limitation of spec 7.2).
    let gguf = fixture.gguf_path();
    let original = std::fs::metadata(&gguf).unwrap().modified().unwrap();
    let mut bytes = std::fs::read(&gguf).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&gguf, bytes).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&gguf)
        .unwrap()
        .set_modified(original)
        .unwrap();
    h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(
        (h.backend.loads(), h.backend.runs()),
        (2, 1),
        "the vectors must not run again"
    );
}

#[test]
fn a_modified_file_or_a_damaged_record_is_verified_again() {
    let fixture = Builder::tiny().build();
    let h = Harness::new();
    h.load(&h.local(fixture.path())).unwrap();
    // a newer modification time: full hash and self-check again
    touch(&fixture.gguf_path(), 120);
    h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(h.backend.runs(), 2);
    // a damaged record counts as no record, never as a pass
    let verified = h.cache.path().join("v1").join("verified");
    for entry in std::fs::read_dir(&verified).unwrap().flatten() {
        std::fs::write(entry.path(), b"{ truncated").unwrap();
    }
    h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(h.backend.runs(), 3);
    h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(h.backend.runs(), 3, "the record was rewritten");
    // another manifest (another declared temperature) has its own record
    let other = Builder::tiny().declared(1.5).build();
    h.load(&h.local(other.path())).unwrap();
    assert_eq!((h.backend.runs(), h.records()), (4, 2));
}

#[test]
fn selfcheck_failures_are_typed_and_never_cached() {
    let cases: Vec<(&str, Perturb)> = vec![
        ("ids", Perturb::IdsOff),
        ("probability", Perturb::LogitShift(10.0)),
        ("nan", Perturb::Nan),
        ("inf", Perturb::Inf),
        ("shape", Perturb::DropQuestion),
    ];
    for (name, perturb) in cases {
        let fixture = Builder::tiny().build();
        let mut h = Harness::new();
        h.backend = FakeBackend::new().perturbed(perturb);
        for attempt in 1..=2 {
            let err = h.load(&h.local(fixture.path())).err().unwrap();
            let Error::ModelVerification(failure) = &err else {
                panic!("{name}: {err:?}")
            };
            let ok = matches!(
                (name, failure),
                ("ids", VerifyFailure::Ids { .. })
                    | ("probability", VerifyFailure::Probability { .. })
                    | ("nan" | "inf", VerifyFailure::NonFinite { .. })
                    | ("shape", VerifyFailure::Shape { .. })
            );
            assert!(ok, "{name}: {failure}");
            assert_eq!(h.records(), 0, "{name}: a failure must not be cached");
            assert_eq!(
                h.backend.runs(),
                attempt,
                "{name}: the vectors run again on every attempt"
            );
        }
    }
}

#[test]
fn the_ids_message_names_the_first_difference() {
    let fixture = Builder::tiny().build();
    let mut h = Harness::new();
    h.backend = FakeBackend::new().perturbed(Perturb::IdsOff);
    let msg = h.load(&h.local(fixture.path())).err().unwrap().to_string();
    assert!(
        msg.contains("differ at position 0") && msg.contains("tiny-1"),
        "{msg}"
    );
}

#[test]
fn tolerance_boundaries_are_exact() {
    use crate::json_strict::Json;
    use crate::models::manifest::{CalibrationDecl, Source, Variant};
    use crate::models::tolerance::{Reference, Tolerance};
    use crate::models::vectors::{Expected, Vector};
    use crate::models::verify::{RunOutput, VectorRunner, run_selfcheck};

    struct Fixed(Vec<f64>);
    impl VectorRunner for Fixed {
        fn run(&self, _: &Json) -> crate::error::Result<RunOutput> {
            Ok(RunOutput {
                input_ids: vec![1],
                questions: vec![("q".to_string(), self.0.clone())],
            })
        }
    }
    let variant = |expected_logits: Vec<f64>, expected_p: Vec<f64>| Variant {
        dtype: "f32".into(),
        source: Source::Reserved {
            kind: "x".into(),
            raw: Json::Null,
        },
        calibration: CalibrationDecl {
            declared: false,
            temperature: None,
            evidence: None,
        },
        reference: Reference::default(),
        vectors: vec![Vector {
            id: "v".into(),
            request: Json::Null,
            input_ids: vec![1],
            questions: vec![(
                "q".into(),
                Expected {
                    logits: expected_logits,
                    probabilities: expected_p,
                },
            )],
        }],
    };
    // logit boundary: expected 0.0, got exactly the tolerance, then one step above
    let tol = Tolerance {
        logit: Some(0.06),
        p: 1.0,
    };
    let v = variant(vec![0.0, 0.0], vec![0.5, 0.5]);
    assert!(run_selfcheck("m", &v, &Fixed(vec![0.06, 0.0]), tol, 1.0).is_ok());
    let above = f64::from_bits(0.06f64.to_bits() + 1);
    assert!(matches!(
        run_selfcheck("m", &v, &Fixed(vec![above, 0.0]), tol, 1.0),
        Err(Error::ModelVerification(VerifyFailure::Logit { .. }))
    ));
    // probability boundary: the tolerance is exactly the distance the engine produces
    let got = crate::calibration::calibrated_softmax(&[1.0, 0.0], 1.0).unwrap();
    let exact = (got[0] - 0.5).abs();
    let v = variant(vec![0.0, 0.0], vec![0.5, 0.5]);
    let runner = Fixed(vec![1.0, 0.0]);
    assert!(
        run_selfcheck(
            "m",
            &v,
            &runner,
            Tolerance {
                logit: None,
                p: exact
            },
            1.0
        )
        .is_ok()
    );
    let below = f64::from_bits(exact.to_bits() - 1);
    assert!(matches!(
        run_selfcheck(
            "m",
            &v,
            &runner,
            Tolerance {
                logit: None,
                p: below
            },
            1.0
        ),
        Err(Error::ModelVerification(VerifyFailure::Probability { .. }))
    ));
}

#[test]
fn the_users_temperature_does_not_enter_the_selfcheck() {
    let fixture = Builder::tiny().build();
    let h = Harness::new();
    let mut req = h.local(fixture.path());
    req.temperature = Some(1.7);
    let model = h.load(&req).unwrap();
    assert_eq!(
        (
            model.data.temperature,
            model.data.calibrated,
            model.data.calibration_source
        ),
        (1.7, true, "user")
    );
    assert_eq!(h.backend.runs(), 1);
}

#[test]
fn calibration_gate_runs_before_the_engine() {
    let fixture = Builder::tiny().build();
    let h = Harness::new();
    let mut req = h.local(fixture.path());
    req.allow_uncalibrated = false;
    let err = h.load(&req).err().unwrap();
    assert!(
        matches!(err, Error::IncompatibleModel(Incompat::Uncalibrated { .. })),
        "{err:?}"
    );
    assert_eq!(h.backend.loads(), 0);
    req.allow_uncalibrated = true;
    let model = h.load(&req).unwrap();
    assert_eq!(
        (
            model.data.calibrated,
            model.data.temperature,
            model.data.calibration_source
        ),
        (false, 1.0, "none")
    );
}

#[test]
fn a_second_family_loads_through_the_same_code() {
    // `test-fake` has another architecture, tensors, special tokens and question kinds; the
    // loader needed no change for it, only a row in the table of families.
    let fixture = Builder::tiny_fake().build();
    let h = Harness::new();
    let model = h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(
        (model.data.family.as_str(), model.data.dtype.as_str()),
        ("test-fake", "f32")
    );
    assert_eq!(h.backend.runs(), 1);
}

#[test]
fn the_first_failing_stage_wins_and_messages_are_stable() {
    let corrupt = |d: &std::path::Path| {
        let p = d.join("model.gguf");
        let mut b = std::fs::read(&p).unwrap();
        let last = b.len() - 1;
        b[last] ^= 1;
        std::fs::write(p, b).unwrap();
    };
    // stage 2 (unknown family) beats stage 3 (bad hash)
    let b = || Builder::tiny().set("family", s("nope")).after_seal(corrupt);
    let h = Harness::new();
    let f = b().build();
    let first = h.load(&h.local(f.path())).err().unwrap();
    assert!(matches!(
        first,
        Error::IncompatibleModel(Incompat::UnknownFamily { .. })
    ));
    assert_eq!(
        first.to_string(),
        h.load(&h.local(f.path())).err().unwrap().to_string()
    );
    // stage 3 (bad hash) beats stage 4 (missing tensor)
    let f = Builder::tiny()
        .drop_tensor("blk.0.attn_q.weight")
        .after_seal(corrupt)
        .build();
    let e = h.load(&h.local(f.path())).err().unwrap();
    assert!(
        matches!(e, Error::IncompatibleModel(Incompat::HashMismatch { .. })),
        "{e:?}"
    );
    // stage 4 (missing tensor) beats stage 5 (tokenizer)
    let f = Builder::tiny()
        .drop_tensor("blk.0.attn_q.weight")
        .edit_tokenizer(|t| {
            crate::models::testing::jsonedit::set(
                t,
                "padding",
                obj(vec![("strategy", s("BatchLongest"))]),
            );
        })
        .build();
    let e = h.load(&h.local(f.path())).err().unwrap();
    assert!(
        matches!(e, Error::IncompatibleModel(Incompat::TensorMissing { .. })),
        "{e:?}"
    );
}

fn registry_harness(builder: Builder) -> Harness {
    let fixture = builder.registry().build();
    let text = std::fs::read_to_string(fixture.manifest_path()).unwrap();
    Harness::with_registry(registry_for(&text))
}

fn named(name: &str) -> Selection {
    Selection {
        name_or_path: NameOrPath::Str(name.to_string()),
        revision: None,
        device: "cpu".to_string(),
        dtype: None,
        manifest: None,
    }
}

#[test]
fn what_the_manifest_alone_decides_happens_before_any_network_or_disk() {
    let h = registry_harness(Builder::tiny());
    let mut req = crate::models::load::LoadRequest {
        selection: named("test/tiny"),
        temperature: None,
        allow_uncalibrated: false,
    };
    // uncalibrated and no consent
    let err = h.load(&req).err().unwrap();
    assert!(matches!(
        err,
        Error::IncompatibleModel(Incompat::Uncalibrated { .. })
    ));
    // a dtype the manifest does not offer
    req.allow_uncalibrated = true;
    req.selection.dtype = Some("bf16".to_string());
    let err = h.load(&req).err().unwrap();
    assert!(
        matches!(
            err,
            Error::IncompatibleModel(Incompat::DtypeNotOffered { .. })
        ),
        "{err:?}"
    );
    // arguments
    req.selection.dtype = None;
    req.selection.device = "cuda".to_string();
    assert!(matches!(
        h.load(&req).err().unwrap(),
        Error::ModelArgument(_)
    ));
    assert_eq!(h.transport.gets.load(Ordering::SeqCst), 0);
    assert!(
        !h.cache.path().join("v1").exists(),
        "nothing may be created in the cache"
    );
    assert_eq!(h.backend.loads(), 0);

    // an unknown family in a registry entry fails before the network too
    let h = registry_harness(Builder::tiny().set("family", s("nope")));
    req.selection.device = "cpu".to_string();
    assert!(matches!(
        h.load(&req).err().unwrap(),
        Error::IncompatibleModel(Incompat::UnknownFamily { .. })
    ));
    assert_eq!(h.transport.gets.load(Ordering::SeqCst), 0);
}

#[test]
fn an_unknown_name_makes_no_request() {
    let h = Harness::new();
    let req = crate::models::load::LoadRequest {
        selection: named("someone/unknown"),
        temperature: None,
        allow_uncalibrated: true,
    };
    let err = h.load(&req).err().unwrap();
    assert!(matches!(
        err,
        Error::IncompatibleModel(Incompat::UnknownName { .. })
    ));
    assert_eq!(h.transport.gets.load(Ordering::SeqCst), 0);
    assert!(!h.cache.path().join("v1").exists());
}

// ---- conversion (the interface to 018) --------------------------------------------------

fn converted_fixture() -> (crate::models::testing::Fixture, Vec<u8>) {
    let good = Builder::tiny().build();
    let bytes = std::fs::read(good.gguf_path()).unwrap();
    let f = Builder::tiny().hf_lora().build();
    (f, bytes)
}

#[test]
fn converter_output_is_validated_like_a_downloaded_gguf() {
    let (f, good) = converted_fixture();
    // a good conversion loads
    let mut h = Harness::new();
    h.converter = Some(std::sync::Arc::new(FakeConverter::new(good)));
    let model = h.load(&h.local(f.path())).unwrap();
    assert_eq!(model.data.dtype, "q8_0");
    assert_eq!(h.backend.runs(), 1);
    assert_eq!(
        h.records(),
        1,
        "a converted model gets a verification record like a downloaded one"
    );
    // a faulty converter (a tensor is missing) cannot get around the checks
    let faulty = Builder::tiny().drop_tensor("blk.0.attn_q.weight").build();
    let bad_bytes = std::fs::read(faulty.gguf_path()).unwrap();
    let mut h = Harness::new();
    h.converter = Some(std::sync::Arc::new(FakeConverter::new(bad_bytes)));
    let e = h.load(&h.local(f.path())).err().unwrap();
    assert!(
        matches!(e, Error::IncompatibleModel(Incompat::TensorMissing { .. })),
        "{e:?}"
    );
    assert_eq!(h.backend.loads(), 0);
    // a converter that fails
    let (f2, good2) = converted_fixture();
    let mut conv = FakeConverter::new(good2);
    conv.fail = true;
    let mut h = Harness::new();
    h.converter = Some(std::sync::Arc::new(conv));
    let e = h.load(&h.local(f2.path())).err().unwrap();
    assert!(
        matches!(
            e,
            Error::IncompatibleModel(Incompat::ConversionFailed { .. })
        ),
        "{e:?}"
    );
    // without a converter the reserved kinds are refused
    let (f3, _) = converted_fixture();
    let h = Harness::new();
    let e = h.load(&h.local(f3.path())).err().unwrap();
    assert!(
        matches!(
            e,
            Error::IncompatibleModel(Incompat::SourceWithoutConverter { .. })
        ),
        "{e:?}"
    );
}

#[test]
fn conversion_is_cached_by_converter_version() {
    let (f, good) = converted_fixture();
    let conv = std::sync::Arc::new(FakeConverter::new(good.clone()));
    let mut h = Harness::new();
    h.converter = Some(conv.clone());
    h.load(&h.local(f.path())).unwrap();
    h.load(&h.local(f.path())).unwrap();
    assert_eq!(conv.calls(), 1, "the second load reuses the converted file");
    // a new version of the converter converts again
    let mut v2 = FakeConverter::new(good);
    v2.version = "fake-2".to_string();
    let v2 = std::sync::Arc::new(v2);
    h.converter = Some(v2.clone());
    h.load(&h.local(f.path())).unwrap();
    assert_eq!(v2.calls(), 1);
}

// ---- notices ---------------------------------------------------------------------------

#[test]
fn license_and_notice_are_reported() {
    let fixture = Builder::tiny()
        .set("notice", s("Demo model: 4 declared tasks"))
        .set("license.restrictions", s("non-commercial use only"))
        .build();
    let h = Harness::new();
    h.load(&h.local(fixture.path())).unwrap();
    let events = h.recorder.events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::License { spdx, .. } if spdx == "Apache-2.0"))
    );
    let warnings: Vec<&String> = events
        .iter()
        .filter_map(|e| {
            if let Event::Warning(w) = e {
                Some(w)
            } else {
                None
            }
        })
        .collect();
    assert!(
        warnings.iter().any(|w| w.contains("non-commercial")),
        "{warnings:?}"
    );
    assert!(
        warnings.iter().any(|w| w.contains("Demo model")),
        "{warnings:?}"
    );
}

#[test]
fn engine_vectors_fail_beyond_tol_p_and_full_precision_ones_do_not() {
    // The engine is off by a logit shift of 5e-4 (a probability shift around 1e-4): inside the
    // loose full-precision row of the family, outside the tight engine-recorded row. The same
    // numbers are accepted or refused only because of `selfcheck.reference` (spec 009, AC-10).
    for (reference, accepted) in [
        (None, true),
        (Some("full_precision"), true),
        (Some("engine"), false),
    ] {
        let mut builder = Builder::tiny_fake();
        if let Some(r) = reference {
            builder = builder.set("variants.0.selfcheck.reference", s(r));
        }
        let fixture = builder.build();
        let mut h = Harness::new();
        h.backend = FakeBackend::new().perturbed(Perturb::LogitShift(5e-4));
        let result = h.load(&h.local(fixture.path()));
        match (accepted, result) {
            (true, Ok(_)) => assert_eq!(h.records(), 1, "{reference:?}"),
            (
                false,
                Err(Error::ModelVerification(VerifyFailure::Probability { tolerance, .. })),
            ) => {
                assert_eq!(tolerance, 1e-6, "the engine row is the one that answered");
                assert_eq!(h.records(), 0, "a failure is never cached");
            }
            (accepted, other) => {
                panic!("{reference:?} accepted={accepted}: {:?}", other.map(|_| ()))
            }
        }
    }
    // an engine-recorded variant whose numbers are exactly what the engine gives passes
    let fixture = Builder::tiny_fake()
        .set("variants.0.selfcheck.reference", s("engine"))
        .build();
    let h = Harness::new();
    h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(h.records(), 1);
}

#[test]
fn a_variant_the_manifest_does_not_offer_lists_the_available_ones() {
    // spec 009 AC-4: asking for a dtype the manifest has no variant for names what it has
    let h = registry_harness(Builder::tiny());
    let req = crate::models::load::LoadRequest {
        selection: Selection {
            dtype: Some("bf16".to_string()),
            ..named("test/tiny")
        },
        temperature: None,
        allow_uncalibrated: true,
    };
    let err = h.load(&req).err().unwrap();
    let Error::IncompatibleModel(Incompat::DtypeNotOffered { dtype, available }) = &err else {
        panic!("{err:?}")
    };
    assert_eq!(
        (dtype.as_str(), available.as_slice()),
        ("bf16", &["q8_0".to_string()][..])
    );
    let text = err.to_string();
    assert!(
        text.contains("'bf16'") && text.contains("available: q8_0"),
        "{text}"
    );
    assert_eq!(h.backend.loads(), 0);
}

/// A second variant (`bf16`) that declares a calibration, sharing the file entry of the first
/// (it is never opened by these tests): a manifest with a calibrated sibling.
pub(super) fn with_declared_sibling(builder: Builder) -> Builder {
    use crate::json_strict::Json;
    use crate::models::testing::jsonedit::{b, f, set};
    builder.edit_manifest(|j| {
        let Json::Object(top) = j else {
            panic!("manifest")
        };
        let (_, variants) = top.iter_mut().find(|(k, _)| k == "variants").unwrap();
        let Json::Array(items) = variants else {
            panic!("variants")
        };
        let mut second = items[0].clone();
        set(&mut second, "dtype", s("bf16"));
        set(
            &mut second,
            "calibration",
            obj(vec![
                ("declared", b(true)),
                ("temperature", f(2.09)),
                ("evidence", s("measured on another set, for bf16 only")),
            ]),
        );
        items.push(second);
    })
}

#[test]
fn an_uncalibrated_variant_names_its_calibrated_sibling_and_touches_nothing() {
    // spec 009 AC-5: refused at stage 2, before the network and before any weights are read
    // (the GGUF is deleted after the manifest is sealed: the refusal must not need it)
    let fixture = with_declared_sibling(Builder::tiny())
        .after_seal(|dir| std::fs::remove_file(dir.join("model.gguf")).unwrap())
        .build();
    let h = Harness::new();
    let mut req = h.local(fixture.path());
    req.allow_uncalibrated = false;
    let err = h.load(&req).err().unwrap();
    let Error::IncompatibleModel(Incompat::Uncalibrated {
        dtype,
        other_declared,
        ..
    }) = &err
    else {
        panic!("{err:?}")
    };
    assert_eq!(dtype, "q8_0");
    assert_eq!(other_declared, &Some(("bf16".to_string(), 2.09)));
    let text = err.to_string();
    for part in ["(q8_0)", "Jev.calibrate()", "'bf16'", "does not transfer"] {
        assert!(text.contains(part), "{part:?} missing from {text}");
    }
    assert_eq!(h.backend.loads(), 0);
    assert_eq!(h.records(), 0);
    assert_eq!(
        h.transport.gets.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    // the same fixture without a calibrated sibling gets the plain message
    let alone = Builder::tiny().build();
    let mut req = h.local(alone.path());
    req.allow_uncalibrated = false;
    let text = h.load(&req).err().unwrap().to_string();
    assert!(
        text.contains("Jev.calibrate()") && !text.contains("does not transfer"),
        "{text}"
    );
}

#[test]
fn a_calibrated_sibling_does_not_calibrate_the_variant_that_is_loaded() {
    // spec 009 AC-6: with consent the variant runs at T = 1.0 and says so on every answer,
    // although the bf16 variant of the same manifest declares 2.09
    let fixture = with_declared_sibling(Builder::tiny()).build();
    let h = Harness::new();
    let model = h.load(&h.local(fixture.path())).unwrap();
    assert_eq!(
        (
            model.data.dtype.as_str(),
            model.data.temperature,
            model.data.calibrated,
            model.data.calibration_source
        ),
        ("q8_0", 1.0, false, "none")
    );
    assert_eq!(
        (
            model.scorer.calibration().temperature(),
            model.scorer.calibration().calibrated()
        ),
        (1.0, false)
    );
    // and with a temperature of the user's: absolute, calibrated, source "user"
    let mut req = h.local(fixture.path());
    req.temperature = Some(1.7);
    let model = h.load(&req).unwrap();
    assert_eq!(
        (
            model.data.temperature,
            model.data.calibrated,
            model.data.calibration_source
        ),
        (1.7, true, "user")
    );
    assert_eq!(model.scorer.calibration().temperature(), 1.7);
}

#[test]
fn a_loaded_model_reports_every_variant_of_its_manifest() {
    // spec 009 AC-3: `variants` lists what the manifest offers, not only what was loaded, and
    // the declared calibration belongs to the variant that declares it
    let fixture = with_declared_sibling(Builder::tiny()).build();
    let h = Harness::new();
    let model = h.load(&h.local(fixture.path())).unwrap();
    let size = std::fs::metadata(fixture.gguf_path()).unwrap().len();
    let got: Vec<_> = model
        .data
        .variants
        .iter()
        .map(|v| {
            (
                v.dtype.as_str(),
                v.size_bytes,
                v.calibrated,
                v.temperature,
                v.is_default,
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            ("q8_0", Some(size), false, None, true),
            ("bf16", Some(size), true, Some(2.09), false),
        ]
    );
    // what was loaded is the first one, uncalibrated, whatever the second declares
    assert_eq!(
        (model.data.dtype.as_str(), model.data.calibrated),
        ("q8_0", false)
    );
}
