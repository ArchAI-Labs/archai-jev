//! Which device a model loads on, end to end with the fake engine and simulated devices (spec 011,
//! section 2): every box of the table, the self-check on the device, the record per device and the
//! order of the stages. No GPU is needed; the fake engine can compute differently on one backend.

use std::sync::atomic::Ordering;

use super::Harness;
use crate::engine::device::{BackendKind, DeviceKind, FakeProbe, Refusal};
use crate::error::Error;
use crate::models::incompat::Incompat;
use crate::models::load::LoadedModel;
use crate::models::testing::Builder;
use crate::models::testing::fake::Perturb;
use crate::schema::{Choice, ChoiceOption, Question, Questions, State, StateValue, TextEntry};

/// The test family with a CUDA row (`test-fake`, `f32`) and one without it (`qwen2-letters`).
fn with_row() -> Builder {
    Builder::tiny_fake()
}
fn without_row() -> Builder {
    Builder::tiny()
}

fn harness(probe: FakeProbe) -> Harness {
    let mut h = Harness::new();
    h.devices = Box::new(probe);
    h
}

fn on(
    h: &Harness,
    fixture: &crate::models::testing::Fixture,
    device: &str,
) -> crate::error::Result<LoadedModel> {
    h.load(&h.local_on(fixture.path(), device))
}

fn nothing_touched(h: &Harness) {
    assert_eq!(
        h.transport.gets.load(Ordering::SeqCst),
        0,
        "no network request"
    );
    assert_eq!(h.backend.loads(), 0, "the engine was not asked to load");
    assert!(
        !h.cache.path().join("v1").exists(),
        "nothing may be created in the cache"
    );
}

fn warnings(h: &Harness) -> Vec<String> {
    h.recorder
        .events()
        .into_iter()
        .filter_map(|e| match e {
            crate::hub::events::Event::Warning(w) => Some(w),
            _ => None,
        })
        .collect()
}

// ---- the table ---------------------------------------------------------------------------------

#[test]
fn cpu_never_uses_a_gpu() {
    // a laptop with CUDA and Vulkan, a model with a CUDA row: the default is still the CPU
    let h = harness(FakeProbe::rtx_laptop());
    let fixture = with_row().build();
    let model = on(&h, &fixture, "cpu").unwrap();
    assert_eq!(model.data.device, "cpu");
    assert_eq!(model.data.fallback, None);
    assert_eq!(h.backend.devices_loaded(), ["cpu"]);
}

#[test]
fn explicit_without_backend() {
    let h = harness(FakeProbe::cpu_only());
    let fixture = with_row().build();
    for device in ["cuda", "vulkan:1", "metal"] {
        let err = on(&h, &fixture, device).err().unwrap();
        let Error::DeviceUnavailable(Refusal::NotCompiled { compiled, .. }) = &err else {
            panic!("{device}: {err:?}");
        };
        assert_eq!(compiled, &[BackendKind::Cpu]);
        assert!(err.to_string().contains("compiled: cpu"), "{err}");
        nothing_touched(&h);
    }
}

#[test]
fn explicit_without_device() {
    let h = harness(FakeProbe::cpu_only().compiled(BackendKind::Cuda));
    let fixture = with_row().build();
    let err = on(&h, &fixture, "cuda").err().unwrap();
    assert!(
        matches!(&err, Error::DeviceUnavailable(Refusal::NoDevice { found, .. }) if found.is_empty()),
        "{err:?}"
    );
    nothing_touched(&h);
    // a wrong index names the devices that exist
    let h = harness(FakeProbe::rtx_laptop());
    let err = on(&h, &fixture, "cuda:3").err().unwrap();
    assert!(err.to_string().contains("cuda:0"), "{err}");
    nothing_touched(&h);
}

#[test]
fn explicit_without_row() {
    // the default family has no tolerance on a GPU: it may answer on the CPU only
    let h = harness(FakeProbe::rtx_laptop());
    let fixture = without_row().build();
    let err = on(&h, &fixture, "cuda").err().unwrap();
    let Error::IncompatibleModel(Incompat::NoDeviceRow {
        backend, verified, ..
    }) = &err
    else {
        panic!("{err:?}");
    };
    assert_eq!(backend, "cuda");
    assert_eq!(verified, &["cpu".to_string()]);
    assert!(err.to_string().contains("no verified tolerance"), "{err}");
    nothing_touched(&h);
}

#[test]
fn the_grammar_is_checked_before_anything_else() {
    let h = harness(FakeProbe::cpu_only());
    let fixture = with_row().build();
    for (device, hint) in [
        ("mps", "use 'metal'"),
        ("gpu", "'auto'"),
        ("CPU", "lowercase"),
    ] {
        let err = on(&h, &fixture, device).err().unwrap();
        assert!(matches!(err, Error::ModelArgument(_)), "{device}: {err:?}");
        assert!(err.to_string().contains(hint), "{err}");
        nothing_touched(&h);
    }
}

#[test]
fn auto_without_backend_is_silent() {
    let h = harness(FakeProbe::cpu_only());
    let fixture = with_row().build();
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(
        (model.data.device.as_str(), &model.data.fallback),
        ("cpu", &None)
    );
    assert!(
        warnings(&h).iter().all(|w| !w.contains("CPU instead")),
        "{:?}",
        warnings(&h)
    );
}

#[test]
fn auto_falls_back_with_a_reason() {
    // a GPU is there, but this model has no tolerance on it
    let h = harness(FakeProbe::rtx_laptop());
    let fixture = without_row().build();
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(model.data.device, "cpu");
    let reason = model.data.fallback.unwrap();
    assert!(reason.contains("no verified tolerance"), "{reason}");
    // a backend compiled in but no device
    let h = harness(FakeProbe::cpu_only().compiled(BackendKind::Cuda));
    let fixture = with_row().build();
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(model.data.device, "cpu");
    assert!(model.data.fallback.unwrap().contains("no device found"));
}

#[test]
fn auto_skips_integrated_gpus() {
    let h = harness(FakeProbe::cpu_only().device(
        BackendKind::Vulkan,
        DeviceKind::Igpu,
        "Intel(R) Graphics",
    ));
    let fixture = with_row().build();
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(model.data.device, "cpu");
    assert!(model.data.fallback.unwrap().contains("integrated"));
}

#[test]
fn an_integrated_gpu_is_refused_by_name() {
    let fixture = with_row().build();
    for device in ["vulkan", "vulkan:0"] {
        let h = harness(FakeProbe::cpu_only().device(
            BackendKind::Vulkan,
            DeviceKind::Igpu,
            "Intel(R) Graphics",
        ));
        let err = on(&h, &fixture, device).err().unwrap();
        assert!(
            matches!(err, Error::DeviceUnavailable(_)),
            "{device}: {err:?}"
        );
        assert!(err.to_string().contains("integrated GPU"), "{err}");
        assert_eq!(h.backend.runs(), 0, "nothing ran on it");
    }
}

#[test]
fn auto_and_explicit_use_the_verified_gpu() {
    let fixture = with_row().build();
    for device in ["auto", "cuda", "cuda:0"] {
        let h = harness(FakeProbe::rtx_laptop());
        let model = on(&h, &fixture, device).unwrap();
        assert_eq!(model.data.device, "cuda:0", "{device}");
        assert_eq!(model.data.fallback, None, "{device}");
        // the engine was told which device: its identity carries the description
        let loaded = h.backend.devices_loaded();
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].starts_with("cuda:0/"), "{loaded:?}");
        assert_eq!(h.backend.runs(), 1, "the self-check ran on the device");
    }
}

#[test]
fn fallback_is_reported_in_model_data() {
    // the engine cannot use the GPU: auto continues on the CPU and says why; explicit refuses
    let fixture = with_row().build();
    let h = harness(FakeProbe::rtx_laptop());
    let h = Harness {
        backend: h.backend.clone().failing_load_on(BackendKind::Cuda),
        ..h
    };
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(model.data.device, "cpu");
    let reason = model.data.fallback.clone().unwrap();
    assert!(
        reason.contains("cuda:0") && reason.contains("cannot use"),
        "{reason}"
    );
    let loaded = h.backend.devices_loaded();
    assert_eq!(loaded.len(), 2);
    assert!(
        loaded[0].starts_with("cuda:0/") && loaded[1] == "cpu",
        "{loaded:?}"
    );
    let err = on(&h, &fixture, "cuda").err().unwrap();
    assert!(matches!(err, Error::Inference { .. }), "{err:?}");
}

// ---- the self-check on the device --------------------------------------------------------------

#[test]
fn device_selfcheck_beyond_tolerance_is_refused_with_explicit_and_auto() {
    let fixture = with_row().build();
    for device in ["cuda", "auto"] {
        let h = harness(FakeProbe::rtx_laptop());
        let h = Harness {
            backend: h
                .backend
                .clone()
                .perturbed_on(BackendKind::Cuda, Perturb::LogitShift(0.5)),
            ..h
        };
        let err = on(&h, &fixture, device).err().unwrap();
        assert!(
            matches!(err, Error::ModelVerification(_)),
            "{device}: {err:?}"
        );
        // no fallback to the CPU after wrong numbers: the engine was asked once, for the GPU
        let loaded = h.backend.devices_loaded();
        assert_eq!(loaded.len(), 1, "{device}: {loaded:?}");
        assert!(loaded[0].starts_with("cuda:0/"));
        assert_eq!(
            h.records(),
            0,
            "{device}: a failed self-check leaves no record"
        );
    }
}

#[test]
fn device_selfcheck_within_tolerance_passes() {
    // the CUDA row is tighter than the CPU one: a gap the CPU would accept is refused on the GPU
    let fixture = with_row().build();
    let shifted = |gap: f64| {
        let h = harness(FakeProbe::rtx_laptop());
        Harness {
            backend: h
                .backend
                .clone()
                .perturbed_on(BackendKind::Cuda, Perturb::LogitShift(gap)),
            ..h
        }
    };
    let h = shifted(2e-4);
    let model = on(&h, &fixture, "cuda").unwrap();
    assert_eq!(model.data.device, "cuda:0");
    assert_eq!(h.records(), 1);
    // 7e-4 is inside the CPU tolerance (1e-3) and outside the CUDA one (5e-4)
    let h = shifted(7e-4);
    assert!(matches!(
        on(&h, &fixture, "cuda").err().unwrap(),
        Error::ModelVerification(_)
    ));
    let cpu_backend = h
        .backend
        .clone()
        .perturbed_on(BackendKind::Cpu, Perturb::LogitShift(7e-4));
    let cpu_harness = Harness {
        backend: cpu_backend,
        ..harness(FakeProbe::rtx_laptop())
    };
    assert!(on(&cpu_harness, &fixture, "cpu").is_ok());
}

#[test]
fn an_engine_failure_during_the_selfcheck_is_not_a_verification_failure() {
    // out of memory while the self-check creates its context: explicit raises the engine error,
    // auto continues on the CPU, and neither is a wrong-number error
    let fixture = with_row().build();
    let make = || {
        let h = harness(FakeProbe::rtx_laptop());
        Harness {
            backend: h.backend.clone().failing_run_on(BackendKind::Cuda),
            ..h
        }
    };
    let h = make();
    let err = on(&h, &fixture, "cuda").err().unwrap();
    assert!(matches!(err, Error::Inference { .. }), "{err:?}");
    assert_eq!(h.records(), 0);
    let h = make();
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(model.data.device, "cpu");
    assert!(model.data.fallback.unwrap().contains("out of memory"));
    assert_eq!(h.records(), 1, "only the CPU run was recorded");
}

#[test]
fn a_record_is_per_device_in_the_pipeline() {
    let fixture = with_row().build();
    let h = harness(FakeProbe::rtx_laptop());
    on(&h, &fixture, "cuda").unwrap();
    on(&h, &fixture, "cuda").unwrap();
    assert_eq!(
        h.backend.runs(),
        1,
        "the second open on the same device runs no vector"
    );
    // the CPU is another device: its own self-check, its own record
    on(&h, &fixture, "cpu").unwrap();
    assert_eq!((h.backend.runs(), h.records()), (2, 2));
    on(&h, &fixture, "cpu").unwrap();
    on(&h, &fixture, "cuda").unwrap();
    assert_eq!(h.backend.runs(), 2);
}

// ---- a request on a loaded model ----------------------------------------------------------------

fn one_question() -> (State, Questions) {
    let state = State::new(StateValue::string("a state")).unwrap();
    let choice = Choice::new(
        TextEntry::text("which?").unwrap(),
        vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
    )
    .unwrap();
    (
        state,
        Questions::new([("q", Question::from(choice))]).unwrap(),
    )
}

#[test]
fn a_request_error_keeps_the_device() {
    let fixture = with_row().build();
    let h = harness(FakeProbe::rtx_laptop());
    let model = on(&h, &fixture, "auto").unwrap();
    assert_eq!(model.data.device, "cuda:0");
    let (state, questions) = one_question();
    h.backend.fail_scores(true);
    let err = model.scorer.score(&state, &questions).unwrap_err();
    assert!(matches!(err, Error::Inference { .. }), "{err:?}");
    // nothing reloaded, nothing moved to the CPU, and the next request still works
    assert_eq!(h.backend.loads(), 1);
    assert_eq!(model.data.device, "cuda:0");
    h.backend.fail_scores(false);
    assert!(model.scorer.score(&state, &questions).is_ok());
    assert_eq!(h.backend.devices_loaded().len(), 1);
}
