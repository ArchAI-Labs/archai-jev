//! Tests of the device grammar, the selection table and the offload plan (spec 011).

use super::*;

fn every_row(_: BackendKind) -> bool {
    true
}
fn no_row(_: BackendKind) -> bool {
    false
}

fn chosen(
    request: &str,
    probe: &FakeProbe,
    has_row: &dyn Fn(BackendKind) -> bool,
) -> Result<Selected, Refusal> {
    select(DeviceRequest::parse(request).unwrap(), probe, has_row)
}

// ---- the grammar ------------------------------------------------------------------------

#[test]
fn grammar_accepts_the_valid_names() {
    use BackendKind::{Cuda, Metal, Vulkan};
    let named = |backend, index| DeviceRequest::Named { backend, index };
    for (text, want) in [
        ("cpu", DeviceRequest::Cpu),
        ("auto", DeviceRequest::Auto),
        ("cuda", named(Cuda, 0)),
        ("cuda:0", named(Cuda, 0)),
        ("cuda:1", named(Cuda, 1)),
        ("vulkan", named(Vulkan, 0)),
        ("vulkan:12", named(Vulkan, 12)),
        ("metal", named(Metal, 0)),
        ("metal:0", named(Metal, 0)),
    ] {
        assert_eq!(DeviceRequest::parse(text), Ok(want), "{text}");
    }
    // "cuda" is "cuda:0"
    assert_eq!(DeviceRequest::parse("cuda").unwrap().name(), "cuda:0");
}

#[test]
fn grammar_rejects_the_rest_and_lists_the_valid_ones() {
    for text in [
        "gpu",
        "CPU",
        "Cuda",
        "AUTO",
        "mps",
        "",
        " ",
        " cpu",
        "cpu ",
        "cuda ",
        "cuda:",
        ":0",
        "cuda:-1",
        "cuda:+1",
        "cuda:01",
        "cuda:1.0",
        "cuda:a",
        "cuda: 1",
        "cuda:1:2",
        "cpu:0",
        "auto:0",
        "opencl",
        "rocm",
        "cuda0",
        "cuda:99999999999999999999999",
    ] {
        let err = DeviceRequest::parse(text).unwrap_err();
        assert_eq!(err.given, text);
        let message = err.to_string();
        for name in [
            "'cpu'", "'auto'", "'cuda'", "'vulkan'", "'metal'", "'cuda:1'",
        ] {
            assert!(message.contains(name), "{text:?}: {message}");
        }
    }
}

#[test]
fn the_error_helps_with_the_common_mistakes() {
    let hint = |text: &str| DeviceRequest::parse(text).unwrap_err().to_string();
    assert!(hint("mps").contains("use 'metal'"));
    assert!(hint("gpu").contains("'auto'"));
    assert!(hint("CPU").contains("did you mean 'cpu'"));
    assert!(hint("CUDA:1").contains("did you mean 'cuda:1'"));
    // no hint when there is nothing useful to say
    assert!(!hint("opencl").contains("did you mean"));
}

// ---- the table of spec 011 section 2 ------------------------------------------------------

#[test]
fn cpu_never_uses_a_gpu() {
    // a build with every backend, a model with a row everywhere: "cpu" is still the CPU
    let probe = FakeProbe::rtx_laptop().compiled(BackendKind::Metal);
    let got = chosen("cpu", &probe, &every_row).unwrap();
    assert_eq!(got.device.backend, BackendKind::Cpu);
    assert_eq!(got.fallback, None);
    assert_eq!(got.name(), "cpu");
    assert!(!got.plan().uses_gpu());
    // and the same on a build that has no GPU backend at all
    let plain = chosen("cpu", &FakeProbe::cpu_only(), &no_row).unwrap();
    assert_eq!(plain.name(), "cpu");
}

#[test]
fn explicit_without_backend() {
    let probe = FakeProbe::cpu_only();
    for (text, backend) in [
        ("cuda", BackendKind::Cuda),
        ("vulkan:1", BackendKind::Vulkan),
        ("metal", BackendKind::Metal),
    ] {
        let err = chosen(text, &probe, &every_row).unwrap_err();
        assert_eq!(
            err,
            Refusal::NotCompiled {
                backend,
                compiled: vec![BackendKind::Cpu]
            },
            "{text}"
        );
        let message = err.to_string();
        assert!(message.contains(&format!("'{backend}'")), "{message}");
        assert!(message.contains("compiled: cpu"), "{message}");
        assert!(message.contains("docs/gpu.md"), "{message}");
    }
    // a build with CUDA does not make Vulkan appear, and the message lists what it has
    let cuda_build = FakeProbe::cpu_only().compiled(BackendKind::Cuda);
    let err = chosen("vulkan", &cuda_build, &every_row).unwrap_err();
    assert!(err.to_string().contains("compiled: cpu, cuda"), "{err}");
}

#[test]
fn explicit_without_device() {
    // compiled in, but the machine has none (no driver, a VM)
    let probe = FakeProbe::cpu_only().compiled(BackendKind::Cuda);
    let err = chosen("cuda", &probe, &every_row).unwrap_err();
    assert_eq!(
        err,
        Refusal::NoDevice {
            wanted: "cuda:0".to_string(),
            found: vec![]
        }
    );
    assert!(err.to_string().contains("driver"), "{err}");
    // a wrong index on a machine that has a device: the message lists the real ones
    let laptop = FakeProbe::rtx_laptop();
    let err = chosen("vulkan:2", &laptop, &every_row).unwrap_err();
    assert_eq!(
        err,
        Refusal::NoDevice {
            wanted: "vulkan:2".to_string(),
            found: vec!["vulkan:0".to_string(), "vulkan:1".to_string()]
        }
    );
    assert!(err.to_string().contains("vulkan:0, vulkan:1"), "{err}");
}

#[test]
fn explicit_without_row() {
    let laptop = FakeProbe::rtx_laptop();
    let only_vulkan = |b: BackendKind| b == BackendKind::Vulkan;
    let err = chosen("cuda", &laptop, &only_vulkan).unwrap_err();
    assert_eq!(
        err,
        Refusal::NoRow {
            backend: BackendKind::Cuda
        }
    );
    assert!(err.to_string().contains("no verified tolerance"), "{err}");
    // the same request with a row works, and says nothing about a fallback
    let ok = chosen("vulkan:1", &laptop, &only_vulkan).unwrap();
    assert_eq!(ok.name(), "vulkan:1");
    assert_eq!(ok.fallback, None);
    // a missing device is reported before a missing row
    let err = chosen("cuda:3", &laptop, &no_row).unwrap_err();
    assert!(matches!(err, Refusal::NoDevice { .. }), "{err:?}");
}

#[test]
fn auto_without_backend_is_silent() {
    let got = chosen("auto", &FakeProbe::cpu_only(), &every_row).unwrap();
    assert_eq!(got.name(), "cpu");
    assert_eq!(got.fallback, None, "there was no other choice: no warning");
}

#[test]
fn auto_falls_back_with_a_reason() {
    // compiled in, no device
    let probe = FakeProbe::cpu_only().compiled(BackendKind::Cuda);
    let got = chosen("auto", &probe, &every_row).unwrap();
    assert_eq!(got.name(), "cpu");
    let reason = got.fallback.unwrap();
    assert!(
        reason.contains("cuda: compiled in but no device"),
        "{reason}"
    );
    // a device but no row for the model
    let laptop = FakeProbe::rtx_laptop();
    let got = chosen("auto", &laptop, &no_row).unwrap();
    assert_eq!(got.name(), "cpu");
    let reason = got.fallback.unwrap();
    assert!(
        reason.contains("cuda: this model has no verified tolerance"),
        "{reason}"
    );
    assert!(
        reason.contains("vulkan: this model has no verified tolerance"),
        "{reason}"
    );
}

#[test]
fn auto_skips_integrated_gpus() {
    // only an integrated Vulkan GPU: auto does not take it and says why
    let intel_only =
        FakeProbe::cpu_only().device(BackendKind::Vulkan, DeviceKind::Igpu, "Intel(R) Graphics");
    let got = chosen("auto", &intel_only, &every_row).unwrap();
    assert_eq!(got.name(), "cpu");
    assert!(got.fallback.unwrap().contains("only integrated GPUs"));
    // naming it is refused too: the tolerance row of the backend is the one of its discrete GPUs
    for name in ["vulkan", "vulkan:0"] {
        let err = chosen(name, &intel_only, &every_row).err().unwrap();
        assert_eq!(
            err,
            Refusal::IntegratedGpu {
                device: "vulkan:0".to_string()
            }
        );
        assert!(err.to_string().contains("integrated GPU"), "{err}");
    }
    // with the NVIDIA GPU also there, auto takes it and not the integrated one that comes first
    let vulkan_only = FakeProbe::cpu_only()
        .device(BackendKind::Vulkan, DeviceKind::Igpu, "Intel(R) Graphics")
        .device(
            BackendKind::Vulkan,
            DeviceKind::Gpu,
            "NVIDIA GeForce RTX 5070",
        );
    let got = chosen("auto", &vulkan_only, &every_row).unwrap();
    assert_eq!(got.name(), "vulkan:1");
    assert_eq!(got.fallback, None);
}

#[test]
fn auto_prefers_cuda_then_metal_then_vulkan() {
    let laptop = FakeProbe::rtx_laptop();
    assert_eq!(
        chosen("auto", &laptop, &every_row).unwrap().name(),
        "cuda:0"
    );
    // without a CUDA row it goes to the next backend instead of falling back to the CPU
    let no_cuda = |b: BackendKind| b != BackendKind::Cuda;
    assert_eq!(
        chosen("auto", &laptop, &no_cuda).unwrap().name(),
        "vulkan:1"
    );
    let mac = FakeProbe::cpu_only()
        .device(BackendKind::Metal, DeviceKind::Gpu, "Apple M2")
        .device(BackendKind::Vulkan, DeviceKind::Gpu, "MoltenVK");
    assert_eq!(chosen("auto", &mac, &every_row).unwrap().name(), "metal:0");
    // a skipped backend does not hide the reason when nothing is left
    let no_vulkan = |b: BackendKind| b != BackendKind::Vulkan;
    let got = chosen(
        "auto",
        &FakeProbe::cpu_only().device(BackendKind::Vulkan, DeviceKind::Gpu, "x"),
        &no_vulkan,
    )
    .unwrap();
    assert!(
        got.fallback
            .unwrap()
            .starts_with("using the CPU instead of a GPU")
    );
}

#[test]
fn a_probe_without_a_cpu_is_an_error_not_a_guess() {
    struct Empty;
    impl DeviceProbe for Empty {
        fn compiled_backends(&self) -> Vec<BackendKind> {
            vec![BackendKind::Cpu]
        }
        fn devices(&self) -> Vec<DeviceInfo> {
            Vec::new()
        }
    }
    let err = select(DeviceRequest::Cpu, &Empty, &every_row).unwrap_err();
    assert!(matches!(err, Refusal::NoDevice { .. }), "{err:?}");
}

// ---- the plan --------------------------------------------------------------------------

#[test]
fn offload_plan_for_cpu() {
    // the CPU of a machine whose list starts with a GPU: the plan names the CPU's own entry
    let mut cpu = FakeProbe::cpu_only().devices().remove(0);
    cpu.engine_index = 4;
    let plan = OffloadPlan::for_device(&cpu);
    assert_eq!(
        plan,
        OffloadPlan {
            layers: Layers::None,
            device: 4,
            op_offload: false
        }
    );
    assert!(!plan.uses_gpu());
}

#[test]
fn offload_plan_for_a_gpu_has_one_device() {
    let laptop = FakeProbe::rtx_laptop();
    let got = chosen("cuda", &laptop, &every_row).unwrap();
    let plan = got.plan();
    assert_eq!(
        plan.layers,
        Layers::All,
        "all or nothing: no partial offload"
    );
    assert_eq!(plan.device, got.device.engine_index);
    assert!(plan.op_offload);
    assert!(plan.uses_gpu());
    // the plan is a plain Copy value: LlamaConfig, which is Copy, will hold it
    let copy = plan;
    assert_eq!(copy, plan);
    // every GPU backend gets the same shape of plan, each with its own engine index
    for text in ["cuda:0", "vulkan:1"] {
        let sel = chosen(text, &laptop, &every_row).unwrap();
        assert_eq!(sel.plan().device, sel.device.engine_index, "{text}");
        assert_eq!(sel.plan().layers, Layers::All, "{text}");
    }
}

#[test]
fn a_device_name_round_trips_through_the_grammar() {
    let laptop = FakeProbe::rtx_laptop();
    for d in laptop.devices() {
        let name = d.name();
        if d.kind == DeviceKind::Igpu {
            // a name the grammar accepts and the library refuses: the integrated GPU
            assert!(chosen(&name, &laptop, &every_row).is_err(), "{name}");
            continue;
        }
        let sel = chosen(&name, &laptop, &every_row).unwrap();
        assert_eq!(sel.device, d, "{name}");
    }
}
