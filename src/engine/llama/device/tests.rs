//! Tests of the device side of the engine (spec 011): the parameters built from a plan, the CPU
//! check for every device, what a ggml device is, and the rule that no model is ever created
//! from the defaults of llama.cpp. None of them needs a model file or a GPU.

use std::path::{Path, PathBuf};

use super::*;
use crate::engine::cpu::REQUIRED;

struct Without(&'static str);
impl CpuFeatures for Without {
    fn has(&self, name: &str) -> bool {
        name != self.0
    }
}

/// The plan of a GPU: every layer, with the device number of the processor because a machine
/// without a GPU backend has no other number to give. What matters here is the shape.
fn gpu_shaped() -> OffloadPlan {
    OffloadPlan {
        layers: Layers::All,
        device: cpu_plan().unwrap().device,
        op_offload: true,
    }
}

#[test]
fn params_are_built_from_the_plan_only() {
    let cpu = cpu_plan().unwrap();
    assert_eq!(cpu.layers, Layers::None);
    assert!(!cpu.op_offload);
    let params = model_params(&cpu).unwrap();
    assert_eq!(params.n_gpu_layers(), 0, "the CPU plan offloads nothing");
    assert_eq!(params.devices(), vec![cpu.device], "and names one device");

    let gpu = model_params(&gpu_shaped()).unwrap();
    assert_eq!(gpu.n_gpu_layers(), 999, "a GPU plan offloads every layer");
    assert_eq!(gpu.devices().len(), 1, "to one device, never several");
}

#[test]
fn a_device_the_engine_does_not_have_is_an_error_not_a_default() {
    let nowhere = OffloadPlan {
        layers: Layers::None,
        device: 999,
        op_offload: false,
    };
    let err = model_params(&nowhere).err().unwrap();
    assert!(matches!(err, Error::Inference { .. }), "{err:?}");
    assert!(err.to_string().contains("no device number 999"), "{err}");
    assert!(device_label(&nowhere).is_err());
}

#[test]
fn the_cpu_check_runs_for_every_device() {
    for plan in [cpu_plan().unwrap(), gpu_shaped()] {
        for feature in REQUIRED {
            let err = prepare_model(&plan, &Without(feature)).err().unwrap();
            assert!(
                err.to_string().contains(&feature.to_uppercase()),
                "{plan:?} {feature}: {err}"
            );
        }
        assert!(prepare_model(&plan, &Without("none")).is_ok(), "{plan:?}");
    }
}

#[test]
fn what_a_ggml_device_is() {
    use BackendKind::{Cpu, Cuda, Metal, Vulkan};
    use DeviceKind as K;
    use LlamaBackendDeviceType as T;
    for (registry, ty, want) in [
        ("CPU", T::Cpu, Some((Cpu, K::Cpu))),
        ("CUDA", T::Gpu, Some((Cuda, K::Gpu))),
        ("cuda", T::Gpu, Some((Cuda, K::Gpu))),
        ("Vulkan", T::Gpu, Some((Vulkan, K::Gpu))),
        ("Vulkan", T::IntegratedGpu, Some((Vulkan, K::Igpu))),
        ("Metal", T::Gpu, Some((Metal, K::Gpu))),
        ("MTL", T::IntegratedGpu, Some((Metal, K::Igpu))),
        // what we do not offer
        ("BLAS", T::Accelerator, None),
        ("CUDA", T::Accelerator, None),
        ("RPC", T::Gpu, None),
        ("HIP", T::Gpu, None),
        ("Vulkan", T::Unknown, None),
    ] {
        assert_eq!(classify(registry, ty), want, "{registry} {ty:?}");
    }
}

#[test]
fn compiled_backends_follow_the_features() {
    use BackendKind::{Cpu, Cuda, Metal, Vulkan};
    for (cuda, vulkan, metal, want) in [
        (false, false, false, vec![Cpu]),
        (true, false, false, vec![Cpu, Cuda]),
        (false, true, false, vec![Cpu, Vulkan]),
        (false, false, true, vec![Cpu, Metal]),
        (true, true, true, vec![Cpu, Cuda, Vulkan, Metal]),
    ] {
        assert_eq!(
            compiled_from(cuda, vulkan, metal),
            want,
            "{cuda} {vulkan} {metal}"
        );
    }
    // and the real list reads the real features of this build
    assert_eq!(
        compiled_backends(),
        compiled_from(
            cfg!(feature = "cuda"),
            cfg!(feature = "vulkan"),
            cfg!(feature = "metal")
        )
    );
    assert_eq!(compiled_backends().contains(&Cuda), cfg!(feature = "cuda"));
    assert_eq!(
        compiled_backends().contains(&Vulkan),
        cfg!(feature = "vulkan")
    );
    assert_eq!(
        compiled_backends().contains(&Metal),
        cfg!(feature = "metal")
    );
    assert_eq!(compiled_backends().first(), Some(&Cpu));
}

#[test]
fn the_probe_lists_this_machine() {
    let devices = LlamaProbe.devices();
    let cpus: Vec<_> = devices
        .iter()
        .filter(|d| d.backend == BackendKind::Cpu)
        .collect();
    assert_eq!(cpus.len(), 1, "{devices:?}");
    assert_eq!(cpus[0].kind, DeviceKind::Cpu);
    assert_eq!(cpus[0].name(), "cpu");
    // what is compiled in follows the features; a CPU wheel lists nothing but the processor
    assert_eq!(LlamaProbe.compiled_backends(), compiled_backends());
    if compiled_backends() == [BackendKind::Cpu] {
        assert_eq!(devices.len(), 1, "{devices:?}");
    }
    // the indices of each backend count from 0 in the order the engine lists them
    for backend in [BackendKind::Cuda, BackendKind::Vulkan, BackendKind::Metal] {
        let own: Vec<usize> = devices
            .iter()
            .filter(|d| d.backend == backend)
            .map(|d| d.index)
            .collect();
        assert_eq!(own, (0..own.len()).collect::<Vec<_>>(), "{backend}");
    }
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The code of a file without its test modules and without comments, so that a line that only
/// talks about the defaults does not count.
fn code_of(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap();
    let cut = text.find("#[cfg(test)]").unwrap_or(text.len());
    text[..cut]
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn no_model_is_created_from_the_defaults_of_llama_cpp() {
    // `LlamaModelParams::default()` is where `n_gpu_layers = -1` comes from: the only place that
    // may call it is `model_params`, which sets the layers and the device right after.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(files.len() > 50, "the source tree was not found");
    let users: Vec<String> = files
        .iter()
        .filter(|f| code_of(f).contains("LlamaModelParams::default()"))
        .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(users, ["device.rs"], "{users:?}");

    let device = code_of(&src.join("engine/llama/device.rs"));
    let at = device.find("LlamaModelParams::default()").unwrap();
    let after: String = device[at..]
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(
        after, "LlamaModelParams::default().with_n_gpu_layers(layers).with_devices(&[plan.device])",
        "the defaults must be followed by the layers and the device of the plan"
    );

    // no model is loaded from anything but the parameters of the plan
    for file in &files {
        let code = code_of(file);
        if code.contains("load_from_file(") {
            assert!(
                code.contains("prepare_model(&config.plan"),
                "{file:?} loads a model without the parameters of its plan"
            );
        }
    }
}
