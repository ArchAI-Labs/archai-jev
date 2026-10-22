//! The device side of the llama.cpp engine (spec 011): which devices llama.cpp sees and how a
//! model is told where to run.
//!
//! llama.cpp has two defaults that would silently put a CPU request on a GPU in a build with a
//! GPU backend: `n_gpu_layers = -1` ("everything on the GPU if there is one") and `op_offload =
//! true`. The model is therefore **never** created from those defaults: [`model_params`] is the
//! only place that creates `LlamaModelParams`, and it always sets the layers and the one device
//! from an [`OffloadPlan`] (a test reads the source to keep it that way).

use std::sync::Once;

use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::{
    LlamaBackendDeviceType, LogOptions, list_llama_ggml_backend_devices, send_logs_to_tracing,
};

use crate::engine::cpu::{self, CpuFeatures};
use crate::engine::device::{
    BackendKind, DeviceInfo, DeviceKind, DeviceProbe, Layers, OffloadPlan,
};
use crate::error::{Error, Result};

fn inference(detail: impl Into<String>) -> Error {
    Error::Inference {
        detail: detail.into(),
    }
}

/// Layers asked of the device for [`Layers::All`]. llama.cpp offloads as many as the model has;
/// 999 is what the spike S14 measured with and no model of ours has a hundred layers.
const ALL_LAYERS: u32 = 999;

/// Silence the logs of ggml and llama.cpp **before** anything starts them. Registering a GPU
/// backend prints a banner on `stderr` ("found 1 CUDA devices...") the first time the devices are
/// listed, which is before `LlamaBackend::init` has a chance to silence it.
pub fn quiet_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| send_logs_to_tracing(LogOptions::default().with_logs_enabled(false)));
}

/// The parameters of a model for `plan`: the layers on the device and the one device it uses.
///
/// # Errors
/// [`Error::Inference`] if the engine has no device with that number.
pub fn model_params(plan: &OffloadPlan) -> Result<LlamaModelParams> {
    let layers = match plan.layers {
        Layers::None => 0,
        Layers::All => ALL_LAYERS,
    };
    LlamaModelParams::default()
        .with_n_gpu_layers(layers)
        .with_devices(&[plan.device])
        .map_err(|e| {
            inference(format!(
                "the engine has no device number {}: {e}",
                plan.device
            ))
        })
}

/// [`model_params`] after the check of the CPU, which runs **for every device**: ggml's CPU
/// backend executes part of the work even when the layers are on a GPU, and a CPU without AVX2
/// would die with "illegal instruction" (spec 011, section 3 rule 6).
///
/// # Errors
/// [`Error::Inference`] naming the missing CPU feature, or the unknown device.
pub fn prepare_model(plan: &OffloadPlan, host: &dyn CpuFeatures) -> Result<LlamaModelParams> {
    cpu::check(host)?;
    model_params(plan)
}

/// What kind of device a ggml device is, and which of our backends drives it. `None` for what
/// this crate does not offer (accelerators such as BLAS, remote devices, backends we do not know).
pub fn classify(
    registry_name: &str,
    device_type: LlamaBackendDeviceType,
) -> Option<(BackendKind, DeviceKind)> {
    let backend = match registry_name.to_ascii_lowercase().as_str() {
        "cuda" => BackendKind::Cuda,
        "vulkan" => BackendKind::Vulkan,
        "metal" | "mtl" => BackendKind::Metal,
        _ => BackendKind::Cpu,
    };
    match (device_type, backend) {
        (LlamaBackendDeviceType::Cpu, _) => Some((BackendKind::Cpu, DeviceKind::Cpu)),
        (_, BackendKind::Cpu) => None,
        (LlamaBackendDeviceType::Gpu, b) => Some((b, DeviceKind::Gpu)),
        (LlamaBackendDeviceType::IntegratedGpu, b) => Some((b, DeviceKind::Igpu)),
        (LlamaBackendDeviceType::Accelerator | LlamaBackendDeviceType::Unknown, _) => None,
    }
}

fn mib(bytes: usize) -> Option<u64> {
    (bytes > 0).then(|| u64::try_from(bytes / (1024 * 1024)).unwrap_or(u64::MAX))
}

/// The backends this build of the engine was compiled with: the processor, and each GPU backend
/// whose Cargo feature is on. The wheels on PyPI have none besides the processor.
///
/// A feature is only a request to CMake; the proof that it produced a backend is the build of the
/// feature in CI, which asserts this list on the artefact. `build.rs` makes the one case where a
/// feature is accepted and produces nothing (`metal` without `GGML_METAL=ON`) a build error.
pub fn compiled_backends() -> Vec<BackendKind> {
    compiled_from(
        cfg!(feature = "cuda"),
        cfg!(feature = "vulkan"),
        cfg!(feature = "metal"),
    )
}

/// The list for these features, in the order the devices are listed.
pub fn compiled_from(cuda: bool, vulkan: bool, metal: bool) -> Vec<BackendKind> {
    let mut backends = vec![BackendKind::Cpu];
    if cuda {
        backends.push(BackendKind::Cuda);
    }
    if vulkan {
        backends.push(BackendKind::Vulkan);
    }
    if metal {
        backends.push(BackendKind::Metal);
    }
    backends
}

/// llama.cpp as a [`DeviceProbe`]: the devices ggml registered, in its own order.
#[derive(Debug, Clone, Copy, Default)]
pub struct LlamaProbe;

impl DeviceProbe for LlamaProbe {
    fn compiled_backends(&self) -> Vec<BackendKind> {
        compiled_backends()
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        quiet_logs();
        let mut counts = [0usize; 4];
        let mut out = Vec::new();
        for d in list_llama_ggml_backend_devices() {
            let Some((backend, kind)) = classify(&d.backend, d.device_type) else {
                continue;
            };
            let slot = match backend {
                BackendKind::Cpu => 0,
                BackendKind::Cuda => 1,
                BackendKind::Vulkan => 2,
                BackendKind::Metal => 3,
            };
            let index = counts.get(slot).copied().unwrap_or(0);
            if let Some(count) = counts.get_mut(slot) {
                *count += 1;
            }
            out.push(DeviceInfo {
                backend,
                index,
                engine_index: d.index,
                kind,
                description: d.description,
                memory_total_mib: mib(d.memory_total),
                memory_free_mib: mib(d.memory_free),
            });
        }
        out
    }
}

/// The plan that runs a model on the processor of this machine: nothing offloaded, and the
/// engine's own CPU device as the only one.
///
/// # Errors
/// [`Error::Inference`] if the engine lists no CPU device.
pub fn cpu_plan() -> Result<OffloadPlan> {
    LlamaProbe
        .devices()
        .iter()
        .find(|d| d.backend == BackendKind::Cpu)
        .map(OffloadPlan::for_device)
        .ok_or_else(|| inference("the engine lists no CPU device"))
}

/// What identifies the device of a plan in the identity of a saved prefix: its backend, its name
/// and its description as the engine reports them, so another GPU in the same slot is another
/// device.
///
/// # Errors
/// [`Error::Inference`] if the engine has no device with that number.
pub fn device_label(plan: &OffloadPlan) -> Result<String> {
    quiet_logs();
    list_llama_ggml_backend_devices()
        .into_iter()
        .find(|d| d.index == plan.device)
        .map(|d| format!("{}/{}/{}", d.backend, d.name, d.description))
        .ok_or_else(|| inference(format!("the engine has no device number {}", plan.device)))
}

#[cfg(test)]
mod tests;
