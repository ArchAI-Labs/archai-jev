//! A [`DeviceProbe`] with a chosen list, so the table of spec 011 section 2 is tested without a GPU.

use super::{BackendKind, DeviceInfo, DeviceKind, DeviceProbe};

/// A probe with a chosen list of backends and devices.
#[derive(Debug, Clone)]
pub struct FakeProbe {
    compiled: Vec<BackendKind>,
    devices: Vec<DeviceInfo>,
}

impl FakeProbe {
    /// A build with the CPU only: the wheels on PyPI.
    pub fn cpu_only() -> Self {
        let mut p = FakeProbe {
            compiled: vec![BackendKind::Cpu],
            devices: Vec::new(),
        };
        p.devices.push(DeviceInfo {
            backend: BackendKind::Cpu,
            index: 0,
            engine_index: 0,
            kind: DeviceKind::Cpu,
            description: "Test CPU".to_string(),
            memory_total_mib: None,
            memory_free_mib: None,
        });
        p
    }

    /// Add a backend to the build (it can have no device).
    #[must_use]
    pub fn compiled(mut self, backend: BackendKind) -> Self {
        if !self.compiled.contains(&backend) {
            self.compiled.push(backend);
        }
        self
    }

    /// Add a device (and its backend to the build). `index` follows the devices of that backend
    /// already added; `engine_index` follows the whole list, like the engine's own.
    #[must_use]
    pub fn device(mut self, backend: BackendKind, kind: DeviceKind, description: &str) -> Self {
        self = self.compiled(backend);
        let index = self.devices.iter().filter(|d| d.backend == backend).count();
        let engine_index = self.devices.len();
        self.devices.push(DeviceInfo {
            backend,
            index,
            engine_index,
            kind,
            description: description.to_string(),
            memory_total_mib: Some(8_151),
            memory_free_mib: Some(7_000),
        });
        self
    }

    /// The development machine: an NVIDIA GPU through CUDA and Vulkan, and an Intel iGPU that
    /// only Vulkan sees, listed in the order Vulkan reports them (Intel 0, NVIDIA 1).
    pub fn rtx_laptop() -> Self {
        FakeProbe::cpu_only()
            .device(
                BackendKind::Cuda,
                DeviceKind::Gpu,
                "NVIDIA GeForce RTX 5070",
            )
            .device(BackendKind::Vulkan, DeviceKind::Igpu, "Intel(R) Graphics")
            .device(
                BackendKind::Vulkan,
                DeviceKind::Gpu,
                "NVIDIA GeForce RTX 5070",
            )
    }
}

impl DeviceProbe for FakeProbe {
    fn compiled_backends(&self) -> Vec<BackendKind> {
        self.compiled.clone()
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        self.devices.clone()
    }
}
