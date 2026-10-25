//! Which device runs the model (spec 011, sections 1-3). Pure: no llama.cpp, no PyO3.
//!
//! Everything that decides *where* a model runs is here, as plain values:
//!
//! * [`DeviceRequest::parse`] turns the `device=` string into a request (`"cpu"`, `"auto"`, or a
//!   backend with an optional index);
//! * [`select`] applies the table of spec 011 section 2 to a request, what the build contains and
//!   what the machine has (a [`DeviceProbe`]), and returns a [`Selected`] device or a typed
//!   [`Refusal`]. Nobody else repeats that reasoning;
//! * [`OffloadPlan`] is the answer in the form the engine needs: how many layers go to which
//!   device. It is built from a [`Selected`] and **never** left to the defaults of llama.cpp,
//!   where `n_gpu_layers = -1` means "everything on the GPU if there is one" (section 3, rule 1).
//!
//! The rows of the table that happen *after* a device is chosen (not enough memory for the
//! self-check context, a self-check outside the tolerance, an error during a request) belong to
//! the loader, not to this module.

use std::fmt;

/// A family of devices that one build of llama.cpp can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// The processor. Always present.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Vulkan: NVIDIA, AMD and Intel (also the integrated GPUs).
    Vulkan,
    /// Apple Metal.
    Metal,
}

impl BackendKind {
    /// The GPU backends, in the order `"auto"` prefers them (spec 011 section 2).
    pub const GPU_BY_PREFERENCE: [BackendKind; 3] =
        [BackendKind::Cuda, BackendKind::Metal, BackendKind::Vulkan];

    /// The canonical lowercase name, the one `device=` and `compiled_backends()` use.
    pub const fn name(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Metal => "metal",
        }
    }

    /// The backend with this canonical name, if there is one.
    pub fn from_name(name: &str) -> Option<BackendKind> {
        match name {
            "cpu" => Some(BackendKind::Cpu),
            "cuda" => Some(BackendKind::Cuda),
            "vulkan" => Some(BackendKind::Vulkan),
            "metal" => Some(BackendKind::Metal),
            _ => None,
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The names `device=` accepts, for the error message.
const VALID_NAMES: &str = "'cpu', 'auto', 'cuda', 'vulkan', 'metal', or one of the last three with an index such as 'cuda:1' (the index counts the devices of that backend)";

/// A `device=` string that is not part of the grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceParseError {
    /// What was given.
    pub given: String,
}

impl fmt::Display for DeviceParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "device '{}' is not valid; valid: {VALID_NAMES}",
            self.given
        )?;
        let lower = self.given.to_ascii_lowercase();
        if self.given == "mps" {
            write!(
                f,
                "; 'mps' is the PyTorch name for Apple's GPU, use 'metal'"
            )
        } else if self.given == "gpu" {
            write!(
                f,
                "; 'gpu' does not name a backend, use 'auto' for the best verified GPU or name one"
            )
        } else if lower != self.given && DeviceRequest::parse(&lower).is_ok() {
            write!(f, "; names are lowercase, did you mean '{lower}'?")
        } else {
            Ok(())
        }
    }
}

impl std::error::Error for DeviceParseError {}

/// What the user asked for with `device=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    /// The processor, never a GPU (the default).
    Cpu,
    /// The best verified GPU, else the processor.
    Auto,
    /// One device of a GPU backend: `index` counts the devices of **that** backend.
    Named {
        /// `Cuda`, `Vulkan` or `Metal` (the parser never produces `Cpu` here).
        backend: BackendKind,
        /// Position among the devices of `backend`; `"cuda"` is `"cuda:0"`.
        index: usize,
    },
}

impl DeviceRequest {
    /// Parse the grammar of spec 011 section 1: `cpu`, `auto`, `cuda|vulkan|metal` with an optional
    /// `:N`. Case matters and nothing else is accepted (no `gpu`, no `CPU`, no `mps`, no spaces).
    ///
    /// # Errors
    /// [`DeviceParseError`], whose message lists the valid names.
    pub fn parse(text: &str) -> Result<DeviceRequest, DeviceParseError> {
        let bad = || DeviceParseError {
            given: text.to_string(),
        };
        match text {
            "cpu" => return Ok(DeviceRequest::Cpu),
            "auto" => return Ok(DeviceRequest::Auto),
            _ => {}
        }
        let (name, index) = match text.split_once(':') {
            None => (text, None),
            Some((name, digits)) => (name, Some(digits)),
        };
        let backend = match BackendKind::from_name(name) {
            Some(BackendKind::Cpu) | None => return Err(bad()),
            Some(backend) => backend,
        };
        let index = match index {
            None => 0,
            Some(digits) => parse_index(digits).ok_or_else(bad)?,
        };
        Ok(DeviceRequest::Named { backend, index })
    }

    /// The canonical text of the request (`"cuda"` becomes `"cuda:0"`).
    pub fn name(&self) -> String {
        match self {
            DeviceRequest::Cpu => "cpu".to_string(),
            DeviceRequest::Auto => "auto".to_string(),
            DeviceRequest::Named { backend, index } => format!("{backend}:{index}"),
        }
    }
}

/// Decimal digits only, no sign, no spaces, no leading zero (except `0` itself).
fn parse_index(digits: &str) -> Option<usize> {
    let well_formed = !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'));
    if well_formed {
        digits.parse().ok()
    } else {
        None
    }
}

/// What kind of device it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// The processor.
    Cpu,
    /// A discrete GPU (own memory).
    Gpu,
    /// An integrated GPU (shares the memory of the processor).
    Igpu,
}

impl DeviceKind {
    /// `"cpu"`, `"gpu"` or `"igpu"`, as `available_devices()` shows it.
    pub const fn name(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
            DeviceKind::Igpu => "igpu",
        }
    }
}

/// One device the engine can run on, as the machine reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The backend that drives it.
    pub backend: BackendKind,
    /// Position among the devices of `backend` (what the user writes after the colon).
    pub index: usize,
    /// Position in the list of the engine's own devices: what the engine needs to be pointed at it.
    pub engine_index: usize,
    /// Discrete GPU, integrated GPU or processor.
    pub kind: DeviceKind,
    /// The human name (`NVIDIA GeForce RTX 5070 Laptop GPU`).
    pub description: String,
    /// Total memory in MiB, when known.
    pub memory_total_mib: Option<u64>,
    /// Free memory in MiB, when known.
    pub memory_free_mib: Option<u64>,
}

impl DeviceInfo {
    /// The name `device=` uses for it: `cpu`, `cuda:0`, `vulkan:1`.
    pub fn name(&self) -> String {
        match self.backend {
            BackendKind::Cpu => "cpu".to_string(),
            other => format!("{other}:{}", self.index),
        }
    }

    /// What a verification record is keyed by: the name **and** the description of a GPU (another
    /// GPU in the same slot runs other kernels), just `cpu` for the processor.
    pub fn identity(&self) -> String {
        match self.backend {
            BackendKind::Cpu => "cpu".to_string(),
            _ => format!("{}/{}", self.name(), self.description),
        }
    }
}

/// What the build contains and what the machine has. The loader asks it; the tests replace it.
pub trait DeviceProbe: Send + Sync {
    /// The backends this build was compiled with (always includes `Cpu`).
    fn compiled_backends(&self) -> Vec<BackendKind>;
    /// Every device the engine sees, the processor included. Listing never loads a model.
    fn devices(&self) -> Vec<DeviceInfo>;
}

/// The probe of a build that has no GPU backend: the wheels on PyPI. It lists the processor only.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuOnlyProbe;

impl DeviceProbe for CpuOnlyProbe {
    fn compiled_backends(&self) -> Vec<BackendKind> {
        vec![BackendKind::Cpu]
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        vec![DeviceInfo {
            backend: BackendKind::Cpu,
            index: 0,
            engine_index: 0,
            kind: DeviceKind::Cpu,
            description: "CPU".to_string(),
            memory_total_mib: None,
            memory_free_mib: None,
        }]
    }
}

/// The answer of [`select`]: the device that will run the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selected {
    /// The device itself.
    pub device: DeviceInfo,
    /// Why the processor was used although a GPU was wanted (`"auto"` only). `None` when the
    /// choice was straightforward, including `"auto"` on a build with no GPU backend at all.
    pub fallback: Option<String>,
}

impl Selected {
    /// The name `ModelInfo.device` reports.
    pub fn name(&self) -> String {
        self.device.name()
    }

    /// How to run the model on this device.
    pub fn plan(&self) -> OffloadPlan {
        OffloadPlan::for_device(&self.device)
    }
}

/// An explicit device that cannot be used (the rows of spec 011 section 2 that raise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The build does not contain the backend (the wheels on PyPI are CPU-only).
    NotCompiled {
        /// What was asked.
        backend: BackendKind,
        /// What the build has.
        compiled: Vec<BackendKind>,
    },
    /// The backend is compiled in, but the machine has no such device (no driver, a VM, a wrong index).
    NoDevice {
        /// The device asked for, as text (`cuda:1`).
        wanted: String,
        /// The devices of that backend that do exist, as `device=` names.
        found: Vec<String>,
    },
    /// There is no verified tolerance for this model on this backend.
    NoRow {
        /// The backend without a row.
        backend: BackendKind,
    },
    /// The device is an integrated GPU: the tolerances were measured on discrete GPUs, and the
    /// integrated one that was measured was slower than the processor and is not admitted.
    IntegratedGpu {
        /// The device asked for (`vulkan:0`).
        device: String,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::NotCompiled { backend, compiled } => write!(
                f,
                "this build of archai_jev was compiled without the '{backend}' backend (compiled: {}); the wheels on PyPI are CPU-only, see docs/gpu.md to build with a GPU backend",
                names(compiled)
            ),
            Refusal::NoDevice { wanted, found } => {
                if found.is_empty() {
                    write!(
                        f,
                        "no '{wanted}' device found: the backend is compiled in but the machine reports none (is the driver installed and is there a GPU?)"
                    )
                } else {
                    write!(
                        f,
                        "no '{wanted}' device found; the devices of that backend are: {}",
                        found.join(", ")
                    )
                }
            }
            Refusal::NoRow { backend } => write!(
                f,
                "this model has no verified tolerance on the '{backend}' backend, so it is not allowed to answer there; use device='cpu' (or 'auto')"
            ),
            Refusal::IntegratedGpu { device } => write!(
                f,
                "'{device}' is an integrated GPU: the tolerances of this library were measured on discrete GPUs only, so a model is not allowed to answer on it; use a discrete device or device='cpu'"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

fn names(backends: &[BackendKind]) -> String {
    let list: Vec<&str> = backends.iter().map(|b| b.name()).collect();
    list.join(", ")
}

/// Apply the table of spec 011 section 2 to `request`.
///
/// `has_row(backend)` says whether the model has a tolerance row on that backend (spec 011 section
/// 3, rule 3); a backend without one is never used. The processor needs no row.
///
/// | Situation | `Cpu` | a named backend | `Auto` |
/// |---|---|---|---|
/// | backend not in the build | CPU | [`Refusal::NotCompiled`] | CPU, no warning |
/// | compiled, no such device | CPU | [`Refusal::NoDevice`] | CPU and a reason |
/// | device, but no tolerance row | CPU | [`Refusal::NoRow`] | CPU and a reason |
///
/// `Auto` skips integrated Vulkan GPUs (provisional, spec 011 section 2) and takes the first device
/// of the first backend in the order CUDA, Metal, Vulkan that has a row.
///
/// # Errors
/// [`Refusal`] for an explicit GPU that cannot be used. `Cpu` and `Auto` fail only if the engine
/// does not list a CPU device at all, reported as [`Refusal::NoDevice`] for `cpu`.
pub fn select(
    request: DeviceRequest,
    probe: &dyn DeviceProbe,
    has_row: &dyn Fn(BackendKind) -> bool,
) -> Result<Selected, Refusal> {
    let devices = probe.devices();
    match request {
        DeviceRequest::Cpu => cpu(&devices, None),
        DeviceRequest::Named { backend, index } => named(backend, index, probe, &devices, has_row),
        DeviceRequest::Auto => auto(probe, &devices, has_row),
    }
}

fn cpu(devices: &[DeviceInfo], fallback: Option<String>) -> Result<Selected, Refusal> {
    devices
        .iter()
        .find(|d| d.backend == BackendKind::Cpu)
        .map(|d| Selected {
            device: d.clone(),
            fallback,
        })
        .ok_or_else(|| Refusal::NoDevice {
            wanted: "cpu".to_string(),
            found: Vec::new(),
        })
}

fn of_backend(devices: &[DeviceInfo], backend: BackendKind) -> impl Iterator<Item = &DeviceInfo> {
    devices.iter().filter(move |d| d.backend == backend)
}

fn named(
    backend: BackendKind,
    index: usize,
    probe: &dyn DeviceProbe,
    devices: &[DeviceInfo],
    has_row: &dyn Fn(BackendKind) -> bool,
) -> Result<Selected, Refusal> {
    if backend == BackendKind::Cpu {
        return cpu(devices, None);
    }
    let compiled = probe.compiled_backends();
    if !compiled.contains(&backend) {
        return Err(Refusal::NotCompiled { backend, compiled });
    }
    let Some(device) = of_backend(devices, backend).find(|d| d.index == index) else {
        return Err(Refusal::NoDevice {
            wanted: format!("{backend}:{index}"),
            found: of_backend(devices, backend).map(DeviceInfo::name).collect(),
        });
    };
    // the rows of a backend are the ones measured on its discrete GPUs
    if device.kind == DeviceKind::Igpu {
        return Err(Refusal::IntegratedGpu {
            device: device.name(),
        });
    }
    if !has_row(backend) {
        return Err(Refusal::NoRow { backend });
    }
    Ok(Selected {
        device: device.clone(),
        fallback: None,
    })
}

fn auto(
    probe: &dyn DeviceProbe,
    devices: &[DeviceInfo],
    has_row: &dyn Fn(BackendKind) -> bool,
) -> Result<Selected, Refusal> {
    let compiled = probe.compiled_backends();
    let mut reasons: Vec<String> = Vec::new();
    for backend in BackendKind::GPU_BY_PREFERENCE {
        if !compiled.contains(&backend) {
            continue;
        }
        let mut usable = of_backend(devices, backend)
            .filter(|d| !(backend == BackendKind::Vulkan && d.kind == DeviceKind::Igpu))
            .peekable();
        if usable.peek().is_none() {
            reasons.push(if of_backend(devices, backend).next().is_some() {
                format!("{backend}: only integrated GPUs, which 'auto' does not use (name one to use it)")
            } else {
                format!("{backend}: compiled in but no device found (driver missing?)")
            });
            continue;
        }
        if !has_row(backend) {
            reasons.push(format!(
                "{backend}: this model has no verified tolerance on it"
            ));
            continue;
        }
        if let Some(device) = usable.next() {
            return Ok(Selected {
                device: device.clone(),
                fallback: None,
            });
        }
    }
    // No GPU backend in the build at all: nothing was lost, so nothing to warn about.
    let fallback = if reasons.is_empty() {
        None
    } else {
        Some(format!(
            "using the CPU instead of a GPU ({})",
            reasons.join("; ")
        ))
    };
    cpu(devices, fallback)
}

/// How many layers go to the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layers {
    /// None: the whole model stays in the memory of the processor.
    None,
    /// All of them: no partial offload (spec 011 section 3, rule 2).
    All,
}

/// How the engine runs the model: a value, never the defaults of llama.cpp (spec 011 section 3,
/// rule 1). It holds **one** device, so splitting a model over two is not expressible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffloadPlan {
    /// Layers on the device.
    pub layers: Layers,
    /// `DeviceInfo::engine_index` of the only device used (the processor's own for the CPU plan).
    pub device: usize,
    /// Whether the scheduler may move operations to a GPU on its own: off for the CPU plan.
    pub op_offload: bool,
}

impl OffloadPlan {
    /// The plan for `device`: nothing offloaded and `op_offload` off for the processor, every layer
    /// on it for a GPU.
    pub fn for_device(device: &DeviceInfo) -> OffloadPlan {
        match device.backend {
            BackendKind::Cpu => OffloadPlan {
                layers: Layers::None,
                device: device.engine_index,
                op_offload: false,
            },
            _ => OffloadPlan {
                layers: Layers::All,
                device: device.engine_index,
                op_offload: true,
            },
        }
    }

    /// True if this plan can put anything on a GPU.
    pub fn uses_gpu(&self) -> bool {
        self.layers != Layers::None || self.op_offload
    }
}

#[cfg(any(test, feature = "testing"))]
mod fake;
#[cfg(any(test, feature = "testing"))]
pub use fake::FakeProbe;

#[cfg(test)]
mod tests;
