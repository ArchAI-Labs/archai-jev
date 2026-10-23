# Running on a GPU

The wheels on PyPI are **CPU-only**. A GPU is used only by a build you make from the source
(`maturin build --release --features cuda`, or `vulkan`, or `metal`; see
[Building](#building-from-source)). Nothing in a CPU wheel changes: `device="cpu"` is the default,
it never touches a GPU, and its answers are bit for bit the ones of a build without GPU support.

> **These measurements come from one machine** (an NVIDIA RTX 5070 Laptop and an Intel integrated
> GPU, Windows 11, CUDA 12.9, Vulkan SDK 1.4.363). They show what is possible, not what you will
> get. Nothing here is a promise of speed on your hardware; measure with your own requests.

## Choosing the device

```python
from archai_jev import Jev, available_devices, compiled_backends

compiled_backends()      # ("cpu", "cuda") in a CUDA build; ("cpu",) in every wheel on PyPI
available_devices()      # the devices this machine has; the processor first (starts the drivers)

Jev.from_pretrained(device="cpu")       # default: never a GPU
Jev.from_pretrained(device="cuda:0")    # insist on this device or fail
Jev.from_pretrained(device="auto")      # the best GPU that is allowed to answer, else the CPU
```

`device` is `"cpu"`, `"auto"`, or `"cuda"`, `"vulkan"`, `"metal"` with an optional index
(`"cuda:1"`). The index counts the devices **of that backend** in the order `available_devices()`
lists them (on a laptop with an integrated GPU, the Intel GPU may be `vulkan:0` and the NVIDIA one
`vulkan:1`: read the list). Anything else (`"gpu"`, `"CPU"`, `"mps"`) is a `ValueError` that lists
the valid names. No environment variable selects the device.

| You ask for | What happens |
|---|---|
| `"cpu"` | The processor, always. |
| `"cuda:0"` (or another named device) | That device, or an error: `DeviceUnavailableError` if the build lacks the backend or the machine lacks the device, or if the model has no verified tolerance on that backend (see below). Never a silent change of device. |
| `"auto"` | The first of CUDA, Metal, Vulkan that exists and is allowed to answer for this model; integrated Vulkan GPUs are skipped. If none is usable, the CPU. A `DeviceFallbackWarning` is raised **only** when a GPU backend is compiled in and could not be used (a missing driver, not enough memory); a CPU wheel says nothing, because there was nothing to fall back from. |

`Jev.model_info.device` always says where the model runs.

### Which models may run on a GPU

A model may answer on a backend only if the library holds a **tolerance row** for that model and
that backend. The self-check at load time (the same one the CPU does) then compares the numbers of
the device with the reference. Today the rows are:

- the default model (`qwen2-keys`, Q8_0) on **CUDA** (tolerance 0.097 in probability) and on
  **Vulkan** (0.057), both below the CPU row (0.1).

Every other model, every other precision, and Metal have **no row**, so `device="cuda"`,
`"vulkan"` or `"metal"` raises `DeviceUnavailableError` for them, and `"auto"` uses the CPU (with the
warning above if a GPU backend is compiled in). A row is added only after the fidelity and the
calibration of that backend were measured. If a GPU gives numbers outside the tolerance the load
fails with `ModelVerificationError`, and `"auto"` does **not** fall back to the CPU in that case:
numbers that should not be trusted are an error, not a detail.

## What a GPU did on the machine we measured

The default model with Q8_0 weights, requests of three sizes (W1, W3 and ten questions about one
state, Q10), the same requests on the best setting of the CPU of that machine, three alternating
blocks, with a control of the device against itself (A/A) kept inside 0.95 to 1.05. The tables are
generated from `tests/data/gpu/report.json` by a test: do not edit them by hand.

<!-- gpu-table:begin -->
**Calls back to back** (p50 of the device, ratio to the tuned CPU in brackets; below 1 is faster)

| Device | Kind | W1 (94 tokens) | W3 (365 tokens) | Q10 (10 questions, 1537 tokens) |
|---|---|---|---|---|
| `cuda:0` NVIDIA GeForce RTX 5070 Laptop GPU | discrete | 52 ms (0.17x) | 66 ms (0.05x) | 161 ms (0.08x) |
| `vulkan:1` NVIDIA GeForce RTX 5070 Laptop GPU | discrete | 206 ms (0.64x) | 228 ms (0.17x) | 360 ms (0.19x) |
| `vulkan:0` Intel(R) Graphics | integrated | 836 ms (2.69x) | 1950 ms (1.54x) | 4298 ms (2.23x) |

**A pause of 0.5 s between calls** (p50 of the device, ratio to the tuned CPU in brackets; below 1 is faster)

| Device | Kind | W1 (94 tokens) | W3 (365 tokens) | Q10 (10 questions, 1537 tokens) |
|---|---|---|---|---|
| `cuda:0` NVIDIA GeForce RTX 5070 Laptop GPU | discrete | 54 ms (0.14x) | 73 ms (0.05x) | 180 ms (0.09x) |
| `vulkan:1` NVIDIA GeForce RTX 5070 Laptop GPU | discrete | 204 ms (0.54x) | 225 ms (0.16x) | 371 ms (0.19x) |
| `vulkan:0` Intel(R) Graphics | integrated | 839 ms (2.23x) | 1957 ms (1.44x) | 4294 ms (2.22x) |

**Fidelity and calibration** (default model, fp32 reference; ECE of the CPU: 0.0111)

| Device | Largest abs. difference in probability | ECE | Tolerance row |
|---|---|---|---|
| `cuda:0` | 0.054 | 0.0108 | yes |
| `vulkan:1` | 0.036 | 0.0123 | yes |
| `vulkan:0` | 0.070 | not measured | no |
<!-- gpu-table:end -->

How to read it, and what **not** to read into it:

- On this machine the NVIDIA GPU through **CUDA** answered every size faster than the CPU, from about
  6 times faster on a short state to about 20 times on a long one. **Vulkan** on the same GPU was
  1.5 to 6 times faster than the CPU, and 2 to 4 times slower than CUDA. The integrated GPU was
  **slower than the CPU** and is not allowed to answer (no tolerance row, skipped by `"auto"`).
- The advantage depends on the CPU you compare with. A fast desktop CPU and a small GPU can give a
  different picture; so can a model other than the default.
- A GPU does not make a call free: the context is created for every request, and the first call of
  a process also loads the model onto the device (about 1.5 s to the first answer on CUDA and 1.9 s
  on Vulkan here; the CUDA build had the kernels of this GPU built in, so nothing was compiled at
  run time).
- Numbers for more than one machine are not available yet.

### Do the answers change?

- On a given device, the same request gives the same numbers every time (20 repetitions, six
  requests, all bit-identical on both GPUs).
- The memory of prefixes ([prefix-cache.md](prefix-cache.md)) behaves on a GPU as on the CPU: a cold
  cache, a warm one and a store that remembers nothing give identical numbers, bit for bit; asking
  without the cache (`prefix_cache_mib=0`) decodes the request in one pass and differs by rounding
  (up to 0.03 in probability on CUDA and 0.015 on Vulkan in our check).
- Between devices the numbers differ by rounding of the arithmetic, within the tolerance row of the
  backend. A GPU is never trusted more than the CPU: its row is never wider.
- Calling `Jev.from_pretrained(device="cpu")` in a build that includes a GPU backend gives
  **exactly** the answers of a CPU wheel.

### Memory

On the machine we measured, the default model took about 2.0 GiB of the memory of the GPU at its
peak. A build with CUDA also creates the CUDA context when you ask for `device="cpu"`: about 126 MiB
of GPU memory (13 MiB with Vulkan), which does not grow, and nothing runs there.

## Building from source

You need what the CPU build needs (a C++ compiler, CMake and `libclang` for `bindgen`), plus the SDK of the backend. The Cargo features only forward to the
llama.cpp wrapper and add no dependency. A GPU build **is not** a wheel for PyPI.

### CUDA

```text
# Linux, CUDA Toolkit 12.8 or later (12.9.1 is what we test) with nvcc on PATH
CMAKE_CUDA_ARCHITECTURES=120 maturin build --release --features cuda

# Windows (PowerShell), Visual Studio Build Tools and the CUDA Toolkit
$env:CUDA_PATH_V12_9 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CMAKE_CUDA_ARCHITECTURES = "120"
$env:CARGO_TARGET_DIR = "C:\t\aj"        # a short path: long ones make MSBuild fail
maturin build --release --features cuda
```

`CMAKE_CUDA_ARCHITECTURES` is the compute capability of **your** GPU without the dot (`120` for an
RTX 50 series, `89` for RTX 40, `86` for RTX 30). One architecture keeps the build short and the
file small; a list (`"86;89;120"`) builds more. Blackwell (`sm_120`) needs CUDA 12.8 or later.

Python on Windows does not search `PATH` for DLLs: add the CUDA `bin` folder with
`os.add_dll_directory(...)` before `import archai_jev`.

`--features cuda-no-vmm` is the same build without a link to `nvcuda.dll`, so that
`import archai_jev` also works on a machine with **no NVIDIA driver** (the `device="cuda"` request
then raises `DeviceUnavailableError`).

### Vulkan

Install the Vulkan SDK and set `VULKAN_SDK`. On Windows the Visual Studio generator of CMake fails;
use Ninja from a Visual Studio command prompt:

```text
# Windows (PowerShell, after VsDevCmd.bat -arch=x64 so that the MSVC compiler is on PATH)
$env:CMAKE_GENERATOR = "Ninja"
maturin build --release --features vulkan

# Linux
maturin build --release --features vulkan
```

### Metal (macOS, Apple silicon)

```text
export GGML_METAL=ON
maturin build --release --features metal
```

`.cargo/config.toml` switches Metal **off** for every build of this crate, so that the CPU wheels
never contain it; the feature therefore needs `GGML_METAL=ON`, and the build stops with a message
if it is missing (or on any system that is not macOS). We have **not** run Metal on a Mac: the
build is checked in CI, the speed and the fidelity are unmeasured, and no model has a tolerance
row for it, so `device="metal"` refuses every model until one is measured.

### Check what you built

```python
import archai_jev
print(archai_jev.compiled_backends())   # for example ("cpu", "cuda")
print(archai_jev.available_devices())
```

`compiled_backends()` says what the build contains; `available_devices()` what the machine has. A
backend in the first list and no device in the second is a missing driver.
