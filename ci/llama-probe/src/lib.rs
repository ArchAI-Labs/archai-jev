//! Build probe for the llama.cpp backend (see `Cargo.toml`). Throwaway code: it only has to prove that the C++ side
//! was compiled, linked into the Python extension and can run on this platform.

use std::ffi::CStr;
use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Initializes llama.cpp once per process (the real engine does the same) and keeps it alive for the whole process.
/// Returns what the backend reports about the platform. ggml's clock is only valid after this call.
fn init_backend() -> Result<&'static str, PyErr> {
    static INIT: OnceLock<Result<String, String>> = OnceLock::new();
    let res = INIT.get_or_init(|| match LlamaBackend::init() {
        Ok(mut backend) => {
            backend.void_logs();
            let info = format!(
                "mmap={} mlock={} gpu_offload={}",
                backend.supports_mmap(),
                backend.supports_mlock(),
                backend.supports_gpu_offload()
            );
            std::mem::forget(backend); // never freed: the engine will own one backend per process
            Ok(info)
        }
        Err(e) => Err(e.to_string()),
    });
    match res {
        Ok(info) => Ok(info.as_str()),
        Err(e) => Err(PyRuntimeError::new_err(format!("llama.cpp backend init failed: {e}"))),
    }
}

#[pymodule]
pub mod llama_probe {
    use super::*;

    /// llama.cpp's own description of this build: the CPU features ggml was compiled with and detected at run time
    /// (AVX2, NEON, ...), then what the backend supports (mmap, mlock, GPU offload). Runs real C++ code in the extension.
    #[pyfunction]
    fn system_info() -> PyResult<String> {
        let backend = init_backend()?;
        // SAFETY: `llama_print_system_info` returns a pointer to a static, NUL-terminated string owned by llama.cpp.
        let cpu = unsafe {
            let ptr = llama_cpp_sys_2::llama_print_system_info();
            if ptr.is_null() {
                return Err(PyRuntimeError::new_err("llama_print_system_info returned NULL"));
            }
            CStr::from_ptr(ptr).to_string_lossy().trim().to_string()
        };
        Ok(format!("{cpu} || {backend}"))
    }

    /// Microseconds on llama.cpp's monotonic clock (two calls must not go backwards).
    #[pyfunction]
    fn time_us() -> PyResult<i64> {
        init_backend()?;
        Ok(llama_cpp_2::llama_time_us())
    }

    /// True when the extension was built with the OpenMP variant.
    #[pyfunction]
    fn built_with_openmp() -> bool {
        cfg!(feature = "openmp")
    }
}
