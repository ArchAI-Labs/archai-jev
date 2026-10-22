//! Binding of model loading (spec 005 section 1): `load_model` and `list_models`.
//!
//! The environment is read **here, once per call**, and turned into an explicit configuration;
//! the core never reads it. Hub events become `logging` records of the logger `archai_jev`; a
//! `Ctrl-C` during a download stops it between blocks.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::classes::PyModel;
use crate::convert::Qwen35Converter;
use crate::convert::headpt::HeadPtReader;
use crate::engine::device::{BackendKind, DeviceProbe};
use crate::engine::llama::{
    LlamaProbe, compiled_backends as llama_compiled_backends, enumerations,
};
use crate::error::Error;
use crate::hub::cache::{self, CacheEnv, Os};
use crate::hub::config::HubConfig;
use crate::hub::download::UreqTransport;
use crate::hub::events::{Cancel, Event, Observer};
use crate::models::llama_backend::{LlamaBackend, default_threads};
use crate::models::load::{self, Context, LoadRequest, ModelData};
use crate::models::manifest::{HeadSpec, Manifest};
use crate::models::registry::Registry;
use crate::models::resolve::{NameOrPath, Selection};
use crate::models::variants::{VariantData, variants_of};

struct ProcessEnv;

impl CacheEnv for ProcessEnv {
    fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

fn env(name: &str) -> Option<String> {
    ProcessEnv.get(name).filter(|v| !v.is_empty())
}

/// Hub events as records of the logger `archai_jev`.
struct LoggingObserver;

impl LoggingObserver {
    fn log(level: &str, message: &str) {
        Python::attach(|py| {
            let result = py
                .import("logging")
                .and_then(|m| m.getattr("getLogger"))
                .and_then(|g| g.call1(("archai_jev",)))
                .and_then(|logger| logger.call_method1(level, (message,)));
            drop(result); // a broken logging setup must not break loading
        });
    }
}

impl Observer for LoggingObserver {
    fn on_event(&self, event: &Event) {
        match event {
            Event::Cached { file } => Self::log("debug", &format!("{file}: found in the cache")),
            Event::DownloadStart {
                file,
                size,
                url,
                destination,
            } => Self::log(
                "info",
                &format!("downloading {file} ({size} bytes) from {url} to {destination}"),
            ),
            Event::DownloadProgress { file, done, total } => {
                Self::log("info", &format!("{file}: {done} of {total} bytes"));
            }
            Event::DownloadDone { file } => Self::log("info", &format!("{file}: done")),
            Event::License { name, spdx, url } => {
                Self::log("info", &format!("model {name}: license {spdx} ({url})"));
            }
            Event::Warning(text) => Self::log("warning", text),
            Event::ConvertStart {
                tensors,
                bytes,
                destination,
                dtype,
            } => Self::log(
                "info",
                &format!(
                    "converting the original files to a {dtype} GGUF ({tensors} tensors, {bytes} bytes) into {destination}; this happens once"
                ),
            ),
            Event::ConvertProgress { done, total } => {
                Self::log("info", &format!("conversion: {done} of {total} tensors"));
            }
            Event::ConvertDone { seconds, bytes } => Self::log(
                "info",
                &format!("conversion done in {seconds:.1} s ({bytes} bytes)"),
            ),
        }
    }
}

/// `Ctrl-C` stops a download: Python's signal check says so.
struct SignalCancel;

impl Cancel for SignalCancel {
    fn cancelled(&self) -> bool {
        Python::attach(|py| py.check_signals().is_err())
    }
}

fn truthy(name: &str) -> bool {
    env(name).is_some_and(|v| !matches!(v.as_str(), "0" | "false" | "no"))
}

/// The variants of a manifest as plain dictionaries (`VariantInfo` is built in Python).
fn variants_dicts<'py>(
    py: Python<'py>,
    variants: &[VariantData],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    variants
        .iter()
        .map(|v| {
            let out = PyDict::new(py);
            out.set_item("dtype", &v.dtype)?;
            out.set_item("size_bytes", v.size_bytes)?;
            out.set_item("calibrated", v.calibrated)?;
            out.set_item("temperature", v.temperature)?;
            out.set_item("is_default", v.is_default)?;
            Ok(out)
        })
        .collect()
}

fn info_dict<'py>(py: Python<'py>, d: &ModelData) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("name", &d.name)?;
    out.set_item("revision", &d.revision)?;
    out.set_item("family", &d.family)?;
    out.set_item("head", &d.head)?;
    out.set_item("template", &d.template)?;
    out.set_item("temperature", d.temperature)?;
    out.set_item("calibrated", d.calibrated)?;
    out.set_item("license", &d.license)?;
    out.set_item("max_context", d.max_context)?;
    out.set_item("dtype", &d.dtype)?;
    out.set_item("source", d.source)?;
    out.set_item("calibration_source", d.calibration_source)?;
    out.set_item("tasks", d.tasks.clone())?;
    out.set_item("notice", d.notice.as_deref())?;
    out.set_item("variants", variants_dicts(py, &d.variants)?)?;
    out.set_item("max_options", d.max_options)?;
    out.set_item("max_key_tokens", d.max_key_tokens)?;
    out.set_item("device", &d.device)?;
    out.set_item("fallback", d.fallback.as_deref())?;
    Ok(out)
}

/// The devices this process can run a model on, as plain dictionaries. Lists them; loads nothing.
#[pyfunction]
pub fn available_devices(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let devices = py.detach(|| LlamaProbe.devices());
    devices
        .iter()
        .map(|d| {
            let out = PyDict::new(py);
            out.set_item("name", d.name())?;
            out.set_item("kind", d.kind.name())?;
            out.set_item("backend", d.backend.name())?;
            out.set_item("description", &d.description)?;
            out.set_item("memory_total_mib", d.memory_total_mib)?;
            out.set_item("memory_free_mib", d.memory_free_mib)?;
            Ok(out)
        })
        .collect()
}

/// The backends this build was compiled with: `cpu`, and each GPU backend of its features.
#[pyfunction]
pub fn compiled_backends() -> Vec<&'static str> {
    llama_compiled_backends()
        .into_iter()
        .map(BackendKind::name)
        .collect()
}

/// How many times the devices were listed in this process (a test reads it: `import` lists none).
#[pyfunction]
pub fn _device_enumerations() -> usize {
    enumerations()
}

/// Memory of the prefix cache when nobody says otherwise, in MiB.
const DEFAULT_PREFIX_CACHE_MIB: u64 = 128;

/// The prefix cache limit in bytes: the argument, else `ARCHAI_JEV_PREFIX_CACHE_MIB` (read here,
/// once per call), else the default. `0` turns the reuse off.
fn prefix_cache_bytes(prefix_cache_mib: Option<u64>) -> PyResult<u64> {
    let mib = match prefix_cache_mib {
        Some(mib) => mib,
        None => match env("ARCHAI_JEV_PREFIX_CACHE_MIB") {
            None => DEFAULT_PREFIX_CACHE_MIB,
            Some(v) => v.trim().parse::<u64>().map_err(|_| {
                PyValueError::new_err(format!(
                    "ARCHAI_JEV_PREFIX_CACHE_MIB must be a whole number of MiB (0 turns the cache off), got {v:?}"
                ))
            })?,
        },
    };
    mib.checked_mul(1 << 20)
        .ok_or_else(|| PyValueError::new_err(format!("prefix_cache_mib is too large: {mib} MiB")))
}

/// Load a model (see `Jev.from_pretrained`). Arguments are already type-checked in Python.
#[pyfunction]
#[pyo3(signature = (kind, value, revision, device, dtype, temperature, allow_uncalibrated, cache_dir, offline, manifest, prefix_cache_mib))]
#[allow(clippy::too_many_arguments)]
pub fn load_model<'py>(
    py: Python<'py>,
    kind: &str,
    value: Option<String>,
    revision: Option<String>,
    device: String,
    dtype: Option<String>,
    temperature: Option<f64>,
    allow_uncalibrated: bool,
    cache_dir: Option<PathBuf>,
    offline: bool,
    manifest: Option<PathBuf>,
    prefix_cache_mib: Option<u64>,
) -> PyResult<(PyModel, Bound<'py, PyDict>)> {
    let name_or_path = match (kind, value) {
        ("default", _) => NameOrPath::Default,
        ("path", Some(p)) => NameOrPath::Path(PathBuf::from(p)),
        ("str", Some(s)) => NameOrPath::Str(s),
        (other, _) => {
            return Err(PyValueError::new_err(format!(
                "unknown name kind {other:?}"
            )));
        }
    };
    let threads = match env("ARCHAI_JEV_NUM_THREADS") {
        None => default_threads(),
        Some(v) => match v.trim().parse::<i32>() {
            Ok(n) if n > 0 => n,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "ARCHAI_JEV_NUM_THREADS must be a whole number greater than 0, got {v:?}"
                )));
            }
        },
    };
    let cache_bytes = prefix_cache_bytes(prefix_cache_mib)?;
    let root = cache::resolve_root(cache_dir.as_deref(), &ProcessEnv, Os::current())
        .map_err(Error::ModelDownload)?;
    let mut hub = HubConfig::new(root);
    hub.offline = offline || truthy("ARCHAI_JEV_OFFLINE") || truthy("HF_HUB_OFFLINE");
    if let Some(endpoint) = env("ARCHAI_JEV_HF_ENDPOINT").or_else(|| env("HF_ENDPOINT")) {
        hub.endpoint = endpoint;
    }

    let request = LoadRequest {
        selection: Selection {
            name_or_path,
            revision,
            device,
            dtype,
            manifest,
        },
        temperature,
        allow_uncalibrated,
    };
    let registry = Registry::builtin().map_err(Error::IncompatibleModel)?;
    let backend = LlamaBackend::new(threads).with_prefix_cache(cache_bytes);
    let transport = UreqTransport::new(&hub);
    let loaded = py.detach(|| {
        let ctx = Context {
            registry,
            hub: &hub,
            backend: &backend,
            devices: &LlamaProbe,
            transport: &transport,
            observer: &LoggingObserver,
            cancel: &SignalCancel,
            head_reader: Some(&HeadPtReader),
            converter: Some(&Qwen35Converter),
        };
        load::load_model(&request, &ctx)
    })?;
    let info = info_dict(py, &loaded.data)?;
    Ok((PyModel::wrap(Arc::clone(&loaded.scorer)), info))
}

fn listed<'py>(py: Python<'py>, registry: &Registry, m: &Manifest) -> PyResult<Bound<'py, PyDict>> {
    let variant = m.variant(&m.default_dtype);
    let (temperature, calibrated) = variant.map_or((1.0, false), |v| {
        (
            v.calibration.temperature.unwrap_or(1.0),
            v.calibration.declared,
        )
    });
    let out = PyDict::new(py);
    out.set_item("name", &m.name)?;
    out.set_item("revision", &m.revision)?;
    out.set_item("family", &m.family)?;
    out.set_item(
        "head",
        match &m.head {
            HeadSpec::Letters { .. } => "letters",
            HeadSpec::KeyLogProb { .. } => "key_logprob",
            HeadSpec::AnswerLetters { .. } => "answer_letters",
            HeadSpec::Pointer { .. } => "pointer",
        },
    )?;
    out.set_item(
        "template",
        format!("{}-v{}", m.template.id, m.template.version),
    )?;
    out.set_item("temperature", temperature)?;
    out.set_item("calibrated", calibrated)?;
    out.set_item("license", &m.license.spdx)?;
    out.set_item("max_context", m.max_context)?;
    out.set_item("dtype", &m.default_dtype)?;
    out.set_item("source", "registry")?;
    out.set_item(
        "calibration_source",
        if calibrated { "manifest" } else { "none" },
    )?;
    out.set_item(
        "tasks",
        m.tasks
            .as_ref()
            .map(|t| t.iter().map(|x| x.id.clone()).collect::<Vec<_>>()),
    )?;
    out.set_item("notice", m.notice.as_deref())?;
    out.set_item("is_default", registry.is_default(m))?;
    out.set_item("variants", variants_dicts(py, &variants_of(m))?)?;
    out.set_item("max_options", m.head.max_options())?;
    out.set_item("max_key_tokens", m.head.max_key_tokens())?;
    Ok(out)
}

/// The models of the registry, as plain dictionaries (static data: no network, no disk): one per
/// name, at its current revision (the others are reached with `revision=`).
#[pyfunction]
pub fn list_models(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let registry = Registry::builtin().map_err(Error::IncompatibleModel)?;
    registry
        .names()
        .iter()
        .map(|name| {
            let current = registry
                .resolve(name, None)
                .map_err(Error::IncompatibleModel)?;
            listed(py, registry, current)
        })
        .collect()
}
