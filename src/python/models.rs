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
    Ok(out)
}

/// Load a model (see `Jev.from_pretrained`). Arguments are already type-checked in Python.
#[pyfunction]
#[pyo3(signature = (kind, value, revision, device, dtype, temperature, allow_uncalibrated, cache_dir, offline, manifest))]
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
    let backend = LlamaBackend::new(threads);
    let transport = UreqTransport::new(&hub);
    let loaded = py.detach(|| {
        let ctx = Context {
            registry,
            hub: &hub,
            backend: &backend,
            transport: &transport,
            observer: &LoggingObserver,
            cancel: &SignalCancel,
            head_reader: None,
            converter: None,
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
    Ok(out)
}

/// The models of the registry, as plain dictionaries (static data: no network, no disk).
#[pyfunction]
pub fn list_models(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let registry = Registry::builtin().map_err(Error::IncompatibleModel)?;
    registry
        .entries()
        .iter()
        .map(|m| listed(py, registry, m))
        .collect()
}
