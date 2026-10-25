use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::engine::device::{CpuOnlyProbe, DeviceProbe};
use crate::error::Result;
use crate::hub::config::HubConfig;
use crate::hub::download::{Response, Transport};
use crate::hub::events::{NeverCancel, Recorder};
use crate::models::load::{Context, LoadRequest, LoadedModel, load_model};
use crate::models::registry::Registry;
use crate::models::resolve::{NameOrPath, Selection};
use crate::models::testing::FakeBackend;

/// A transport that refuses to touch the network and counts the attempts.
#[derive(Default)]
pub struct NoNetwork {
    pub gets: AtomicUsize,
}

impl Transport for NoNetwork {
    fn get(&self, _: &str, _: Option<u64>, _: Duration) -> std::result::Result<Response, String> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        Err("the network is disabled in this test".to_string())
    }
}

/// Everything a pipeline test needs: a private cache, a registry, a fake backend, counters.
pub struct Harness {
    pub cache: tempfile::TempDir,
    pub registry: Registry,
    pub backend: FakeBackend,
    pub recorder: Arc<Recorder>,
    pub transport: Arc<NoNetwork>,
    pub offline: bool,
    pub endpoint: Option<String>,
    pub real_transport: Option<Arc<dyn Transport>>,
    pub converter: Option<Arc<dyn crate::models::convert::Converter>>,
    pub head_reader: Option<Arc<dyn crate::models::head::HeadReader>>,
    pub cancel: Option<Arc<dyn crate::hub::events::Cancel>>,
    /// What the build contains and the machine has; the CPU alone unless a test says otherwise.
    pub devices: Box<dyn DeviceProbe>,
}

pub const EMPTY_INDEX: &str = r#"{"default": null, "current": {}}"#;

impl Harness {
    pub fn new() -> Self {
        Harness::with_registry(Registry::from_sources(&[], EMPTY_INDEX).unwrap())
    }

    pub fn with_registry(registry: Registry) -> Self {
        Harness {
            cache: tempfile::tempdir().unwrap(),
            registry,
            backend: FakeBackend::new(),
            recorder: Arc::new(Recorder::default()),
            transport: Arc::new(NoNetwork::default()),
            offline: false,
            endpoint: None,
            real_transport: None,
            converter: None,
            head_reader: None,
            cancel: None,
            devices: Box::new(CpuOnlyProbe),
        }
    }

    pub fn hub(&self) -> HubConfig {
        let mut hub = HubConfig::new(self.cache.path().to_path_buf());
        hub.offline = self.offline;
        if let Some(e) = &self.endpoint {
            hub.endpoint.clone_from(e);
        }
        hub.backoff = vec![Duration::ZERO];
        hub
    }

    pub fn local(&self, dir: &Path) -> LoadRequest {
        self.local_on(dir, "cpu")
    }

    pub fn local_on(&self, dir: &Path, device: &str) -> LoadRequest {
        LoadRequest {
            selection: Selection {
                name_or_path: NameOrPath::Path(dir.to_path_buf()),
                revision: None,
                device: device.to_string(),
                dtype: None,
                manifest: None,
            },
            temperature: None,
            allow_uncalibrated: true,
        }
    }

    pub fn load(&self, req: &LoadRequest) -> Result<LoadedModel> {
        let hub = self.hub();
        let ctx = Context {
            registry: &self.registry,
            hub: &hub,
            backend: &self.backend,
            devices: self.devices.as_ref(),
            transport: self
                .real_transport
                .as_deref()
                .unwrap_or(self.transport.as_ref()),
            observer: self.recorder.as_ref(),
            cancel: self.cancel.as_deref().unwrap_or(&NeverCancel),
            head_reader: self.head_reader.as_deref(),
            converter: self.converter.as_deref(),
        };
        load_model(req, &ctx)
    }

    /// How many verification records are in the cache.
    pub fn records(&self) -> usize {
        std::fs::read_dir(self.cache.path().join("v1").join("verified"))
            .map(|d| {
                d.flatten()
                    .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                    .count()
            })
            .unwrap_or(0)
    }
}
