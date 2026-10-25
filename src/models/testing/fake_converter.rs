//! A fake converter: writes a prepared GGUF into the output folder, counts its calls.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::models::convert::{ConvertJob, Converted, Converter};
use crate::models::incompat::Incompat;

/// Writes `gguf` as `model.gguf`, or fails.
pub struct FakeConverter {
    pub gguf: Vec<u8>,
    pub version: String,
    pub calls: Arc<AtomicUsize>,
    pub fail: bool,
}

impl FakeConverter {
    pub fn new(gguf: Vec<u8>) -> Self {
        FakeConverter {
            gguf,
            version: "fake-1".to_string(),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }
    }
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Converter for FakeConverter {
    fn kinds(&self) -> Vec<String> {
        vec!["hf-lora".to_string()]
    }

    fn version(&self) -> String {
        self.version.clone()
    }

    fn convert(&self, _job: &ConvertJob<'_>, out_dir: &Path) -> Result<Converted> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(Error::IncompatibleModel(Incompat::ConversionFailed {
                detail: "the fake converter was told to fail".to_string(),
            }));
        }
        std::fs::write(out_dir.join("model.gguf"), &self.gguf).unwrap();
        Ok(Converted {
            model: "model.gguf".into(),
            head: None,
            sha256: None,
            size: None,
        })
    }
}
