//! The interface to the checkpoint converter (spec 018).
//!
//! A manifest whose variant has an `hf-lora` source needs its original files turned into a GGUF.
//! 005 does not know how: it hands a [`Converter`] a [`ConvertJob`] (the manifest, the variant's
//! dtype and the **verified** original files), keeps the result in the cache under a key that
//! includes the converter's version, and then applies **the same checks** to the output as to a
//! downloaded GGUF, so a faulty converter cannot get around them.

use std::path::{Path, PathBuf};

use super::families::ArchParams;
use super::manifest::{HfLoraSource, Manifest};
use crate::error::Result;
use crate::hub::events::{Cancel, Observer};

/// The files a conversion produced, relative to the output folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    /// The GGUF file.
    pub model: PathBuf,
    /// The head weights file, for pointer heads (the head file itself is read where it is, so
    /// converters leave this `None`).
    pub head: Option<PathBuf>,
    /// SHA-256 of the GGUF, if the converter computed it while writing (otherwise the cache
    /// computes it).
    pub sha256: Option<String>,
    /// Size of the GGUF in bytes, if the converter knows.
    pub size: Option<u64>,
}

/// Where the original files of an `hf-lora` source are (all already checked against their SHA-256).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFiles {
    /// `config.json` of the base model.
    pub base_config: PathBuf,
    /// The safetensors weights of the base model.
    pub base_weights: PathBuf,
    /// `tokenizer.json` of the base model.
    pub base_tokenizer: PathBuf,
    /// `tokenizer_config.json` of the base model.
    pub base_tokenizer_config: PathBuf,
    /// `adapter_config.json`.
    pub adapter_config: PathBuf,
    /// The safetensors weights of the adapter.
    pub adapter_weights: PathBuf,
}

/// What a converter is asked to do.
pub struct ConvertJob<'a> {
    /// The manifest.
    pub manifest: &'a Manifest,
    /// The variant to produce (`f32`, `bf16`).
    pub dtype: &'a str,
    /// The architecture parameters of the manifest, checked against the family.
    pub params: &'a ArchParams,
    /// The declared source (names, repositories and revisions).
    pub source: &'a HfLoraSource,
    /// Where the verified originals are.
    pub files: SourceFiles,
    /// The decision-head weights file (`head.pt`), if the head has one.
    pub head_path: Option<&'a Path>,
    /// The folder the result will end up in (the conversion itself runs in a temporary one),
    /// for the messages.
    pub destination: PathBuf,
    /// Receives progress.
    pub observer: &'a dyn Observer,
    /// Asked between blocks whether to stop.
    pub cancel: &'a dyn Cancel,
}

/// Turns the original files of a checkpoint into a GGUF (018 implements it).
pub trait Converter: Send + Sync {
    /// The source kinds this converter handles (`hf-lora`).
    fn kinds(&self) -> Vec<String>;

    /// A version string that changes whenever the output may change: it is part of the cache key.
    fn version(&self) -> String;

    /// Convert the source of `job` into `out_dir`.
    ///
    /// # Errors
    /// `IncompatibleModelError` for anything wrong with the checkpoint, an I/O error when a file
    /// cannot be read or written, `Cancelled` when asked to stop.
    fn convert(&self, job: &ConvertJob<'_>, out_dir: &Path) -> Result<Converted>;
}

/// A folder-name-safe form of a version string.
pub fn safe_component(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
