//! The interface to the checkpoint converter (spec 018).
//!
//! A manifest whose variant has a `hf-lora` or `hf-full` source needs its original files turned
//! into a GGUF. 005 does not know how: it asks a [`Converter`], keeps the result in the cache
//! under a key that includes the converter's version, and then applies **the same checks** to
//! the output as to a downloaded GGUF, so a faulty converter cannot get around them.

use std::path::{Path, PathBuf};

use super::incompat::Incompat;
use crate::json_strict::Json;

/// The files a conversion produced, relative to the output folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    /// The GGUF file.
    pub model: PathBuf,
    /// The head weights file, for pointer heads.
    pub head: Option<PathBuf>,
}

/// Turns the original files of a checkpoint into a GGUF (018 implements it).
pub trait Converter: Send + Sync {
    /// The source kinds this converter handles (`hf-lora`, `hf-full`).
    fn kinds(&self) -> Vec<String>;

    /// A version string that changes whenever the output may change: it is part of the cache key.
    fn version(&self) -> String;

    /// Convert the source described by `raw` (the variant's `source` object) into `out_dir`.
    ///
    /// # Errors
    /// Any [`Incompat`]; the loader reports it as a failed conversion.
    fn convert(&self, raw: &Json, out_dir: &Path) -> Result<Converted, Incompat>;
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
