//! The cache of converted models (spec 018, AC-43..AC-47): one folder per (manifest, dtype,
//! converter version) under `<cache>/v1/materialized/`.
//!
//! A conversion is written into a private temporary folder next to the final one and renamed into
//! place **after** its record (`conversion.json`) is written, so a folder that has a record is a
//! complete one and nothing half-written ever has the final name. The record says which file is
//! the GGUF, how big it is and its SHA-256 (computed while the converter wrote it): the next
//! loads trust the file only if it still is what the record says.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::convert::{ConvertJob, Converter};
use super::hash::sha256_file;
use super::incompat::Incompat;
use crate::error::{Error, Result};
use crate::json_strict::{self, Obj};

/// Name of the record inside a converted folder.
pub const RECORD: &str = "conversion.json";

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A converted model in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Materialized {
    /// The GGUF file.
    pub model: PathBuf,
    /// Its SHA-256, computed when it was written.
    pub sha256: String,
    /// Its size in bytes.
    pub size: u64,
}

fn fail(detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::ConversionFailed {
        detail: detail.into(),
    })
}

fn io(path: &Path, e: &std::io::Error) -> Error {
    crate::convert::safetensors::io_error(path, e)
}

/// The folder of a converted model.
pub fn folder(materialized_root: &Path, manifest_sha: &str, dtype: &str, version: &str) -> PathBuf {
    materialized_root
        .join(manifest_sha.get(..16).unwrap_or(manifest_sha))
        .join(dtype)
        .join(super::convert::safe_component(version))
}

/// The record of the folder `out`, if it is complete and describes this `version` and `dtype`
/// and a GGUF of the recorded size. Anything else (no record, unreadable, other version, file
/// missing or of another size) is `None`: the model is not there.
pub fn read(out: &Path, version: &str, dtype: &str) -> Option<Materialized> {
    let bytes = std::fs::read(out.join(RECORD)).ok()?;
    let json = json_strict::parse(&bytes).ok()?;
    let mut o = Obj::new(&json, RECORD).ok()?;
    let recipe = o.str("recipe").ok()?;
    let d = o.str("dtype").ok()?;
    let model = o.str("model").ok()?;
    let size = o.u64("size").ok()?;
    let sha256 = o.str("sha256").ok()?.to_string();
    o.finish().ok()?;
    if recipe != version
        || d != dtype
        || model.contains(['/', '\\'])
        || !super::files::is_sha256(&sha256)
    {
        return None;
    }
    let model = out.join(model);
    (std::fs::metadata(&model).ok()?.len() == size).then_some(Materialized {
        model,
        sha256,
        size,
    })
}

/// Remove a converted folder (best effort).
pub fn discard(out: &Path) {
    let _ = std::fs::remove_dir_all(out);
}

/// Convert into `out`: a temporary folder, then the record, then an atomic rename. If another
/// process (or thread) finished first, its result is used. The flag says whether the file was
/// written by this very call (`true`) or found already there (`false`, to be checked as any file
/// of the cache).
///
/// # Errors
/// Whatever the converter reports; I/O errors; `ConversionFailed` if the result does not match
/// what the converter said it wrote.
pub fn build(
    converter: &dyn Converter,
    job: &ConvertJob<'_>,
    out: &Path,
) -> Result<(Materialized, bool)> {
    let version = converter.version();
    let parent = out.parent().ok_or_else(|| fail("no cache folder"))?;
    std::fs::create_dir_all(parent).map_err(|e| io(parent, &e))?;
    let tmp = parent.join(format!(
        "{}.tmp-{}-{}",
        out.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| io(&tmp, &e))?;
    let result = (|| -> Result<()> {
        let converted = converter.convert(job, &tmp)?;
        let rel = converted.model.to_string_lossy().replace('\\', "/");
        let model = tmp.join(&rel);
        let actual = std::fs::metadata(&model).map_err(|e| io(&model, &e))?.len();
        let (sha256, size) = match (converted.sha256, converted.size) {
            (Some(s), Some(n)) => (s, n),
            _ => sha256_file(&model).map_err(|e| io(&model, &e))?,
        };
        if size != actual {
            return Err(fail(format!(
                "the converter says it wrote {size} bytes to {rel}, the file has {actual}"
            )));
        }
        let record = format!(
            "{{\"recipe\":{},\"dtype\":{},\"model\":{},\"size\":{size},\"sha256\":\"{sha256}\"}}",
            serde_json::to_string(&version).unwrap_or_default(),
            serde_json::to_string(job.dtype).unwrap_or_default(),
            serde_json::to_string(&rel).unwrap_or_default(),
        );
        let path = tmp.join(RECORD);
        std::fs::write(&path, record).map_err(|e| io(&path, &e))
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }
    let unusable = || {
        fail(format!(
            "the converted folder {} is not usable",
            out.display()
        ))
    };
    if std::fs::rename(&tmp, out).is_ok() {
        return read(out, &version, job.dtype)
            .map(|m| (m, true))
            .ok_or_else(unusable);
    }
    // The folder is already there: either another process finished first (its result is used,
    // and the caller must check it, it was not written here), or something unusable is in the way
    // (no record, another version): that is replaced, once.
    if let Some(other) = read(out, &version, job.dtype) {
        let _ = std::fs::remove_dir_all(&tmp);
        return Ok((other, false));
    }
    discard(out);
    let renamed = std::fs::rename(&tmp, out);
    let _ = std::fs::remove_dir_all(&tmp);
    match (renamed, read(out, &version, job.dtype)) {
        (Ok(()), Some(m)) => Ok((m, true)),
        // lost a second race against a process that did finish
        (Err(_), Some(m)) => Ok((m, false)),
        _ => Err(unusable()),
    }
}
