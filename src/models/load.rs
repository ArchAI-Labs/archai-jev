//! `load_model`: the whole path from the arguments of `from_pretrained` to a loaded model,
//! in the fixed order of spec 005, section 1.4. The first check that fails wins.

use std::path::PathBuf;
use std::sync::Arc;

use super::backend::{Backend, ValidatedCheckpoint};
use super::calibration_gate::{self, CalibrationSource};
use super::convert::{Converter, safe_component};
use super::files::FileEntry;
use super::gguf;
use super::hash::{sha256_file, sha256_hex};
use super::head::{HeadReader, check_pointer};
use super::incompat::Incompat;
use super::manifest::{HeadSpec, Manifest, ManifestOrigin, Source, Variant};
use super::record::{self, Record, Stamp};
use super::registry::Registry;
use super::resolve::{Selection, Target, resolve};
use super::tokenizer_check;
use super::tolerance;
use super::validate;
use super::verify::run_selfcheck;
use crate::error::{Error, Result};
use crate::hub::blobs::BlobStore;
use crate::hub::cache::Layout;
use crate::hub::config::HubConfig;
use crate::hub::download::{Transport, ensure_blob};
use crate::hub::events::{Cancel, Event, Observer};
use crate::scorer::Scorer;

/// What to load.
#[derive(Debug, Clone)]
pub struct LoadRequest {
    /// Which model and variant.
    pub selection: Selection,
    /// The user's calibration (`temperature=`).
    pub temperature: Option<f64>,
    /// The user's consent to uncalibrated probabilities.
    pub allow_uncalibrated: bool,
}

/// Everything `load_model` needs from the outside.
pub struct Context<'a> {
    /// The known models.
    pub registry: &'a Registry,
    /// Cache, endpoint, offline.
    pub hub: &'a HubConfig,
    /// The inference engine.
    pub backend: &'a dyn Backend,
    /// The network.
    pub transport: &'a dyn Transport,
    /// Receives progress and notices.
    pub observer: &'a dyn Observer,
    /// Asked whether to stop.
    pub cancel: &'a dyn Cancel,
    /// Reads pointer-head weight files (018), if available.
    pub head_reader: Option<&'a dyn HeadReader>,
    /// Converts `hf-lora` / `hf-full` sources into a GGUF (018), if available.
    pub converter: Option<&'a dyn Converter>,
}

/// Facts about a loaded model, for `ModelInfo`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelData {
    /// Model name.
    pub name: String,
    /// Revision (commit hash, or the label of a local model).
    pub revision: String,
    /// Family key.
    pub family: String,
    /// Head kind.
    pub head: String,
    /// Template, e.g. `chatml-letters-v1`.
    pub template: String,
    /// Weights type.
    pub dtype: String,
    /// Temperature in use.
    pub temperature: f64,
    /// Whether the calibration is declared or given by the user.
    pub calibrated: bool,
    /// `manifest`, `user` or `none`.
    pub calibration_source: &'static str,
    /// SPDX license.
    pub license: String,
    /// Maximum served context in tokens.
    pub max_context: u64,
    /// Ids of the declared tasks, if the model has a closed task set.
    pub tasks: Option<Vec<String>>,
    /// Notice for the user.
    pub notice: Option<String>,
    /// `registry` or `local`.
    pub source: &'static str,
}

/// A model ready to answer.
pub struct LoadedModel {
    /// Facts about it.
    pub data: ModelData,
    /// Answers requests.
    pub scorer: Arc<dyn Scorer>,
}

fn incompat(i: Incompat) -> Error {
    Error::IncompatibleModel(i)
}

fn files_of<'m>(m: &'m Manifest, v: &'m Variant) -> Vec<&'m FileEntry> {
    let mut out = Vec::new();
    if let Source::Gguf { file } = &v.source {
        out.push(file);
    }
    out.push(&m.tokenizer.file);
    if let HeadSpec::Pointer { weights, .. } = &m.head {
        out.push(weights);
    }
    out
}

/// A file ready to be checked: where it is and what the manifest says about it.
struct Located<'m> {
    entry: &'m FileEntry,
    path: PathBuf,
    verified_now: bool,
}

fn locate<'m>(
    ctx: &Context<'_>,
    target: &Target<'_>,
    entries: Vec<&'m FileEntry>,
) -> Result<Vec<Located<'m>>> {
    let layout = Layout::new(&ctx.hub.cache_root);
    let store = BlobStore::new(layout.blobs);
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        match target {
            Target::Registry(_) => {
                let origin = entry.origin.as_ref().ok_or_else(|| {
                    incompat(Incompat::ManifestBadValue {
                        path: entry.path.clone(),
                        detail: "a registry file needs an origin".to_string(),
                    })
                })?;
                let fetched = ensure_blob(
                    ctx.hub,
                    &store,
                    entry,
                    origin,
                    ctx.transport,
                    ctx.observer,
                    ctx.cancel,
                )?;
                out.push(Located {
                    entry,
                    path: fetched.path,
                    verified_now: fetched.verified_now,
                });
            }
            Target::Local { dir, .. } => {
                let path = dir.join(&entry.path);
                if !path.exists() {
                    return Err(incompat(Incompat::FileMissing {
                        path: path.display().to_string(),
                    }));
                }
                out.push(Located {
                    entry,
                    path,
                    verified_now: false,
                });
            }
        }
    }
    Ok(out)
}

fn verify_integrity(files: &[Located<'_>]) -> Result<()> {
    for f in files {
        let shown = f.path.display().to_string();
        let size = std::fs::metadata(&f.path).map(|m| m.len()).map_err(|_| {
            incompat(Incompat::FileMissing {
                path: shown.clone(),
            })
        })?;
        if size != f.entry.size {
            return Err(incompat(Incompat::SizeMismatch {
                path: shown,
                expected: f.entry.size,
                got: size,
            }));
        }
        if f.verified_now {
            continue;
        }
        let (hash, _) = sha256_file(&f.path).map_err(|_| {
            incompat(Incompat::FileMissing {
                path: shown.clone(),
            })
        })?;
        if hash != f.entry.sha256 {
            return Err(incompat(Incompat::HashMismatch {
                path: shown,
                expected: f.entry.sha256.clone(),
                got: hash,
                size,
            }));
        }
    }
    Ok(())
}

/// Run the converter into the cache once per (manifest, dtype, converter version).
fn materialize(
    converter: &dyn Converter,
    raw: &crate::json_strict::Json,
    out: &std::path::Path,
) -> Result<PathBuf> {
    let marker = out.join("DONE");
    if let Ok(done) = std::fs::read_to_string(&marker) {
        return Ok(out.join(done.trim()));
    }
    let fail = |detail: String| incompat(Incompat::ConversionFailed { detail });
    let parent = out
        .parent()
        .ok_or_else(|| fail("no cache folder".to_string()))?;
    std::fs::create_dir_all(parent).map_err(|e| fail(e.to_string()))?;
    let tmp = parent.join(format!(
        "{}.tmp-{}",
        out.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| fail(e.to_string()))?;
    let converted = converter.convert(raw, &tmp).map_err(incompat)?;
    let rel = converted.model.to_string_lossy().replace('\\', "/");
    std::fs::write(tmp.join("DONE"), &rel).map_err(|e| fail(e.to_string()))?;
    if std::fs::rename(&tmp, out).is_err() {
        // another process finished first: use its result
        let _ = std::fs::remove_dir_all(&tmp);
    }
    let done = std::fs::read_to_string(&marker).map_err(|e| fail(e.to_string()))?;
    Ok(out.join(done.trim()))
}

/// Load a model: resolve, read and check the manifest, fetch and verify the files, check the
/// structure, load the weights, self-check (first time only) and return the model.
///
/// # Errors
/// See spec 005: [`Error::ModelArgument`], [`Error::IncompatibleModel`],
/// [`Error::ModelDownload`], [`Error::ModelVerification`], [`Error::Cancelled`].
pub fn load_model(req: &LoadRequest, ctx: &Context<'_>) -> Result<LoadedModel> {
    // Stage 0: arguments and which model.
    let (target, dtype_arg) = resolve(&req.selection, ctx.registry)?;

    // Stage 1: the manifest.
    let manifest: Manifest = match &target {
        Target::Registry(m) => (*m).clone(),
        Target::Local { manifest_path, .. } => {
            let bytes = std::fs::read(manifest_path).map_err(|_| {
                incompat(Incompat::ManifestAbsent {
                    expected: manifest_path.display().to_string(),
                })
            })?;
            let m = Manifest::from_bytes(&bytes, ManifestOrigin::Local).map_err(incompat)?;
            if ctx.registry.has(&m.name) {
                return Err(incompat(Incompat::NameCollision { name: m.name }));
            }
            m
        }
    };

    // Stage 2: what the manifest declares, the variant, the calibration. No file, no network.
    let converter_kinds = ctx.converter.map(Converter::kinds).unwrap_or_default();
    let resolved = validate::semantics(&manifest, &converter_kinds).map_err(incompat)?;
    let dtype = dtype_arg.map_or_else(|| manifest.default_dtype.clone(), str::to_string);
    let variant = manifest.variant(&dtype).ok_or_else(|| {
        incompat(Incompat::DtypeNotOffered {
            dtype: dtype.clone(),
            available: manifest.variants.iter().map(|v| v.dtype.clone()).collect(),
        })
    })?;
    let decision = calibration_gate::decide(
        &manifest.name,
        variant,
        req.temperature,
        req.allow_uncalibrated,
    )?;

    // Stage 3: the files, present and (unless a record vouches for them) verified.
    let layout = Layout::new(&ctx.hub.cache_root);
    let located = locate(ctx, &target, files_of(&manifest, variant))?;
    let manifest_sha = sha256_hex(manifest.canonical_text().as_bytes());
    let key = record::key(&manifest_sha, &dtype);
    let current: Vec<(String, PathBuf, String)> = located
        .iter()
        .map(|f| (f.entry.path.clone(), f.path.clone(), f.entry.sha256.clone()))
        .collect();
    let by_entry = |e: &FileEntry| {
        located
            .iter()
            .find(|f| std::ptr::eq(f.entry, e))
            .map(|f| f.path.clone())
    };
    let tokenizer_path = by_entry(&manifest.tokenizer.file).unwrap_or_default();
    let head_path = match &manifest.head {
        HeadSpec::Pointer { weights, .. } => by_entry(weights),
        HeadSpec::Letters { .. } => None,
    };
    let (model_path, converted) = match &variant.source {
        Source::Gguf { file } => (by_entry(file).unwrap_or_default(), false),
        Source::Reserved { kind, raw } => {
            let converter = ctx
                .converter
                .ok_or_else(|| incompat(Incompat::SourceWithoutConverter { kind: kind.clone() }))?;
            let out = layout
                .materialized
                .join(manifest_sha.get(..16).unwrap_or(&manifest_sha))
                .join(&dtype)
                .join(safe_component(&converter.version()));
            (materialize(converter, raw, &out)?, true)
        }
    };
    // A converted model has no hash in the manifest to vouch for, so it is always re-checked.
    let vouched = !converted
        && record::load(&layout.verified, &key).is_some_and(|r| record::matches(&r, &current));

    if !vouched {
        verify_integrity(&located)?;
        // Stage 4: GGUF.
        let info = gguf::read_header(&model_path).map_err(incompat)?;
        validate::gguf_vs_manifest(&info, &resolved, &manifest, &dtype).map_err(incompat)?;
        // Stage 5: tokenizer and tokens.
        let n_vocab = (resolved.family.n_vocab)(&resolved.params);
        tokenizer_check::check(&tokenizer_path, &manifest, resolved.family, n_vocab)
            .map_err(incompat)?;
        // Stage 6: head weights.
        if let (HeadSpec::Pointer { .. }, Some(path)) = (&manifest.head, &head_path) {
            let reader = ctx.head_reader.ok_or_else(|| {
                incompat(Incompat::HeadWeights {
                    detail: "no reader for head weight files is available in this build"
                        .to_string(),
                })
            })?;
            let declared = variant
                .calibration
                .declared
                .then_some(variant.calibration.temperature)
                .flatten();
            check_pointer(&manifest.head, reader, path, declared).map_err(incompat)?;
        }
    }

    // Stage 7: the engine loads the weights.
    let checkpoint = ValidatedCheckpoint {
        manifest: manifest.clone(),
        dtype: dtype.clone(),
        model_path,
        tokenizer_path,
        head_path,
        calibration: decision,
    };
    let engine = ctx.backend.load(&checkpoint)?;

    // Stage 8: self-check, once per (files, library, platform).
    if !vouched {
        let tol = tolerance::lookup(resolved.family.key, &dtype).ok_or_else(|| {
            incompat(Incompat::NoTolerance {
                family: resolved.family.key.to_string(),
                dtype: dtype.clone(),
            })
        })?;
        run_selfcheck(
            &manifest.name,
            variant,
            engine.runner.as_ref(),
            tol,
            decision.manifest_temperature,
        )?;
        let stamps: Vec<Stamp> = located
            .iter()
            .filter_map(|f| {
                let (size, mtime_ns) = record::stat(&f.path).ok()?;
                Some(Stamp {
                    name: f.entry.path.clone(),
                    size,
                    mtime_ns,
                    sha256: f.entry.sha256.clone(),
                })
            })
            .collect();
        if !converted
            && stamps.len() == located.len()
            && record::save(&layout.verified, &Record { key, files: stamps }).is_err()
        {
            ctx.observer.on_event(&Event::Warning(
                "could not write the verification record; the self-check will run again next time"
                    .to_string(),
            ));
        }
    }

    // Notices.
    ctx.observer.on_event(&Event::License {
        name: manifest.name.clone(),
        spdx: manifest.license.spdx.clone(),
        url: manifest.license.url.clone(),
    });
    if let Some(r) = manifest
        .license
        .restrictions
        .as_deref()
        .filter(|r| !r.is_empty())
    {
        ctx.observer.on_event(&Event::Warning(format!(
            "license restrictions of {}: {r}",
            manifest.name
        )));
    }
    if let Some(n) = manifest.notice.as_deref().filter(|n| !n.is_empty()) {
        ctx.observer
            .on_event(&Event::Warning(format!("{}: {n}", manifest.name)));
    }

    let data = ModelData {
        name: manifest.name.clone(),
        revision: manifest.revision.clone(),
        family: manifest.family.clone(),
        head: manifest.head.kind().to_string(),
        template: format!("{}-v{}", manifest.template.id, manifest.template.version),
        dtype,
        temperature: decision.calibration.temperature(),
        calibrated: decision.calibration.calibrated(),
        calibration_source: match decision.source {
            CalibrationSource::Manifest => "manifest",
            CalibrationSource::User => "user",
            CalibrationSource::None => "none",
        },
        license: manifest.license.spdx.clone(),
        max_context: manifest.max_context,
        tasks: manifest
            .tasks
            .as_ref()
            .map(|t| t.iter().map(|x| x.id.clone()).collect()),
        notice: manifest.notice.clone(),
        source: match target {
            Target::Registry(_) => "registry",
            Target::Local { .. } => "local",
        },
    };
    Ok(LoadedModel {
        data,
        scorer: engine.scorer,
    })
}
