//! `load_model`: the whole path from the arguments of `from_pretrained` to a loaded model,
//! in the fixed order of spec 005, section 1.4. The first check that fails wins.

use std::path::PathBuf;
use std::sync::Arc;

use super::backend::{Backend, ValidatedCheckpoint};
use super::calibration_gate::{self, CalibrationSource};
use super::convert::{ConvertJob, Converter, SourceFiles};
use super::files::FileEntry;
use super::gguf;
use super::hash::{sha256_file, sha256_hex};
use super::head::{HeadReader, check_pointer};
use super::incompat::Incompat;
use super::manifest::{HeadSpec, Manifest, ManifestOrigin, Source, Variant};
use super::materialize::{self, Materialized};
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

/// A converted model of this call: how it was made and what the cache recorded about it.
struct Converted<'a> {
    converter: &'a dyn Converter,
    job: ConvertJob<'a>,
    out: PathBuf,
    done: Materialized,
    /// Converted in this very call: its SHA-256 was computed while it was written, so it is
    /// not read again to be compared with itself.
    fresh: bool,
}

impl Converted<'_> {
    /// The name of the GGUF in the verification record.
    fn stamp_name(&self) -> String {
        format!(
            "converted:{}",
            self.done
                .model
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        )
    }

    /// Whether the GGUF in the cache still has the SHA-256 the conversion recorded.
    fn matches_record(&self) -> Result<bool> {
        if self.fresh {
            return Ok(true);
        }
        let (sha, _) = sha256_file(&self.done.model).map_err(|_| {
            incompat(Incompat::FileMissing {
                path: self.done.model.display().to_string(),
            })
        })?;
        Ok(sha == self.done.sha256)
    }
}

fn files_of<'m>(m: &'m Manifest, v: &'m Variant) -> Vec<&'m FileEntry> {
    let mut out = Vec::new();
    match &v.source {
        Source::Gguf { file } => out.push(file),
        Source::HfLora(source) => out.extend(source.files()),
        Source::Reserved { .. } => {}
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
    // `originals_verified` says whether the SHA-256 of every original file was already checked
    // in this call (it is done before a conversion starts, and not twice).
    let mut originals_verified = false;
    let mut converted_here: Option<Converted<'_>> = None;
    let mut model_path = match &variant.source {
        Source::Gguf { file } => by_entry(file).unwrap_or_default(),
        Source::Reserved { kind, .. } => {
            return Err(incompat(Incompat::SourceWithoutConverter {
                kind: kind.clone(),
            }));
        }
        Source::HfLora(source) => {
            let converter = ctx.converter.ok_or_else(|| {
                incompat(Incompat::SourceWithoutConverter {
                    kind: "hf-lora".to_string(),
                })
            })?;
            let version = converter.version();
            let out = materialize::folder(&layout.materialized, &manifest_sha, &dtype, &version);
            let path_of = |e: &FileEntry| by_entry(e).unwrap_or_default();
            let job = ConvertJob {
                manifest: &manifest,
                dtype: &dtype,
                params: &resolved.params,
                source,
                files: SourceFiles {
                    base_config: path_of(&source.base.config),
                    base_weights: path_of(&source.base.weights),
                    base_tokenizer: path_of(&source.base.tokenizer),
                    base_tokenizer_config: path_of(&source.base.tokenizer_config),
                    adapter_config: path_of(&source.adapter.config),
                    adapter_weights: path_of(&source.adapter.weights),
                },
                head_path: head_path.as_deref(),
                destination: out.clone(),
                observer: ctx.observer,
                cancel: ctx.cancel,
            };
            let (done, fresh) = match materialize::read(&out, &version, &dtype) {
                Some(m) => (m, false),
                None => {
                    // The originals are checked **before** anything is converted from them.
                    verify_integrity(&located)?;
                    originals_verified = true;
                    materialize::build(converter, &job, &out)?
                }
            };
            let path = done.model.clone();
            converted_here = Some(Converted {
                converter,
                job,
                out,
                done,
                fresh,
            });
            path
        }
    };
    // What the verification record covers: the originals and, for a converted model, the GGUF
    // as the conversion recorded it.
    let mut current = current;
    if let Some(c) = &converted_here {
        current.push((c.stamp_name(), c.done.model.clone(), c.done.sha256.clone()));
    }
    let vouched =
        record::load(&layout.verified, &key).is_some_and(|r| record::matches(&r, &current));

    if !vouched {
        if !originals_verified {
            verify_integrity(&located)?;
        }
        if let Some(c) = converted_here.as_mut() {
            // The converted file must still be what the conversion wrote. If not (a damaged or
            // altered cache), it is derived data: convert again from the verified originals,
            // once; if that does not match either, something is wrong with the converter.
            if !c.matches_record()? {
                ctx.observer.on_event(&Event::Warning(
                    "the converted model in the cache does not match its record; converting again"
                        .to_string(),
                ));
                materialize::discard(&c.out);
                (c.done, c.fresh) = materialize::build(c.converter, &c.job, &c.out)?;
                if !c.matches_record()? {
                    return Err(incompat(Incompat::ConversionFailed {
                        detail: "the converted file does not match its record even after converting again"
                            .to_string(),
                    }));
                }
                model_path = c.done.model.clone();
                if let Some(last) = current.last_mut() {
                    *last = (c.stamp_name(), c.done.model.clone(), c.done.sha256.clone());
                }
            }
        }
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
            let base = match &variant.source {
                Source::HfLora(s) => Some((s.base.repo.as_str(), s.base.revision.as_str())),
                _ => None,
            };
            check_pointer(&manifest.head, reader, path, declared, base).map_err(incompat)?;
        }
    }

    // Stage 7: the engine loads the weights.
    let model_sha256 = match (&variant.source, &converted_here) {
        (Source::Gguf { file }, _) => file.sha256.clone(),
        (_, Some(c)) => c.done.sha256.clone(),
        _ => String::new(),
    };
    let checkpoint = ValidatedCheckpoint {
        manifest: manifest.clone(),
        dtype: dtype.clone(),
        model_path,
        model_sha256,
        manifest_sha256: manifest_sha.clone(),
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
        let stamps: Vec<Stamp> = current
            .iter()
            .filter_map(|(name, path, sha256)| {
                let (size, mtime_ns) = record::stat(path).ok()?;
                Some(Stamp {
                    name: name.clone(),
                    size,
                    mtime_ns,
                    sha256: sha256.clone(),
                })
            })
            .collect();
        if stamps.len() == current.len()
            && record::save(&layout.verified, &Record { key, files: stamps }).is_err()
        {
            ctx.observer.on_event(&Event::Warning(
                "could not write the verification record; the self-check will run again next time"
                    .to_string(),
            ));
        }
    }

    // The self-check ran through the very path of real requests and may have saved prefixes: the
    // first request of the user starts cold, with nothing counted.
    engine.scorer.clear_cache();
    engine.scorer.reset_cache_counters();

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
