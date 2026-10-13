//! Stage 0 of loading: turn the arguments of `from_pretrained` into a model to load.

use std::path::{Path, PathBuf};

use super::failures::ArgFailure;
use super::incompat::Incompat;
use super::manifest::Manifest;
use super::recipes;
use super::registry::Registry;
use crate::error::{Error, Result};

/// File name of the manifest in a local model folder.
pub const MANIFEST_FILE: &str = "archai-jev-manifest.json";

/// What the user passed as the first argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameOrPath {
    /// Nothing: the default model.
    Default,
    /// A `pathlib.Path` (or other `os.PathLike`): always a folder.
    Path(PathBuf),
    /// A `str`: a registry name or a folder.
    Str(String),
}

/// The arguments of `from_pretrained` that decide which model to load.
#[derive(Debug, Clone)]
pub struct Selection {
    /// First argument.
    pub name_or_path: NameOrPath,
    /// `revision=`.
    pub revision: Option<String>,
    /// `device=`.
    pub device: String,
    /// `dtype=` as given.
    pub dtype: Option<String>,
    /// `manifest=`.
    pub manifest: Option<PathBuf>,
}

/// The model chosen.
#[derive(Debug, Clone)]
pub enum Target<'r> {
    /// An entry of the registry (pinned).
    Registry(&'r Manifest),
    /// A local folder with its manifest file.
    Local {
        /// The model folder.
        dir: PathBuf,
        /// The manifest file to read.
        manifest_path: PathBuf,
    },
}

fn arg(a: ArgFailure) -> Error {
    Error::ModelArgument(a)
}

/// Normalise a `dtype` argument to a canonical name.
///
/// # Errors
/// [`ArgFailure::Dtype`] for a name outside the global list.
pub fn canonical_dtype(given: &str) -> Result<&'static str> {
    recipes::canonical(given).ok_or_else(|| {
        arg(ArgFailure::Dtype {
            got: given.to_string(),
        })
    })
}

/// Check the argument-only rules and choose the model.
///
/// # Errors
/// Wrong arguments ([`Error::ModelArgument`]) or a name that is not known
/// ([`Incompat::UnknownName`]), a revision that is not pinned, a missing manifest.
pub fn resolve<'r>(
    sel: &Selection,
    registry: &'r Registry,
) -> Result<(Target<'r>, Option<&'static str>)> {
    if sel.device != "cpu" {
        return Err(arg(ArgFailure::Device {
            got: sel.device.clone(),
        }));
    }
    let dtype = sel.dtype.as_deref().map(canonical_dtype).transpose()?;

    let local = |dir: PathBuf| -> Result<Target<'r>> {
        if sel.revision.is_some() {
            return Err(arg(ArgFailure::RevisionForLocal));
        }
        let manifest_path = sel
            .manifest
            .clone()
            .unwrap_or_else(|| dir.join(MANIFEST_FILE));
        if !manifest_path.is_file() {
            return Err(Error::IncompatibleModel(Incompat::ManifestAbsent {
                expected: manifest_path.display().to_string(),
            }));
        }
        Ok(Target::Local { dir, manifest_path })
    };
    let from_registry = |name: &str| -> Result<Target<'r>> {
        if sel.manifest.is_some() {
            return Err(arg(ArgFailure::ManifestWithRegistryName {
                name: name.to_string(),
            }));
        }
        let m = registry
            .resolve(name, sel.revision.as_deref())
            .map_err(Error::IncompatibleModel)?;
        Ok(Target::Registry(m))
    };

    let target = match &sel.name_or_path {
        NameOrPath::Default => {
            let default = registry.default_entry().ok_or_else(|| Error::NotAvailable {
                detail: "this build has no default model registered yet: it arrives with the first model-backed release; pass a model name or a local folder with a manifest".to_string(),
            })?;
            from_registry(&default.name.clone())?
        }
        NameOrPath::Path(p) => {
            if !p.is_dir() {
                return Err(arg(ArgFailure::PathNotFound {
                    path: p.display().to_string(),
                }));
            }
            local(p.clone())?
        }
        NameOrPath::Str(s) => {
            let is_name = registry.has(s);
            let is_dir = Path::new(s).is_dir();
            match (is_name, is_dir) {
                (true, true) => return Err(arg(ArgFailure::Ambiguous { name: s.clone() })),
                (true, false) => from_registry(s)?,
                (false, true) => local(PathBuf::from(s))?,
                (false, false) => {
                    return Err(Error::IncompatibleModel(Incompat::UnknownName {
                        name: s.clone(),
                        known: registry.names(),
                    }));
                }
            }
        }
    };
    Ok((target, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn message(given: &str) -> String {
        match canonical_dtype(given) {
            Err(Error::ModelArgument(a)) => a.to_string(),
            other => panic!("{given:?}: expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn canonical_dtype_accepts_the_canonical_names_and_two_synonyms() {
        assert_eq!(canonical_dtype("f32").unwrap(), "f32");
        assert_eq!(canonical_dtype("float32").unwrap(), "f32");
        assert_eq!(canonical_dtype("bfloat16").unwrap(), "bf16");
        assert_eq!(canonical_dtype("q8_0").unwrap(), "q8_0");
    }

    #[test]
    fn unknown_dtype_names_list_the_supported_ones() {
        // spec 009 AC-1: "int8", "q4", "fp16" and a differently written "Q8_0" are all refused
        for bad in ["int8", "q4", "fp16", "Q8_0"] {
            let m = message(bad);
            assert!(m.contains(&format!("'{bad}' is not a known dtype")), "{m}");
            for name in recipes::DTYPES {
                assert!(m.contains(name), "{m} should list {name}");
            }
        }
    }

    #[test]
    fn int8_is_pointed_to_q8_0_and_nothing_else_is() {
        assert!(message("int8").contains("'q8_0' is the 8-bit integer format"));
        assert!(!message("q4").contains("8-bit integer"));
        assert!(!message("Q8_0").contains("8-bit integer"));
    }
}
