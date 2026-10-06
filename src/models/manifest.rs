//! The manifest (schema v1): what a checkpoint is, in our terms (D19).
//!
//! Reading is strict and has two steps. This file does the *structural* one (stage 1: JSON,
//! fields, types, domains of values). What the declarations mean (family, tolerances,
//! calibration, tasks) is checked in `validate` (stage 2).

use super::files::{FileEntry, is_commit};
use super::incompat::Incompat;
use super::vectors::{Vector, parse_vectors};
use crate::json_strict::{self, FieldProblem, Json, Obj};

/// The only schema version this build reads.
pub const SCHEMA_VERSION: u64 = 1;

/// Where a manifest comes from: the built-in registry or a user's folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestOrigin {
    /// Compiled into the wheel: files carry an `origin` to download from.
    Registry,
    /// Provided by the user next to a local model: nothing is downloaded.
    Local,
}

/// A token declared for a role of the template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    /// The token text, e.g. `<|im_start|>`.
    pub text: String,
    /// Its id in the tokenizer.
    pub id: u32,
}

/// Tokenizer file and special tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerSpec {
    /// The `tokenizer.json`.
    pub file: FileEntry,
    /// Role (as named by the template) to token.
    pub special_tokens: Vec<(String, SpecialToken)>,
}

/// One answer letter or word and its token id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// The label, e.g. `A` or `TRUE`.
    pub label: String,
    /// The token id of the label.
    pub id: u32,
}

/// The decision head declared by the manifest.
#[derive(Debug, Clone, PartialEq)]
pub enum HeadSpec {
    /// Read the logits of the answer letters (D29): restricted softmax over fixed ids.
    Letters {
        /// Only `restricted_softmax` exists.
        rule: String,
        /// Targets for the 1st, 2nd, ... option of a Choice.
        choice_targets: Vec<Target>,
        /// `(no, yes)` targets for YesNo, if supported.
        yes_no: Option<(Target, Target)>,
    },
    /// Kev's pointer head: two projections and a dot product.
    Pointer {
        /// Hidden size of the backbone.
        d_model: u64,
        /// Projection size (256 for Kev).
        proj_dim: u64,
        /// The head weights file.
        weights: FileEntry,
    },
}

impl HeadSpec {
    /// The kind name used in manifests.
    pub fn kind(&self) -> &'static str {
        match self {
            HeadSpec::Letters { .. } => "letters",
            HeadSpec::Pointer { .. } => "pointer",
        }
    }
}

/// A prompt template and its version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateSpec {
    /// Template id, e.g. `chatml-letters`.
    pub id: String,
    /// Template version.
    pub version: u64,
}

/// Declared license.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct License {
    /// SPDX identifier (or a `LicenseRef-`).
    pub spdx: String,
    /// Where the license text is.
    pub url: String,
    /// Restrictions to show the user, if any.
    pub restrictions: Option<String>,
}

/// One declared task of a closed task set (D29).
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// Unique id.
    pub id: String,
    /// `choice` or `yes_no` (checked in stage 2).
    pub kind: String,
    /// How a question is matched to the task: opaque here, defined by the engine (006).
    pub matching: Json,
}

/// Where the weights of a variant come from.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// A ready-made GGUF file.
    Gguf {
        /// The GGUF.
        file: FileEntry,
    },
    /// Reserved for the converter (018): kind and raw fields.
    Reserved {
        /// `hf-lora` or `hf-full`.
        kind: String,
        /// The fields, unchecked.
        raw: Json,
    },
}

/// Calibration as written in the manifest; its consistency is checked in stage 2.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationDecl {
    /// Whether a calibration is declared.
    pub declared: bool,
    /// The temperature (only with `declared`).
    pub temperature: Option<f64>,
    /// Where the temperature comes from (only with `declared`).
    pub evidence: Option<String>,
}

/// One variant of the weights (one dtype), with its own calibration and self-check.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// Weights type name, e.g. `q8_0`.
    pub dtype: String,
    /// Where the weights come from.
    pub source: Source,
    /// Declared calibration.
    pub calibration: CalibrationDecl,
    /// Self-check vectors.
    pub vectors: Vec<Vector>,
}

/// The `architecture` object: a name and free-form structural parameters whose allowed
/// names and types are decided by the family (stage 2).
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureSpec {
    /// Architecture name, e.g. `qwen2`.
    pub name: String,
    /// The other fields, in document order.
    pub params: Vec<(String, Json)>,
}

/// A parsed manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// Registry or local.
    pub origin: ManifestOrigin,
    /// Model name.
    pub name: String,
    /// Revision (commit hash in the registry, free label for local models).
    pub revision: String,
    /// Family key.
    pub family: String,
    /// Architecture.
    pub architecture: ArchitectureSpec,
    /// Decision head.
    pub head: HeadSpec,
    /// Prompt template.
    pub template: TemplateSpec,
    /// Tokenizer.
    pub tokenizer: TokenizerSpec,
    /// Maximum served context in tokens.
    pub max_context: u64,
    /// Context seen in training, informational.
    pub trained_context: Option<u64>,
    /// License.
    pub license: License,
    /// Notice shown to the user.
    pub notice: Option<String>,
    /// Closed task set, or `None` for no restriction.
    pub tasks: Option<Vec<Task>>,
    /// Name of the variant loaded by default.
    pub default_dtype: String,
    /// The variants, in manifest order.
    pub variants: Vec<Variant>,
    /// The parsed tree (its canonical form is what gets hashed).
    pub raw: Json,
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_alphanumeric());
    first_ok
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/'))
}

fn valid_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 64
        && label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn valid_dtype(dtype: &str) -> bool {
    !dtype.is_empty()
        && dtype.len() <= 16
        && dtype
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

fn bad(path: &str, detail: String) -> Incompat {
    Incompat::ManifestBadValue {
        path: path.to_string(),
        detail,
    }
}

fn token_id(path: &str, v: &Json) -> Result<u32, Incompat> {
    match v {
        Json::Int(n) if (0..=i128::from(u32::MAX)).contains(n) => {
            Ok(u32::try_from(*n).unwrap_or(0))
        }
        other => Err(bad(
            path,
            format!(
                "must be a token id (a non-negative integer), got {}",
                other.kind()
            ),
        )),
    }
}

fn target(json: &Json, path: &str) -> Result<Target, Incompat> {
    let mut o = Obj::new(json, path)?;
    let label = o.str("label")?.to_string();
    let id = token_id(&o.child_path("id"), o.req("id")?)?;
    o.finish()?;
    Ok(Target { label, id })
}

fn parse_head(o: &mut Obj<'_>, family: &str, registry: bool) -> Result<HeadSpec, Incompat> {
    let mut h = o.obj("head")?;
    let kind = h.str("kind")?;
    let spec = match kind {
        "letters" => {
            let rule = h.str("rule")?.to_string();
            if rule != "restricted_softmax" {
                return Err(bad(
                    &h.child_path("rule"),
                    format!("must be \"restricted_softmax\", got {rule:?}"),
                ));
            }
            let arr = h.arr("choice_targets")?;
            if arr.is_empty() {
                return Err(bad(
                    &h.child_path("choice_targets"),
                    "must have at least one target".to_string(),
                ));
            }
            let base = h.child_path("choice_targets");
            let choice_targets = arr
                .iter()
                .enumerate()
                .map(|(i, t)| target(t, &format!("{base}[{i}]")))
                .collect::<Result<Vec<_>, _>>()?;
            let yes_no = match h.req("yes_no_targets")? {
                Json::Null => None,
                other => {
                    let p = h.child_path("yes_no_targets");
                    let mut yo = Obj::new(other, &p)?;
                    let no = target(yo.req("no")?, &format!("{p}.no"))?;
                    let yes = target(yo.req("yes")?, &format!("{p}.yes"))?;
                    yo.finish()?;
                    Some((no, yes))
                }
            };
            HeadSpec::Letters {
                rule,
                choice_targets,
                yes_no,
            }
        }
        "pointer" => {
            let d_model = h.u64("d_model")?;
            let proj_dim = h.u64("proj_dim")?;
            let wp = h.child_path("weights");
            let weights = FileEntry::from_json(h.req("weights")?, &wp, registry)?;
            HeadSpec::Pointer {
                d_model,
                proj_dim,
                weights,
            }
        }
        other => {
            return Err(Incompat::HeadMismatch {
                family: family.to_string(),
                expected: "one of: letters, pointer".to_string(),
                got: other.to_string(),
            });
        }
    };
    h.finish()?;
    Ok(spec)
}

fn parse_source(v: &Json, path: &str, registry: bool) -> Result<Source, Incompat> {
    let mut s = Obj::new(v, path)?;
    let kind = s.str("kind")?;
    match kind {
        "gguf" => {
            let file = FileEntry::from_json(s.req("file")?, &s.child_path("file"), registry)?;
            s.finish()?;
            Ok(Source::Gguf { file })
        }
        "hf-lora" | "hf-full" => Ok(Source::Reserved {
            kind: kind.to_string(),
            raw: v.clone(),
        }),
        other => Err(bad(
            &s.child_path("kind"),
            format!("must be \"gguf\", \"hf-lora\" or \"hf-full\", got {other:?}"),
        )),
    }
}

fn parse_calibration(v: &Json, path: &str) -> Result<CalibrationDecl, Incompat> {
    let mut c = Obj::new(v, path)?;
    let declared = c.bool("declared")?;
    let temperature = match c.opt("temperature") {
        None => None,
        Some(Json::Int(n)) => Some(*n as f64),
        Some(Json::Float(f)) => Some(*f),
        Some(other) => {
            return Err(bad(
                &c.child_path("temperature"),
                format!("must be a number, got {}", other.kind()),
            ));
        }
    };
    let evidence = match c.opt("evidence") {
        None => None,
        Some(Json::Str(s)) => Some(s.clone()),
        Some(other) => {
            return Err(bad(
                &c.child_path("evidence"),
                format!("must be a string, got {}", other.kind()),
            ));
        }
    };
    c.finish()?;
    Ok(CalibrationDecl {
        declared,
        temperature,
        evidence,
    })
}

fn parse_variants(o: &mut Obj<'_>, registry: bool) -> Result<Vec<Variant>, Incompat> {
    let items = o.arr("variants")?;
    if items.is_empty() {
        return Err(bad(
            "variants",
            "must have at least one variant".to_string(),
        ));
    }
    let mut out: Vec<Variant> = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let vp = format!("variants[{i}]");
        let mut v = Obj::new(item, &vp)?;
        let dtype = v.str("dtype")?.to_string();
        if !valid_dtype(&dtype) {
            return Err(bad(
                &format!("{vp}.dtype"),
                format!("must be 1-16 characters of a-z, 0-9 and _, got {dtype:?}"),
            ));
        }
        if out.iter().any(|x| x.dtype == dtype) {
            return Err(bad(
                &format!("{vp}.dtype"),
                format!("{dtype:?} is used by two variants"),
            ));
        }
        let source = parse_source(v.req("source")?, &format!("{vp}.source"), registry)?;
        let calibration = parse_calibration(v.req("calibration")?, &format!("{vp}.calibration"))?;
        let mut sc = v.obj("selfcheck")?;
        let vectors = parse_vectors(sc.arr("vectors")?, &format!("{vp}.selfcheck.vectors"))?;
        sc.finish()?;
        v.finish()?;
        out.push(Variant {
            dtype,
            source,
            calibration,
            vectors,
        });
    }
    Ok(out)
}

fn parse_tasks(o: &mut Obj<'_>) -> Result<Option<Vec<Task>>, Incompat> {
    match o.req("tasks")? {
        Json::Null => Ok(None),
        Json::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let tp = format!("tasks[{i}]");
                let mut t = Obj::new(item, &tp)?;
                let id = t.str("id")?.to_string();
                let kind = t.str("kind")?.to_string();
                let matching = t.req("match")?.clone();
                t.finish()?;
                out.push(Task { id, kind, matching });
            }
            Ok(Some(out))
        }
        other => Err(bad(
            "tasks",
            format!("must be an array or null, got {}", other.kind()),
        )),
    }
}

impl Manifest {
    /// Read a manifest from bytes (stage 1).
    ///
    /// # Errors
    /// [`Incompat`] naming the first structural problem.
    pub fn from_bytes(bytes: &[u8], origin: ManifestOrigin) -> Result<Manifest, Incompat> {
        let raw = json_strict::parse(bytes)?;
        Self::from_json(raw, origin)
    }

    /// Read a manifest from an already parsed tree (stage 1).
    ///
    /// # Errors
    /// [`Incompat`] naming the first structural problem.
    pub fn from_json(raw: Json, origin: ManifestOrigin) -> Result<Manifest, Incompat> {
        let registry = origin == ManifestOrigin::Registry;
        let mut o = Obj::new(&raw, "<manifest>")?;

        match o.opt("schema_version") {
            None => {
                return Err(Incompat::ManifestMissingField {
                    path: "schema_version".to_string(),
                });
            }
            Some(Json::Int(n)) if u64::try_from(*n).ok() == Some(SCHEMA_VERSION) => {}
            Some(other) => {
                return Err(Incompat::SchemaVersion {
                    got: other.to_canonical_string(),
                    supported: SCHEMA_VERSION.to_string(),
                });
            }
        }

        let name = o.str("name")?.to_string();
        if !valid_name(&name) {
            return Err(bad(
                "name",
                format!(
                    "must start with a letter or digit and use only letters, digits and ._/- (at most 128), got {name:?}"
                ),
            ));
        }
        let revision = o.str("revision")?.to_string();
        let revision_ok = if registry {
            is_commit(&revision)
        } else {
            valid_label(&revision)
        };
        if !revision_ok {
            return Err(bad(
                "revision",
                if registry {
                    format!("must be a full 40-digit commit hash, got {revision:?}")
                } else {
                    format!("must be 1-64 characters of letters, digits and ._-, got {revision:?}")
                },
            ));
        }
        let family = o.str("family")?.to_string();

        let mut a = o.obj("architecture")?;
        let arch_name = a.str("name")?.to_string();
        let params = a
            .rest()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let architecture = ArchitectureSpec {
            name: arch_name,
            params,
        };

        let head = parse_head(&mut o, &family, registry)?;

        let mut t = o.obj("template")?;
        let template = TemplateSpec {
            id: t.str("id")?.to_string(),
            version: t.u64("version")?,
        };
        t.finish()?;

        let mut tk = o.obj("tokenizer")?;
        let file = FileEntry::from_json(tk.req("file")?, "tokenizer.file", registry)?;
        let Json::Object(st) = tk.req("special_tokens")? else {
            return Err(bad(
                "tokenizer.special_tokens",
                "must be an object of role to token".to_string(),
            ));
        };
        let mut special_tokens = Vec::with_capacity(st.len());
        for (role, tok) in st {
            let p = format!("tokenizer.special_tokens.{role}");
            let mut to = Obj::new(tok, &p)?;
            let text = to.str("text")?.to_string();
            let id = token_id(&format!("{p}.id"), to.req("id")?)?;
            to.finish()?;
            special_tokens.push((role.clone(), SpecialToken { text, id }));
        }
        tk.finish()?;
        let tokenizer = TokenizerSpec {
            file,
            special_tokens,
        };

        let max_context = o.u64("max_context")?;
        if max_context == 0 {
            return Err(bad("max_context", "must be at least 1".to_string()));
        }
        let trained_context = match o.opt("trained_context") {
            None | Some(Json::Null) => None,
            Some(Json::Int(n)) if *n >= 1 => u64::try_from(*n).ok(),
            Some(other) => {
                return Err(bad(
                    "trained_context",
                    format!("must be a positive integer or null, got {}", other.kind()),
                ));
            }
        };

        let mut l = o.obj("license")?;
        let spdx = l.str("spdx")?.to_string();
        let url = l.str("url")?.to_string();
        let restrictions = match l.opt("restrictions") {
            None | Some(Json::Null) => None,
            Some(Json::Str(s)) => Some(s.clone()),
            Some(other) => {
                return Err(bad(
                    "license.restrictions",
                    format!("must be a string or null, got {}", other.kind()),
                ));
            }
        };
        l.finish()?;
        let license = License {
            spdx,
            url,
            restrictions,
        };

        let notice = match o.opt("notice") {
            None | Some(Json::Null) => None,
            Some(Json::Str(s)) => Some(s.clone()),
            Some(other) => {
                return Err(bad(
                    "notice",
                    format!("must be a string or null, got {}", other.kind()),
                ));
            }
        };
        let tasks = parse_tasks(&mut o)?;
        let default_dtype = o.str("default_dtype")?.to_string();
        let variants = parse_variants(&mut o, registry)?;
        o.finish()?;

        Ok(Manifest {
            origin,
            name,
            revision,
            family,
            architecture,
            head,
            template,
            tokenizer,
            max_context,
            trained_context,
            license,
            notice,
            tasks,
            default_dtype,
            variants,
            raw,
        })
    }

    /// The variant called `dtype`.
    pub fn variant(&self, dtype: &str) -> Option<&Variant> {
        self.variants.iter().find(|v| v.dtype == dtype)
    }

    /// Canonical text of the parsed manifest (what the verification record hashes).
    pub fn canonical_text(&self) -> String {
        self.raw.to_canonical_string()
    }
}

impl From<json_strict::JsonProblem> for FieldProblem {
    fn from(p: json_strict::JsonProblem) -> Self {
        FieldProblem::Bad {
            path: "<manifest>".to_string(),
            detail: p.to_string(),
        }
    }
}
