//! Fail-closed validation (D20, level 1): what the manifest declares (stage 2) and whether a
//! GGUF file matches it (stage 4). Each failure is one [`Incompat`] reason.

use super::families::{self, ArchParams, Family, MetaExpect, Role};
use super::gguf::{GgmlType, GgufInfo, MetaValue};
use super::incompat::Incompat;
use super::manifest::{CalibrationDecl, HeadSpec, Manifest, Source, Variant};
use super::tolerance;
use super::vectors::check_vectors;

/// The family and architecture parameters a manifest resolved to.
#[derive(Clone)]
pub struct Resolved {
    /// The family of the table.
    pub family: &'static Family,
    /// The architecture parameters, checked against the family.
    pub params: ArchParams,
}

fn tasks_kinds(m: &Manifest, family: &Family) -> Vec<String> {
    match &m.tasks {
        None => family
            .question_types
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        Some(tasks) => {
            let mut kinds: Vec<String> = Vec::new();
            for t in tasks {
                let q = if t.kind == "yes_no" {
                    "noul"
                } else {
                    t.kind.as_str()
                };
                if !kinds.iter().any(|k| k == q) {
                    kinds.push(q.to_string());
                }
            }
            kinds
        }
    }
}

fn check_calibration(c: &CalibrationDecl) -> Result<(), Incompat> {
    let bad = |detail: &str| Incompat::BadCalibration {
        detail: detail.to_string(),
    };
    if c.declared {
        match c.temperature {
            Some(t) if t.is_finite() && t > 0.0 => {}
            Some(t) => {
                return Err(bad(&format!(
                    "temperature must be a finite number > 0, got {t}"
                )));
            }
            None => return Err(bad("declared is true but temperature is missing")),
        }
        if c.evidence.as_deref().is_none_or(|e| e.trim().is_empty()) {
            return Err(bad(
                "declared is true but evidence (where the temperature comes from) is missing or empty",
            ));
        }
    } else if c.temperature.is_some() || c.evidence.is_some() {
        return Err(bad(
            "declared is false, so temperature and evidence must be absent",
        ));
    }
    Ok(())
}

fn check_variant(
    v: &Variant,
    m: &Manifest,
    family: &Family,
    n_vocab: u64,
    converter_kinds: &[String],
) -> Result<(), Incompat> {
    if tolerance::lookup(family.key, &v.dtype).is_none() {
        return Err(Incompat::NoTolerance {
            family: family.key.to_string(),
            dtype: v.dtype.clone(),
        });
    }
    if let Source::Reserved { kind, .. } = &v.source
        && !converter_kinds.contains(kind)
    {
        return Err(Incompat::SourceWithoutConverter { kind: kind.clone() });
    }
    for kind in tasks_kinds(m, family) {
        if !v
            .vectors
            .iter()
            .any(|vec| vec.question_types().contains(&kind))
        {
            return Err(Incompat::MissingVectors {
                dtype: v.dtype.clone(),
                kind,
            });
        }
    }
    check_vectors(&v.vectors, n_vocab)?;
    check_calibration(&v.calibration)
}

fn check_tasks(m: &Manifest) -> Result<(), Incompat> {
    let Some(tasks) = &m.tasks else { return Ok(()) };
    let bad = |index: usize, detail: String| Incompat::BadTasks { index, detail };
    if tasks.is_empty() {
        return Err(bad(
            0,
            "the task list is empty; use null for no restriction".to_string(),
        ));
    }
    for (i, t) in tasks.iter().enumerate() {
        if t.id.trim().is_empty() {
            return Err(bad(i, "the task id is empty".to_string()));
        }
        if !matches!(t.kind.as_str(), "choice" | "yes_no") {
            return Err(bad(
                i,
                format!("kind must be \"choice\" or \"yes_no\", got {:?}", t.kind),
            ));
        }
        if let Some(j) = tasks.iter().take(i).position(|o| o.id == t.id) {
            return Err(bad(
                i,
                format!("the id {:?} is already used by tasks[{j}]", t.id),
            ));
        }
    }
    Ok(())
}

/// Stage 2: check what the manifest declares against the table of families, the table of
/// tolerances and itself. Needs no file and no network.
///
/// # Errors
/// The first [`Incompat`] found, in the order of spec 005 section 6.
pub fn semantics(m: &Manifest, converter_kinds: &[String]) -> Result<Resolved, Incompat> {
    let family = families::lookup(&m.family).ok_or_else(|| Incompat::UnknownFamily {
        got: m.family.clone(),
        supported: families::keys(),
    })?;
    if m.architecture.name != family.arch {
        return Err(Incompat::ArchitectureMismatch {
            family: family.key.to_string(),
            expected: family.arch.to_string(),
            got: m.architecture.name.clone(),
        });
    }
    let params = family.read_params(&m.architecture)?;
    if m.head.kind() != family.head_kind {
        return Err(Incompat::HeadMismatch {
            family: family.key.to_string(),
            expected: family.head_kind.to_string(),
            got: m.head.kind().to_string(),
        });
    }
    if m.template.id != family.template_id || m.template.version != family.template_version {
        return Err(Incompat::TemplateMismatch {
            family: family.key.to_string(),
            expected: format!("{} v{}", family.template_id, family.template_version),
            got: format!("{} v{}", m.template.id, m.template.version),
        });
    }
    if m.variant(&m.default_dtype).is_none() {
        return Err(Incompat::ManifestBadValue {
            path: "default_dtype".to_string(),
            detail: format!("{:?} is not one of the variants", m.default_dtype),
        });
    }
    check_tasks(m)?;
    if family.head_kind == "pointer" {
        if m.tasks.is_some() {
            return Err(Incompat::TasksNotSupported {
                family: family.key.to_string(),
            });
        }
        if let HeadSpec::Pointer { d_model, .. } = &m.head
            && *d_model != params.int("n_embd")
        {
            return Err(Incompat::HeadDimension {
                d_model: *d_model,
                n_embd: params.int("n_embd"),
            });
        }
    }
    let n_vocab = (family.n_vocab)(&params);
    for v in &m.variants {
        check_variant(v, m, family, n_vocab, converter_kinds)?;
    }
    if m.license.spdx.trim().is_empty() {
        return Err(Incompat::MissingLicense);
    }
    if let HeadSpec::Letters { choice_targets, .. } = &m.head
        && choice_targets.len() > 255
    {
        return Err(Incompat::ManifestBadValue {
            path: "head.choice_targets".to_string(),
            detail: "has more than 255 targets".to_string(),
        });
    }
    Ok(Resolved { family, params })
}

fn describe(v: Option<&MetaValue>) -> String {
    match v {
        None => "missing".to_string(),
        Some(MetaValue::Str(s)) => format!("{s:?}"),
        Some(MetaValue::UInt(n)) => n.to_string(),
        Some(MetaValue::Int(n)) => n.to_string(),
        Some(other) => format!("{other:?}"),
    }
}

fn as_uint(v: &MetaValue) -> Option<u64> {
    match v {
        MetaValue::UInt(n) => Some(*n),
        MetaValue::Int(n) => u64::try_from(*n).ok(),
        _ => None,
    }
}

/// Stage 4: does the GGUF header match what the manifest and the family say?
///
/// # Errors
/// Metadata that differs, a context shorter than `max_context`, and tensors that are missing,
/// unexpected, of another shape or of another type.
pub fn gguf_vs_manifest(
    info: &GgufInfo,
    resolved: &Resolved,
    manifest: &Manifest,
    dtype: &str,
) -> Result<(), Incompat> {
    let family = resolved.family;
    for (key, expect) in (family.metadata)(&resolved.params) {
        let got = info.get(&key);
        let ok = match (&expect, got) {
            (MetaExpect::Str(s), Some(MetaValue::Str(g))) => s == g,
            (MetaExpect::UInt(n), Some(g)) => as_uint(g) == Some(*n),
            _ => false,
        };
        if !ok {
            return Err(Incompat::GgufMetadata {
                key,
                expected: match expect {
                    MetaExpect::Str(s) => format!("{s:?}"),
                    MetaExpect::UInt(n) => n.to_string(),
                },
                got: describe(got),
            });
        }
    }
    let context = info.get(family.context_key).and_then(as_uint);
    let Some(context_length) = context else {
        return Err(Incompat::GgufMetadata {
            key: family.context_key.to_string(),
            expected: "a context length".to_string(),
            got: describe(info.get(family.context_key)),
        });
    };
    if manifest.max_context > context_length {
        return Err(Incompat::ContextTooLong {
            max_context: manifest.max_context,
            context_length,
        });
    }

    let matrix = families::matrix_type(dtype);
    let expected = (family.tensors)(&resolved.params);
    for spec in &expected {
        let Some(t) = info.tensor(&spec.name) else {
            return Err(Incompat::TensorMissing {
                name: spec.name.clone(),
            });
        };
        if t.dims != spec.dims {
            return Err(Incompat::TensorShape {
                name: spec.name.clone(),
                expected: spec.dims.clone(),
                got: t.dims.clone(),
            });
        }
        let want = match spec.role {
            Role::Vector => Some(GgmlType::F32),
            Role::Matrix => matrix,
        };
        if want != Some(t.ty) {
            return Err(Incompat::TensorType {
                name: spec.name.clone(),
                expected: want.map_or_else(|| "a supported type".to_string(), GgmlType::name),
                got: t.ty.name(),
            });
        }
    }
    if let Some(extra) = info
        .tensors
        .iter()
        .find(|t| !expected.iter().any(|s| s.name == t.name))
    {
        return Err(Incompat::TensorUnexpected {
            name: extra.name.clone(),
        });
    }
    Ok(())
}
