//! The plan of a conversion (spec 018, 3.2): which tensor of the GGUF comes from which tensors of
//! the checkpoint, and every check that the checkpoint is what the layout says it is.
//!
//! The plan is built from the **headers** only (names, shapes, types): nothing is read from the
//! data, so a wrong checkpoint is refused before a single weight is converted.

use super::layout::{Dims, Entry};
use super::lora::{Half, LoraConfig, base_of};
use super::safetensors::{Dtype, SafeTensors};
use crate::error::{Error, Result};
use crate::models::families::Role;
use crate::models::gguf::GgmlType;
use crate::models::incompat::Incompat;

const BASE: &str = "base weights";
const ADAPTER: &str = "adapter weights";

/// The names of the two tensors of the LoRA pair that adapts a weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    /// `lora_A`, `[r, in]`.
    pub a: String,
    /// `lora_B`, `[out, r]`.
    pub b: String,
}

/// One tensor to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Planned {
    /// The layout entry.
    pub entry: Entry,
    /// The type written to the GGUF.
    pub ty: GgmlType,
    /// Bytes written.
    pub bytes: u64,
    /// The adapter pair merged into it, if a LoRA adapts this weight.
    pub merge: Option<Pair>,
}

fn refuse(file: &str, detail: String) -> Error {
    Error::IncompatibleModel(Incompat::SourceFile {
        file: file.to_string(),
        detail,
    })
}

/// The type a variant stores its matrices in.
///
/// # Errors
/// `SourceUnsupported` for a dtype the converter does not write.
pub fn matrix_type(dtype: &str) -> Result<GgmlType> {
    match dtype {
        "f32" => Ok(GgmlType::F32),
        "bf16" => Ok(GgmlType::BF16),
        other => Err(Error::IncompatibleModel(Incompat::SourceUnsupported {
            what: format!("the dtype {other:?}"),
            detail: "the converter writes f32 and bf16; quantised variants are not produced here"
                .to_string(),
        })),
    }
}

fn pair_of(entry: &Entry) -> Option<Pair> {
    let module = entry.hf.strip_suffix(".weight")?;
    let module = module.strip_prefix("model.language_model.")?;
    Some(Pair {
        a: format!("base_model.model.{module}.lora_A.weight"),
        b: format!("base_model.model.{module}.lora_B.weight"),
    })
}

/// Check `base` and `adapter` against the layout and build the plan.
///
/// # Errors
/// `IncompatibleModelError` naming the tensor for anything missing, unexpected, of another shape
/// or type, or an adapter that does not match the weights it adapts.
pub fn build(
    dims: &Dims,
    dtype: &str,
    base: &SafeTensors,
    adapter: &SafeTensors,
    lora: &LoraConfig,
) -> Result<Plan> {
    let variant_type = matrix_type(dtype)?;
    let entries = dims.entries();

    // the checkpoint holds exactly the expected tensors (and the vision tower, which we drop)
    for t in base.tensors() {
        if !entries.iter().any(|e| e.hf == t.name) && !t.name.starts_with("model.visual.") {
            return Err(refuse(
                BASE,
                format!(
                    "tensor '{}' is not expected for this model (nothing is ignored silently)",
                    t.name
                ),
            ));
        }
    }
    let mut planned = Vec::with_capacity(entries.len());
    let mut merged_pairs: Vec<Pair> = Vec::new();
    for entry in entries {
        let Some(t) = base.get(&entry.hf) else {
            return Err(refuse(BASE, format!("tensor '{}' is missing", entry.hf)));
        };
        if t.shape != entry.hf_shape {
            return Err(refuse(
                BASE,
                format!(
                    "tensor '{}' has shape {:?}, expected {:?}",
                    entry.hf, t.shape, entry.hf_shape
                ),
            ));
        }
        if t.dtype != entry.hf_dtype {
            return Err(refuse(
                BASE,
                format!(
                    "tensor '{}' has dtype {}, expected {}",
                    entry.hf,
                    t.dtype.name(),
                    entry.hf_dtype.name()
                ),
            ));
        }
        let ty = if entry.role == Role::Vector {
            GgmlType::F32
        } else {
            variant_type
        };
        let bytes = ty.byte_size(t.elements()).ok_or_else(|| {
            refuse(
                BASE,
                format!(
                    "tensor '{}' has a size that does not fit its type",
                    entry.hf
                ),
            )
        })?;
        let merge = match entry.module {
            Some(module)
                if lora.targets(module) && entry.hf.starts_with("model.language_model.") =>
            {
                let pair = pair_of(&entry);
                if let Some(p) = &pair {
                    merged_pairs.push(p.clone());
                }
                pair
            }
            _ => None,
        };
        planned.push(Planned {
            entry,
            ty,
            bytes,
            merge,
        });
    }

    check_adapter(adapter, lora, &planned, base)?;
    Ok(Plan { tensors: planned })
}

fn check_adapter(
    adapter: &SafeTensors,
    lora: &LoraConfig,
    planned: &[Planned],
    base: &SafeTensors,
) -> Result<()> {
    // every tensor of the adapter is one half of a pair that belongs to an adapted weight
    for t in adapter.tensors() {
        let known = base_of(&t.name).is_some_and(|(hf, _)| {
            planned
                .iter()
                .any(|p| p.merge.is_some() && p.entry.hf == hf)
        });
        if !known {
            return Err(refuse(
                ADAPTER,
                format!(
                    "tensor '{}' does not belong to a weight the adapter's target_modules adapt",
                    t.name
                ),
            ));
        }
        if t.dtype != Dtype::F32 {
            return Err(refuse(
                ADAPTER,
                format!(
                    "tensor '{}' has dtype {}, expected F32",
                    t.name,
                    t.dtype.name()
                ),
            ));
        }
    }
    for p in planned {
        let Some(pair) = &p.merge else { continue };
        let w = base
            .get(&p.entry.hf)
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        let (out_dim, in_dim) = (
            w.first().copied().unwrap_or(0),
            w.get(1).copied().unwrap_or(0),
        );
        for (name, half) in [(&pair.a, Half::A), (&pair.b, Half::B)] {
            let Some(t) = adapter.get(name) else {
                return Err(refuse(
                    ADAPTER,
                    format!(
                        "tensor '{name}' is missing: the weight '{}' is adapted according to adapter_config.json",
                        p.entry.hf
                    ),
                ));
            };
            let expected = match half {
                Half::A => vec![lora.r, in_dim],
                Half::B => vec![out_dim, lora.r],
            };
            if t.shape != expected {
                return Err(refuse(
                    ADAPTER,
                    format!(
                        "tensor '{name}' has shape {:?}, expected {expected:?} (r = {}, weight {w:?})",
                        t.shape, lora.r
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// What to write, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The tensors, in the order they are written.
    pub tensors: Vec<Planned>,
}

impl Plan {
    /// Number of tensors that a LoRA adapts.
    pub fn merged(&self) -> usize {
        self.tensors.iter().filter(|p| p.merge.is_some()).count()
    }
}
