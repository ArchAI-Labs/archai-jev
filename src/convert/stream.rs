//! Writing one tensor of the GGUF, in blocks (spec 018, 3.2).
//!
//! The numbers are read from the checkpoint a block at a time, transformed in `f32` (the merge of
//! the LoRA, `+1`, `-exp`), checked and written in the type of the variant. Nothing larger than a
//! block is ever held, so the embedding (one GiB as `f32`) costs the same memory as a norm.

use super::gguf_out::Out;
use super::layout::Transform;
use super::lora::LoraConfig;
use super::numeric::{bf16_to_f32, f32_to_bf16, merge_row, neg_exp};
use super::plan::Planned;
use super::safetensors::{Dtype, SafeTensors, Tensor};
use crate::error::{Error, Result};
use crate::hub::events::Cancel;
use crate::models::gguf::GgmlType;
use crate::models::incompat::Incompat;

/// Elements per block (4 MiB as `f32`).
const BLOCK: u64 = 1 << 20;

/// A deliberate mistake in the conversion, for tests that prove the checks would notice it
/// (spec 018, AC-3 and AC-16). Only reachable from test builds; release builds never mutate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutant {
    /// Convert correctly.
    None,
    /// Leave out the `+ 1` of the norms.
    NoPlusOne,
    /// Leave out the `-exp` of `A_log`.
    NoNegExp,
    /// Merge with a scale of 1 instead of `alpha / r`.
    ScaleOne,
    /// Ignore the adapter.
    NoMerge,
    /// Write `conv1d` with its two dimensions mixed up.
    ConvTransposed,
}

#[cfg(any(test, feature = "testing"))]
thread_local! {
    static MUTANT: std::cell::Cell<Mutant> = const { std::cell::Cell::new(Mutant::None) };
}

/// Make the conversions of **this thread** go wrong in the given way (test builds only).
#[cfg(any(test, feature = "testing"))]
pub fn set_mutant(m: Mutant) {
    MUTANT.with(|c| c.set(m));
}

#[cfg(any(test, feature = "testing"))]
fn mutated(m: Mutant) -> bool {
    MUTANT.with(std::cell::Cell::get) == m
}

#[cfg(not(any(test, feature = "testing")))]
const fn mutated(_: Mutant) -> bool {
    false
}

#[cfg(any(test, feature = "testing"))]
mod mutants {
    #![allow(clippy::indexing_slicing)]

    /// `[channels][kernel]` read as `[kernel][channels]`.
    pub fn transpose(values: &mut [f32], kernel: usize) {
        let channels = values.len() / kernel.max(1);
        let src = values.to_vec();
        for c in 0..channels {
            for k in 0..kernel {
                values[k * channels + c] = src[c * kernel + k];
            }
        }
    }
}

fn refuse(file: &str, detail: String) -> Error {
    Error::IncompatibleModel(Incompat::SourceFile {
        file: file.to_string(),
        detail,
    })
}

fn decode(dtype: Dtype, raw: &[u8], out: &mut Vec<f32>) {
    out.clear();
    match dtype {
        Dtype::F32 => out.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c)),
        ),
        Dtype::Bf16 => out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c))),
        ),
    }
}

fn check_finite(file: &str, values: &[f32], what: &str, tensor: &str, first: u64) -> Result<()> {
    match values.iter().position(|v| !v.is_finite()) {
        None => Ok(()),
        Some(i) => Err(refuse(
            file,
            format!(
                "tensor '{tensor}' has a non-finite {what} at element {} ({})",
                first + i as u64,
                values.get(i).copied().unwrap_or(f32::NAN)
            ),
        )),
    }
}

fn encode(ty: GgmlType, values: &[f32], out: &mut Vec<u8>) {
    out.clear();
    if ty == GgmlType::BF16 {
        out.extend(values.iter().flat_map(|v| f32_to_bf16(*v).to_le_bytes()));
    } else {
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }
}

/// Buffers reused from tensor to tensor.
#[derive(Default)]
pub struct Scratch {
    raw: Vec<u8>,
    values: Vec<f32>,
    delta: Vec<f32>,
    merged: Vec<f32>,
    bytes: Vec<u8>,
}

/// A size or an index of the file as a `usize`; one that does not fit is refused, never read as 0.
fn index(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        refuse(
            "weights",
            format!("a size of {value} does not fit in the memory of this machine"),
        )
    })
}

fn read_f32(st: &mut SafeTensors, t: &Tensor) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; index(t.len)?];
    st.read(t, 0, &mut raw)?;
    let mut out = Vec::new();
    decode(t.dtype, &raw, &mut out);
    check_finite("adapter weights", &out, "value", &t.name, 0)?;
    Ok(out)
}

/// Write the data of `planned` to `out`.
///
/// # Errors
/// `IncompatibleModelError` for a non-finite number (in the checkpoint or produced by the
/// transformation), an I/O error, or `Cancelled` when `cancel` says so between blocks.
pub fn write_tensor(
    planned: &Planned,
    base: &mut SafeTensors,
    adapter: &mut SafeTensors,
    lora: &LoraConfig,
    out: &mut Out,
    cancel: &dyn Cancel,
    scratch: &mut Scratch,
) -> Result<()> {
    let start = out.position();
    let Some(t) = base.get(&planned.entry.hf).cloned() else {
        return Err(refuse(
            "base weights",
            format!("tensor '{}' is missing", planned.entry.hf),
        ));
    };
    let size = t.dtype.size();
    let total = t.elements();
    let name = &planned.entry.hf;

    if let Some(pair) = planned.merge.as_ref().filter(|_| !mutated(Mutant::NoMerge)) {
        let (Some(a), Some(b)) = (adapter.get(&pair.a).cloned(), adapter.get(&pair.b).cloned())
        else {
            return Err(refuse(
                "adapter weights",
                format!("the pair of '{name}' is missing"),
            ));
        };
        let a_vals = read_f32(adapter, &a)?;
        let b_vals = read_f32(adapter, &b)?;
        let in_dim = t.shape.get(1).copied().unwrap_or(1);
        let rows_per_block = (BLOCK / in_dim.max(1)).max(1);
        let Some(rows) = t.shape.first().copied() else {
            return Err(refuse(
                "base weights",
                format!("tensor '{name}' has no shape"),
            ));
        };
        let n_in = index(in_dim)?;
        let r = index(lora.r)?;
        let mut row = 0u64;
        while row < rows {
            if cancel.cancelled() {
                return Err(Error::Cancelled);
            }
            let count = rows_per_block.min(rows - row);
            let n = index(count * in_dim)?;
            scratch.raw.resize(n * size as usize, 0);
            base.read(&t, row * in_dim * size, &mut scratch.raw)?;
            decode(t.dtype, &scratch.raw, &mut scratch.values);
            check_finite("base weights", &scratch.values, "value", name, row * in_dim)?;
            scratch.merged.clear();
            scratch.delta.resize(n_in, 0.0);
            let mut merged_row = vec![0.0f32; n_in];
            for (i, w_row) in scratch.values.chunks_exact(n_in.max(1)).enumerate() {
                let global = index(row)? + i;
                let Some(b_row) = b_vals.get(global * r..(global + 1) * r) else {
                    return Err(refuse(
                        "adapter weights",
                        format!("the B matrix of '{name}' has no row {global}"),
                    ));
                };
                merge_row(
                    w_row,
                    b_row,
                    &a_vals,
                    if mutated(Mutant::ScaleOne) {
                        1.0
                    } else {
                        lora.scale()
                    },
                    &mut scratch.delta,
                    &mut merged_row,
                );
                scratch.merged.extend_from_slice(&merged_row);
            }
            check_finite(
                "base weights",
                &scratch.merged,
                "merged value",
                name,
                row * in_dim,
            )?;
            encode(planned.ty, &scratch.merged, &mut scratch.bytes);
            out.write(&scratch.bytes)?;
            row += count;
        }
    } else {
        let mut first = 0u64;
        while first < total {
            if cancel.cancelled() {
                return Err(Error::Cancelled);
            }
            let count = BLOCK.min(total - first);
            let n = index(count)?;
            scratch.raw.resize(n * size as usize, 0);
            base.read(&t, first * size, &mut scratch.raw)?;
            decode(t.dtype, &scratch.raw, &mut scratch.values);
            check_finite("base weights", &scratch.values, "value", name, first)?;
            match planned.entry.transform {
                Transform::Copy => {}
                Transform::Squeeze =>
                {
                    #[cfg(any(test, feature = "testing"))]
                    if mutated(Mutant::ConvTransposed) {
                        let kernel = planned.entry.dims.first().copied().unwrap_or(1);
                        mutants::transpose(
                            &mut scratch.values,
                            usize::try_from(kernel).unwrap_or(1),
                        );
                    }
                }
                Transform::PlusOne if mutated(Mutant::NoPlusOne) => {}
                Transform::NegExp if mutated(Mutant::NoNegExp) => {}
                Transform::PlusOne => scratch.values.iter_mut().for_each(|v| *v += 1.0),
                Transform::NegExp => scratch.values.iter_mut().for_each(|v| *v = neg_exp(*v)),
            }
            if planned.entry.transform != Transform::Copy
                && planned.entry.transform != Transform::Squeeze
            {
                check_finite("base weights", &scratch.values, "result", name, first)?;
            }
            encode(planned.ty, &scratch.values, &mut scratch.bytes);
            out.write(&scratch.bytes)?;
            first += count;
        }
    }
    if out.position() != start + planned.bytes {
        return Err(Error::IncompatibleModel(Incompat::ConversionFailed {
            detail: format!(
                "tensor '{}' wrote {} bytes, the plan said {}",
                planned.entry.gguf,
                out.position() - start,
                planned.bytes
            ),
        }));
    }
    Ok(())
}
