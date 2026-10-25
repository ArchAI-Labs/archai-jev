//! Reading `head.pt` without running it (spec 018, section 4; decisions D26 and D35).
//!
//! `head.pt` is a file written by `torch.save`: an uncompressed zip with a pickle that describes
//! a dictionary (the four tensors of the pointer head, plus metadata) and one raw data entry per
//! tensor. [`HeadPtReader`] checks the zip ([`zip`]), interprets the pickle symbolically
//! ([`pickle`]) and reads the tensors as little-endian `f32`. It never calls any function a
//! file names.

pub mod pickle;
pub mod zip;

use std::fs::File;
use std::path::Path;

use self::pickle::{Val, interpret};
use crate::models::head::{HeadReader, HeadTensors};
use crate::models::incompat::Incompat;

const MAX_TENSORS: usize = 16;
const MAX_DATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIMS: usize = 4;

/// The reader of `head.pt` files.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadPtReader;

fn bad(detail: impl Into<String>) -> Incompat {
    Incompat::HeadWeights {
        detail: detail.into(),
    }
}

fn number(v: &Val) -> Option<f64> {
    match v {
        Val::Float(f) => Some(*f),
        Val::Int(n) => Some(*n as f64),
        _ => None,
    }
}

fn contiguous(shape: &[u64]) -> Option<Vec<u64>> {
    let mut stride = Vec::with_capacity(shape.len());
    let mut acc = 1u64;
    for d in shape.iter().rev() {
        stride.push(acc);
        acc = acc.checked_mul(*d)?;
    }
    stride.reverse();
    Some(stride)
}

impl HeadReader for HeadPtReader {
    fn read(&self, path: &Path) -> Result<HeadTensors, Incompat> {
        let shown = path.display();
        let mut file = File::open(path).map_err(|e| bad(format!("cannot open '{shown}': {e}")))?;
        let len = file
            .metadata()
            .map_err(|e| bad(format!("cannot read '{shown}': {e}")))?
            .len();
        let entries = zip::read_entries(&mut file, len)
            .map_err(|e| bad(format!("'{shown}' is not an acceptable torch file: {e}")))?;

        let pkl: Vec<&zip::Entry> = entries
            .iter()
            .filter(|e| e.name == "data.pkl" || e.name.ends_with("/data.pkl"))
            .collect();
        let [pkl] = pkl.as_slice() else {
            return Err(bad(format!(
                "'{shown}' must have exactly one data.pkl entry, it has {}",
                pkl.len()
            )));
        };
        let prefix = pkl.name.strip_suffix("data.pkl").unwrap_or("");
        if prefix.matches('/').count() > 1 {
            return Err(bad(format!(
                "the entry name '{}' is nested too deep",
                pkl.name
            )));
        }
        if pkl.len > pickle::MAX_PICKLE_BYTES as u64 {
            return Err(bad(format!(
                "the pickle is {} bytes; the maximum is {}",
                pkl.len,
                pickle::MAX_PICKLE_BYTES
            )));
        }
        let bytes = zip::read_entry(&mut file, pkl).map_err(bad)?;
        let value = interpret(&bytes).map_err(|e| bad(format!("the pickle of '{shown}': {e}")))?;
        let Some(Val::Dict(head)) = value.get("head") else {
            return Err(bad("the file has no 'head' dictionary of tensors"));
        };
        if head.len() > MAX_TENSORS {
            return Err(bad(format!(
                "the head has {} tensors; the maximum is {MAX_TENSORS}",
                head.len()
            )));
        }

        let mut tensors = Vec::with_capacity(head.len());
        let mut used_keys: Vec<&str> = Vec::new();
        let mut total = 0u64;
        for (name, tensor) in head {
            let Some(name) = name.as_str() else {
                return Err(bad("a tensor name is not a string"));
            };
            let Val::Tensor {
                key,
                numel,
                offset,
                shape,
                stride,
            } = tensor
            else {
                return Err(bad(format!("'{name}' is not a tensor")));
            };
            if *offset != 0 {
                return Err(bad(format!(
                    "tensor '{name}' starts at offset {offset} of its storage; shared storages are not accepted"
                )));
            }
            if shape.is_empty() || shape.len() > MAX_DIMS || shape.contains(&0) {
                return Err(bad(format!(
                    "tensor '{name}' has shape {shape:?}; 1 to {MAX_DIMS} non-empty dimensions are expected"
                )));
            }
            let elements = shape
                .iter()
                .try_fold(1u64, |a, d| a.checked_mul(*d))
                .ok_or_else(|| bad(format!("tensor '{name}' is too large")))?;
            if elements != *numel {
                return Err(bad(format!(
                    "tensor '{name}' has {elements} elements but its storage has {numel}"
                )));
            }
            if contiguous(shape).as_ref() != Some(stride) {
                return Err(bad(format!(
                    "tensor '{name}' has strides {stride:?}, which are not contiguous (row-major) for shape {shape:?}"
                )));
            }
            if used_keys.contains(&key.as_str()) {
                return Err(bad(format!("two tensors share the storage '{key}'")));
            }
            used_keys.push(key);
            let bytes_len = elements
                .checked_mul(4)
                .ok_or_else(|| bad(format!("tensor '{name}' is too large")))?;
            total = total.saturating_add(bytes_len);
            if total > MAX_DATA_BYTES {
                return Err(bad(format!(
                    "the head has more than {MAX_DATA_BYTES} bytes of data"
                )));
            }
            let entry_name = format!("{prefix}data/{key}");
            let Some(entry) = entries.iter().find(|e| e.name == entry_name) else {
                return Err(bad(format!(
                    "tensor '{name}' needs the entry '{entry_name}', which is not in the archive"
                )));
            };
            if entry.len != bytes_len {
                return Err(bad(format!(
                    "the entry '{entry_name}' has {} bytes, tensor '{name}' needs {bytes_len} (f32)",
                    entry.len
                )));
            }
            let raw = zip::read_entry(&mut file, entry).map_err(bad)?;
            let values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                return Err(bad(format!(
                    "tensor '{name}' has a non-finite value at element {i}"
                )));
            }
            tensors.push((name.to_string(), shape.clone(), values));
        }

        let temperature = match value.get("temperature") {
            None => None,
            Some(v) => Some(
                number(v)
                    .filter(|t| t.is_finite())
                    .ok_or_else(|| bad("'temperature' is not a finite number"))?,
            ),
        };
        let head_dim = match value.get("head_dim") {
            None => None,
            Some(Val::Int(n)) => {
                Some(u64::try_from(*n).map_err(|_| bad("'head_dim' is not a positive integer"))?)
            }
            Some(_) => return Err(bad("'head_dim' is not an integer")),
        };
        let text = |key: &str| match value.get(key) {
            None => Ok(None),
            Some(Val::Str(s)) => Ok(Some(s.clone())),
            Some(_) => Err(bad(format!("'{key}' is not a string"))),
        };
        let d_model = tensors
            .iter()
            .find(|(n, _, _)| n == "q.weight")
            .and_then(|(_, shape, _)| shape.get(1).copied());
        Ok(HeadTensors {
            tensors,
            temperature,
            d_model,
            head_dim,
            base: text("base")?,
            base_revision: text("base_revision")?,
        })
    }
}

#[cfg(test)]
mod tests;
