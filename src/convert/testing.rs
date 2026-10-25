//! Test support for the converter (feature `testing`, off in release wheels): writers of the
//! file formats the converter reads, so tests can build tiny checkpoints and break them in
//! exactly one way. These writers are **only** for tests: the library never writes safetensors,
//! zip or pickle.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

use super::numeric::{bf16_to_f32, f32_to_bf16};
use super::safetensors::Dtype;

/// A tensor to put in a safetensors file.
#[derive(Debug, Clone)]
pub struct StEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl StEntry {
    /// An `F32` tensor.
    pub fn f32(name: &str, shape: &[u64], values: &[f32]) -> StEntry {
        assert_eq!(
            shape.iter().product::<u64>() as usize,
            values.len(),
            "{name}"
        );
        StEntry {
            name: name.to_string(),
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// A `BF16` tensor from `f32` values (rounded).
    pub fn bf16(name: &str, shape: &[u64], values: &[f32]) -> StEntry {
        assert_eq!(
            shape.iter().product::<u64>() as usize,
            values.len(),
            "{name}"
        );
        StEntry {
            name: name.to_string(),
            dtype: Dtype::Bf16,
            shape: shape.to_vec(),
            data: values
                .iter()
                .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                .collect(),
        }
    }

    /// The values as `f32` (any dtype).
    pub fn values(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F32 => self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::Bf16 => self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
        }
    }
}

/// The JSON header of a safetensors file for `entries`, laid out back to back.
pub fn safetensors_header(entries: &[StEntry]) -> String {
    let mut parts = Vec::new();
    let mut at = 0u64;
    for e in entries {
        let shape: Vec<String> = e.shape.iter().map(u64::to_string).collect();
        parts.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            serde_json::to_string(&e.name).unwrap(),
            e.dtype.name(),
            shape.join(","),
            at,
            at + e.data.len() as u64
        ));
        at += e.data.len() as u64;
    }
    format!("{{{}}}", parts.join(","))
}

/// A whole safetensors file.
pub fn safetensors_bytes(entries: &[StEntry]) -> Vec<u8> {
    safetensors_with_header(&safetensors_header(entries), entries)
}

/// A safetensors file with a given header text (to write broken ones) and the data of `entries`.
pub fn safetensors_with_header(header: &str, entries: &[StEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for e in entries {
        out.extend_from_slice(&e.data);
    }
    out
}

/// Read every tensor of a safetensors file (to rewrite it with one defect).
pub fn load_entries(path: &std::path::Path) -> Vec<StEntry> {
    let mut st = super::safetensors::SafeTensors::open(path, "test").unwrap();
    let tensors: Vec<_> = st.tensors().to_vec();
    tensors
        .into_iter()
        .map(|t| {
            let mut data = vec![0u8; t.len as usize];
            st.read(&t, 0, &mut data).unwrap();
            StEntry {
                name: t.name,
                dtype: t.dtype,
                shape: t.shape,
                data,
            }
        })
        .collect()
}

/// Write `entries` as a safetensors file at `path`.
pub fn save_entries(path: &std::path::Path, entries: &[StEntry]) {
    std::fs::write(path, safetensors_bytes(entries)).unwrap();
}
