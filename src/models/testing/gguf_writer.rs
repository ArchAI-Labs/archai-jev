//! Writes GGUF v3 files with a real header and zero-filled tensor data. Not loadable by
//! llama.cpp (the numbers are zeros): it serves the header checks, which are what spec 005 tests.

use crate::models::gguf::GgmlType;

/// A metadata value to write.
#[derive(Debug, Clone)]
pub enum Meta {
    U32(u32),
    U64(u64),
    Str(String),
    StrArray(Vec<String>),
}

/// A tensor to write: descriptor only, data is zeros.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_meta(out: &mut Vec<u8>, v: &Meta) {
    match v {
        Meta::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Meta::U64(x) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Meta::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, s);
        }
        Meta::StrArray(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                put_str(out, s);
            }
        }
    }
}

/// Serialise a GGUF file.
pub fn write(metadata: &[(String, Meta)], tensors: &[TensorDesc], alignment: u64) -> Vec<u8> {
    write_with_offsets(metadata, tensors, alignment, None)
}

/// Like [`write`], optionally forcing the offsets written in the tensor descriptors.
pub fn write_with_offsets(
    metadata: &[(String, Meta)],
    tensors: &[TensorDesc],
    alignment: u64,
    forced: Option<&[u64]>,
) -> Vec<u8> {
    let sizes: Vec<u64> = tensors
        .iter()
        .map(|t| {
            let elements: u64 = t.dims.iter().product();
            GgmlType(t.ty)
                .byte_size(elements)
                .unwrap_or_else(|| panic!("bad tensor type or size: {t:?}"))
        })
        .collect();
    let mut offsets = Vec::new();
    let mut next = 0u64;
    for s in &sizes {
        offsets.push(next);
        next = (next + s).div_ceil(alignment) * alignment;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        put_str(&mut out, k);
        put_meta(&mut out, v);
    }
    let offsets = forced.map_or(offsets.clone(), <[u64]>::to_vec);
    for (t, off) in tensors.iter().zip(&offsets) {
        put_str(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ty.to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
    }
    let data_start = (out.len() as u64).div_ceil(alignment) * alignment;
    out.resize(data_start as usize, 0);
    out.resize((data_start + next) as usize, 0);
    out
}
