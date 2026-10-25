//! A streaming writer of GGUF v3 files (spec 018, 3.2).
//!
//! The header (metadata and tensor descriptors) is computed first from the plan, so every offset
//! is known before any data is written; the data then follows tensor by tensor, each at an
//! aligned offset, and a SHA-256 of the whole file is computed while writing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::safetensors::io_error;
use crate::error::Result;
use crate::models::gguf::GgmlType;
use crate::models::hash::Hasher;

/// Alignment of the data of every tensor (the GGUF default).
pub const ALIGNMENT: u64 = 32;

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `uint32`
    U32(u32),
    /// `float32`
    F32(f32),
    /// `bool`
    Bool(bool),
    /// A string.
    Str(String),
    /// An array of `int32`.
    I32s(Vec<i32>),
    /// An array of `bool`.
    Bools(Vec<bool>),
    /// An array of strings.
    Strs(Vec<String>),
}

/// A tensor descriptor of the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    /// Name.
    pub name: String,
    /// Dimensions, `ne[0]` first.
    pub dims: Vec<u64>,
    /// Type.
    pub ty: GgmlType,
    /// Bytes of data.
    pub bytes: u64,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::Bool(x) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*x));
        }
        Value::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, s);
        }
        Value::I32s(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Value::Bools(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&7u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            out.extend(items.iter().map(|b| u8::from(*b)));
        }
        Value::Strs(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                put_str(out, s);
            }
        }
    }
}

fn align(n: u64) -> u64 {
    n.div_ceil(ALIGNMENT) * ALIGNMENT
}

/// The header of the file, the offset of each tensor (from the start of the data area) and the
/// length of the whole file.
pub struct Header {
    /// Header bytes, padded to the alignment: the data area starts right after.
    pub bytes: Vec<u8>,
    /// Offset of each tensor from the start of the data area, in tensor order.
    pub offsets: Vec<u64>,
    /// Length of the file once all data is written.
    pub total_len: u64,
}

/// Compute the header for `metadata` and `tensors`.
pub fn header(metadata: &[(String, Value)], tensors: &[TensorDesc]) -> Header {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        put_str(&mut out, k);
        put_value(&mut out, v);
    }
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut next = 0u64;
    for t in tensors {
        offsets.push(next);
        put_str(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ty.0.to_le_bytes());
        out.extend_from_slice(&next.to_le_bytes());
        next = align(next + t.bytes);
    }
    let data_start = align(out.len() as u64);
    out.resize(usize::try_from(data_start).unwrap_or(out.len()), 0);
    let last_end = tensors
        .iter()
        .zip(&offsets)
        .next_back()
        .map_or(0, |(t, o)| o + t.bytes);
    Header {
        bytes: out,
        offsets,
        total_len: data_start + last_end,
    }
}

/// The file being written: counts bytes and hashes them as they go.
pub struct Out {
    path: PathBuf,
    w: BufWriter<File>,
    hasher: Hasher,
    written: u64,
}

impl Out {
    /// Create (or truncate) the file at `path`.
    ///
    /// # Errors
    /// An I/O error.
    pub fn create(path: &Path) -> Result<Out> {
        let file = File::create(path).map_err(|e| io_error(path, &e))?;
        Ok(Out {
            path: path.to_path_buf(),
            w: BufWriter::with_capacity(1 << 20, file),
            hasher: Hasher::new(),
            written: 0,
        })
    }

    /// Append `bytes`.
    ///
    /// # Errors
    /// An I/O error (a full disk included).
    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.w
            .write_all(bytes)
            .map_err(|e| io_error(&self.path, &e))?;
        self.hasher.update(bytes);
        self.written += bytes.len() as u64;
        Ok(())
    }

    /// Write zeros until the file is `position` bytes long.
    ///
    /// # Errors
    /// An I/O error.
    pub fn pad_to(&mut self, position: u64) -> Result<()> {
        const ZEROS: [u8; 4096] = [0; 4096];
        while self.written < position {
            let n = usize::try_from((position - self.written).min(4096)).unwrap_or(4096);
            self.write(ZEROS.get(..n).unwrap_or(&ZEROS))?;
        }
        Ok(())
    }

    /// Bytes written so far.
    pub fn position(&self) -> u64 {
        self.written
    }

    /// Flush and return the SHA-256 and the length of the file.
    ///
    /// # Errors
    /// An I/O error.
    pub fn finish(mut self) -> Result<(String, u64)> {
        self.finish_ref()
    }

    /// Like [`Out::finish`], from a mutable borrow (the writer is not used afterwards).
    ///
    /// # Errors
    /// An I/O error.
    pub fn finish_ref(&mut self) -> Result<(String, u64)> {
        self.w.flush().map_err(|e| io_error(&self.path, &e))?;
        let hasher = std::mem::take(&mut self.hasher);
        Ok((hasher.finish_hex(), self.written))
    }
}
