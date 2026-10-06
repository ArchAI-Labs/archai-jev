//! A strict reader of GGUF headers: metadata and tensor descriptors, never the weights.
//!
//! It exists so that a model file is checked by memory-safe Rust *before* llama.cpp (C++) sees
//! it, and so that names, shapes and types can be compared with what the family expects.
//! Every count and size is checked against the real file length with checked arithmetic.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::incompat::Incompat;

/// The only GGUF version this reader accepts.
pub const SUPPORTED_VERSION: u32 = 3;
const MAX_KEY_BYTES: u64 = 65_535;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TENSORS: u64 = 100_000;
const MAX_KEYS: u64 = 100_000;
const MAX_DIMS: u32 = 4;

/// A ggml tensor type id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlType(pub u32);

impl GgmlType {
    /// 32-bit float.
    pub const F32: GgmlType = GgmlType(0);
    /// 16-bit float.
    pub const F16: GgmlType = GgmlType(1);
    /// 8-bit quantisation, blocks of 32.
    pub const Q8_0: GgmlType = GgmlType(8);
    /// bfloat16.
    pub const BF16: GgmlType = GgmlType(30);

    /// `(elements per block, bytes per block, name)` for the types we know the size of.
    fn layout(self) -> Option<(u64, u64, &'static str)> {
        Some(match self.0 {
            0 => (1, 4, "F32"),
            1 => (1, 2, "F16"),
            2 => (32, 18, "Q4_0"),
            3 => (32, 20, "Q4_1"),
            6 => (32, 22, "Q5_0"),
            7 => (32, 24, "Q5_1"),
            8 => (32, 34, "Q8_0"),
            9 => (32, 36, "Q8_1"),
            10 => (256, 84, "Q2_K"),
            11 => (256, 110, "Q3_K"),
            12 => (256, 144, "Q4_K"),
            13 => (256, 176, "Q5_K"),
            14 => (256, 210, "Q6_K"),
            15 => (256, 292, "Q8_K"),
            24 => (1, 1, "I8"),
            25 => (1, 2, "I16"),
            26 => (1, 4, "I32"),
            27 => (1, 8, "I64"),
            28 => (1, 8, "F64"),
            30 => (1, 2, "BF16"),
            _ => return None,
        })
    }

    /// Display name (`Q8_0`, `F32`, ...), or `type#N` for an unknown id.
    pub fn name(self) -> String {
        self.layout()
            .map_or_else(|| format!("type#{}", self.0), |(_, _, n)| n.to_string())
    }

    /// Bytes needed for `elements` values, if the type is known and the count is a whole
    /// number of blocks.
    pub fn byte_size(self, elements: u64) -> Option<u64> {
        let (block_elems, block_bytes, _) = self.layout()?;
        if !elements.is_multiple_of(block_elems) {
            return None;
        }
        (elements / block_elems).checked_mul(block_bytes)
    }
}

/// A metadata value; arrays keep only their element type and length.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    /// Any unsigned integer.
    UInt(u64),
    /// Any signed integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A string.
    Str(String),
    /// An array: element type id and length.
    Array {
        /// GGUF type id of the elements.
        elem_type: u32,
        /// Number of elements.
        len: u64,
    },
}

/// A tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Dimensions, `ne[0]` first.
    pub dims: Vec<u64>,
    /// Element type.
    pub ty: GgmlType,
    /// Offset from the start of the data section.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

/// The parsed header of a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufInfo {
    /// Format version.
    pub version: u32,
    /// Metadata in file order.
    pub metadata: Vec<(String, MetaValue)>,
    /// Tensor descriptors in file order.
    pub tensors: Vec<TensorInfo>,
    /// Absolute offset of the data section.
    pub data_offset: u64,
}

impl GgufInfo {
    /// A metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

struct Cursor {
    r: BufReader<File>,
    pos: u64,
    len: u64,
    path: String,
}

impl Cursor {
    fn unreadable(&self, detail: impl std::fmt::Display) -> Incompat {
        Incompat::GgufUnreadable {
            path: self.path.clone(),
            detail: format!("{detail} (at byte {})", self.pos),
        }
    }

    fn impossible(&self, detail: impl std::fmt::Display) -> Incompat {
        Incompat::GgufImpossible {
            path: self.path.clone(),
            detail: detail.to_string(),
        }
    }

    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], Incompat> {
        if self.remaining() < N as u64 {
            return Err(self.unreadable("the file ends before the header is complete"));
        }
        let mut buf = [0u8; N];
        self.r
            .read_exact(&mut buf)
            .map_err(|e| self.unreadable(format!("read error: {e}")))?;
        self.pos += N as u64;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, Incompat> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    fn u64(&mut self) -> Result<u64, Incompat> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn skip(&mut self, n: u64) -> Result<(), Incompat> {
        if n > self.remaining() {
            return Err(self.unreadable("a length field points past the end of the file"));
        }
        let step = i64::try_from(n).map_err(|_| self.unreadable("length too large"))?;
        self.r
            .seek_relative(step)
            .map_err(|e| self.unreadable(format!("seek error: {e}")))?;
        self.pos += n;
        Ok(())
    }

    fn string(&mut self, max: u64) -> Result<String, Incompat> {
        let n = self.u64()?;
        if n > max || n > self.remaining() {
            return Err(self.impossible(format!(
                "a string of {n} bytes at byte {} is longer than allowed or than the rest of the file",
                self.pos
            )));
        }
        let mut buf =
            vec![0u8; usize::try_from(n).map_err(|_| self.unreadable("string too large"))?];
        self.r
            .read_exact(&mut buf)
            .map_err(|e| self.unreadable(format!("read error: {e}")))?;
        self.pos += n;
        String::from_utf8(buf).map_err(|_| self.unreadable("a string is not valid UTF-8"))
    }
}

fn scalar_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_value(c: &mut Cursor, ty: u32) -> Result<MetaValue, Incompat> {
    Ok(match ty {
        0 => MetaValue::UInt(u64::from(c.take::<1>()?[0])),
        1 => MetaValue::Int(i64::from(i8::from_le_bytes(c.take::<1>()?))),
        2 => MetaValue::UInt(u64::from(u16::from_le_bytes(c.take::<2>()?))),
        3 => MetaValue::Int(i64::from(i16::from_le_bytes(c.take::<2>()?))),
        4 => MetaValue::UInt(u64::from(c.u32()?)),
        5 => MetaValue::Int(i64::from(i32::from_le_bytes(c.take::<4>()?))),
        6 => MetaValue::Float(f64::from(f32::from_le_bytes(c.take::<4>()?))),
        7 => MetaValue::Bool(c.take::<1>()?[0] != 0),
        8 => MetaValue::Str(c.string(MAX_STRING_BYTES)?),
        10 => MetaValue::UInt(c.u64()?),
        11 => MetaValue::Int(i64::from_le_bytes(c.take::<8>()?)),
        12 => MetaValue::Float(f64::from_le_bytes(c.take::<8>()?)),
        9 => {
            let elem_type = c.u32()?;
            let len = c.u64()?;
            if elem_type == 8 {
                if len > c.remaining() / 8 {
                    return Err(
                        c.impossible(format!("an array of {len} strings cannot fit in the file"))
                    );
                }
                for _ in 0..len {
                    let n = c.u64()?;
                    c.skip(n)?;
                }
            } else if let Some(size) = scalar_size(elem_type) {
                let total = len
                    .checked_mul(size)
                    .ok_or_else(|| c.impossible("an array is too large"))?;
                c.skip(total)?;
            } else {
                return Err(c.unreadable(format!("unsupported array element type {elem_type}")));
            }
            MetaValue::Array { elem_type, len }
        }
        other => return Err(c.unreadable(format!("unknown metadata value type {other}"))),
    })
}

/// Read and check the header of the GGUF file at `path`.
///
/// # Errors
/// [`Incompat::GgufUnreadable`] for a file that is not GGUF v3 or is cut short,
/// [`Incompat::GgufImpossible`] for counts, offsets or sizes that cannot be true.
pub fn read_header(path: &Path) -> Result<GgufInfo, Incompat> {
    let shown = path.display().to_string();
    let file = File::open(path).map_err(|e| Incompat::GgufUnreadable {
        path: shown.clone(),
        detail: format!("cannot open: {e}"),
    })?;
    let len = file
        .metadata()
        .map_err(|e| Incompat::GgufUnreadable {
            path: shown.clone(),
            detail: format!("cannot stat: {e}"),
        })?
        .len();
    let mut c = Cursor {
        r: BufReader::with_capacity(1 << 16, file),
        pos: 0,
        len,
        path: shown,
    };

    let magic = c.take::<4>()?;
    if &magic != b"GGUF" {
        return Err(c.unreadable(format!("bad magic {magic:?}, expected \"GGUF\"")));
    }
    let version = c.u32()?;
    if version != SUPPORTED_VERSION {
        return Err(c.unreadable(format!(
            "GGUF version {version} is not supported (supported: {SUPPORTED_VERSION})"
        )));
    }
    let n_tensors = c.u64()?;
    let n_keys = c.u64()?;
    if n_tensors > MAX_TENSORS || n_tensors > c.remaining() / 24 {
        return Err(c.impossible(format!(
            "{n_tensors} tensors cannot be true for a file of {len} bytes"
        )));
    }
    if n_keys > MAX_KEYS || n_keys > c.remaining() / 13 {
        return Err(c.impossible(format!(
            "{n_keys} metadata keys cannot be true for a file of {len} bytes"
        )));
    }

    let mut metadata: Vec<(String, MetaValue)> = Vec::new();
    for _ in 0..n_keys {
        let key = c.string(MAX_KEY_BYTES)?;
        let ty = c.u32()?;
        let value = read_value(&mut c, ty)?;
        if metadata.iter().any(|(k, _)| *k == key) {
            return Err(c.impossible(format!("the metadata key {key:?} appears twice")));
        }
        metadata.push((key, value));
    }

    let mut tensors: Vec<TensorInfo> = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string(1024)?;
        let n_dims = c.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(c.impossible(format!("tensor {name:?} has {n_dims} dimensions")));
        }
        let mut dims = Vec::new();
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let d = c.u64()?;
            if d == 0 {
                return Err(c.impossible(format!("tensor {name:?} has a zero dimension")));
            }
            elements = elements
                .checked_mul(d)
                .ok_or_else(|| c.impossible(format!("tensor {name:?} has too many elements")))?;
            dims.push(d);
        }
        let ty = GgmlType(c.u32()?);
        let offset = c.u64()?;
        let size = if ty.layout().is_none() {
            return Err(c.unreadable(format!(
                "tensor {name:?} has the unsupported type id {}",
                ty.0
            )));
        } else {
            ty.byte_size(elements).ok_or_else(|| {
                c.impossible(format!(
                    "tensor {name:?} has {elements} elements, not a whole number of {} blocks",
                    ty.name()
                ))
            })?
        };
        if tensors.iter().any(|t| t.name == name) {
            return Err(c.impossible(format!("the tensor {name:?} appears twice")));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
            size,
        });
    }

    let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
        Some((_, MetaValue::UInt(a))) if *a > 0 && a.is_power_of_two() => *a,
        Some(_) => return Err(c.impossible("general.alignment is not a power of two")),
        None => 32,
    };
    let data_offset = c
        .pos
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| c.impossible("the data offset overflows"))?;

    let mut spans: Vec<(u64, u64, &str)> = Vec::with_capacity(tensors.len());
    for t in &tensors {
        if t.offset % alignment != 0 {
            return Err(c.impossible(format!(
                "tensor {:?} is not aligned to {alignment} bytes",
                t.name
            )));
        }
        let start = data_offset
            .checked_add(t.offset)
            .ok_or_else(|| c.impossible(format!("tensor {:?} offset overflows", t.name)))?;
        let end = start
            .checked_add(t.size)
            .ok_or_else(|| c.impossible(format!("tensor {:?} size overflows", t.name)))?;
        if end > len {
            return Err(c.impossible(format!(
                "tensor {:?} ends at byte {end}, past the end of the file ({len} bytes)",
                t.name
            )));
        }
        spans.push((start, end, t.name.as_str()));
    }
    spans.sort_unstable();
    for pair in spans.windows(2) {
        if let [(_, end_a, a), (start_b, _, b)] = pair
            && start_b < end_a
        {
            return Err(c.impossible(format!("tensors {a:?} and {b:?} overlap")));
        }
    }

    Ok(GgufInfo {
        version,
        metadata,
        tensors,
        data_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_sizes() {
        assert_eq!(GgmlType::F32.byte_size(10), Some(40));
        assert_eq!(GgmlType::Q8_0.byte_size(64), Some(68));
        assert_eq!(GgmlType::Q8_0.byte_size(33), None);
        assert_eq!(GgmlType::BF16.byte_size(3), Some(6));
        assert_eq!(GgmlType(9999).byte_size(1), None);
        assert_eq!(GgmlType::Q8_0.name(), "Q8_0");
        assert_eq!(GgmlType(9999).name(), "type#9999");
    }

    #[test]
    fn missing_file_is_unreadable() {
        let err = read_header(Path::new("definitely/not/here.gguf")).unwrap_err();
        assert!(matches!(err, Incompat::GgufUnreadable { .. }));
    }
}
