//! A strict reader of `.safetensors` files (spec 018, section 3).
//!
//! The format is a little-endian `u64` `N`, then `N` bytes of JSON (name -> dtype, shape,
//! `[begin, end]` offsets), then the raw tensor bytes. We read the header with the strict JSON
//! reader and then **check everything** before any data is read: offsets must tile the data area
//! exactly (no overlap, no hole, nothing after the end), sizes must equal `shape x dtype`, and
//! only the dtypes the converter knows are accepted. Data is read tensor by tensor, in blocks.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::problem_text;
use crate::error::{Error, Result};
use crate::json_strict::{self, Json, Obj};
use crate::models::failures::DownloadFailure;
use crate::models::incompat::Incompat;

/// Largest header we accept, in bytes.
pub const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_TENSORS: usize = 100_000;
const MAX_DIMS: usize = 8;

/// The element types of the tensors we accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// bfloat16.
    Bf16,
}

impl Dtype {
    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 => 2,
        }
    }

    /// The name used in the file.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::Bf16 => "BF16",
        }
    }
}

/// One tensor of the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    /// Its name.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Dimensions, outermost first (row-major data).
    pub shape: Vec<u64>,
    /// First byte, relative to the start of the data area.
    pub start: u64,
    /// Number of bytes.
    pub len: u64,
}

impl Tensor {
    /// Number of elements.
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// An open, checked `.safetensors` file.
#[derive(Debug)]
pub struct SafeTensors {
    path: PathBuf,
    /// What the messages call the file (its role or name).
    label: String,
    file: File,
    data_start: u64,
    tensors: Vec<Tensor>,
}

fn invalid(label: &str, detail: impl Into<String>) -> Error {
    Error::IncompatibleModel(Incompat::SourceFile {
        file: label.to_string(),
        detail: detail.into(),
    })
}

/// An I/O failure while reading a checkpoint file.
pub fn io_error(path: &Path, cause: &std::io::Error) -> Error {
    Error::ModelDownload(DownloadFailure::Io {
        path: path.display().to_string(),
        cause: cause.to_string(),
    })
}

fn parse_dtype(label: &str, name: &str, text: &str) -> Result<Dtype> {
    match text {
        "F32" => Ok(Dtype::F32),
        "BF16" => Ok(Dtype::Bf16),
        other => Err(invalid(
            label,
            format!("tensor '{name}' has dtype {other}; only F32 and BF16 are supported"),
        )),
    }
}

fn read_tensor(label: &str, name: &str, json: &Json) -> Result<Tensor> {
    let bad = |detail: String| invalid(label, format!("tensor '{name}': {detail}"));
    let mut o = Obj::new(json, name).map_err(|e| bad(problem_text(&e)))?;
    let dtype = parse_dtype(
        label,
        name,
        o.str("dtype").map_err(|e| bad(problem_text(&e)))?,
    )?;
    let shape_json = o.arr("shape").map_err(|e| bad(problem_text(&e)))?;
    if shape_json.len() > MAX_DIMS {
        return Err(bad(format!("has more than {MAX_DIMS} dimensions")));
    }
    let mut shape = Vec::with_capacity(shape_json.len());
    for d in shape_json {
        match d {
            Json::Int(n) if *n > 0 => shape
                .push(u64::try_from(*n).map_err(|_| bad("a dimension is too large".to_string()))?),
            other => {
                return Err(bad(format!(
                    "every dimension must be a positive integer, got {}",
                    other.to_canonical_string()
                )));
            }
        }
    }
    let offsets = o
        .arr("data_offsets")
        .map_err(|e| bad(problem_text(&e)))?
        .iter()
        .map(|v| match v {
            Json::Int(n) => u64::try_from(*n).ok(),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [Some(begin), Some(end)] = offsets.as_slice() else {
        return Err(bad(
            "data_offsets must be two non-negative integers".to_string()
        ));
    };
    o.finish().map_err(|e| bad(problem_text(&e)))?;
    let elements = shape
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(*d))
        .ok_or_else(|| bad("the shape is too large".to_string()))?;
    let expected = elements
        .checked_mul(dtype.size())
        .ok_or_else(|| bad("the size is too large".to_string()))?;
    if end < begin || end - begin != expected {
        return Err(bad(format!(
            "data_offsets [{begin}, {end}] span {} bytes, but shape {shape:?} of {} needs {expected}",
            end.saturating_sub(*begin),
            dtype.name()
        )));
    }
    Ok(Tensor {
        name: name.to_string(),
        dtype,
        shape,
        start: *begin,
        len: expected,
    })
}

impl SafeTensors {
    /// Open `path` and check its header. `label` names the file in messages.
    ///
    /// # Errors
    /// `IncompatibleModelError` (`SourceFile`) for anything wrong with the header or the layout;
    /// an I/O error if the file cannot be read.
    pub fn open(path: &Path, label: &str) -> Result<SafeTensors> {
        let mut file = File::open(path).map_err(|e| io_error(path, &e))?;
        let file_len = file.metadata().map_err(|e| io_error(path, &e))?.len();
        if file_len < 8 {
            return Err(invalid(
                label,
                format!("it is {file_len} bytes long, too short for a header"),
            ));
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|e| io_error(path, &e))?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > MAX_HEADER_BYTES {
            return Err(invalid(
                label,
                format!("the header is {header_len} bytes; the maximum is {MAX_HEADER_BYTES}"),
            ));
        }
        let data_start = 8 + header_len;
        if data_start > file_len {
            return Err(invalid(
                label,
                format!("the header says {header_len} bytes but the file has only {file_len}"),
            ));
        }
        let mut header = vec![0u8; usize::try_from(header_len).unwrap_or(0)];
        file.read_exact(&mut header)
            .map_err(|e| io_error(path, &e))?;
        let json = json_strict::parse(&header)
            .map_err(|e| invalid(label, format!("the header is not valid JSON: {e}")))?;
        let Json::Object(pairs) = &json else {
            return Err(invalid(label, "the header is not a JSON object"));
        };
        let mut tensors = Vec::with_capacity(pairs.len());
        for (name, value) in pairs {
            if name == "__metadata__" {
                if !matches!(value, Json::Object(_)) {
                    return Err(invalid(label, "__metadata__ is not an object"));
                }
                continue;
            }
            if tensors.len() >= MAX_TENSORS {
                return Err(invalid(
                    label,
                    format!("more than {MAX_TENSORS} tensors in the header"),
                ));
            }
            tensors.push(read_tensor(label, name, value)?);
        }
        let data_len = file_len - data_start;
        check_layout(label, &tensors, data_len)?;
        Ok(SafeTensors {
            path: path.to_path_buf(),
            label: label.to_string(),
            file,
            data_start,
            tensors,
        })
    }

    /// The tensors, in the order of the header.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// The tensor called `name`.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The name of the file in messages.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Read `buf.len()` bytes of `tensor`, starting `offset` bytes into it.
    ///
    /// # Errors
    /// `IncompatibleModelError` if the range leaves the tensor; an I/O error if the read fails
    /// (including a file that shrank since it was checked).
    pub fn read(&mut self, tensor: &Tensor, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or_else(|| invalid(&self.label, "a read range overflows"))?;
        if end > tensor.len {
            return Err(invalid(
                &self.label,
                format!(
                    "tensor '{}': reading bytes {offset}..{end} of {}",
                    tensor.name, tensor.len
                ),
            ));
        }
        let position = self.data_start + tensor.start + offset;
        self.file
            .seek(SeekFrom::Start(position))
            .and_then(|_| self.file.read_exact(buf))
            .map_err(|e| io_error(&self.path, &e))
    }
}

/// The tensors must tile the data area exactly, in some order.
fn check_layout(label: &str, tensors: &[Tensor], data_len: u64) -> Result<()> {
    let mut order: Vec<&Tensor> = tensors.iter().collect();
    order.sort_by_key(|t| t.start);
    let mut next = 0u64;
    for t in order {
        if t.start < next {
            return Err(invalid(
                label,
                format!(
                    "tensor '{}' starts at byte {} but the previous one ends at {next}: overlapping data",
                    t.name, t.start
                ),
            ));
        }
        if t.start > next {
            return Err(invalid(
                label,
                format!(
                    "there is a hole of {} bytes before tensor '{}' (at byte {next})",
                    t.start - next,
                    t.name
                ),
            ));
        }
        next = t
            .start
            .checked_add(t.len)
            .ok_or_else(|| invalid(label, "an offset overflows"))?;
    }
    if next > data_len {
        return Err(invalid(
            label,
            format!(
                "the tensors end at byte {next} but the file has only {data_len} bytes of data"
            ),
        ));
    }
    if next < data_len {
        return Err(invalid(
            label,
            format!(
                "{} unexplained bytes after the last tensor (data ends at {next}, file at {data_len})",
                data_len - next
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::testing::{
        StEntry, safetensors_bytes, safetensors_header, safetensors_with_header,
    };

    fn open_bytes(bytes: &[u8]) -> Result<SafeTensors> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        std::fs::write(&path, bytes).unwrap();
        SafeTensors::open(&path, "base weights")
    }

    fn reason(r: Result<SafeTensors>) -> String {
        match r {
            Err(Error::IncompatibleModel(Incompat::SourceFile { file, detail })) => {
                assert_eq!(file, "base weights");
                detail
            }
            Err(other) => panic!("not a source-file error: {other}"),
            Ok(_) => panic!("the file was accepted"),
        }
    }

    fn two() -> Vec<StEntry> {
        vec![
            StEntry::bf16("a.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            StEntry::f32("b", &[2], &[0.5, -0.25]),
        ]
    }

    #[test]
    fn a_valid_file_is_read_back_exactly() {
        let entries = two();
        let mut st = open_bytes(&safetensors_bytes(&entries)).unwrap();
        assert_eq!(st.tensors().len(), 2);
        let a = st.get("a.weight").unwrap().clone();
        assert_eq!(
            (a.dtype, a.shape.clone(), a.len, a.elements()),
            (Dtype::Bf16, vec![2, 3], 12, 6)
        );
        let mut buf = vec![0u8; 12];
        st.read(&a, 0, &mut buf).unwrap();
        assert_eq!(buf, entries[0].data);
        let b = st.get("b").unwrap().clone();
        let mut part = [0u8; 4];
        st.read(&b, 4, &mut part).unwrap();
        assert_eq!(f32::from_le_bytes(part), -0.25);
        assert!(st.get("nope").is_none());
        assert_eq!(st.label(), "base weights");
    }

    #[test]
    fn reading_outside_a_tensor_is_refused() {
        let mut st = open_bytes(&safetensors_bytes(&two())).unwrap();
        let b = st.get("b").unwrap().clone();
        let mut buf = [0u8; 4];
        assert!(matches!(
            st.read(&b, 5, &mut buf),
            Err(Error::IncompatibleModel(_))
        ));
        assert!(matches!(
            st.read(&b, u64::MAX, &mut buf),
            Err(Error::IncompatibleModel(_))
        ));
    }

    #[test]
    fn metadata_is_allowed_and_ignored() {
        let entries = two();
        let header = safetensors_header(&entries);
        let with_meta = format!("{{\"__metadata__\":{{\"format\":\"pt\"}},{}", &header[1..]);
        assert_eq!(
            open_bytes(&safetensors_with_header(&with_meta, &entries))
                .unwrap()
                .tensors()
                .len(),
            2
        );
    }

    #[test]
    fn broken_headers_are_each_refused_with_the_reason() {
        let entries = two();
        let good = safetensors_header(&entries);
        let data = |h: &str| safetensors_with_header(h, &entries);

        // too short, header longer than the file, header over the limit
        assert!(reason(open_bytes(b"abc")).contains("too short"));
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(reason(open_bytes(&huge)).contains("the maximum"));
        let mut long = 1000u64.to_le_bytes().to_vec();
        long.extend_from_slice(b"{}");
        assert!(reason(open_bytes(&long)).contains("only"));

        // JSON problems
        assert!(reason(open_bytes(&data("{not json"))).contains("not valid JSON"));
        assert!(reason(open_bytes(&data("[1]"))).contains("not a JSON object"));
        let dup = good.replacen("\"b\"", "\"a.weight\"", 1);
        assert!(reason(open_bytes(&data(&dup))).contains("more than once"));
        assert!(
            reason(open_bytes(&data(&format!(
                "{{\"__metadata__\":3,{}",
                &good[1..]
            ))))
            .contains("__metadata__")
        );

        // dtype
        let f16 = good.replace("\"F32\"", "\"F16\"");
        let r = reason(open_bytes(&data(&f16)));
        assert!(r.contains("'b'") && r.contains("F16"), "{r}");
        // shape and offsets
        assert!(reason(open_bytes(&data(&good.replace("[2,3]", "[2,0]")))).contains("positive"));
        assert!(reason(open_bytes(&data(&good.replace("[2,3]", "[3,3]")))).contains("needs"));
        assert!(
            reason(open_bytes(&data(
                &good.replace("[2,3]", "[18446744073709551615,2]")
            )))
            .contains("too large")
        );
        assert!(
            reason(open_bytes(&data(
                &good.replace("\"shape\":[2]", "\"shape\":[2],\"extra\":1")
            )))
            .contains("unknown field")
        );
        assert!(
            reason(open_bytes(&data(&good.replace("[0,12]", "[\"0\",12]"))))
                .contains("data_offsets")
        );
        assert!(reason(open_bytes(&data(&good.replace("[0,12]", "[12,0]")))).contains("span"));
    }

    #[test]
    fn layout_problems_are_each_refused() {
        let entries = two();
        let good = safetensors_header(&entries);
        // overlap: b starts inside a
        let overlap = good.replace("[12,20]", "[8,16]");
        assert!(
            reason(open_bytes(&safetensors_with_header(&overlap, &entries)))
                .contains("overlapping")
        );
        // hole: b starts after a gap
        let hole = good.replace("[12,20]", "[16,24]");
        assert!(reason(open_bytes(&safetensors_with_header(&hole, &entries))).contains("hole"));
        // data shorter than the header says
        let mut short = safetensors_bytes(&entries);
        short.truncate(short.len() - 1);
        assert!(reason(open_bytes(&short)).contains("only"));
        // bytes after the end
        let mut long = safetensors_bytes(&entries);
        long.push(0);
        assert!(reason(open_bytes(&long)).contains("unexplained"));
    }

    #[test]
    fn a_missing_file_is_an_io_error_not_a_panic() {
        let r = SafeTensors::open(Path::new("/nonexistent/x.safetensors"), "base weights");
        assert!(matches!(
            r,
            Err(Error::ModelDownload(DownloadFailure::Io { .. }))
        ));
    }
}
