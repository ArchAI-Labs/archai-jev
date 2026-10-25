//! A complete GGUF reader and a comparison of two GGUF files, for tests (feature `testing`).
//!
//! The reader of the library only looks at headers; this one also reads every metadata value
//! (arrays included) and hashes the data of each tensor, so a converted file can be compared with
//! the one the official script wrote: tensor by tensor, key by key, ignoring what does not matter
//! (the order of the tensors, descriptive metadata).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Value, json};

use crate::models::gguf::GgmlType;
use crate::models::hash::sha256_hex;

/// A tensor with the digest of its data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullTensor {
    pub dims: Vec<u64>,
    pub ty: String,
    pub sha256: String,
    pub bytes: Vec<u8>,
}

/// A whole GGUF file, parsed.
#[derive(Debug, Clone)]
pub struct FullGguf {
    pub meta: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, FullTensor>,
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8(self.take(n).to_vec()).unwrap()
    }
    fn value(&mut self, ty: u32) -> Value {
        match ty {
            0 => json!(self.take(1)[0]),
            1 => json!(self.take(1)[0] as i8),
            2 => json!(u16::from_le_bytes(self.take(2).try_into().unwrap())),
            3 => json!(i16::from_le_bytes(self.take(2).try_into().unwrap())),
            4 => json!(self.u32()),
            5 => json!(self.u32() as i32),
            6 => json!(f64::from(f32::from_le_bytes(
                self.take(4).try_into().unwrap()
            ))),
            7 => json!(self.take(1)[0] != 0),
            8 => json!(self.string()),
            9 => {
                let elem = self.u32();
                let n = self.u64() as usize;
                Value::Array((0..n).map(|_| self.value(elem)).collect())
            }
            10 => json!(self.u64()),
            11 => json!(self.u64() as i64),
            12 => json!(f64::from_le_bytes(self.take(8).try_into().unwrap())),
            other => panic!("unknown GGUF value type {other}"),
        }
    }
}

/// Parse the GGUF file at `path` completely.
pub fn read_full(path: &Path) -> FullGguf {
    let bytes = std::fs::read(path).unwrap();
    let mut c = Cursor { b: &bytes, at: 0 };
    assert_eq!(c.take(4), b"GGUF");
    assert_eq!(c.u32(), 3);
    let n_tensors = c.u64() as usize;
    let n_kv = c.u64() as usize;
    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string();
        let ty = c.u32();
        let v = c.value(ty);
        assert!(meta.insert(key.clone(), v).is_none(), "duplicate key {key}");
    }
    let alignment = meta
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32) as usize;
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string();
        let nd = c.u32() as usize;
        let dims: Vec<u64> = (0..nd).map(|_| c.u64()).collect();
        let ty = c.u32();
        let offset = c.u64() as usize;
        infos.push((name, dims, GgmlType(ty), offset));
    }
    let data_start = c.at.div_ceil(alignment) * alignment;
    let mut tensors = BTreeMap::new();
    for (name, dims, ty, offset) in infos {
        let elements: u64 = dims.iter().product();
        let size = ty.byte_size(elements).unwrap() as usize;
        let data = &bytes[data_start + offset..data_start + offset + size];
        let t = FullTensor {
            dims,
            ty: ty.name(),
            sha256: sha256_hex(data),
            bytes: data.to_vec(),
        };
        assert!(
            tensors.insert(name.clone(), t).is_none(),
            "duplicate tensor {name}"
        );
    }
    FullGguf { meta, tensors }
}

/// Metadata that describes the file rather than the model: it is not compared.
pub const DESCRIPTIVE: &[&str] = &[
    "general.name",
    "general.type",
    "general.finetune",
    "general.basename",
    "general.size_label",
    "tokenizer.chat_template",
];

/// What differs between two GGUF files, tensor by tensor and key by key (empty when equal).
/// The order of the tensors and the descriptive metadata are ignored.
pub fn compare(a: &FullGguf, b: &FullGguf) -> Vec<String> {
    let mut out = Vec::new();
    for name in a.tensors.keys().chain(b.tensors.keys()) {
        match (a.tensors.get(name), b.tensors.get(name)) {
            (Some(x), Some(y)) => {
                if x.dims != y.dims || x.ty != y.ty {
                    out.push(format!(
                        "tensor {name}: {:?} {} against {:?} {}",
                        x.dims, x.ty, y.dims, y.ty
                    ));
                } else if x.sha256 != y.sha256 {
                    let differing = x.bytes.iter().zip(&y.bytes).filter(|(p, q)| p != q).count();
                    out.push(format!("tensor {name}: data differs in {differing} bytes"));
                }
            }
            (Some(_), None) => out.push(format!("tensor {name}: only in the first file")),
            (None, Some(_)) => out.push(format!("tensor {name}: only in the second file")),
            (None, None) => {}
        }
    }
    out.sort();
    out.dedup();
    let keys = |m: &BTreeMap<String, Value>| {
        m.keys()
            .filter(|k| !DESCRIPTIVE.contains(&k.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    };
    for key in keys(&a.meta).into_iter().chain(keys(&b.meta)) {
        match (a.meta.get(&key), b.meta.get(&key)) {
            (Some(x), Some(y)) if x != y => out.push(format!("metadata {key} differs")),
            (Some(_), None) => out.push(format!("metadata {key}: only in the first file")),
            (None, Some(_)) => out.push(format!("metadata {key}: only in the second file")),
            _ => {}
        }
    }
    out.sort();
    out.dedup();
    out
}

/// How exactly a tensor must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tol {
    /// Every byte.
    Exact,
    /// Every element within this many units in the last place.
    Ulp(u32),
    /// Not compared here: another test checks it by a different measure.
    Skip,
}

fn ordered(bits: u32) -> i64 {
    // map float bit patterns to integers that grow with the value
    if bits & 0x8000_0000 != 0 {
        -i64::from(bits & 0x7FFF_FFFF)
    } else {
        i64::from(bits)
    }
}

/// `(number of elements that differ, largest distance in ulps)` between two tensors of type `ty`.
pub fn ulp_distance(ty: &str, a: &[u8], b: &[u8]) -> (usize, u64) {
    let bits: Vec<(u32, u32)> = match ty {
        "F32" => a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0)
            .map(|(x, y)| (u32::from_le_bytes(*x), u32::from_le_bytes(*y)))
            .collect(),
        "BF16" => a
            .as_chunks::<2>()
            .0
            .iter()
            .zip(b.as_chunks::<2>().0)
            .map(|(x, y)| {
                (
                    u32::from(u16::from_le_bytes(*x)) << 16,
                    u32::from(u16::from_le_bytes(*y)) << 16,
                )
            })
            .collect(),
        other => panic!("ulp distance of {other}"),
    };
    let shift = if ty == "BF16" { 16 } else { 0 };
    let mut differing = 0;
    let mut worst = 0u64;
    for (x, y) in bits {
        if x != y {
            differing += 1;
            let d = (ordered(x) >> shift).abs_diff(ordered(y) >> shift);
            worst = worst.max(d);
        }
    }
    (differing, worst)
}

/// Like [`compare`], but each tensor has its own tolerance (`tol_of(name)`).
pub fn compare_tol(a: &FullGguf, b: &FullGguf, tol_of: &dyn Fn(&str) -> Tol) -> Vec<String> {
    let mut out = Vec::new();
    for name in a.tensors.keys().chain(b.tensors.keys()) {
        match (a.tensors.get(name), b.tensors.get(name)) {
            (Some(x), Some(y)) => {
                if x.dims != y.dims || x.ty != y.ty {
                    out.push(format!(
                        "tensor {name}: {:?} {} against {:?} {}",
                        x.dims, x.ty, y.dims, y.ty
                    ));
                } else if x.sha256 != y.sha256 {
                    let (n, worst) = ulp_distance(&x.ty, &x.bytes, &y.bytes);
                    match tol_of(name) {
                        Tol::Exact => out.push(format!(
                            "tensor {name}: {n} elements differ (up to {worst} ulp), none may"
                        )),
                        Tol::Ulp(max) if worst > u64::from(max) => out.push(format!(
                            "tensor {name}: {n} elements differ by up to {worst} ulp, at most {max} allowed"
                        )),
                        Tol::Ulp(_) | Tol::Skip => {}
                    }
                }
            }
            (Some(_), None) => out.push(format!("tensor {name}: only in the first file")),
            (None, Some(_)) => out.push(format!("tensor {name}: only in the second file")),
            (None, None) => {}
        }
    }
    let rest = compare(
        &FullGguf {
            meta: a.meta.clone(),
            tensors: BTreeMap::new(),
        },
        &FullGguf {
            meta: b.meta.clone(),
            tensors: BTreeMap::new(),
        },
    );
    out.extend(rest);
    out.sort();
    out.dedup();
    out
}

/// The tolerance of each tensor of a Qwen3.5 GGUF made from a base model and a LoRA adapter, when
/// compared with the one the official script wrote (spec 018, AC-11/AC-12): everything is bit for
/// bit **except**
/// - `ssm_a` (PyTorch's `exp` is not correctly rounded): one ulp;
/// - the weights a LoRA adapted. In `bf16` the rounding hides the differences of the merge: one
///   ulp. In `f32` they show: PyTorch multiplies the two small matrices with a kernel that changes
///   with the shape, and where the result is close to zero a difference far below the size of the
///   operands is many ulps *of the result*. Those tensors are `Skip`ped here and checked against
///   the size of their operands instead (`merge_error_in_units`).
pub fn qwen35_lora_tolerance(name: &str, n_layers: u64, f32_file: bool) -> Tol {
    const ADAPTED: &[&str] = &[
        "attn_qkv",
        "attn_gate",
        "ssm_alpha",
        "ssm_beta",
        "ssm_out",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    let mut parts = name.splitn(3, '.');
    let (Some("blk"), Some(index), Some(rest)) = (parts.next(), parts.next(), parts.next()) else {
        return Tol::Exact;
    };
    let Ok(layer) = index.parse::<u64>() else {
        return Tol::Exact;
    };
    if rest == "ssm_a" {
        return Tol::Ulp(1);
    }
    let adapted = layer < n_layers
        && rest
            .strip_suffix(".weight")
            .is_some_and(|m| ADAPTED.contains(&m));
    match (adapted, f32_file) {
        (false, _) => Tol::Exact,
        (true, true) => Tol::Skip,
        (true, false) => Tol::Ulp(1),
    }
}

/// The largest `|a - b| / max(|w|, |2 * delta|)` over the elements of a merged `f32` tensor, in
/// units of `2^-24` (the rounding unit of `f32`): the difference between two merges measured
/// against the size of the numbers that were added, which is what rounding errors scale with.
/// `w` is the base weight, `delta` the product `B @ A` computed in `f64`, `scale` is `alpha / r`.
pub fn merge_error_in_units(w: &[f32], delta: &[f64], scale: f64, a: &[f32], b: &[f32]) -> f64 {
    let mut worst = 0.0f64;
    for (((w, d), x), y) in w.iter().zip(delta).zip(a).zip(b) {
        let size = f64::from(w.abs()).max((scale * d).abs());
        if size > 0.0 {
            let err = (f64::from(*x) - f64::from(*y)).abs();
            worst = worst.max(err / size / 2f64.powi(-24));
        }
    }
    worst
}
