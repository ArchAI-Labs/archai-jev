//! The converter against the official script, on the tiny checkpoint of `tests/data/qwen35-mini`
//! (level T0: no assets, every OS). The expected files are what `convert_hf_to_gguf.py` wrote
//! for the same inputs (see `spec/playground/s7-convert/make_mini.py`).

use std::path::PathBuf;

use super::layout::Dims;
use super::qwen35::{MODEL_FILE, RECIPE, convert_files};
use super::safetensors::{Dtype, SafeTensors};
use super::stream::{Mutant, set_mutant};
use super::testing_gguf::{
    Tol, compare, compare_tol, merge_error_in_units, qwen35_lora_tolerance, read_full,
};
use crate::hub::events::{Event, NeverCancel, NoObserver, Recorder};
use crate::models::convert::{Converter, SourceFiles};
use crate::models::families::{self, ArchParams, ParamValue, Role};
use crate::models::gguf::{self, GgmlType};
use crate::models::hash::{sha256_file, sha256_hex};

pub(crate) fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("qwen35-mini")
}

pub(crate) fn files() -> SourceFiles {
    let d = fixture();
    SourceFiles {
        base_config: d.join("base/config.json"),
        base_weights: d.join("base/model.safetensors"),
        base_tokenizer: d.join("base/tokenizer.json"),
        base_tokenizer_config: d.join("base/tokenizer_config.json"),
        adapter_config: d.join("adapter/adapter_config.json"),
        adapter_weights: d.join("adapter/adapter_model.safetensors"),
    }
}

pub(crate) fn params() -> ArchParams {
    let i = ParamValue::Int;
    ArchParams::from_pairs(vec![
        ("n_layers".to_string(), i(4)),
        ("n_mtp_layers".to_string(), i(1)),
        ("n_embd".to_string(), i(16)),
        ("n_ff".to_string(), i(32)),
        ("n_heads".to_string(), i(2)),
        ("n_kv_heads".to_string(), i(1)),
        ("head_dim".to_string(), i(8)),
        ("n_vocab".to_string(), i(40)),
        ("tie_embeddings".to_string(), ParamValue::Bool(true)),
        ("full_attention_interval".to_string(), i(4)),
        ("ssm_conv_kernel".to_string(), i(4)),
        ("ssm_state_size".to_string(), i(4)),
        ("ssm_group_count".to_string(), i(2)),
        ("ssm_time_step_rank".to_string(), i(2)),
        ("ssm_inner_size".to_string(), i(8)),
        ("rope_theta".to_string(), i(10_000_000)),
        ("rope_dim".to_string(), i(2)),
    ])
}

pub(crate) const BASE_REPO: &str = "Org/Mini-Base";

/// Convert the fixture into a fresh folder and return it with the path of the GGUF.
pub(crate) fn convert(dtype: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    convert_files(
        &files(),
        &params(),
        BASE_REPO,
        dtype,
        dir.path(),
        &NoObserver,
        &NeverCancel,
    )
    .unwrap();
    let path = dir.path().join(MODEL_FILE);
    (dir, path)
}

fn tolerance(name: &str) -> Tol {
    qwen35_lora_tolerance(name, 4, false)
}

fn tolerance_f32(name: &str) -> Tol {
    qwen35_lora_tolerance(name, 4, true)
}

#[test]
fn the_f32_gguf_equals_the_one_of_the_official_script() {
    let (_dir, path) = convert("f32");
    let oracle = read_full(&fixture().join("oracle-f32.gguf"));
    let differences = compare_tol(&read_full(&path), &oracle, &tolerance_f32);
    assert!(differences.is_empty(), "{differences:#?}");
}

/// The tensors a LoRA adapted differ from PyTorch's in f32 by a few rounding units of the size
/// of the numbers added (measured: 3.4 units on this fixture, whose products have 2 terms).
#[test]
fn the_merged_f32_tensors_differ_from_pytorch_only_by_rounding_of_their_operands() {
    let (_dir, path) = convert("f32");
    let ours = read_full(&path);
    let oracle = read_full(&fixture().join("oracle-f32.gguf"));
    let mut base = SafeTensors::open(&files().base_weights, "base").unwrap();
    let mut adapter = SafeTensors::open(&files().adapter_weights, "adapter").unwrap();
    let mut worst = 0.0f64;
    let mut checked = 0;
    for e in Dims::from_params(&params()).entries() {
        let Some(module) =
            e.hf.strip_prefix("model.language_model.")
                .and_then(|m| m.strip_suffix(".weight"))
        else {
            continue;
        };
        if qwen35_lora_tolerance(&e.gguf, 4, true) != Tol::Skip {
            continue;
        }
        let w = values(&mut base, &e.hf);
        let a = values(
            &mut adapter,
            &format!("base_model.model.{module}.lora_A.weight"),
        );
        let b = values(
            &mut adapter,
            &format!("base_model.model.{module}.lora_B.weight"),
        );
        let (rows, cols) = (e.hf_shape[0] as usize, e.hf_shape[1] as usize);
        let r = a.len() / cols;
        let delta: Vec<f64> = (0..rows * cols)
            .map(|i| {
                (0..r)
                    .map(|k| f64::from(b[(i / cols) * r + k]) * f64::from(a[k * cols + i % cols]))
                    .sum()
            })
            .collect();
        let as_f32 = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let (x, y) = (
            as_f32(&ours.tensors[&e.gguf].bytes),
            as_f32(&oracle.tensors[&e.gguf].bytes),
        );
        worst = worst.max(merge_error_in_units(&w, &delta, 2.0, &x, &y));
        checked += 1;
    }
    assert_eq!(checked, 31);
    // the bound: 4 units (two ulps of the operands), the largest measured 3.4
    assert!(
        worst <= 4.0,
        "the merge differs from PyTorch's by {worst} rounding units of its operands"
    );
}

#[test]
fn the_bf16_gguf_equals_the_one_of_the_official_script() {
    let (_dir, path) = convert("bf16");
    let oracle = read_full(&fixture().join("oracle-bf16.gguf"));
    let differences = compare_tol(&read_full(&path), &oracle, &tolerance);
    assert!(differences.is_empty(), "{differences:#?}");
}

#[test]
fn tensors_not_touched_by_the_adapter_are_identical_to_the_byte() {
    for dtype in ["f32", "bf16"] {
        let (_dir, path) = convert(dtype);
        let ours = read_full(&path);
        let oracle = read_full(&fixture().join(format!("oracle-{dtype}.gguf")));
        let mut exact = 0;
        for (name, t) in &ours.tensors {
            if tolerance(name) == Tol::Exact {
                assert_eq!(t.sha256, oracle.tensors[name].sha256, "{dtype}: {name}");
                exact += 1;
            }
        }
        // 70 tensors, 31 adapted by the LoRA and 3 `ssm_a`: the other 36 are bit for bit
        assert_eq!(exact, 36, "{dtype}");
    }
}

#[test]
fn the_output_is_deterministic_and_described_by_what_the_converter_returns() {
    let dir1 = tempfile::tempdir().unwrap();
    let c1 = convert_files(
        &files(),
        &params(),
        BASE_REPO,
        "bf16",
        dir1.path(),
        &NoObserver,
        &NeverCancel,
    )
    .unwrap();
    let (_d2, p2) = convert("bf16");
    let p1 = dir1.path().join(MODEL_FILE);
    let (a, b) = (std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    assert_eq!(
        a, b,
        "two conversions of the same files give the same bytes"
    );
    let digest = sha256_hex(&a);
    // the digest and size the converter computed while writing are the file's
    assert_eq!(c1.sha256.as_deref(), Some(digest.as_str()));
    assert_eq!(c1.size, Some(a.len() as u64));
    assert_eq!(sha256_file(&p1).unwrap().0, digest);
    assert_eq!(super::qwen35::Qwen35Converter.version(), RECIPE);
    // the whole file, on every OS: it changes only together with the recipe version
    assert_eq!(
        (RECIPE, digest.as_str()),
        (RECIPE, PINNED_BF16_SHA256),
        "the output changed: bump the recipe version and this digest together"
    );
    let (_d3, p3) = convert("f32");
    assert_eq!(
        sha256_file(&p3).unwrap().0,
        PINNED_F32_SHA256,
        "the output changed: bump the recipe version and this digest together"
    );
}

/// SHA-256 of the whole GGUF written for the fixture by recipe `qwen35-lora/1`.
const PINNED_BF16_SHA256: &str = "0d1d2a1b56c6e45a93a58bdc227f0f0a6568953386a3dca6f8c193db8c93e3ad";
const PINNED_F32_SHA256: &str = "55385a01f716c124f4af67062badd84ce757db784d0469c8620bcf02bd5a229a";

#[test]
fn the_file_is_a_valid_gguf_that_the_family_row_describes() {
    let family = families::lookup("qwen35-pointer").unwrap();
    for (dtype, matrix) in [("f32", GgmlType::F32), ("bf16", GgmlType::BF16)] {
        let (_dir, path) = convert(dtype);
        let info = gguf::read_header(&path).unwrap();
        assert_eq!(info.tensors.len(), 70);
        for spec in (family.tensors)(&params()) {
            let t = info
                .tensor(&spec.name)
                .unwrap_or_else(|| panic!("no {}", spec.name));
            assert_eq!(t.dims, spec.dims, "{}", spec.name);
            let want = if spec.role == Role::Vector {
                GgmlType::F32
            } else {
                matrix
            };
            assert_eq!(t.ty, want, "{}", spec.name);
            assert_eq!(t.offset % 32, 0, "{} is aligned", spec.name);
        }
        for (key, expect) in (family.metadata)(&params()) {
            let got = info.get(&key).unwrap_or_else(|| panic!("no {key}"));
            match (expect, got) {
                (families::MetaExpect::Str(s), gguf::MetaValue::Str(g)) => assert_eq!(&s, g),
                (families::MetaExpect::UInt(n), gguf::MetaValue::UInt(g)) => {
                    assert_eq!(n, *g, "{key}")
                }
                (e, g) => panic!("{key}: {e:?} against {g:?}"),
            }
        }
    }
}

#[test]
fn a_file_compares_equal_to_itself_and_differences_are_named() {
    let (_d, path) = convert("bf16");
    let a = read_full(&path);
    assert!(compare(&a, &a).is_empty());
    let mut b = a.clone();
    let name = "blk.0.ffn_up.weight".to_string();
    b.tensors.get_mut(&name).unwrap().sha256 = "0".repeat(64);
    b.tensors.get_mut(&name).unwrap().bytes[0] ^= 1;
    b.meta.insert("general.name".into(), "other".into());
    b.meta.insert("qwen35.block_count".into(), 7.into());
    let diff = compare(&a, &b);
    assert_eq!(diff.len(), 2, "{diff:?}");
    assert!(diff.iter().any(|d| d.contains(&name)));
    assert!(diff.iter().any(|d| d.contains("qwen35.block_count")));
    // a one-ulp difference is within `Ulp(1)` and outside `Exact`
    let one = compare_tol(&a, &b, &|_| Tol::Ulp(1));
    assert!(one.iter().all(|d| !d.contains(&name)), "{one:?}");
    let exact = compare_tol(&a, &b, &|_| Tol::Exact);
    assert!(exact.iter().any(|d| d.contains(&name)), "{exact:?}");
}

fn values(st: &mut SafeTensors, name: &str) -> Vec<f32> {
    let t = st.get(name).unwrap().clone();
    let mut raw = vec![0u8; t.len as usize];
    st.read(&t, 0, &mut raw).unwrap();
    match t.dtype {
        Dtype::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| super::numeric::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
    }
}

#[test]
fn the_merge_is_the_documented_sequential_fused_sum() {
    // recompute one merged tensor of the f32 file from the inputs, by the documented order
    let (_d, path) = convert("f32");
    let out = read_full(&path);
    let mut base = SafeTensors::open(&files().base_weights, "base").unwrap();
    let mut adapter = SafeTensors::open(&files().adapter_weights, "adapter").unwrap();
    let w = values(
        &mut base,
        "model.language_model.layers.1.mlp.up_proj.weight",
    ); // [32, 16]
    let a = values(
        &mut adapter,
        "base_model.model.layers.1.mlp.up_proj.lora_A.weight",
    ); // [2, 16]
    let b = values(
        &mut adapter,
        "base_model.model.layers.1.mlp.up_proj.lora_B.weight",
    ); // [32, 2]
    let got = &out.tensors["blk.1.ffn_up.weight"].bytes;
    for row in 0..32 {
        for col in 0..16 {
            let delta = b[row * 2].mul_add(a[col], 0.0);
            let delta = b[row * 2 + 1].mul_add(a[16 + col], delta);
            let want = w[row * 16 + col] + 2.0 * delta;
            let at = (row * 16 + col) * 4;
            let have = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
            assert_eq!(have.to_bits(), want.to_bits(), "row {row} col {col}");
        }
    }
}

#[test]
fn progress_is_reported_at_the_start_and_the_end() {
    let dir = tempfile::tempdir().unwrap();
    let rec = Recorder::default();
    convert_files(
        &files(),
        &params(),
        BASE_REPO,
        "bf16",
        dir.path(),
        &rec,
        &NeverCancel,
    )
    .unwrap();
    let events = rec.events();
    assert!(
        matches!(
            events.first(),
            Some(Event::ConvertStart { tensors: 70, .. })
        ),
        "{events:?}"
    );
    assert!(
        matches!(events.last(), Some(Event::ConvertDone { .. })),
        "{events:?}"
    );
}

/// Puts the converter of this thread back to normal when dropped, even if the test fails.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        set_mutant(Mutant::None);
    }
}

/// Every deliberate mistake of the converter is noticed by the comparison with the official
/// script: the test of the comparison itself (a comparison that passes whatever it is given
/// proves nothing).
#[test]
fn each_mutant_of_the_converter_is_caught_by_the_comparison() {
    let _restore = Restore;
    for dtype in ["bf16", "f32"] {
        let oracle = read_full(&fixture().join(format!("oracle-{dtype}.gguf")));
        let tol = |n: &str| qwen35_lora_tolerance(n, 4, dtype == "f32");
        for mutant in [
            Mutant::NoPlusOne,
            Mutant::NoNegExp,
            Mutant::ScaleOne,
            Mutant::NoMerge,
            Mutant::ConvTransposed,
        ] {
            set_mutant(mutant);
            let (_dir, path) = convert(dtype);
            set_mutant(Mutant::None);
            let differences = compare_tol(&read_full(&path), &oracle, &tol);
            // the merge is only checked by the size of the operands in f32 (another test): there
            // the mutants of the merge are caught by that measure instead
            let merge = matches!(mutant, Mutant::ScaleOne | Mutant::NoMerge);
            if dtype == "f32" && merge {
                assert!(
                    differences.is_empty(),
                    "f32 merge mutants are Skip here: {differences:?}"
                );
            } else {
                assert!(
                    !differences.is_empty(),
                    "{dtype}: {mutant:?} was not noticed"
                );
            }
        }
    }
    // and without a mutant, the same comparison is clean
    set_mutant(Mutant::None);
    let (_dir, path) = convert("bf16");
    let oracle = read_full(&fixture().join("oracle-bf16.gguf"));
    assert!(compare_tol(&read_full(&path), &oracle, &tolerance).is_empty());
}

/// The merge mutants in f32 are caught by the measure against the operands' size.
#[test]
fn the_merge_mutants_are_caught_in_f32_by_the_size_of_the_operands() {
    let _restore = Restore;
    for mutant in [Mutant::ScaleOne, Mutant::NoMerge] {
        set_mutant(mutant);
        let (_dir, path) = convert("f32");
        set_mutant(Mutant::None);
        let ours = read_full(&path);
        let oracle = read_full(&fixture().join("oracle-f32.gguf"));
        let mut base = SafeTensors::open(&files().base_weights, "base").unwrap();
        let mut adapter = SafeTensors::open(&files().adapter_weights, "adapter").unwrap();
        let mut worst = 0.0f64;
        for e in Dims::from_params(&params()).entries() {
            let Some(module) =
                e.hf.strip_prefix("model.language_model.")
                    .and_then(|m| m.strip_suffix(".weight"))
            else {
                continue;
            };
            if qwen35_lora_tolerance(&e.gguf, 4, true) != Tol::Skip {
                continue;
            }
            let w = values(&mut base, &e.hf);
            let a = values(
                &mut adapter,
                &format!("base_model.model.{module}.lora_A.weight"),
            );
            let b = values(
                &mut adapter,
                &format!("base_model.model.{module}.lora_B.weight"),
            );
            let (rows, cols) = (e.hf_shape[0] as usize, e.hf_shape[1] as usize);
            let r = a.len() / cols;
            let delta: Vec<f64> = (0..rows * cols)
                .map(|i| {
                    (0..r)
                        .map(|k| {
                            f64::from(b[(i / cols) * r + k]) * f64::from(a[k * cols + i % cols])
                        })
                        .sum()
                })
                .collect();
            let as_f32 = |bytes: &[u8]| -> Vec<f32> {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            };
            let (x, y) = (
                as_f32(&ours.tensors[&e.gguf].bytes),
                as_f32(&oracle.tensors[&e.gguf].bytes),
            );
            worst = worst.max(merge_error_in_units(&w, &delta, 2.0, &x, &y));
        }
        assert!(
            worst > 4.0,
            "{mutant:?} is only {worst} units away, inside the bound"
        );
    }
}
