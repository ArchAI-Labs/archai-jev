//! The real conversion of Kev-0.8B (spec 018, test level T2: run by hand, `#[ignore]`).
//!
//! Needs the original files (about 1.8 GB) and the GGUF files the official llama.cpp script wrote
//! from them (spike S4, the oracle). Point at them with
//!
//! - `ARCHAI_JEV_TEST_KEV_SNAPSHOT`: a folder with `adapter_config.json`, `adapter_model.safetensors`
//!   (the files of `jaredpalmer/kev-0.8b` @ 9a45d25e);
//! - `ARCHAI_JEV_TEST_BASE_SNAPSHOT`: a folder with `config.json`, `tokenizer.json`,
//!   `tokenizer_config.json`, `model.safetensors-00001-of-00001.safetensors` (the files of
//!   `Qwen/Qwen3.5-0.8B-Base` @ dc7cdfe2);
//! - `ARCHAI_JEV_TEST_KEV_DIR`: a folder with `kev-0.8b-f32.gguf` and `kev-0.8b-bf16.gguf`.
//!
//! Run with `cargo test --release --features testing --test kev_conversion -- --ignored --nocapture`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;
use std::time::Instant;

use _core::convert::RECIPE;
use _core::convert::testing_gguf::{
    Tol, compare_tol, qwen35_lora_tolerance, read_full, ulp_distance,
};
use _core::hub::events::{NeverCancel, NoObserver};
use _core::models::convert::SourceFiles;
use _core::models::registry::Registry;
use _core::models::validate;

fn dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

fn files() -> SourceFiles {
    let (kev, base) = (
        dir("ARCHAI_JEV_TEST_KEV_SNAPSHOT"),
        dir("ARCHAI_JEV_TEST_BASE_SNAPSHOT"),
    );
    SourceFiles {
        base_config: base.join("config.json"),
        base_weights: base.join("model.safetensors-00001-of-00001.safetensors"),
        base_tokenizer: base.join("tokenizer.json"),
        base_tokenizer_config: base.join("tokenizer_config.json"),
        adapter_config: kev.join("adapter_config.json"),
        adapter_weights: kev.join("adapter_model.safetensors"),
    }
}

/// Convert the real files into `out` (timed) and compare with the oracle.
fn convert_and_compare(dtype: &str, out: &std::path::Path) {
    let reg = Registry::builtin().unwrap();
    let manifest = reg.resolve("jaredpalmer/kev-0.8b", None).unwrap();
    let resolved = validate::semantics(manifest, &["hf-lora".to_string()]).unwrap();
    let started = Instant::now();
    let done = _core::convert::qwen35::convert_files(
        &files(),
        &resolved.params,
        "Qwen/Qwen3.5-0.8B-Base",
        dtype,
        out,
        &NoObserver,
        &NeverCancel,
    )
    .unwrap();
    println!(
        "{dtype}: converted in {:.1} s, {} bytes, recipe {RECIPE}",
        started.elapsed().as_secs_f64(),
        done.size.unwrap()
    );
    let ours = read_full(&out.join("model.gguf"));
    let oracle = read_full(&dir("ARCHAI_JEV_TEST_KEV_DIR").join(format!("kev-0.8b-{dtype}.gguf")));
    assert_eq!(ours.tensors.len(), 335);

    // untouched tensors: bit for bit; the others: how far
    let f32_file = dtype == "f32";
    let mut exact = 0;
    let mut report: Vec<String> = Vec::new();
    for (name, t) in &ours.tensors {
        let o = &oracle.tensors[name];
        match qwen35_lora_tolerance(name, 24, false) {
            Tol::Exact => {
                assert_eq!(
                    t.sha256, o.sha256,
                    "{dtype}: {name} must be identical to the byte"
                );
                exact += 1;
            }
            _ if t.sha256 == o.sha256 => {}
            _ => {
                let (n, worst) = ulp_distance(&t.ty, &t.bytes, &o.bytes);
                let elements: u64 = t.dims.iter().product();
                report.push(format!(
                    "{name}: {n} of {elements} elements differ ({:.2e}), up to {worst} ulp",
                    n as f64 / elements as f64
                ));
            }
        }
    }
    println!(
        "{dtype}: {exact} tensors identical to the byte, {} adapted or exp tensors differ",
        report.len()
    );
    for line in &report {
        println!("  {line}");
    }
    let differences = compare_tol(&ours, &oracle, &|n| qwen35_lora_tolerance(n, 24, f32_file));
    // the bf16 clause (<= 1 ulp and few elements) and the metadata must hold; the f32 merged
    // tensors are judged by the size of their operands (the measure of the tiny fixture)
    assert!(differences.is_empty(), "{dtype}: {differences:#?}");
}

#[test]
#[ignore = "needs the original files of Kev-0.8B and the oracle GGUF files (about 6 GB)"]
fn the_bf16_conversion_agrees_with_the_official_script() {
    let out = tempfile::tempdir().unwrap();
    convert_and_compare("bf16", out.path());
}

#[test]
#[ignore = "needs the original files of Kev-0.8B and the oracle GGUF files (about 6 GB)"]
fn the_f32_conversion_agrees_with_the_official_script() {
    let out = tempfile::tempdir().unwrap();
    convert_and_compare("f32", out.path());
}

#[test]
#[ignore = "needs the original files of Kev-0.8B"]
fn the_registry_entry_describes_the_real_config() {
    // `check_against` runs inside the conversion; here only the cheap half: the manifest's
    // architecture agrees with the real config.json
    let hp = _core::convert::config::Hparams::read(&files().base_config).unwrap();
    let reg = Registry::builtin().unwrap();
    let manifest = reg.resolve("jaredpalmer/kev-0.8b", None).unwrap();
    let resolved = validate::semantics(manifest, &["hf-lora".to_string()]).unwrap();
    hp.check_against(&resolved.params).unwrap();
}
