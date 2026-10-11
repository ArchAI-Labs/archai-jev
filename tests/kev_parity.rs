//! Kev-0.8B end to end against the golden of the reference implementation (spec 018 AC-14/15,
//! spec 006b AC-20…24; test level T2, by hand, `#[ignore]`).
//!
//! The whole real path: the original files are put in a cache as if downloaded, `load_model`
//! converts them to GGUF, llama.cpp loads the result, the self-check vectors of the registry run
//! on it, and then every request of the golden (65 + 4) goes through `ask` and is compared with
//! what Kev's own code computed in fp32.
//!
//! Needs `ARCHAI_JEV_TEST_KEV_SNAPSHOT` and `ARCHAI_JEV_TEST_BASE_SNAPSHOT` (see
//! `kev_conversion.rs`). Run with
//! `cargo test --release --features testing --test kev_parity -- --ignored --nocapture`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use _core::Answer;
use _core::calibration::calibrated_softmax;
use _core::convert::Qwen35Converter;
use _core::convert::headpt::HeadPtReader;
use _core::hub::config::HubConfig;
use _core::hub::download::{Response, Transport};
use _core::hub::events::{NeverCancel, Recorder};
use _core::json_strict::Json;
use _core::models::hash::sha256_file;
use _core::models::llama_backend::LlamaBackend;
use _core::models::load::{Context, LoadRequest, load_model};
use _core::models::registry::Registry;
use _core::models::resolve::{NameOrPath, Selection};
use _core::models::testing::golden::{self, GoldenCase};

struct NoNetwork;

impl Transport for NoNetwork {
    fn get(&self, _: &str, _: Option<u64>, _: Duration) -> Result<Response, String> {
        Err("the network is disabled in this test".to_string())
    }
}

fn snapshot(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

/// Put the original files in a cache folder, named by their SHA-256 like a download would.
fn populate(cache: &Path) {
    let blobs = cache.join("v1").join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    let kev = snapshot("ARCHAI_JEV_TEST_KEV_SNAPSHOT");
    let base = snapshot("ARCHAI_JEV_TEST_BASE_SNAPSHOT");
    for path in [
        kev.join("adapter_config.json"),
        kev.join("adapter_model.safetensors"),
        kev.join("head.pt"),
        kev.join("tokenizer.json"),
        base.join("config.json"),
        base.join("model.safetensors-00001-of-00001.safetensors"),
        base.join("tokenizer.json"),
        base.join("tokenizer_config.json"),
    ] {
        let (sha, _) = sha256_file(&path).unwrap();
        let target = blobs.join(&sha);
        if !target.exists() {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Worst {
    logit: f64,
    prob: f64,
    choice_confidence: f64,
    score_confidence: f64,
    score_value: f64,
    n: usize,
}

fn band(state_len: usize) -> usize {
    match state_len {
        0..100 => 0,
        100..400 => 1,
        400..3000 => 2,
        _ => 3,
    }
}

fn number(j: &Json) -> f64 {
    match j {
        Json::Float(f) => *f,
        Json::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

fn run(dtype: &str, cache_mib: u64) {
    let cache = tempfile::tempdir().unwrap();
    populate(cache.path());
    let registry = Registry::builtin().unwrap();
    let mut hub = HubConfig::new(cache.path().to_path_buf());
    hub.offline = true;
    let threads = _core::models::llama_backend::default_threads();
    let backend = LlamaBackend::new(threads).with_prefix_cache(cache_mib << 20);
    let recorder = Recorder::default();
    let ctx = Context {
        registry,
        hub: &hub,
        backend: &backend,
        transport: &NoNetwork,
        observer: &recorder,
        cancel: &NeverCancel,
        head_reader: Some(&HeadPtReader),
        converter: Some(&Qwen35Converter),
    };
    let request = LoadRequest {
        selection: Selection {
            name_or_path: NameOrPath::Str("jaredpalmer/kev-0.8b".to_string()),
            revision: None,
            device: "cpu".to_string(),
            dtype: Some(dtype.to_string()),
            manifest: None,
        },
        temperature: None,
        allow_uncalibrated: false,
    };
    let started = Instant::now();
    let loaded = load_model(&request, &ctx).unwrap_or_else(|e| panic!("{dtype}: {e}"));
    println!(
        "{dtype}: loaded in {:.1} s (conversion, GGUF, engine, the 8 self-check vectors)",
        started.elapsed().as_secs_f64()
    );
    for e in recorder.events() {
        println!("  event: {e:?}");
    }
    assert_eq!(loaded.data.family, "qwen35-pointer");
    assert!(loaded.data.calibrated);

    let mut cases: Vec<GoldenCase> = golden::load("golden.jsonl");
    cases.extend(golden::load("golden_extra.jsonl"));
    assert_eq!(cases.len(), 69);

    // With the reuse on the golden goes through twice: the second pass is warm and must give
    // the very same numbers as the first, besides the parity with the reference.
    let passes = if cache_mib > 0 { 2 } else { 1 };
    let mut first: Vec<Vec<Vec<f64>>> = Vec::new();
    for pass in 0..passes {
        let mut seen: Vec<Vec<Vec<f64>>> = Vec::new();

        let mut worst = [Worst::default(); 4];
        let mut refused = 0;
        let (mut argmax_checked, mut near_ties, mut argmax_flipped) = (0, 0, 0);
        let mut longest_ok = 0;
        for case in &cases {
            let (state, questions) = match golden::request_to_domain(&case.request) {
                Ok(x) => x,
                Err(_) => {
                    refused += 1;
                    continue;
                }
            };
            let answers = _core::ask(loaded.scorer.as_ref(), &state, &questions)
                .unwrap_or_else(|e| panic!("{dtype}: {}: {e}", case.id));
            let logits = loaded.scorer.score(&state, &questions).unwrap();
            seen.push(logits.clone());
            longest_ok = longest_ok.max(case.ids.len());
            let w = &mut worst[band(case.state_len)];
            for ((golden_q, z), (name, answer)) in
                case.questions.iter().zip(&logits).zip(answers.iter())
            {
                assert_eq!(name, golden_q.name);
                assert_eq!(z.len(), golden_q.logits_raw.len(), "{}: {name}", case.id);
                w.n += 1;
                for (zz, g) in z.iter().zip(&golden_q.logits_raw) {
                    w.logit = w
                        .logit
                        .max((zz / case.temperature - g / case.temperature).abs());
                }
                let p = calibrated_softmax(z, case.temperature).unwrap();
                for (a, b) in p.iter().zip(&golden_q.probs) {
                    w.prob = w.prob.max((a - b).abs());
                }
                // the argmax, except where the golden itself has a near tie
                if golden_q.kind != "score" && golden_q.probs.len() > 1 {
                    let mut sorted = golden_q.probs.clone();
                    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let top = |v: &[f64]| {
                        v.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap()
                            .0
                    };
                    argmax_checked += 1;
                    if sorted[0] - sorted[1] < 3e-2 {
                        near_ties += 1;
                    } else if top(&p) != top(&golden_q.probs) {
                        argmax_flipped += 1;
                        println!("  argmax differs: {} / {name}", case.id);
                    }
                }
                match answer {
                    Answer::Choice(c) => {
                        w.choice_confidence = w
                            .choice_confidence
                            .max((c.confidence() - golden_q.confidence).abs());
                    }
                    Answer::Score(s) => {
                        w.score_confidence = w
                            .score_confidence
                            .max((s.confidence() - golden_q.confidence).abs());
                        w.score_value = w
                            .score_value
                            .max((s.value() - number(&golden_q.value)).abs());
                    }
                    Answer::YesNo(_) => {}
                }
            }
        }
        println!(
            "{dtype}: {refused} requests refused on purpose, longest prompt {longest_ok} tokens"
        );
        println!(
            "{dtype}: argmax checked {argmax_checked}, near ties {near_ties}, flipped {argmax_flipped}"
        );
        for (i, label) in ["< 100", "100-400", "400-3000", "> 3000"]
            .iter()
            .enumerate()
        {
            let w = worst[i];
            println!(
                "{dtype}: state {label:>9} tokens, {:3} questions: max |d(z/T)| {:.2e}  |dp| {:.2e}  conf choice {:.2e}  conf score {:.2e}  value score {:.2e}",
                w.n, w.logit, w.prob, w.choice_confidence, w.score_confidence, w.score_value
            );
        }
        assert_eq!(refused, 8, "the requests the library refuses on purpose");
        assert_eq!(argmax_flipped, 0);
        for w in worst {
            assert!(w.logit <= 6e-2, "|d(z/T)| {}", w.logit);
            assert!(w.prob <= 1.5e-2, "|dp| {}", w.prob);
            assert!(
                w.choice_confidence <= 2.5e-2,
                "choice confidence {}",
                w.choice_confidence
            );
            assert!(
                w.score_confidence <= 1e-2,
                "score confidence {}",
                w.score_confidence
            );
            assert!(w.score_value <= 1e-2, "score value {}", w.score_value);
        }
        if pass == 0 {
            first = seen;
        } else {
            assert_eq!(
                first, seen,
                "{dtype}: the warm pass must equal the cold one bit for bit"
            );
        }
        if let Some(info) = loaded.scorer.cache_info() {
            println!("{dtype}: pass {pass}: {info:?}");
            assert!(info.bytes <= info.limit_bytes);
        }
    }
}

#[test]
#[ignore = "needs the original files of Kev-0.8B (1.8 GB): see the header of this file"]
fn kev_bf16_matches_the_golden() {
    run("bf16", 0);
}

#[test]
#[ignore = "needs the original files of Kev-0.8B (1.8 GB): see the header of this file"]
fn kev_f32_matches_the_golden() {
    run("f32", 0);
}

/// Every deliberate mistake of the converter makes the self-check of the registry fail on the
/// real engine: the vectors are what stops a wrong conversion from answering (spec 018, AC-16).
#[test]
#[ignore = "needs the original files of Kev-0.8B (1.8 GB): see the header of this file"]
fn each_converter_mutant_is_stopped_by_the_selfcheck() {
    use _core::Error;
    use _core::convert::stream::{Mutant, set_mutant};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_mutant(Mutant::None);
        }
    }
    let _restore = Restore;
    let cache = tempfile::tempdir().unwrap();
    populate(cache.path());
    let registry = Registry::builtin().unwrap();
    let mut hub = HubConfig::new(cache.path().to_path_buf());
    hub.offline = true;
    let backend = LlamaBackend::new(_core::models::llama_backend::default_threads());
    let ctx = Context {
        registry,
        hub: &hub,
        backend: &backend,
        transport: &NoNetwork,
        observer: &Recorder::default(),
        cancel: &NeverCancel,
        head_reader: Some(&HeadPtReader),
        converter: Some(&Qwen35Converter),
    };
    let request = LoadRequest {
        selection: Selection {
            name_or_path: NameOrPath::Str("jaredpalmer/kev-0.8b".to_string()),
            revision: None,
            device: "cpu".to_string(),
            dtype: Some("bf16".to_string()),
            manifest: None,
        },
        temperature: None,
        allow_uncalibrated: false,
    };
    for mutant in [
        Mutant::NoPlusOne,
        Mutant::NoNegExp,
        Mutant::ScaleOne,
        Mutant::NoMerge,
        Mutant::ConvTransposed,
    ] {
        // a fresh conversion every time; the originals stay
        let _ = std::fs::remove_dir_all(cache.path().join("v1").join("materialized"));
        let _ = std::fs::remove_dir_all(cache.path().join("v1").join("verified"));
        set_mutant(mutant);
        let r = load_model(&request, &ctx);
        set_mutant(Mutant::None);
        match r {
            Err(Error::ModelVerification(f)) => {
                println!("{mutant:?}: stopped by the self-check: {f}")
            }
            Err(other) => println!("{mutant:?}: stopped by another check: {other}"),
            Ok(_) => panic!("{mutant:?}: the self-check did not notice"),
        }
    }
}

#[test]
#[ignore = "needs the original files of Kev-0.8B (1.8 GB): see the header of this file"]
fn kev_bf16_matches_the_golden_with_the_prefix_cache() {
    run("bf16", 128);
}

#[test]
#[ignore = "needs the original files of Kev-0.8B (1.8 GB): see the header of this file"]
fn kev_f32_matches_the_golden_with_the_prefix_cache() {
    run("f32", 128);
}
