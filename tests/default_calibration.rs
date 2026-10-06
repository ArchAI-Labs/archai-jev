//! The declared calibration of the default model, reproduced with our own engine (spec 006c,
//! AC on calibration): accuracy, Brier and ECE on the 1,449 prompts of the author's test split,
//! with the temperature 1 and with the declared 2.09.
//!
//! Manual test (needs the GGUF and the ids of the test split, which come from public datasets
//! and are not redistributed): set `ARCHAI_JEV_TEST_DEFAULT_GGUF` and `ARCHAI_JEV_TEST_EVAL_DIR`
//! (the `eval2` folder of spike S5: `inputs.txt` and `meta.json`) and run
//! `cargo test --features testing --test default_calibration -- --ignored --nocapture`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use _core::calibration::calibrated_softmax;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::engine::{Forward, Output, OutputSpec};
use _core::json_strict::{Json, parse};
use _core::models::testing::assets;

struct Item {
    task: String,
    target: usize,
    ids: Vec<u32>,
}

fn load(dir: &std::path::Path) -> Vec<Item> {
    let meta = parse(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let Json::Array(rows) = meta else {
        panic!("meta")
    };
    let text = std::fs::read_to_string(dir.join("inputs.txt")).unwrap();
    let mut prompts: Vec<(String, Vec<u32>)> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("R") => prompts.push((it.next().unwrap().to_string(), Vec::new())),
            Some("S") => prompts.last_mut().unwrap().1 = it.map(|x| x.parse().unwrap()).collect(),
            _ => {}
        }
    }
    rows.iter()
        .map(|r| {
            let Some(Json::Str(name)) = r.get("name") else {
                panic!("name")
            };
            let Some(Json::Str(task)) = r.get("task") else {
                panic!("task")
            };
            let Some(Json::Int(target)) = r.get("target") else {
                panic!("target")
            };
            let ids = prompts.iter().find(|(n, _)| n == name).unwrap().1.clone();
            Item {
                task: task.clone(),
                target: *target as usize,
                ids,
            }
        })
        .collect()
}

/// Top-label ECE with 10 bins closed on the right, Brier as the sum over classes, as in spec 004.
fn metrics(probs: &[Vec<f64>], targets: &[usize]) -> (f64, f64, f64) {
    let n = probs.len() as f64;
    let mut correct = 0.0;
    let mut brier = 0.0;
    let mut bins = [(0.0f64, 0.0f64, 0.0f64); 10]; // (count, confidence sum, hits)
    for (p, &t) in probs.iter().zip(targets) {
        let top = p
            .iter()
            .enumerate()
            .fold(0, |b, (i, v)| if *v > p[b] { i } else { b });
        let hit = f64::from(u8::from(top == t));
        correct += hit;
        brier += p
            .iter()
            .enumerate()
            .map(|(k, v)| (v - f64::from(u8::from(k == t))).powi(2))
            .sum::<f64>();
        let conf = p[top];
        let bin = ((conf * 10.0).ceil() as usize).clamp(1, 10) - 1;
        bins[bin].0 += 1.0;
        bins[bin].1 += conf;
        bins[bin].2 += hit;
    }
    let ece = bins
        .iter()
        .filter(|b| b.0 > 0.0)
        .map(|b| (b.0 / n) * (b.2 / b.0 - b.1 / b.0).abs())
        .sum();
    (correct / n, brier / n, ece)
}

#[test]
#[ignore = "manual: needs the GGUF and the test-split ids (see the module comment)"]
fn the_declared_temperature_is_reproduced_by_our_engine() {
    let (Some(gguf), Ok(dir)) = (
        assets::default_gguf(),
        std::env::var("ARCHAI_JEV_TEST_EVAL_DIR"),
    ) else {
        return;
    };
    let items = load(std::path::Path::new(&dir));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let forward = LlamaForward::load(
        &gguf,
        LlamaConfig {
            threads,
            max_context: 512,
            hidden_states: false,
        },
    )
    .unwrap();
    // The letters in the order of the training targets: A-D, and TRUE then FALSE.
    let letters = [32u32, 33, 34, 35];
    let booleans = [20_611u32, 30_351];
    let mut raw: Vec<Vec<f64>> = Vec::with_capacity(items.len());
    let started = std::time::Instant::now();
    for (n, item) in items.iter().enumerate() {
        let ids: Vec<u32> = match item.task.as_str() {
            "safety" => letters[..2].to_vec(),
            "intent" => letters.to_vec(),
            _ => booleans.to_vec(),
        };
        let mut session = forward.begin(&[]).unwrap();
        let out = session
            .row(
                &item.ids,
                &[OutputSpec::Logits {
                    position: item.ids.len() - 1,
                    ids,
                }],
            )
            .unwrap();
        let Output::Logits(v) = &out[0] else {
            panic!("logits")
        };
        raw.push(v.iter().map(|x| f64::from(*x)).collect());
        if n % 200 == 0 {
            eprintln!("{n}/{} prompts, {:?}", items.len(), started.elapsed());
        }
    }
    let targets: Vec<usize> = items.iter().map(|i| i.target).collect();
    let at = |t: f64| -> Vec<Vec<f64>> {
        raw.iter()
            .map(|z| calibrated_softmax(z, t).unwrap())
            .collect()
    };
    let (acc1, brier1, ece1) = metrics(&at(1.0), &targets);
    let (acc2, brier2, ece2) = metrics(&at(2.09), &targets);
    eprintln!("T=1.00: accuracy {acc1:.4} brier {brier1:.4} ece {ece1:.4}");
    eprintln!("T=2.09: accuracy {acc2:.4} brier {brier2:.4} ece {ece2:.4}");
    // S6 measured 0.9199 / 0.1363 / 0.0584 and 0.9199 / 0.1209 / 0.0129 with the spike's own reader.
    assert!((acc1 - 0.92).abs() < 0.005, "accuracy {acc1}");
    assert!((brier1 - 0.1363).abs() < 0.003, "brier at T=1: {brier1}");
    assert!((ece1 - 0.0584).abs() < 0.005, "ece at T=1: {ece1}");
    assert!((brier2 - 0.1209).abs() < 0.003, "brier at T=2.09: {brier2}");
    assert!((ece2 - 0.0129).abs() < 0.005, "ece at T=2.09: {ece2}");
}
