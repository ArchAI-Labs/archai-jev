//! P1 to P3 of the performance requirements of the `spark2_5-letters` family (spec 010b), with
//! real weights: by hand, `#[ignore]`, **release** build only, one heavy job at a time.
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_SPARK_DIR = "<local model folder>"
//! $env:ARCHAI_JEV_BENCH_OUT = "benches/answer_letters/results/p1_p3.json"   # optional
//! cargo test --release --features testing --test spark_letters_bench -- --ignored --nocapture
//! ```
//!
//! Method (D23/D42/D47): the two things to compare run in **alternating** order inside the same
//! interval, in two regimes (back to back, and with 0.5 s of pause before each call), with an
//! A/A control (the same thing against itself) that says how big the noise of the machine is.
//! Medians and quartiles are reported; the thresholds were written in the spec before the data.
//! The decisions per second on the Snake bench (P4) were measured in the spike with its own
//! scripts and are not repeated here.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod spark_common;

use std::time::{Duration, Instant};

use _core::calibration::calibrated_softmax;
use _core::engine::pick_logits;
use _core::scorer::Scorer;
use _core::{Choice, ChoiceOption, Question, Questions, State, StateValue, TextEntry};
use spark_common::Model;

const BLOCKS: usize = 14;

fn state(n: usize) -> State {
    // a different text every time, so that nothing of the content is remembered
    let text = (0..14)
        .map(|i| {
            format!(
                "Report {n}.{i}: order {} was packed in bay {} and left at {}h.",
                4000 + n * 31 + i,
                i % 5,
                6 + i % 12
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    State::new(StateValue::string(text)).unwrap()
}

fn question(instructions: &str, options: usize) -> Question {
    Choice::new(
        TextEntry::text(instructions).unwrap(),
        (0..options)
            .map(|i| {
                ChoiceOption::described(
                    format!("k{i}"),
                    &format!("The order was handled by team number {i}"),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
    .into()
}

fn one(options: usize) -> Questions {
    Questions::new([(
        "q",
        question("Which team handled the first order?", options),
    )])
    .unwrap()
}

fn three() -> Questions {
    Questions::new([
        ("a", question("Which team handled the first order?", 4)),
        ("b", question("Which team handled the last order?", 4)),
        (
            "c",
            question("Which team handled an order packed in bay 3?", 4),
        ),
    ])
    .unwrap()
}

fn ms(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1000.0
}

fn quartiles(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    (at(0.25), at(0.5), at(0.75))
}

fn pause(regime: &str) {
    if regime == "paused" {
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Alternate `a` and `b` for `BLOCKS` pairs, in both orders, and give the ratio b/a of medians.
fn compare(
    regime: &str,
    mut a: impl FnMut(usize) -> f64,
    mut b: impl FnMut(usize) -> f64,
) -> (f64, (f64, f64, f64), (f64, f64, f64)) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..BLOCKS {
        let flip = i % 2 == 1;
        for first in [!flip, flip] {
            pause(regime);
            if first {
                xs.push(a(i));
            } else {
                ys.push(b(i));
            }
        }
    }
    let (qa, qb) = (quartiles(xs), quartiles(ys));
    (qb.1 / qa.1, qa, qb)
}

#[test]
#[ignore = "needs the model folder: ARCHAI_JEV_TEST_SPARK_DIR; release build; minutes"]
fn p1_and_p3() {
    use _core::engine::Forward;
    let Some(model) = Model::open("ARCHAI_JEV_TEST_SPARK_DIR") else {
        return;
    };
    // the path with cuts and no memory: what every request of a model without reuse runs
    let bare = model.scorer(Some(512), 1);
    let kept = model.scorer(Some(512), 256 << 20);
    let forward = model.forward();
    for n in 0..3 {
        // warm the machine and the weights
        bare.score(&state(900 + n), &one(4)).unwrap();
    }
    let mut lines = Vec::new();
    for regime in ["consecutive", "paused"] {
        let (aa, ..) = compare(
            regime,
            |i| ms(|| drop(bare.score(&state(1000 + i), &one(4)).unwrap())),
            |i| ms(|| drop(bare.score(&state(2000 + i), &one(4)).unwrap())),
        );
        lines.push(format!("{regime}: A/A (the same call twice) ratio {aa:.3}"));

        // P1: a question of 4 options against only the prefill of the same row
        let (p1, prefill, score) = compare(
            regime,
            |i| {
                let ids = model
                    .build(&state(3000 + i), &one(4))
                    .unwrap()
                    .row(0)
                    .unwrap();
                ms(|| {
                    let mut s = forward.begin_empty().unwrap();
                    s.extend(&ids).unwrap();
                })
            },
            |i| ms(|| drop(bare.score(&state(3000 + i), &one(4)).unwrap())),
        );
        lines.push(format!(
            "{regime}: P1 score / prefill = {p1:.3} (prefill p25/p50/p75 {:.0}/{:.0}/{:.0} ms, score {:.0}/{:.0}/{:.0} ms); threshold 1.15",
            prefill.0, prefill.1, prefill.2, score.0, score.1, score.2
        ));

        // P3: three questions about one state in one request against three requests, no memory
        let (p3, separate, together) = compare(
            regime,
            |i| {
                let st = state(4000 + i);
                let qs = three();
                ms(|| {
                    for (name, q) in qs.iter() {
                        let single = Questions::new([(name, q.clone())]).unwrap();
                        bare.score(&st, &single).unwrap();
                    }
                })
            },
            |i| ms(|| drop(bare.score(&state(4000 + i), &three()).unwrap())),
        );
        lines.push(format!(
            "{regime}: P3 together / separate = {p3:.3} (separate p50 {:.0} ms, together p50 {:.0} ms); threshold 0.7",
            separate.1, together.1
        ));
        // the same with the memory on (the fixed head is warm after the first request)
        let (p3k, _, together_kept) = compare(
            regime,
            |i| {
                let st = state(5000 + i);
                let qs = three();
                ms(|| {
                    for (name, q) in qs.iter() {
                        let single = Questions::new([(name, q.clone())]).unwrap();
                        bare.score(&st, &single).unwrap();
                    }
                })
            },
            |i| ms(|| drop(kept.score(&state(5000 + i), &three()).unwrap())),
        );
        lines.push(format!(
            "{regime}: P3 with the memory on = {p3k:.3} (together p50 {:.0} ms)",
            together_kept.1
        ));
    }
    for line in &lines {
        eprintln!("{line}");
    }
    if let Ok(out) = std::env::var("ARCHAI_JEV_BENCH_OUT") {
        std::fs::write(
            out,
            lines.join(
                "
",
            ),
        )
        .unwrap();
    }
}

/// P2: the finite check over a whole row, the extraction of the letters and the softmax.
#[test]
#[ignore = "release build"]
fn p2_the_checks_cost_next_to_a_decision() {
    let row: Vec<f32> = (0..131_072)
        .map(|i| ((i * 37 % 1000) as f32) / 100.0 - 5.0)
        .collect();
    let ids: Vec<u32> = (46..72).collect();
    let n = 2000;
    let t = Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..n {
        let picked = pick_logits(&row, &ids, 0).unwrap();
        let z: Vec<f64> = picked.iter().map(|v| f64::from(*v)).collect();
        sink += calibrated_softmax(&z, 1.0).unwrap()[0];
    }
    let per_call_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(n);
    eprintln!(
        "finite check over 131072 values + 26 letters + softmax: {per_call_us:.1} us per question (sink {sink:.3})"
    );
    assert!(per_call_us > 0.0);
}

/// Where the gap of P1 comes from: the same row decoded as one block, in the segments its cuts
/// define (what the canonical path does), and through `score` (tokenizing, the template, the
/// head, the checks).
#[test]
#[ignore = "needs the model folder: ARCHAI_JEV_TEST_SPARK_DIR; release build; minutes"]
fn p1_decomposition() {
    use _core::engine::{Forward, OutputSpec};
    let Some(model) = Model::open("ARCHAI_JEV_TEST_SPARK_DIR") else {
        return;
    };
    let bare = model.scorer(Some(512), 1);
    let forward = model.forward();
    bare.score(&state(900), &one(4)).unwrap();
    let (mut whole, mut cut, mut scored, mut build) = (vec![], vec![], vec![], vec![]);
    for i in 0..8 {
        let st = state(6000 + i);
        let qs = one(4);
        let mut prompt = None;
        build.push(ms(|| prompt = Some(model.build(&st, &qs).unwrap())));
        let prompt = prompt.unwrap();
        let row = prompt.row(0).unwrap();
        let cuts: Vec<usize> = prompt.row_cuts(0).unwrap().iter().map(|c| c.pos).collect();
        whole.push(ms(|| {
            let mut s = forward.begin_empty().unwrap();
            s.extend(&row).unwrap();
        }));
        cut.push(ms(|| {
            let mut s = forward.begin_empty().unwrap();
            let mut from = 0;
            for c in &cuts {
                s.extend(&row[from..*c]).unwrap();
                from = *c;
            }
            let last = row.len() - 1;
            let spec = OutputSpec::Logits {
                position: last - from,
                ids: vec![46, 47, 48, 49],
            };
            s.row_segments(&[&row[from..]], &[spec]).unwrap();
        }));
        scored.push(ms(|| drop(bare.score(&st, &qs).unwrap())));
        eprintln!("{} tokens, cuts {cuts:?}", row.len());
    }
    let med = |v: Vec<f64>| quartiles(v).1;
    eprintln!(
        "p50: build the prompt {:.1} ms; one block {:.0} ms; in segments with the answer {:.0} ms; score {:.0} ms",
        med(build),
        med(whole),
        med(cut),
        med(scored)
    );
}
