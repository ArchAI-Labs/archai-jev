//! The v2 model (spec 010) against the reference engine of the Decision Index kit.
//!
//! `golden.jsonl` has requests of ours and, per question, what the kit's `transformers` engine
//! (float32 on CPU) produced for the published fp16 weights: the ids of the prompt, the ids of
//! every option key, the score of every option. Two levels:
//!
//! - **T1** (needs the base `tokenizer.json`): the prompt and the key ids this library builds are
//!   **identical** to the reference's, for every question.
//! - **T2** (needs a GGUF of the v2 weights, `#[ignore]`): the scores of the real engine are close
//!   to the reference's.
//!
//! The golden is `tests/data/v2-golden/golden.jsonl`; `ARCHAI_JEV_TEST_V2_GOLDEN` points elsewhere.
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_DEFAULT_TOKENIZER = "<base tokenizer.json>"
//! $env:ARCHAI_JEV_TEST_V2_GGUF = "<f16 or q8_0 gguf of the v2 weights>"      # T2 only
//! cargo test --features testing --test v2_parity -- --include-ignored --nocapture
//! ```
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;
use std::sync::Arc;

use _core::Calibration;
use _core::Scorer;
use _core::calibration::calibrated_softmax;
use _core::engine::llama::{LlamaConfig, LlamaForward, cpu_plan};
use _core::heads::KeyLogProbHead;
use _core::json_strict::{Json, parse};
use _core::model_scorer::ModelScorer;
use _core::models::testing::assets;
use _core::prompt::di::DecisionIndexChatMl;
use _core::prompt::{PromptTokenizer, RequestLimits, Roles, Template};
use _core::request::request_to_domain;

const MAX_KEY_TOKENS: usize = 8;
const MAX_OPTIONS: usize = 48;
const MAX_CONTEXT: u32 = 1024;

fn roles() -> Roles {
    Roles::new([("im_start", 151_644u32), ("im_end", 151_645)])
}

struct Question {
    name: String,
    ids: Vec<u32>,
    key_ids: Vec<Vec<u32>>,
    scores: Vec<f64>,
}

struct Case {
    id: String,
    request: Json,
    questions: Vec<Question>,
}

fn u32s(j: &Json) -> Vec<u32> {
    let Json::Array(items) = j else {
        panic!("an array of ids")
    };
    items
        .iter()
        .map(|v| match v {
            Json::Int(i) => u32::try_from(*i).unwrap(),
            other => panic!("not an id: {other:?}"),
        })
        .collect()
}

fn f64s(j: &Json) -> Vec<f64> {
    let Json::Array(items) = j else {
        panic!("an array of numbers")
    };
    items
        .iter()
        .map(|v| match v {
            Json::Float(f) => *f,
            Json::Int(i) => *i as f64,
            other => panic!("not a number: {other:?}"),
        })
        .collect()
}

fn load() -> Option<Vec<Case>> {
    let path = std::env::var("ARCHAI_JEV_TEST_V2_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/v2-golden/golden.jsonl")
        });
    let text = std::fs::read_to_string(path).unwrap();
    Some(
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                let row = parse(line.as_bytes()).unwrap();
                let Some(Json::Str(id)) = row.get("id") else {
                    panic!("id")
                };
                let Some(Json::Object(qs)) = row.get("questions") else {
                    panic!("questions")
                };
                let questions = qs
                    .iter()
                    .map(|(name, q)| {
                        let Some(Json::Array(keys)) = q.get("key_ids") else {
                            panic!("key_ids")
                        };
                        Question {
                            name: name.clone(),
                            ids: u32s(q.get("ids").unwrap()),
                            key_ids: keys.iter().map(u32s).collect(),
                            scores: f64s(q.get("scores").unwrap()),
                        }
                    })
                    .collect();
                Case {
                    id: id.clone(),
                    request: row.get("request").unwrap().clone(),
                    questions,
                }
            })
            .collect(),
    )
}

#[test]
fn prompt_and_key_ids_are_identical_to_the_reference() {
    let (Some(cases), Some(tokenizer)) = (load(), assets::default_tokenizer()) else {
        eprintln!("SKIPPED: set the default tokenizer (ARCHAI_JEV_TEST_DEFAULT_TOKENIZER)");
        return;
    };
    let tok = PromptTokenizer::from_file(&tokenizer, roles()).unwrap();
    let template = DecisionIndexChatMl::new(MAX_KEY_TOKENS);
    let limits = RequestLimits {
        model: "v2",
        max_context: 100_000,
        question_types: &["choice", "noul"],
        support: &KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS),
    };
    let mut questions = 0usize;
    for case in &cases {
        let (state, qs) =
            request_to_domain(&case.request).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let prompt = template
            .build(&state, &qs, &tok, &limits)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(prompt.branches().len(), case.questions.len(), "{}", case.id);
        for (branch, want) in prompt.branches().iter().zip(&case.questions) {
            assert_eq!(
                branch.ids(),
                want.ids.as_slice(),
                "{} question {}: the prompt ids differ from the reference",
                case.id,
                want.name
            );
            assert_eq!(
                branch.keys(),
                want.key_ids.as_slice(),
                "{} question {}: the key ids differ from the reference",
                case.id,
                want.name
            );
            questions += 1;
        }
    }
    println!(
        "{} requests, {questions} questions: ids identical to the reference",
        cases.len()
    );
    assert!(questions >= 80);
}

fn scorer(gguf: &std::path::Path, tokenizer: &std::path::Path, cache_bytes: u64) -> ModelScorer {
    let tok = PromptTokenizer::from_file(tokenizer, roles()).unwrap();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let forward = LlamaForward::load(
        gguf,
        LlamaConfig {
            threads,
            max_context: MAX_CONTEXT,
            hidden_states: false,
            plan: cpu_plan().unwrap(),
        },
    )
    .unwrap()
    .with_key_scoring(MAX_KEY_TOKENS as u32);
    ModelScorer::new(
        "v2",
        Box::new(DecisionIndexChatMl::new(MAX_KEY_TOKENS)),
        tok,
        Arc::new(forward),
        Box::new(KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS)),
        Calibration::new(1.0, false).unwrap(),
        u64::from(MAX_CONTEXT),
        vec!["choice".to_string(), "noul".to_string()],
    )
    .with_prefix_cache(cache_bytes)
}

/// The real engine on the real weights against the reference: the scores, the probabilities
/// (softmax over the options) and the choice.
#[test]
#[ignore = "needs a GGUF of the v2 weights: set ARCHAI_JEV_TEST_V2_GGUF"]
fn the_engine_scores_like_the_reference() {
    let (Some(cases), Some(tokenizer), Ok(gguf)) = (
        load(),
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_V2_GGUF"),
    ) else {
        eprintln!("SKIPPED: set the tokenizer and ARCHAI_JEV_TEST_V2_GGUF");
        return;
    };
    let tolerance: f64 = std::env::var("ARCHAI_JEV_TEST_V2_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.2);
    let scorer = scorer(std::path::Path::new(&gguf), &tokenizer, 0);
    let started = std::time::Instant::now();
    let (mut worst_score, mut worst_p, mut flips, mut questions) = (0.0f64, 0.0f64, 0usize, 0usize);
    let mut worst_case = String::new();
    for case in &cases {
        let (state, qs) = request_to_domain(&case.request).unwrap();
        let run = scorer.run(&state, &qs).unwrap();
        for ((name, mine), want) in run.questions.iter().zip(&case.questions) {
            assert_eq!(name, &want.name, "{}", case.id);
            assert_eq!(mine.len(), want.scores.len(), "{} {name}", case.id);
            let ds = mine
                .iter()
                .zip(&want.scores)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let pm = calibrated_softmax(mine, 1.0).unwrap();
            let pw = calibrated_softmax(&want.scores, 1.0).unwrap();
            let dp = pm
                .iter()
                .zip(&pw)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let top = |p: &[f64]| {
                p.iter()
                    .enumerate()
                    .fold(0, |b, (i, v)| if *v > p[b] { i } else { b })
            };
            let mut sorted = pw.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            if top(&pm) != top(&pw) && (sorted[0] - sorted[1]) >= 0.25 {
                flips += 1;
                eprintln!(
                    "{} {name}: a flip with a margin of {:.3}",
                    case.id,
                    sorted[0] - sorted[1]
                );
            }
            if dp > worst_p {
                worst_p = dp;
                worst_case = format!("{} {name}", case.id);
            }
            worst_score = worst_score.max(ds);
            questions += 1;
        }
    }
    println!(
        "{questions} questions in {:.1} s | max |score diff| {worst_score:.4} | max |dp| {worst_p:.4} ({worst_case}) | flips beyond a 0.25 margin: {flips}",
        started.elapsed().as_secs_f64()
    );
    assert!(
        worst_p <= tolerance,
        "max |dp| {worst_p} is beyond {tolerance}"
    );
    assert_eq!(flips, 0);
}

/// Softmax of the scores of one question (what the library turns into probabilities).
fn softmax(scores: &[f64]) -> Vec<f64> {
    let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

/// The cache of prefixes (spec 007) with the cut points of the template: cold, warm and a store
/// that remembers nothing give the very same scores, bit for bit; the plain path (no cache, the
/// row in one decode) stays within the noise of the engine.
#[test]
#[ignore = "needs a GGUF of the v2 weights: set ARCHAI_JEV_TEST_V2_GGUF"]
fn the_cache_of_prefixes_changes_nothing() {
    let (Some(cases), Some(tokenizer), Ok(gguf)) = (
        load(),
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_V2_GGUF"),
    ) else {
        eprintln!("SKIPPED: set the tokenizer and ARCHAI_JEV_TEST_V2_GGUF");
        return;
    };
    let gguf = std::path::PathBuf::from(gguf);
    // the requests with several questions about one state, and a spread of the others
    let picked: Vec<&Case> = cases
        .iter()
        .enumerate()
        .filter(|(i, c)| c.questions.len() > 1 || i % 4 == 0)
        .map(|(_, c)| c)
        .collect();
    let remembering = scorer(&gguf, &tokenizer, 64 << 20);
    let amnesic = scorer(&gguf, &tokenizer, 1); // the segmented path, but nothing is kept
    let plain = scorer(&gguf, &tokenizer, 0);
    let mut worst_plain = 0.0f64;
    for case in &picked {
        let (state, qs) = request_to_domain(&case.request).unwrap();
        let reference = amnesic.run(&state, &qs).unwrap();
        let cold = remembering.run(&state, &qs).unwrap();
        let warm = remembering.run(&state, &qs).unwrap();
        assert_eq!(cold.questions, reference.questions, "{}: cold", case.id);
        assert_eq!(warm.questions, reference.questions, "{}: warm", case.id);
        let one_decode = plain.run(&state, &qs).unwrap();
        for ((_, a), (_, b)) in reference.questions.iter().zip(&one_decode.questions) {
            for (x, y) in softmax(a).iter().zip(softmax(b)) {
                worst_plain = worst_plain.max((x - y).abs());
            }
        }
    }
    let info = remembering.cache_info().unwrap();
    println!(
        "{} requests: cold = warm = amnesic bit for bit; hits {} misses {} entries {}; plain path within {worst_plain:.4} (in probability) of the segmented one",
        picked.len(),
        info.hits,
        info.misses,
        info.entries
    );
    assert!(info.hits >= 1, "the store was used");
    // the unit of the tolerances (probabilities); 0.05 is what spec 009 allows a variant against
    // its own recorded numbers, and a different split of the same row is the same kind of noise
    assert!(worst_plain < 0.05, "the plain path is {worst_plain} away");
}
