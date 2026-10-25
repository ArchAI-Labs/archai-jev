//! The default model (v2, Q8_0) on a GPU: the real engine on the named device against the reference
//! of the Decision Index kit (fp32), with the tolerance row that the library holds for that
//! backend (spec 011, level T2-GPU). The tests do nothing without a GPU named in the environment.
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_DEFAULT_TOKENIZER = "<base tokenizer.json>"
//! $env:ARCHAI_JEV_TEST_V2_GGUF = "<q8_0 gguf of the v2 weights>"
//! $env:ARCHAI_JEV_TEST_DEVICE = "cuda:0"      # or vulkan:1, the names available_devices() prints
//! cargo test --release --features "testing cuda" --test gpu_parity -- --ignored --nocapture
//! ```
//!
//! * `parity_and_selfcheck_on_the_device`: the scores agree with the reference within the row of
//!   the backend (a backend with no row is refused, as the library refuses it), and the same
//!   request twice gives the same numbers.
//! * `the_prefix_cache_on_the_device`: the identity of the memory of prefixes (spec 007) on the
//!   device: which of its guarantees hold there is printed, and the ones that must hold are asserted.
//! * `sixteen_callers_on_one_slot`: the invariants of the pool (spec 008) with `device=`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use _core::Calibration;
use _core::Scorer;
use _core::calibration::calibrated_softmax;
use _core::engine::device::{BackendKind, DeviceInfo, DeviceProbe, DeviceRequest, OffloadPlan};
use _core::engine::llama::{LlamaConfig, LlamaForward, LlamaProbe};
use _core::heads::KeyLogProbHead;
use _core::json_strict::{Json, parse};
use _core::model_scorer::ModelScorer;
use _core::models::testing::assets;
use _core::models::tolerance::{Reference, lookup};
use _core::prompt::di::DecisionIndexChatMl;
use _core::prompt::{PromptTokenizer, Roles};
use _core::request::request_to_domain;
use _core::schema::{Questions, State};

const MAX_KEY_TOKENS: usize = 8;
const MAX_OPTIONS: usize = 48;
const MAX_CONTEXT: u32 = 1024;
/// The temperature of the default model's manifest: the self-check compares probabilities at the
/// declared temperature, so the rows of the tolerance table are in that unit.
const DECLARED_TEMPERATURE: f64 = 1.7497;

struct Case {
    id: String,
    state: State,
    questions: Questions,
    /// The reference scores, per question in order.
    scores: Vec<Vec<f64>>,
}

struct Setup {
    gguf: PathBuf,
    tokenizer: PathBuf,
    device: DeviceInfo,
    cases: Vec<Case>,
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

fn cases() -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/v2-golden/golden.jsonl");
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let row = parse(line.as_bytes()).unwrap();
            let Some(Json::Str(id)) = row.get("id") else {
                panic!("id")
            };
            let Some(Json::Object(qs)) = row.get("questions") else {
                panic!("questions")
            };
            let (state, questions) = request_to_domain(row.get("request").unwrap()).unwrap();
            Case {
                id: id.clone(),
                state,
                questions,
                scores: qs
                    .iter()
                    .map(|(_, q)| f64s(q.get("scores").unwrap()))
                    .collect(),
            }
        })
        .collect()
}

/// What the environment names, or `None` (and a line saying so) if a test cannot run.
fn setup() -> Option<Setup> {
    let (Some(tokenizer), Ok(gguf), Ok(name)) = (
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_V2_GGUF"),
        std::env::var("ARCHAI_JEV_TEST_DEVICE"),
    ) else {
        eprintln!(
            "SKIPPED: set ARCHAI_JEV_TEST_DEFAULT_TOKENIZER, ARCHAI_JEV_TEST_V2_GGUF and \
             ARCHAI_JEV_TEST_DEVICE (for example cuda:0)"
        );
        return None;
    };
    let DeviceRequest::Named { backend, index } = DeviceRequest::parse(&name).unwrap() else {
        panic!("name a GPU device, not {name:?}");
    };
    let devices = LlamaProbe.devices();
    let device = devices
        .iter()
        .find(|d| d.backend == backend && d.index == index)
        .unwrap_or_else(|| panic!("{name} is not among {devices:?}"))
        .clone();
    eprintln!("on {} ({})", device.name(), device.description);
    Some(Setup {
        gguf: PathBuf::from(gguf),
        tokenizer,
        device,
        cases: cases(),
    })
}

fn forward(gguf: &Path, plan: OffloadPlan, k: usize) -> Arc<LlamaForward> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    Arc::new(
        LlamaForward::load(
            gguf,
            LlamaConfig {
                threads,
                max_context: MAX_CONTEXT,
                hidden_states: false,
                plan,
            },
        )
        .unwrap()
        .with_key_scoring(MAX_KEY_TOKENS as u32)
        .with_max_concurrency(k),
    )
}

fn scorer(forward: Arc<LlamaForward>, tokenizer: &Path, cache_bytes: u64) -> Arc<ModelScorer> {
    Arc::new(
        ModelScorer::new(
            "v2",
            Box::new(DecisionIndexChatMl::new(MAX_KEY_TOKENS)),
            PromptTokenizer::from_file(
                tokenizer,
                Roles::new([("im_start", 151_644u32), ("im_end", 151_645)]),
            )
            .unwrap(),
            forward,
            Box::new(KeyLogProbHead::new(MAX_OPTIONS, MAX_KEY_TOKENS)),
            Calibration::new(1.0, false).unwrap(),
            u64::from(MAX_CONTEXT),
            vec!["choice".to_string(), "noul".to_string()],
        )
        .with_prefix_cache(cache_bytes),
    )
}

fn probs(scores: &[f64], temperature: f64) -> Vec<f64> {
    calibrated_softmax(scores, temperature).unwrap()
}

fn largest_gap(a: &[f64], b: &[f64], temperature: f64) -> f64 {
    probs(a, temperature)
        .iter()
        .zip(probs(b, temperature))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn top(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold(0, |b, (i, v)| if *v > p[b] { i } else { b })
}

#[test]
#[ignore = "needs a GPU, the v2 GGUF and its tokenizer: see the header of this file"]
fn parity_and_selfcheck_on_the_device() {
    let Some(s) = setup() else { return };
    // the row the library holds for this backend: without one the model is not allowed to answer
    let row = lookup(
        "qwen2-keys",
        "q8_0",
        Reference::FullPrecision,
        s.device.backend,
    )
    .unwrap_or_else(|| panic!("{} has no tolerance row: not admitted", s.device.name()));
    assert_ne!(s.device.backend, BackendKind::Cpu);

    let scorer = scorer(
        forward(&s.gguf, OffloadPlan::for_device(&s.device), 1),
        &s.tokenizer,
        0,
    );
    let (mut worst, mut worst_case, mut flips, mut questions) = (0.0f64, String::new(), 0, 0);
    for case in &s.cases {
        let run = scorer.run(&case.state, &case.questions).unwrap();
        let again = scorer.run(&case.state, &case.questions).unwrap();
        assert_eq!(
            run.questions, again.questions,
            "{}: not repeatable",
            case.id
        );
        for ((name, mine), want) in run.questions.iter().zip(&case.scores) {
            assert_eq!(mine.len(), want.len(), "{} {name}", case.id);
            assert!(mine.iter().all(|x| x.is_finite()), "{} {name}", case.id);
            let gap = largest_gap(mine, want, DECLARED_TEMPERATURE);
            if gap > worst {
                worst = gap;
                worst_case = format!("{} {name}", case.id);
            }
            let (pm, pw) = (
                probs(mine, DECLARED_TEMPERATURE),
                probs(want, DECLARED_TEMPERATURE),
            );
            let mut sorted = pw.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            if top(&pm) != top(&pw) && sorted[0] - sorted[1] >= 0.25 {
                flips += 1;
            }
            questions += 1;
        }
    }
    println!(
        "{}: {questions} questions, max |dp| {worst:.4} ({worst_case}) at T = {DECLARED_TEMPERATURE}, row {}, flips beyond a 0.25 margin {flips}",
        s.device.name(),
        row.p
    );
    assert!(questions >= 80);
    assert!(
        worst <= row.p,
        "{}: max |dp| {worst} is beyond the row {}",
        s.device.name(),
        row.p
    );
    assert_eq!(flips, 0);
}

/// The memory of prefixes on the device. Cold, warm and a store that remembers nothing run the
/// same segments, so they must be identical bit for bit as on the CPU; the plain path (the row in
/// one decode) splits the work differently and is only close, as on the CPU.
#[test]
#[ignore = "needs a GPU, the v2 GGUF and its tokenizer: see the header of this file"]
fn the_prefix_cache_on_the_device() {
    let Some(s) = setup() else { return };
    let forward = forward(&s.gguf, OffloadPlan::for_device(&s.device), 1);
    let remembering = scorer(forward.clone(), &s.tokenizer, 64 << 20);
    let amnesic = scorer(forward.clone(), &s.tokenizer, 1);
    let plain = scorer(forward, &s.tokenizer, 0);
    let picked: Vec<&Case> = s
        .cases
        .iter()
        .enumerate()
        .filter(|(i, c)| c.questions.len() > 1 || i % 4 == 0)
        .map(|(_, c)| c)
        .collect();
    let mut worst_plain = 0.0f64;
    for case in &picked {
        let reference = amnesic.run(&case.state, &case.questions).unwrap();
        let cold = remembering.run(&case.state, &case.questions).unwrap();
        let warm = remembering.run(&case.state, &case.questions).unwrap();
        assert_eq!(cold.questions, reference.questions, "{}: cold", case.id);
        assert_eq!(warm.questions, reference.questions, "{}: warm", case.id);
        let one_decode = plain.run(&case.state, &case.questions).unwrap();
        for ((_, a), (_, b)) in reference.questions.iter().zip(&one_decode.questions) {
            worst_plain = worst_plain.max(largest_gap(a, b, 1.0));
        }
    }
    let info = remembering.cache_info().unwrap();
    println!(
        "{}: {} requests, cold = warm = amnesic bit for bit; hits {}; plain path within {worst_plain:.4}",
        s.device.name(),
        picked.len(),
        info.hits
    );
    assert!(info.hits >= 1, "the store was used");
    assert!(worst_plain < 0.05, "the plain path is {worst_plain} away");
}

/// Sixteen threads over a pool of one slot: every answer equals the one given to a single caller
/// bit for bit, and the threads of the contexts that run together never exceed the budget.
#[test]
#[ignore = "needs a GPU, the v2 GGUF and its tokenizer: see the header of this file"]
fn sixteen_callers_on_one_slot() {
    let Some(s) = setup() else { return };
    let forward = forward(&s.gguf, OffloadPlan::for_device(&s.device), 1);
    let scorer = scorer(forward.clone(), &s.tokenizer, 0);
    let cases = Arc::new(s.cases);
    let single: Vec<_> = cases
        .iter()
        .map(|c| scorer.run(&c.state, &c.questions).unwrap().questions)
        .collect();
    let handles: Vec<_> = (0..16)
        .map(|w| {
            let (scorer, cases) = (Arc::clone(&scorer), Arc::clone(&cases));
            thread::spawn(move || {
                (w..cases.len())
                    .step_by(16)
                    .map(|i| (i, scorer.run(&cases[i].state, &cases[i].questions).unwrap()))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let mut served = 0;
    for handle in handles {
        for (i, run) in handle.join().unwrap() {
            assert_eq!(run.questions, single[i], "{}: another answer", cases[i].id);
            served += 1;
        }
    }
    assert_eq!(served, cases.len());
    let stats = forward.pool_stats();
    println!(
        "{} peak threads {} of {}",
        s.device.name(),
        stats.peak_threads,
        stats.thread_budget
    );
    assert!(stats.peak_threads <= stats.thread_budget);
    assert_eq!(stats.max_concurrency, 1);
}
