//! The default model (spec 006c): the prompt ids against the training formatters (test level T1,
//! needs the base `tokenizer.json`) and the real engine against the PyTorch oracle (level T2,
//! needs the 1.65 GB GGUF: `cargo test --features testing --test default_model -- --ignored`).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use _core::Calibration;
use _core::calibration::calibrated_softmax;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::heads::LettersHead;
use _core::model_scorer::ModelScorer;
use _core::models::manifest::Target;
use _core::models::testing::assets;
use _core::models::testing::golden::{self, DefaultCase};
use _core::prompt::chatml::ChatMlLetters;
use _core::prompt::tasks::{self, default_tasks};
use _core::prompt::{NoLimits, PromptTokenizer, RequestLimits, Roles, Template};

const TYPES: [&str; 2] = ["choice", "noul"];

fn roles() -> Roles {
    Roles::new([("im_start", 151_644u32), ("im_end", 151_645)])
}

fn template() -> ChatMlLetters {
    ChatMlLetters::new(Some(tasks::parse(&default_tasks(), 4).unwrap()))
}

fn target(label: &str, id: u32) -> Target {
    Target {
        label: label.to_string(),
        id,
    }
}

fn head() -> LettersHead {
    LettersHead::new(
        &[
            target("A", 32),
            target("B", 33),
            target("C", 34),
            target("D", 35),
        ],
        Some(&(target("FALSE", 30_351), target("TRUE", 20_611))),
    )
}

#[test]
fn prompt_ids_equal_the_training_formatters() {
    let Some(path) = assets::default_tokenizer() else {
        return;
    };
    let tok = PromptTokenizer::from_file(&path, roles()).unwrap();
    let limits = RequestLimits {
        model: "default",
        max_context: 512,
        question_types: &TYPES,
        support: &NoLimits,
    };
    let cases = golden::load_default();
    assert_eq!(cases.len(), 48);
    for case in cases {
        let (state, questions) =
            golden::request_to_domain(&case.request).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let prompt = template()
            .build(&state, &questions, &tok, &limits)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(prompt.state().len(), 0, "{}", case.id);
        assert_eq!(
            prompt.branches()[0].ids(),
            case.ids.as_slice(),
            "{}: the ids differ from the training formatter",
            case.id
        );
    }
}

fn restricted_probs(logits: &[f64], t: f64) -> Vec<f64> {
    calibrated_softmax(logits, t).unwrap()
}

/// The real engine on the real model: `|Δp|` against the PyTorch oracle within the tolerance of
/// the Q8_0 file (D29: 0.2), and the argmax unless the oracle's top two are closer than 0.25.
#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF and the default tokenizer"]
fn the_engine_matches_the_oracle_on_the_default_model() {
    let (Some(tokenizer), Some(gguf)) = (assets::default_tokenizer(), assets::default_gguf())
    else {
        return;
    };
    let tok = PromptTokenizer::from_file(&tokenizer, roles()).unwrap();
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
    let scorer = ModelScorer::new(
        "default",
        Box::new(template()),
        tok,
        Arc::new(forward),
        Box::new(head()),
        Calibration::new(1.0, false).unwrap(),
        512,
        TYPES.iter().map(|t| (*t).to_string()).collect(),
    );
    let mut worst = 0.0f64;
    let mut flips = 0;
    let started = std::time::Instant::now();
    let cases: Vec<DefaultCase> = golden::load_default();
    for case in &cases {
        let (state, questions) = golden::request_to_domain(&case.request).unwrap();
        let run = scorer.run(&state, &questions).unwrap();
        assert_eq!(run.input_ids, case.ids, "{}: ids", case.id);
        let mine = restricted_probs(&run.questions[0].1, 1.0);
        let theirs = restricted_probs(&case.oracle_logits, 1.0);
        let dp = mine
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        worst = worst.max(dp);
        let top = |p: &[f64]| {
            p.iter()
                .enumerate()
                .fold(0, |b, (i, v)| if *v > p[b] { i } else { b })
        };
        if top(&mine) != top(&theirs) {
            let mut sorted = theirs.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let gap = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
            assert!(
                gap < 0.25,
                "{}: argmax differs with a clear gap {gap}",
                case.id
            );
            flips += 1;
        }
        assert!(
            dp <= 0.2,
            "{}: |dp| = {dp} exceeds the Q8_0 tolerance of 0.2",
            case.id
        );
        eprintln!(
            "{:<14} dp={dp:.4} oracle={theirs:?} engine={mine:?}",
            case.id
        );
    }
    eprintln!(
        "max |dp| = {worst:.4} over {} prompts, {flips} argmax flips (all near ties), {:?} in total",
        cases.len(),
        started.elapsed()
    );
}
