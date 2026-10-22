//! The self-check vectors of a quantised variant, recorded by the engine on the very file
//! (spec 009, section 4), and what they are worth. The Q4_K_M of the v2 was NOT admitted because
//! of the numbers these tests print on other platforms (D60); they stay as the evidence and as the
//! way to measure the next candidate.
//!
//! Three steps, all by hand (T2, `#[ignore]`) because they need the GGUF of the variant:
//!
//! 1. `record_vectors_of_the_engine` runs the requests of the registry's `q8_0` variant through
//!    the engine on the Q4_K_M file and writes `{id, input_ids, scores}` lines. A script turns
//!    them into a manifest (`benches/quantization/make_v2_manifest.py --engine-vectors`).
//! 2. `the_numbers_do_not_depend_on_the_threads` repeats the run with 1, 4 and the machine's
//!    threads and prints the largest difference in probability: the noise the tolerance has to
//!    cover on one machine.
//! 3. `the_recorded_vectors_reproduce` reads the lines committed in `tests/data/quantization/` and
//!    prints the largest difference: run it on another platform with the same file to measure the
//!    cross-platform difference (the manual workflow `quantized-vectors.yml` does it).
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_DEFAULT_TOKENIZER = "<base tokenizer.json>"
//! $env:ARCHAI_JEV_TEST_V2_GGUF = "<q4_k_m.gguf>"
//! $env:ARCHAI_JEV_TEST_ENGINE_VECTORS = "<lines.jsonl>"     # step 1 writes it; step 3 defaults to the committed one
//! cargo test --release --features testing --test quantized_vectors -- --include-ignored --nocapture --test-threads 1
//! ```
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use _core::Calibration;
use _core::calibration::calibrated_softmax;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::heads::KeyLogProbHead;
use _core::json_strict::{Json, parse};
use _core::model_scorer::ModelScorer;
use _core::models::testing::assets;
use _core::prompt::di::DecisionIndexChatMl;
use _core::prompt::{PromptTokenizer, Roles};
use _core::request::request_to_domain;

const MAX_KEY_TOKENS: usize = 4;
const MAX_OPTIONS: usize = 20;
const MAX_CONTEXT: u32 = 512;

/// The vectors of the registry's `q8_0` variant that the Q4_K_M variant also carries: both kinds
/// of question (`choice` with 2 to 6 options, `noul`) and short to long states.
const PICKED: [&str; 12] = [
    "v2-000", "v2-001", "v2-003", "v2-004", "v2-005", "v2-006", "v2-009", "v2-012", "v2-013",
    "v2-014", "v2-016", "v2-018",
];

fn registry_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/models/registry/nickprock__archai-jev-qwen-1.5b@7730248d.json")
}

/// The requests of the picked vectors, in order, with their ids.
fn requests() -> Vec<(String, Json)> {
    let text = std::fs::read_to_string(registry_manifest()).unwrap();
    let manifest = parse(text.as_bytes()).unwrap();
    let Some(Json::Array(variants)) = manifest.get("variants") else {
        panic!("variants")
    };
    let q8 = variants
        .iter()
        .find(|v| matches!(v.get("dtype"), Some(Json::Str(d)) if d == "q8_0"))
        .unwrap();
    let Some(Json::Array(vectors)) = q8.get("selfcheck").and_then(|s| s.get("vectors")) else {
        panic!("vectors")
    };
    PICKED
        .iter()
        .map(|id| {
            let v = vectors
                .iter()
                .find(|v| matches!(v.get("id"), Some(Json::Str(s)) if s == id))
                .unwrap_or_else(|| panic!("{id} is not a vector of the q8_0 variant"));
            ((*id).to_string(), v.get("request").unwrap().clone())
        })
        .collect()
}

/// The cache of prefixes the library gives a model by default (128 MiB): the numbers of a
/// variant are the ones of the path with cuts, which is what a default `Jev` runs.
const DEFAULT_CACHE: u64 = 128 << 20;

fn scorer(gguf: &Path, tokenizer: &Path, threads: i32) -> ModelScorer {
    scorer_with_cache(gguf, tokenizer, threads, DEFAULT_CACHE)
}

fn scorer_with_cache(gguf: &Path, tokenizer: &Path, threads: i32, cache: u64) -> ModelScorer {
    let roles = Roles::new([("im_start", 151_644u32), ("im_end", 151_645)]);
    let tok = PromptTokenizer::from_file(tokenizer, roles).unwrap();
    let forward = LlamaForward::load(
        gguf,
        LlamaConfig {
            threads,
            max_context: MAX_CONTEXT,
            hidden_states: false,
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
    .with_prefix_cache(cache)
}

fn machine_threads() -> i32 {
    std::thread::available_parallelism().map_or(4, |n| n.get() as i32)
}

struct Inputs {
    gguf: PathBuf,
    tokenizer: PathBuf,
}

fn inputs() -> Option<Inputs> {
    let (Some(tokenizer), Ok(gguf)) = (
        assets::default_tokenizer(),
        std::env::var("ARCHAI_JEV_TEST_V2_GGUF"),
    ) else {
        eprintln!("SKIPPED: set the tokenizer and ARCHAI_JEV_TEST_V2_GGUF");
        return None;
    };
    Some(Inputs {
        gguf: PathBuf::from(gguf),
        tokenizer,
    })
}

/// The raw scores of every question of `request`, in order (one question per vector here).
fn scores(scorer: &ModelScorer, request: &Json) -> Vec<(String, Vec<f64>)> {
    let (state, qs) = request_to_domain(request).unwrap();
    scorer.run(&state, &qs).unwrap().questions
}

/// Largest `|dp|` between two sets of scores (probabilities at T = 1).
fn gap(a: &[f64], b: &[f64]) -> f64 {
    let (pa, pb) = (
        calibrated_softmax(a, 1.0).unwrap(),
        calibrated_softmax(b, 1.0).unwrap(),
    );
    pa.iter()
        .zip(&pb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn lines_path() -> Option<PathBuf> {
    std::env::var("ARCHAI_JEV_TEST_ENGINE_VECTORS")
        .ok()
        .map(PathBuf::from)
}

#[test]
#[ignore = "needs the Q4_K_M GGUF: set ARCHAI_JEV_TEST_V2_GGUF and ARCHAI_JEV_TEST_ENGINE_VECTORS"]
fn record_vectors_of_the_engine() {
    let (Some(inp), Some(out)) = (inputs(), lines_path()) else {
        return;
    };
    let scorer = scorer(&inp.gguf, &inp.tokenizer, machine_threads());
    let mut text = String::new();
    for (id, request) in requests() {
        let (state, qs) = request_to_domain(&request).unwrap();
        let run = scorer.run(&state, &qs).unwrap();
        assert_eq!(run.questions.len(), 1, "{id}: one question per vector");
        let (name, s) = &run.questions[0];
        let join = |items: &[String]| items.join(",");
        text.push_str(&format!(
            "{{\"id\":\"{id}\",\"question\":{name:?},\"input_ids\":[{}],\"scores\":[{}]}}\n",
            join(&run.input_ids.iter().map(u32::to_string).collect::<Vec<_>>()),
            // `{:?}` of an f64 is the shortest text that reads back as the same number
            join(&s.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>()),
        ));
    }
    std::fs::write(&out, text).unwrap();
    println!("{} vectors written to {}", PICKED.len(), out.display());
}

#[test]
#[ignore = "needs the Q4_K_M GGUF: set ARCHAI_JEV_TEST_V2_GGUF"]
fn the_numbers_do_not_depend_on_the_threads() {
    let Some(inp) = inputs() else { return };
    let reqs = requests();
    let reference = scorer(&inp.gguf, &inp.tokenizer, machine_threads());
    let base: Vec<_> = reqs.iter().map(|(_, r)| scores(&reference, r)).collect();
    // same scorer twice: the run-to-run noise on the same threads
    let again: Vec<_> = reqs.iter().map(|(_, r)| scores(&reference, r)).collect();
    let mut worst = Vec::new();
    for (label, other) in [
        ("same threads, second run", again),
        ("1 thread", run_all(&inp, 1, &reqs)),
        ("4 threads", run_all(&inp, 4, &reqs)),
    ] {
        let d = base
            .iter()
            .zip(&other)
            .map(|(a, b)| gap(&a[0].1, &b[0].1))
            .fold(0.0, f64::max);
        println!("{label}: max |dp| = {d:.6}");
        worst.push(d);
    }
    // the tolerance of an engine-recorded variant is at most 0.05 (spec 009 section 4); on one
    // machine the threads must stay far below it
    assert!(worst.iter().all(|d| *d < 0.01), "{worst:?}");
}

fn run_all(inp: &Inputs, threads: i32, reqs: &[(String, Json)]) -> Vec<Vec<(String, Vec<f64>)>> {
    let s = scorer(&inp.gguf, &inp.tokenizer, threads);
    reqs.iter().map(|(_, r)| scores(&s, r)).collect()
}

#[test]
#[ignore = "needs the Q4_K_M GGUF: set ARCHAI_JEV_TEST_V2_GGUF"]
fn the_recorded_vectors_reproduce() {
    let Some(inp) = inputs() else { return };
    // the lines committed with the repository, unless another file is given
    let path = lines_path().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/quantization/q4_k_m_engine_vectors.jsonl")
    });
    let recorded: Vec<Json> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| parse(l.as_bytes()).unwrap())
        .collect();
    assert_eq!(recorded.len(), PICKED.len());
    let scorer = scorer(&inp.gguf, &inp.tokenizer, machine_threads());
    let reqs = requests();
    let mut worst = 0.0f64;
    for ((id, request), line) in reqs.iter().zip(&recorded) {
        let Some(Json::Array(want)) = line.get("scores") else {
            panic!("scores")
        };
        let want: Vec<f64> = want
            .iter()
            .map(|v| match v {
                Json::Float(f) => *f,
                Json::Int(i) => *i as f64,
                other => panic!("{other:?}"),
            })
            .collect();
        let got = scores(&scorer, request);
        let d = gap(&got[0].1, &want);
        println!("{id}: max |dp| = {d:.6}");
        worst = worst.max(d);
    }
    println!("the largest difference from the recorded numbers: {worst:.6}");
    // one line a script can read (the manual CI job collects it per platform)
    println!(
        "CROSS_PLATFORM os={} arch={} max_dp={worst:.6}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    assert!(worst < 0.05, "{worst}");
}

/// The row decoded in one piece (no cache) and in the segments of its cuts (the default) are two
/// different computations of the same numbers: this prints how far apart they are on this file.
#[test]
#[ignore = "needs a GGUF: set ARCHAI_JEV_TEST_V2_GGUF"]
fn the_plain_path_and_the_segmented_path_agree() {
    let Some(inp) = inputs() else { return };
    let threads = machine_threads();
    let plain = scorer_with_cache(&inp.gguf, &inp.tokenizer, threads, 0);
    let cut = scorer_with_cache(&inp.gguf, &inp.tokenizer, threads, DEFAULT_CACHE);
    let mut worst = 0.0f64;
    for (id, request) in requests() {
        let d = gap(&scores(&plain, &request)[0].1, &scores(&cut, &request)[0].1);
        println!("{id}: plain vs segmented max |dp| = {d:.6}");
        worst = worst.max(d);
    }
    println!("plain vs segmented: largest |dp| = {worst:.6}");
}

/// What the self-check of the **default model** does at load, on this platform: the 16 vectors of
/// its registry entry (expected numbers from the reference engine in full precision) run through
/// the engine on the published Q8_0 file, in the segments of the template's cuts, with the
/// tolerance of the library's table. Prints the largest difference of probability at the declared
/// temperature (the quantity the check compares) and fails if it is beyond the tolerance.
#[test]
#[ignore = "needs the Q8_0 GGUF of the default model: set ARCHAI_JEV_TEST_V2_GGUF"]
fn the_selfcheck_of_the_default_passes_on_this_platform() {
    use _core::engine::device::BackendKind;
    use _core::models::tolerance::{Reference, lookup};
    let Some(inp) = inputs() else { return };
    let text = std::fs::read_to_string(registry_manifest()).unwrap();
    let manifest = parse(text.as_bytes()).unwrap();
    let Some(Json::Array(variants)) = manifest.get("variants") else {
        panic!("variants")
    };
    let variant = variants
        .iter()
        .find(|v| matches!(v.get("dtype"), Some(Json::Str(d)) if d == "q8_0"))
        .unwrap();
    let number = |j: &Json| match j {
        Json::Float(f) => *f,
        Json::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    };
    let temperature = number(
        variant
            .get("calibration")
            .unwrap()
            .get("temperature")
            .unwrap(),
    );
    let Some(Json::Array(vectors)) = variant.get("selfcheck").and_then(|s| s.get("vectors")) else {
        panic!("vectors")
    };
    let tolerance = lookup(
        "qwen2-keys",
        "q8_0",
        Reference::FullPrecision,
        BackendKind::Cpu,
    )
    .unwrap()
    .p;
    let scorer = scorer(&inp.gguf, &inp.tokenizer, machine_threads());
    let mut worst = 0.0f64;
    for vector in vectors {
        let Some(Json::Str(id)) = vector.get("id") else {
            panic!("id")
        };
        let request = vector.get("request").unwrap();
        let expected = vector.get("expected").unwrap();
        let (state, qs) = request_to_domain(request).unwrap();
        let run = scorer.run(&state, &qs).unwrap();
        let Some(Json::Array(want_ids)) = expected.get("input_ids") else {
            panic!("input_ids")
        };
        let want_ids: Vec<u32> = want_ids
            .iter()
            .map(|v| u32::try_from(number(v) as i64).unwrap())
            .collect();
        assert_eq!(run.input_ids, want_ids, "{id}: the prompt ids differ");
        let Some(Json::Object(questions)) = expected.get("questions") else {
            panic!("questions")
        };
        for ((name, got), (want_name, want)) in run.questions.iter().zip(questions) {
            assert_eq!(name, want_name, "{id}");
            let Some(Json::Array(want_p)) = want.get("probabilities") else {
                panic!("probabilities")
            };
            let want_p: Vec<f64> = want_p.iter().map(number).collect();
            let p = calibrated_softmax(got, temperature).unwrap();
            let d = p
                .iter()
                .zip(&want_p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            println!("{id}: max |dp| = {d:.6}");
            worst = worst.max(d);
        }
    }
    println!(
        "SELFCHECK_DEFAULT os={} arch={} vectors={} max_dp={worst:.6} tolerance={tolerance}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        vectors.len()
    );
    assert!(
        worst <= tolerance,
        "{worst} is beyond the tolerance {tolerance}"
    );
}
