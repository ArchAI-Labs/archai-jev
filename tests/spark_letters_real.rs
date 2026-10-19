//! The `spark2_5-letters` family with real weights (spec 010b, test level T2: by hand,
//! `#[ignore]`, `--release`; see `spark_common` for the folders and the golden they read).
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_SPARK_DIR = "<local model folder>"            # GGUF + tokenizer + manifest
//! $env:ARCHAI_JEV_TEST_SPARK_DIR_OTHER = "<the folder of the other size>"
//! $env:ARCHAI_JEV_TEST_SPARK_GOLDEN = "<golden.jsonl>"
//! cargo test --release --features testing --test spark_letters_real -- --ignored --nocapture --test-threads 1
//! ```
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod spark_common;

use std::path::{Path, PathBuf};

use _core::Error;
use _core::calibration::calibrated_softmax;
use _core::hub::config::HubConfig;
use _core::hub::download::{Response, Transport};
use _core::hub::events::{NeverCancel, Recorder};
use _core::json_strict::{Json, parse};
use _core::models::hash::sha256_file;
use _core::models::llama_backend::LlamaBackend;
use _core::models::load::{Context, LoadRequest, LoadedModel, load_model};
use _core::models::registry::Registry;
use _core::models::resolve::{NameOrPath, Selection};
use _core::models::testing::jsonedit::{remove, set};
use _core::models::tolerance::{Reference, lookup};
use spark_common::{golden, numbers, path_of};

struct NoNetwork;

impl Transport for NoNetwork {
    fn get(&self, _: &str, _: Option<u64>, _: std::time::Duration) -> Result<Response, String> {
        Err("the network is disabled in this test".to_string())
    }
}

/// Load the model in `dir` through the whole real path (manifest, files, GGUF header, tokenizer,
/// engine, self-check), with a private cache so that the self-check always runs.
fn load(dir: &Path, cache_mib: u64) -> Result<LoadedModel, Error> {
    let cache = tempfile::tempdir().unwrap();
    let mut hub = HubConfig::new(cache.path().to_path_buf());
    hub.offline = true;
    let backend = LlamaBackend::new(_core::models::llama_backend::default_threads())
        .with_prefix_cache(cache_mib << 20);
    let ctx = Context {
        registry: Registry::builtin().unwrap(),
        hub: &hub,
        backend: &backend,
        transport: &NoNetwork,
        observer: &Recorder::default(),
        cancel: &NeverCancel,
        head_reader: None,
        converter: None,
    };
    let request = LoadRequest {
        selection: Selection {
            name_or_path: NameOrPath::Path(dir.to_path_buf()),
            revision: None,
            device: "cpu".to_string(),
            dtype: None,
            manifest: None,
        },
        temperature: None,
        allow_uncalibrated: true,
    };
    load_model(&request, &ctx)
}

fn dir() -> Option<PathBuf> {
    path_of("ARCHAI_JEV_TEST_SPARK_DIR", "a local model folder")
}

/// A copy of the folder `base` where the manifest is edited by `edit`; the (large) files are
/// hard-linked when the drive allows it.
fn mutated(base: &Path, edit: impl FnOnce(&mut Json)) -> tempfile::TempDir {
    let out = tempfile::tempdir().unwrap();
    for name in ["model.gguf", "tokenizer.json"] {
        let (from, to) = (base.join(name), out.path().join(name));
        if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to).unwrap();
        }
    }
    let bytes = std::fs::read(base.join("archai-jev-manifest.json")).unwrap();
    let mut manifest = parse(&bytes).unwrap();
    edit(&mut manifest);
    std::fs::write(
        out.path().join("archai-jev-manifest.json"),
        manifest.to_canonical_string(),
    )
    .unwrap();
    out
}

fn vector_path(i: usize, tail: &str) -> String {
    format!("variants.0.selfcheck.vectors.{i}.{tail}")
}

fn vector_count(manifest: &Json) -> usize {
    match manifest.get("variants") {
        Some(Json::Array(v)) => match v[0].get("selfcheck").and_then(|s| s.get("vectors")) {
            Some(Json::Array(vectors)) => vectors.len(),
            other => panic!("vectors: {other:?}"),
        },
        other => panic!("variants: {other:?}"),
    }
}

/// Apply `f` to every vector of the manifest, by its path.
fn each_vector(manifest: &mut Json, f: impl Fn(&mut Json, usize)) {
    for i in 0..vector_count(manifest) {
        f(manifest, i);
    }
}

fn reverse_numbers(manifest: &mut Json, path: &str) {
    let Some(Json::Array(items)) = get_mut(manifest, path) else {
        panic!("no list at {path}")
    };
    items.reverse();
}

fn get_mut<'a>(root: &'a mut Json, path: &str) -> Option<&'a mut Json> {
    let mut node = root;
    for part in path.split('.') {
        node = match node {
            Json::Object(pairs) => &mut pairs.iter_mut().find(|(k, _)| k == part)?.1,
            Json::Array(items) => items.get_mut(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(node)
}

fn question_names(manifest: &Json, i: usize) -> Vec<String> {
    let v = match manifest.get("variants") {
        Some(Json::Array(v)) => &v[0],
        other => panic!("variants: {other:?}"),
    };
    match v
        .get("selfcheck")
        .and_then(|s| s.get("vectors"))
        .and_then(|vs| match vs {
            Json::Array(a) => a.get(i),
            _ => None,
        })
        .and_then(|vec| vec.get("expected"))
        .and_then(|e| e.get("questions"))
    {
        Some(Json::Object(p)) => p.iter().map(|(k, _)| k.clone()).collect(),
        other => panic!("questions: {other:?}"),
    }
}

/// The self-check passes with the weights, and a manifest that is wrong in one of the ways the
/// library has to notice is refused (AC-13).
#[test]
#[ignore = "needs the model folder: ARCHAI_JEV_TEST_SPARK_DIR"]
fn the_selfcheck_passes_and_every_mutant_is_caught() {
    let Some(base) = dir() else { return };
    let started = std::time::Instant::now();
    let loaded = load(&base, 0).unwrap_or_else(|e| panic!("the genuine folder is refused: {e}"));
    eprintln!(
        "the genuine folder loads and passes its self-check in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(loaded.data.head, "answer_letters");
    assert_eq!(loaded.data.max_options, Some(26));
    assert!(!loaded.data.calibrated, "the model is not calibrated");
    assert_eq!(loaded.data.temperature, 1.0);
    drop(loaded);

    type Edit = Box<dyn Fn(&mut Json)>;
    let tokenizer_path = base.join("tokenizer.json");
    let space_letters: Vec<u32> = {
        let tk = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
        (0..26u8)
            .map(|i| {
                let ids = tk
                    .encode(format!(" {}", char::from(b'A' + i)), false)
                    .unwrap();
                assert_eq!(ids.get_ids().len(), 1, "a letter with a space is one token");
                ids.get_ids()[0]
            })
            .collect()
    };
    let mutants: Vec<(&str, Edit)> = vec![
        (
            "the template differs: one id of a prompt is another",
            Box::new(|m| {
                let at = vector_path(0, "expected.input_ids.20");
                let Some(Json::Int(n)) = get_mut(m, &at) else {
                    panic!("an id")
                };
                *n += 1;
            }),
        ),
        (
            "keys instead of descriptions in the request",
            Box::new(|m| {
                // the request of the first vector reads its options without descriptions
                let at = vector_path(0, "request.questions");
                let Some(Json::Object(qs)) = get_mut(m, &at) else {
                    panic!("questions")
                };
                for (_, q) in qs.iter_mut() {
                    if let Some(Json::Object(criteria)) = get_mut(q, "criteria") {
                        for (_, d) in criteria.iter_mut() {
                            *d = Json::Null;
                        }
                    }
                }
            }),
        ),
        (
            "the options in another order (the logits come reversed)",
            Box::new(|m| {
                each_vector(m, |m, i| {
                    for q in question_names(m, i) {
                        reverse_numbers(
                            m,
                            &vector_path(i, &format!("expected.questions.{q}.logits")),
                        );
                        reverse_numbers(
                            m,
                            &vector_path(i, &format!("expected.questions.{q}.probabilities")),
                        );
                    }
                });
            }),
        ),
        (
            "false and true swapped in the yes/no vectors, or any other normalisation: probabilities at another temperature",
            Box::new(|m| {
                each_vector(m, |m, i| {
                    for q in question_names(m, i) {
                        let at = vector_path(i, &format!("expected.questions.{q}.probabilities"));
                        let logits_at = vector_path(i, &format!("expected.questions.{q}.logits"));
                        let Some(Json::Array(l)) = get_mut(m, &logits_at).cloned() else {
                            panic!("logits")
                        };
                        let z: Vec<f64> = l
                            .iter()
                            .map(|v| match v {
                                Json::Float(x) => *x,
                                Json::Int(n) => *n as f64,
                                other => panic!("{other:?}"),
                            })
                            .collect();
                        let p = calibrated_softmax(&z, 0.35).unwrap();
                        *get_mut(m, &at).unwrap() =
                            Json::Array(p.into_iter().map(Json::Float).collect());
                    }
                });
            }),
        ),
        (
            "the letters have a leading space",
            Box::new(move |m| {
                for (i, id) in space_letters.iter().enumerate() {
                    set(
                        m,
                        &format!("head.letters.{i}.id"),
                        Json::Int(i128::from(*id)),
                    );
                }
            }),
        ),
        (
            "the letters are shifted by one",
            Box::new(|m| {
                for i in 0..26usize {
                    let at = format!("head.letters.{i}.id");
                    let Some(Json::Int(n)) = get_mut(m, &at) else {
                        panic!("an id")
                    };
                    *n += 1;
                }
            }),
        ),
        (
            "the sanitisation is incomplete: a control token is not listed",
            Box::new(|m| remove(m, "tokenizer.control_tokens.7")),
        ),
        (
            "the end of the prompt is not the token the answer is read at",
            Box::new(|m| {
                set(m, "tokenizer.special_tokens.think_end.id", Json::Int(3));
                set(
                    m,
                    "tokenizer.special_tokens.think_end.text",
                    Json::Str("<think>".into()),
                );
            }),
        ),
        (
            "the prompt is read for a longer context than the file has",
            Box::new(|m| set(m, "max_context", Json::Int(5_000_000))),
        ),
    ];
    let mut caught = 0usize;
    for (what, edit) in mutants {
        let folder = mutated(&base, edit);
        match load(folder.path(), 0) {
            Ok(_) => panic!("NOT CAUGHT: {what}"),
            Err(e @ (Error::ModelVerification(_) | Error::IncompatibleModel(_))) => {
                eprintln!(
                    "caught ({}): {what}\n    {e}",
                    if matches!(e, Error::ModelVerification(_)) {
                        "self-check"
                    } else {
                        "load checks"
                    }
                );
                caught += 1;
            }
            Err(other) => panic!("{what}: refused for another reason: {other}"),
        }
    }
    assert_eq!(caught, 9);
}

/// A GGUF of another size, or of another architecture, under this manifest is refused, even
/// when the manifest is made to agree with the file about its hash and size (AC-13).
#[test]
#[ignore = "needs two model folders: ARCHAI_JEV_TEST_SPARK_DIR and ARCHAI_JEV_TEST_SPARK_DIR_OTHER"]
fn a_gguf_under_the_wrong_manifest_is_refused() {
    let (Some(base), Some(other)) = (
        dir(),
        path_of(
            "ARCHAI_JEV_TEST_SPARK_DIR_OTHER",
            "the local model folder of the other size",
        ),
    ) else {
        return;
    };
    let mut wrong_files = vec![other.join("model.gguf")];
    if let Some(v1) = std::env::var_os("ARCHAI_JEV_TEST_DEFAULT_GGUF").map(PathBuf::from) {
        wrong_files.push(v1);
    }
    for gguf in wrong_files {
        // 1. the file is not the one the manifest names: its SHA-256 says so
        let folder = mutated(&base, |_| {});
        std::fs::remove_file(folder.path().join("model.gguf")).unwrap();
        std::fs::copy(&gguf, folder.path().join("model.gguf")).unwrap();
        let err = load(folder.path(), 0)
            .err()
            .expect("the wrong file under the right manifest");
        eprintln!("{}: refused by the hash: {err}", gguf.display());
        assert!(matches!(err, Error::IncompatibleModel(_)), "{err}");
        // 2. the manifest is made to agree with that file about hash and size: the header
        //    (architecture, layers, tensors) must say no
        let (sha, size) = sha256_file(&gguf).unwrap();
        let folder2 = mutated(&base, |m| {
            set(m, "variants.0.source.file.sha256", Json::Str(sha.clone()));
            set(
                m,
                "variants.0.source.file.size",
                Json::Int(i128::from(size)),
            );
        });
        std::fs::remove_file(folder2.path().join("model.gguf")).unwrap();
        std::fs::copy(&gguf, folder2.path().join("model.gguf")).unwrap();
        let err = load(folder2.path(), 0)
            .err()
            .expect("the wrong file with a matching manifest");
        eprintln!("{}: refused by the header: {err}", gguf.display());
        assert!(matches!(err, Error::IncompatibleModel(_)), "{err}");
    }
}

/// The probabilities of the engine against the reference runtime on the golden (AC-14).
#[test]
#[ignore = "needs the model folder and the golden: ARCHAI_JEV_TEST_SPARK_DIR, ARCHAI_JEV_TEST_SPARK_GOLDEN"]
fn the_engine_matches_the_reference() {
    let (Some(base), Some(rows)) = (dir(), golden()) else {
        return;
    };
    let tol = lookup("spark2_5-letters", "q8_0", Reference::FullPrecision)
        .unwrap()
        .p;
    let loaded = load(&base, 128).unwrap();
    let (mut worst_p, mut worst_logit) = (0.0f64, 0.0f64);
    let (mut questions, mut flips, mut near_ties) = (0usize, 0usize, 0usize);
    let mut worst_at = String::new();
    let mut too_long = 0usize;
    for row in &rows {
        let (state, qs) = row.domain().unwrap();
        // a request with a row over max_context is refused whole, never cut (AC-5); the golden
        // has a few such long ones, and the refusal must be exactly for those
        let longest = row
            .questions()
            .iter()
            .map(|(_, q)| spark_common::ints(q.get("tokens_llama").unwrap()).len())
            .max()
            .unwrap();
        let scored = match loaded.scorer.score(&state, &qs) {
            Ok(s) => {
                assert!(
                    longest as u64 <= loaded.data.max_context,
                    "{}: {longest} tokens were answered",
                    row.id
                );
                s
            }
            Err(Error::Unsupported(u)) => {
                assert!(
                    matches!(u, _core::prompt::Unsupported::RowTooLong { .. })
                        && longest as u64 > loaded.data.max_context,
                    "{}: refused ({u}) with a row of {longest} tokens",
                    row.id
                );
                too_long += 1;
                continue;
            }
            Err(e) => panic!("{}: {e}", row.id),
        };
        assert_eq!(scored.len(), row.questions().len());
        for ((name, theirs), ours) in row.questions().iter().zip(&scored) {
            let ref_logits = numbers(theirs.get("letter_logits_direct").unwrap());
            let ref_p = match theirs.get("probabilities_T1") {
                Some(Json::Object(p)) => p
                    .iter()
                    .map(|(_, v)| match v {
                        Json::Float(x) => *x,
                        other => panic!("{other:?}"),
                    })
                    .collect::<Vec<f64>>(),
                other => panic!("probabilities_T1: {other:?}"),
            };
            let our_p = calibrated_softmax(ours, 1.0).unwrap();
            assert_eq!(our_p.len(), ref_p.len(), "{}/{name}", row.id);
            questions += 1;
            for (a, b) in our_p.iter().zip(&ref_p) {
                if (a - b).abs() > worst_p {
                    worst_p = (a - b).abs();
                    worst_at = format!("{}/{name}", row.id);
                }
            }
            for (a, b) in ours.iter().zip(&ref_logits) {
                worst_logit = worst_logit.max((a - b).abs());
            }
            let top = |p: &[f64]| {
                p.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, _)| i)
                    .unwrap()
            };
            if top(&our_p) != top(&ref_p) {
                flips += 1;
                let mut sorted = ref_p.clone();
                sorted.sort_by(|x, y| y.total_cmp(x));
                if sorted[0] - sorted[1] <= tol {
                    near_ties += 1;
                }
            }
        }
    }
    eprintln!("{too_long} requests refused for being longer than max_context");
    eprintln!(
        "{questions} questions; largest |dp| {worst_p:.4} at {worst_at} (tolerance {tol}); largest |dlogit| {worst_logit:.3}; argmax flips {flips} (near ties {near_ties})"
    );
    assert!(questions >= 60);
    assert!(
        worst_p <= tol,
        "|dp| {worst_p} at {worst_at} is over the tolerance {tol}"
    );
    assert_eq!(flips, near_ties, "an argmax flip that is not a near tie");
}

// ---- the cache of prefixes with a sliding window (AC-15, spec 007) ---------------------------

use _core::scorer::Scorer;
use _core::{Choice, ChoiceOption, Questions, State, StateValue, TextEntry};

fn long_state(entries: usize) -> State {
    let text = (0..entries)
        .map(|i| {
            format!(
                "Entry {i}: the shipment {} left depot {} on day {}.",
                1000 + i * 7,
                i % 13,
                i % 28 + 1
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    State::new(StateValue::string(text)).unwrap()
}

fn three_questions() -> Questions {
    let q = |instructions: &str| -> _core::Question {
        Choice::new(
            TextEntry::text(instructions).unwrap(),
            vec![
                ChoiceOption::described("a", "The shipment left on an early day of the month")
                    .unwrap(),
                ChoiceOption::described("b", "The shipment left on a late day of the month")
                    .unwrap(),
                ChoiceOption::described("c", "The report does not say").unwrap(),
            ],
        )
        .unwrap()
        .into()
    };
    Questions::new([
        ("one", q("When did the first shipment leave?")),
        ("two", q("When did the last shipment leave?")),
        ("three", q("When did a shipment from depot 3 leave?")),
    ])
    .unwrap()
}

/// How many tokens the shared part of the request (up to the end of the state) has.
fn shared_tokens(model: &spark_common::Model, state: &State, qs: &Questions) -> usize {
    let prompt = model.build(state, qs).unwrap();
    prompt.row_cuts(0).unwrap().last().unwrap().pos
}

/// Cold, warm and without memory give the same bits, past the window, with several questions
/// about one state (AC-15).
#[test]
#[ignore = "needs the model folder: ARCHAI_JEV_TEST_SPARK_DIR"]
fn cold_warm_and_no_cache_agree_past_the_window() {
    let Some(model) = spark_common::Model::open("ARCHAI_JEV_TEST_SPARK_DIR") else {
        return;
    };
    let window = 512u64;
    let bare = model.scorer(Some(window), 1);
    let kept = model.scorer(Some(window), 256 << 20);
    let qs = three_questions();
    for entries in [2usize, 6, 12, 25, 40, 60, 70] {
        let st = long_state(entries);
        let n = shared_tokens(&model, &st, &qs);
        let reference = bare.score(&st, &qs).unwrap();
        let before = kept.cache_info().unwrap();
        let cold = kept.score(&st, &qs).unwrap();
        let warm = kept.score(&st, &qs).unwrap();
        let after = kept.cache_info().unwrap();
        assert_eq!(cold, reference, "cold, {n} shared tokens");
        assert_eq!(warm, reference, "warm, {n} shared tokens");
        eprintln!(
            "{n:>5} shared tokens: cold = warm = bare; entries {} -> {}, hits {} -> {}",
            before.entries, after.entries, before.hits, after.hits
        );
        if n as u64 > window {
            // nothing of this state was kept: only the short fixed head can be
            assert!(
                after.entries <= before.entries.max(1),
                "a prefix past the window was saved"
            );
        }
    }
}

/// Where does the identity of a restored snapshot end? Without the limit of the family, compare
/// cold, warm and bare for shared parts from 440 to 830 tokens (the spike found it between 512
/// and 830; the limit of the family is the window, 512). Prints the table; asserts that
/// everything up to the window is identical.
#[test]
#[ignore = "needs the model folder: ARCHAI_JEV_TEST_SPARK_DIR; takes minutes"]
fn the_boundary_of_identity_is_the_window() {
    let Some(model) = spark_common::Model::open("ARCHAI_JEV_TEST_SPARK_DIR") else {
        return;
    };
    let bare = model.scorer(None, 1);
    let qs = three_questions();
    let mut first_difference: Option<usize> = None;
    for entries in 16..=30 {
        let st = long_state(entries);
        let n = shared_tokens(&model, &st, &qs);
        if !(440..=830).contains(&n) {
            continue;
        }
        let kept = model.scorer(None, 256 << 20); // a fresh memory per point
        let reference = bare.score(&st, &qs).unwrap();
        let cold = kept.score(&st, &qs).unwrap();
        let warm = kept.score(&st, &qs).unwrap();
        let worst = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| {
            a.iter()
                .flatten()
                .zip(b.iter().flatten())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        };
        let (c, w) = (worst(&cold, &reference), worst(&warm, &reference));
        eprintln!("{n:>4} shared tokens: cold differs {c:e}, warm differs {w:e}");
        if n <= 512 {
            assert_eq!(
                (c, w),
                (0.0, 0.0),
                "not identical at {n} tokens, inside the window"
            );
        } else if w != 0.0 && first_difference.is_none() {
            first_difference = Some(n);
        }
    }
    eprintln!("first shared length past the window with a difference: {first_difference:?}");
}
