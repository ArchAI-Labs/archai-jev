//! Shared by the by-hand tests of the `spark2_5-letters` family (spec 010b, levels T1 and T2).
//!
//! Nothing here is a file of the model: the tests read, from the computer of whoever runs them,
//! a **local model folder** (`ARCHAI_JEV_TEST_SPARK_DIR`: GGUF, `tokenizer.json` and
//! `archai-jev-manifest.json`) and a **golden** (`ARCHAI_JEV_TEST_SPARK_GOLDEN`: requests of ours
//! with the ids and the letter logits of the reference runtime). Both live outside the
//! repository. The tests are `#[ignore]`d (CI never runs them); run them with `--ignored`.
//! A variable that is missing makes the test say so and pass, unless
//! `ARCHAI_JEV_REQUIRE_ASSETS=1`, which makes it fail.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;

use std::sync::Arc;

use _core::Calibration;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::heads::AnswerLettersHead;
use _core::json_big::parse_with_big_ints;
use _core::json_strict::Json;
use _core::model_scorer::ModelScorer;
use _core::models::manifest::{Manifest, ManifestOrigin};
use _core::prompt::template::{TemplateInputs, instantiate};
use _core::prompt::{NoLimits, PromptTokenizer, RequestLimits, Roles, Template};
use _core::request::request_to_domain;
use _core::{Error, Questions, State};

pub const TYPES: [&str; 2] = ["choice", "noul"];

fn require() -> bool {
    std::env::var("ARCHAI_JEV_REQUIRE_ASSETS").is_ok_and(|v| v == "1")
}

/// The path in `variable`, or `None` (after saying so) when it is missing and not required.
pub fn path_of(variable: &str, help: &str) -> Option<PathBuf> {
    match std::env::var(variable) {
        Ok(p) if !p.is_empty() && PathBuf::from(&p).exists() => Some(PathBuf::from(p)),
        _ => {
            assert!(!require(), "{variable} must be set: {help}");
            eprintln!("SKIPPED: {variable} is missing: {help}");
            None
        }
    }
}

/// A local model folder, read: the manifest, the tokenizer and the template it names.
pub struct Model {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub tokenizer: PromptTokenizer,
    pub template: Box<dyn Template>,
}

impl Model {
    pub fn open(variable: &str) -> Option<Model> {
        let dir = path_of(
            variable,
            "a local model folder of the spark2_5-letters family",
        )?;
        let bytes = std::fs::read(dir.join("archai-jev-manifest.json")).expect("the manifest");
        let manifest =
            Manifest::from_bytes(&bytes, ManifestOrigin::Local).expect("a valid manifest");
        let roles = Roles::new(
            manifest
                .tokenizer
                .special_tokens
                .iter()
                .map(|(role, t)| (role.clone(), t.id)),
        );
        let tokenizer = PromptTokenizer::from_file(&dir.join(&manifest.tokenizer.file.path), roles)
            .expect("tokenizer");
        let template = instantiate(
            &manifest.template.id,
            manifest.template.version,
            &TemplateInputs {
                tasks: None,
                max_options: 26,
                max_key_tokens: 1,
                control_tokens: &manifest.tokenizer.control_tokens,
            },
        )
        .expect("the template of the manifest");
        Some(Model {
            dir,
            manifest,
            tokenizer,
            template,
        })
    }

    pub fn build(
        &self,
        state: &State,
        questions: &Questions,
    ) -> Result<_core::prompt::Prompt, Error> {
        let limits = RequestLimits {
            model: "spark-test",
            max_context: u64::MAX,
            question_types: &TYPES,
            support: &NoLimits,
        };
        self.template
            .build(state, questions, &self.tokenizer, &limits)
    }

    /// A scorer of this model made by hand, so a test can choose what the library chooses for
    /// itself: the longest prefix kept between requests (`snapshot_limit`) and the memory for
    /// them (`cache_bytes`; 1 keeps nothing but runs the same path with cuts).
    pub fn scorer(&self, snapshot_limit: Option<u64>, cache_bytes: u64) -> ModelScorer {
        let roles = Roles::new(
            self.manifest
                .tokenizer
                .special_tokens
                .iter()
                .map(|(role, t)| (role.clone(), t.id)),
        );
        let tokenizer =
            PromptTokenizer::from_file(&self.dir.join(&self.manifest.tokenizer.file.path), roles)
                .expect("tokenizer");
        let template = instantiate(
            &self.manifest.template.id,
            self.manifest.template.version,
            &TemplateInputs {
                tasks: None,
                max_options: 26,
                max_key_tokens: 1,
                control_tokens: &self.manifest.tokenizer.control_tokens,
            },
        )
        .expect("the template");
        let forward = LlamaForward::load(
            &self.dir.join("model.gguf"),
            LlamaConfig {
                threads: _core::models::llama_backend::default_threads(),
                max_context: 2048,
                hidden_states: false,
            },
        )
        .expect("the model loads");
        let head = match &self.manifest.head {
            _core::models::manifest::HeadSpec::AnswerLetters {
                letters,
                max_options,
                ..
            } => AnswerLettersHead::new(letters, usize::try_from(*max_options).unwrap()),
            other => panic!("not an answer-letters head: {other:?}"),
        };
        ModelScorer::new(
            "spark-test",
            template,
            tokenizer,
            Arc::new(forward),
            Box::new(head),
            Calibration::new(1.0, false).unwrap(),
            2048,
            TYPES.iter().map(|t| (*t).to_string()).collect(),
        )
        .with_snapshot_limit(snapshot_limit)
        .with_prefix_cache(cache_bytes)
    }

    /// The engine alone, for a test that times a plain prefill.
    pub fn forward(&self) -> LlamaForward {
        LlamaForward::load(
            &self.dir.join("model.gguf"),
            LlamaConfig {
                threads: _core::models::llama_backend::default_threads(),
                max_context: 2048,
                hidden_states: false,
            },
        )
        .expect("the model loads")
    }

    /// Letter ids of the manifest, `A` first.
    pub fn letters(&self) -> Vec<u32> {
        match &self.manifest.head {
            _core::models::manifest::HeadSpec::AnswerLetters { letters, .. } => {
                letters.iter().map(|t| t.id).collect()
            }
            other => panic!("not an answer-letters head: {other:?}"),
        }
    }
}

/// One row of the golden: our request and what the reference gave for each of its questions.
pub struct Row {
    pub id: String,
    pub json: Json,
}

impl Row {
    fn field(&self, key: &str) -> &Json {
        self.json
            .get(key)
            .unwrap_or_else(|| panic!("golden field {key}"))
    }

    /// Our request, as written by `make_requests.py`.
    pub fn ours(&self) -> &Json {
        self.field("ours")
    }

    /// The question results of the reference, by name.
    pub fn questions(&self) -> &[(String, Json)] {
        match self.field("questions") {
            Json::Object(p) => p,
            other => panic!("questions: {other:?}"),
        }
    }

    /// The request in the domain types.
    pub fn domain(&self) -> Result<(State, Questions), String> {
        request_to_domain(&typesafe(self.ours()))
    }
}

pub fn golden() -> Option<Vec<Row>> {
    golden_from("ARCHAI_JEV_TEST_SPARK_GOLDEN")
}

/// The golden of requests with control tokens in their text (reference numbers for the text
/// after our neutralisation; see `make_inject.py` in the spike folder).
pub fn golden_inject() -> Option<Vec<Row>> {
    golden_from("ARCHAI_JEV_TEST_SPARK_GOLDEN_INJECT")
}

fn golden_from(variable: &str) -> Option<Vec<Row>> {
    let path = path_of(
        variable,
        "a golden of the reference (a JSONL file of requests of ours with ids and letter logits)",
    )?;
    let text = std::fs::read_to_string(path).expect("the golden");
    Some(
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let json = parse_with_big_ints(l.as_bytes()).expect("a JSON line");
                let id = match json.get("id") {
                    Some(Json::Str(s)) => s.clone(),
                    other => panic!("id: {other:?}"),
                };
                Row { id, json }
            })
            .collect(),
    )
}

/// What comparing our prompts with the ids of the reference found.
#[derive(Default)]
pub struct Comparison {
    pub compared: usize,
    pub refused: Vec<String>,
    pub different: Vec<String>,
    pub fixed_cuts: usize,
    pub content_cuts: usize,
}

/// Build every request of `rows` and compare the ids with the reference's, question by question.
pub fn compare_ids(model: &Model, rows: &[Row]) -> Comparison {
    let mut c = Comparison::default();
    for row in rows {
        let built = row.domain().and_then(|(state, questions)| {
            model.build(&state, &questions).map_err(|e| e.to_string())
        });
        let prompt = match built {
            Ok(p) => p,
            Err(e) => {
                c.refused.push(format!("{}: {e}", row.id));
                continue;
            }
        };
        for ((name, theirs), branch) in row.questions().iter().zip(prompt.branches()) {
            let reference = ints(theirs.get("tokens_llama").expect("ids of the reference"));
            c.compared += 1;
            if branch.ids() != reference.as_slice() {
                let at = branch
                    .ids()
                    .iter()
                    .zip(&reference)
                    .position(|(a, b)| a != b)
                    .unwrap_or(branch.len().min(reference.len()));
                c.different.push(format!(
                    "{}/{name}: {} ids against {}, first difference at {at}",
                    row.id,
                    branch.len(),
                    reference.len()
                ));
            }
            let count = |kind| branch.cuts().iter().filter(|cut| cut.kind == kind).count();
            c.fixed_cuts += count(_core::prompt::CutKind::Fixed);
            c.content_cuts += count(_core::prompt::CutKind::Content);
        }
    }
    c
}

fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Our request (`kind`, `options`, `true`/`false`) in the TypeSafe JSON shape.
fn typesafe(ours: &Json) -> Json {
    let questions = match ours.get("questions") {
        Some(Json::Object(p)) => p,
        other => panic!("questions: {other:?}"),
    };
    let converted = questions
        .iter()
        .map(|(name, q)| {
            let kind = match q.get("kind") {
                Some(Json::Str(k)) => k.as_str(),
                other => panic!("kind: {other:?}"),
            };
            let instructions = q.get("instructions").cloned().unwrap_or(Json::Null);
            let converted = if kind == "choice" {
                let criteria = match q.get("options") {
                    Some(Json::Array(options)) => options
                        .iter()
                        .map(|pair| match pair {
                            Json::Array(kv) => match (kv.first(), kv.get(1)) {
                                (Some(Json::Str(k)), Some(d)) => (k.clone(), d.clone()),
                                other => panic!("option: {other:?}"),
                            },
                            other => panic!("option: {other:?}"),
                        })
                        .collect(),
                    other => panic!("options: {other:?}"),
                };
                obj(vec![
                    ("type", Json::Str("choice".into())),
                    ("instructions", instructions),
                    ("criteria", Json::Object(criteria)),
                ])
            } else {
                let mut criteria = Vec::new();
                for key in ["true", "false"] {
                    match q.get(key) {
                        None | Some(Json::Null) => {}
                        Some(d) => criteria.push((key.to_string(), d.clone())),
                    }
                }
                let mut pairs = vec![
                    ("type", Json::Str("noul".into())),
                    ("instructions", instructions),
                ];
                if !criteria.is_empty() {
                    pairs.push(("criteria", Json::Object(criteria)));
                }
                obj(pairs)
            };
            (name.clone(), converted)
        })
        .collect();
    obj(vec![
        ("state", ours.get("state").cloned().unwrap_or(Json::Null)),
        ("questions", Json::Object(converted)),
    ])
}

/// A list of integers of the golden.
pub fn ints(value: &Json) -> Vec<u32> {
    match value {
        Json::Array(items) => items
            .iter()
            .map(|i| match i {
                Json::Int(n) => u32::try_from(*n).expect("an id"),
                other => panic!("an integer, got {other:?}"),
            })
            .collect(),
        other => panic!("a list, got {other:?}"),
    }
}

/// A list of numbers of the golden.
pub fn numbers(value: &Json) -> Vec<f64> {
    match value {
        Json::Array(items) => items
            .iter()
            .map(|i| match i {
                Json::Int(n) => *n as f64,
                Json::Float(x) => *x,
                other => panic!("a number, got {other:?}"),
            })
            .collect(),
        other => panic!("a list, got {other:?}"),
    }
}

pub fn text<'a>(value: &'a Json, what: &str) -> &'a str {
    match value {
        Json::Str(s) => s,
        other => panic!("{what}: {other:?}"),
    }
}
