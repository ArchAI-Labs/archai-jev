//! Reader of the Kev golden fixture (`tests/data/kev-golden/`) and a converter from the golden's
//! TypeSafe-shaped requests to the domain types. Test support only (feature `testing`).
//!
//! The converter is deliberately **strict like the library** (D30, D32): a request Kev accepts
//! but the library refuses (a state that is a bare number, instructions that are `null`, unknown
//! keys in a yes/no `criteria`) comes back as `Err(reason)`; the tests list those on purpose.

use std::path::{Path, PathBuf};

use crate::json_strict::{Json, parse};
use crate::schema::{Questions, State, StateValue};

/// What the golden says about one question.
#[derive(Debug, Clone)]
pub struct GoldenQuestion {
    /// Question name.
    pub name: String,
    /// `choice`, `noul` or `score`.
    pub kind: String,
    /// Start of the branch inside the packed ids.
    pub branch_start: usize,
    /// Length of the branch.
    pub branch_len: usize,
    /// Packed position of the `decide` token.
    pub decide_pos: usize,
    /// Packed positions of the `box_end` tokens.
    pub opt_pos: Vec<usize>,
    /// Raw logits of the head (before the temperature).
    pub logits_raw: Vec<f64>,
    /// Logits divided by the temperature.
    pub logits_tempered: Vec<f64>,
    /// Probabilities, float32 unrounded.
    pub probs: Vec<f64>,
}

/// One request of the golden with Kev's answer.
#[derive(Debug, Clone)]
pub struct GoldenCase {
    /// Case id.
    pub id: String,
    /// The request as written.
    pub request: Json,
    /// Packed prompt ids.
    pub ids: Vec<u32>,
    /// Number of leading ids that are the state (delimiter included).
    pub state_len: usize,
    /// Checkpoint temperature.
    pub temperature: f64,
    /// `usage.input_tokens` of Kev.
    pub usage: usize,
    /// One entry per question, in request order.
    pub questions: Vec<GoldenQuestion>,
}

/// Directory of the fixture.
pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("kev-golden")
}

fn num_vec(j: &Json) -> Vec<f64> {
    match j {
        Json::Array(items) => items
            .iter()
            .map(|v| match v {
                Json::Float(f) => *f,
                Json::Int(i) => *i as f64,
                _ => f64::NAN,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn idx_vec(j: &Json) -> Vec<usize> {
    num_vec(j).into_iter().map(|f| f as usize).collect()
}

fn usize_of(j: Option<&Json>) -> usize {
    match j {
        Some(Json::Int(i)) => *i as usize,
        _ => 0,
    }
}

/// Marks an integer too big for `json_strict` (which reads at most 64 bits) so that it survives
/// parsing; `value_of` turns it back into an exact integer. Test support only.
const BIG_INT: &str = "@@bigint@@:";

fn protect_big_ints(line: &str) -> String {
    let mut out = String::with_capacity(line.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
        } else if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
        } else if c == '-' || c.is_ascii_digit() {
            let mut j = i;
            while chars
                .get(j)
                .is_some_and(|d| d.is_ascii_digit() || "+-.eE".contains(*d))
            {
                j += 1;
            }
            let token: String = chars.get(i..j).unwrap_or_default().iter().collect();
            let digits = token.strip_prefix('-').unwrap_or(&token);
            if digits.len() >= 19 && digits.chars().all(|d| d.is_ascii_digit()) {
                out.push('"');
                out.push_str(BIG_INT);
                out.push_str(&token);
                out.push('"');
            } else {
                out.push_str(&token);
            }
            i = j;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Read a `.jsonl` file of the fixture.
///
/// # Panics
/// If the file is missing or malformed: the fixture is part of the repository.
pub fn load(file: &str) -> Vec<GoldenCase> {
    let path = data_dir().join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let row = parse(protect_big_ints(line).as_bytes()).expect("golden line is valid JSON");
            let get = |k: &str| row.get(k).unwrap_or_else(|| panic!("golden field {k}"));
            let Json::Str(id) = get("id") else {
                panic!("id")
            };
            let ids = idx_vec(get("ids")).into_iter().map(|i| i as u32).collect();
            let questions = match get("questions") {
                Json::Array(qs) => qs
                    .iter()
                    .map(|q| {
                        let field = |k: &str| q.get(k).unwrap_or(&Json::Null);
                        let text = |k: &str| match field(k) {
                            Json::Str(s) => s.clone(),
                            _ => String::new(),
                        };
                        GoldenQuestion {
                            name: text("id"),
                            kind: text("type"),
                            branch_start: usize_of(q.get("branch_start")),
                            branch_len: usize_of(q.get("branch_len")),
                            decide_pos: usize_of(q.get("decide_pos")),
                            opt_pos: idx_vec(field("opt_pos")),
                            logits_raw: num_vec(field("logits_raw")),
                            logits_tempered: num_vec(field("logits_tempered")),
                            probs: num_vec(field("probs")),
                        }
                    })
                    .collect(),
                _ => panic!("questions"),
            };
            let temperature = match get("temperature") {
                Json::Float(f) => *f,
                other => panic!("temperature {other:?}"),
            };
            GoldenCase {
                id: id.clone(),
                request: get("request").clone(),
                ids,
                state_len: usize_of(row.get("state_len")),
                temperature,
                usage: usize_of(row.get("usage_input_tokens")),
                questions,
            }
        })
        .collect()
}

/// Convert a golden request into the domain types, or say why the library refuses it.
///
/// # Errors
/// The reason, as text.
pub fn request_to_domain(request: &Json) -> Result<(State, Questions), String> {
    crate::request::request_to_domain_with(request, &|s| {
        s.strip_prefix(BIG_INT)
            .and_then(|digits| StateValue::int_text(digits).ok())
    })
}

/// One prompt of the default-model golden with the oracle's logits.
#[derive(Debug, Clone)]
pub struct DefaultCase {
    /// Case id.
    pub id: String,
    /// `safety`, `intent`, `entailment` or `similarity`.
    pub task: String,
    /// The request as written.
    pub request: Json,
    /// Ids of the whole prompt, as the training formatter builds them.
    pub ids: Vec<u32>,
    /// Token ids the logits were read at.
    pub answer_ids: Vec<u32>,
    /// PyTorch fp32 logits of the model at `answer_ids`.
    pub oracle_logits: Vec<f64>,
}

/// Directory of the default-model golden.
pub fn default_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("default-golden")
}

/// Read `tests/data/default-golden/golden.jsonl`.
///
/// # Panics
/// If the file is missing or malformed: it is part of the repository.
pub fn load_default() -> Vec<DefaultCase> {
    let path = default_data_dir().join("golden.jsonl");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let row = parse(line.as_bytes()).expect("golden line is valid JSON");
            let get = |k: &str| row.get(k).unwrap_or_else(|| panic!("golden field {k}"));
            let text = |k: &str| match get(k) {
                Json::Str(s) => s.clone(),
                other => panic!("{k}: {other:?}"),
            };
            DefaultCase {
                id: text("id"),
                task: text("task"),
                request: get("request").clone(),
                ids: idx_vec(get("ids")).into_iter().map(|i| i as u32).collect(),
                answer_ids: idx_vec(get("answer_ids"))
                    .into_iter()
                    .map(|i| i as u32)
                    .collect(),
                oracle_logits: num_vec(get("oracle_logits")),
            }
        })
        .collect()
}
