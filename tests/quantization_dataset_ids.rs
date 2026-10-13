//! The JSONL datasets of the quantization study build, in this library, exactly the prompt ids the
//! user's training formatters produce (spec 009, T7; level T2, by hand).
//!
//! `benches/quantization/export_dataset.py` writes `val.jsonl`, `test.jsonl` and `external.jsonl`
//! (requests with a `label` per question) and, next to each, `*.ids.jsonl` with the ids of the
//! training formatter for the same example. This test builds the prompt of every request with the
//! library (template, sanitisation, NFC, tokenizer) and compares. A record where the library gives
//! other ids would be scored on a different prompt than the one the model was trained and
//! calibrated on, so each difference is listed and the test fails.
//!
//! ```text
//! $env:ARCHAI_JEV_TEST_EVAL_DATA = "<the OUT_DIR of the export>"
//! $env:ARCHAI_JEV_TEST_DEFAULT_TOKENIZER = "<base tokenizer.json>"
//! cargo test --features testing --test quantization_dataset_ids -- --ignored --nocapture
//! ```
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use _core::json_strict::{Json, parse};
use _core::models::testing::assets;
use _core::models::testing::golden;
use _core::prompt::chatml::ChatMlLetters;
use _core::prompt::tasks::{self, default_tasks};
use _core::prompt::{PromptTokenizer, RequestLimits, Roles, Template};

const TYPES: [&str; 2] = ["choice", "noul"];

fn roles() -> Roles {
    Roles::new([("im_start", 151_644u32), ("im_end", 151_645)])
}

fn template() -> ChatMlLetters {
    ChatMlLetters::new(Some(tasks::parse(&default_tasks(), 4).unwrap()))
}

/// The request of a record: no `id`, no `group`, no `label` (the loader of the library reads them,
/// the prompt builder does not want them).
fn request_of(record: &Json) -> Json {
    let Json::Object(pairs) = record else {
        panic!("a record is an object")
    };
    let mut out = Vec::new();
    for (key, value) in pairs {
        match (key.as_str(), value) {
            ("id" | "group", _) => {}
            ("questions", Json::Object(questions)) => {
                let cleaned = questions
                    .iter()
                    .map(|(name, q)| {
                        let Json::Object(fields) = q else {
                            panic!("a question is an object")
                        };
                        let kept = fields
                            .iter()
                            .filter(|(k, _)| k != "label")
                            .cloned()
                            .collect();
                        (name.clone(), Json::Object(kept))
                    })
                    .collect();
                out.push((key.clone(), Json::Object(cleaned)));
            }
            _ => out.push((key.clone(), value.clone())),
        }
    }
    Json::Object(out)
}

/// The state of a record as text (the datasets of the study have text states).
fn state_text(record: &Json) -> String {
    match record.get("state") {
        Some(Json::Str(s)) => s.clone(),
        other => panic!("a state that is not text: {other:?}"),
    }
}

fn read_lines(path: &PathBuf) -> Vec<Json> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| parse(l.as_bytes()).unwrap())
        .collect()
}

fn ids_of(row: &Json) -> (String, Vec<u32>) {
    let Some(Json::Str(id)) = row.get("id") else {
        panic!("ids row without id")
    };
    let Some(Json::Array(ids)) = row.get("ids") else {
        panic!("ids row without ids")
    };
    let ids = ids
        .iter()
        .map(|v| match v {
            Json::Int(i) => u32::try_from(*i).unwrap(),
            other => panic!("not an id: {other:?}"),
        })
        .collect();
    (id.clone(), ids)
}

#[test]
#[ignore = "needs the exported datasets (ARCHAI_JEV_TEST_EVAL_DATA) and the base tokenizer"]
fn the_library_builds_the_ids_of_the_training_formatters() {
    let Ok(dir) = std::env::var("ARCHAI_JEV_TEST_EVAL_DATA") else {
        eprintln!(
            "SKIPPED: set ARCHAI_JEV_TEST_EVAL_DATA to the output folder of export_dataset.py"
        );
        return;
    };
    let Some(tokenizer) = assets::default_tokenizer() else {
        return;
    };
    let dir = PathBuf::from(dir);
    let tok = PromptTokenizer::from_file(&tokenizer, roles()).unwrap();
    let template = template();
    let limits = RequestLimits {
        model: "default",
        max_context: 512,
        question_types: &TYPES,
        support: &_core::prompt::NoLimits,
    };

    let mut total = 0usize;
    let mut problems: Vec<String> = Vec::new();
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    let mut sanitised: Vec<String> = Vec::new();
    for name in ["test", "val", "external"] {
        let records = read_lines(&dir.join(format!("{name}.jsonl")));
        let wanted: BTreeMap<String, Vec<u32>> = read_lines(&dir.join(format!("{name}.ids.jsonl")))
            .iter()
            .map(ids_of)
            .collect();
        assert_eq!(
            records.len(),
            wanted.len(),
            "{name}: records and ids differ in number"
        );
        for record in &records {
            total += 1;
            let Some(Json::Str(id)) = record.get("id") else {
                panic!("{name}: a record without id")
            };
            let request = request_of(record);
            let (state, questions) = match golden::request_to_domain(&request) {
                Ok(x) => x,
                Err(e) => {
                    *by_kind
                        .entry("the library refuses the request")
                        .or_default() += 1;
                    problems.push(format!("{id}: refused: {e}"));
                    continue;
                }
            };
            let prompt = match template.build(&state, &questions, &tok, &limits) {
                Ok(p) => p,
                Err(e) => {
                    *by_kind.entry("the prompt cannot be built").or_default() += 1;
                    problems.push(format!("{id}: {e}"));
                    continue;
                }
            };
            let got = prompt.branches()[0].ids();
            let want = &wanted[id];
            if got != want.as_slice() {
                let at = got.iter().zip(want).position(|(a, b)| a != b);
                // The one explained difference: the library neutralises every `<|name|>` in a
                // state (D36, Kev's sanitisation, hole included), the training formatter does
                // not. Anything else is a real difference.
                if state_text(record).contains("<|") {
                    sanitised.push(format!("{id} (first difference at {at:?})"));
                    continue;
                }
                *by_kind.entry("other ids").or_default() += 1;
                problems.push(format!(
                    "{id}: ids differ (library {} tokens, formatter {}; first difference at {at:?})",
                    got.len(),
                    want.len()
                ));
            }
        }
        println!("{name}: {} records checked", records.len());
    }
    println!(
        "total {total}; identical or explained {}; sanitised special-looking text: {sanitised:?}; problems {}: {by_kind:?}",
        total - problems.len(),
        problems.len()
    );
    for p in problems.iter().take(40) {
        println!("  {p}");
    }
    assert!(
        problems.is_empty(),
        "{} of {total} records do not give the formatter's ids: {by_kind:?}",
        problems.len()
    );
}
