//! Reading the JSONL dataset format of Kev (spec 017 §1.2): one request of the TypeSafe shape per
//! line, plus a `label` for every question, and the optional `id` and `group`.
//!
//! Every line is read as strictly as the library reads anything: RFC 8259 JSON, no `NaN`, no
//! repeated keys, no BOM. Every field the format does not have is an error with the list of the
//! names, never ignored; every error names the file and the line.

use crate::error::{Error, Result};
use crate::json_big::parse_with_big_ints;
use crate::json_strict::{Json, JsonProblem};
use crate::request::{question_of, value_of};
use crate::schema::{Questions, State};

use super::dataset::{Dataset, LabelledRecord, RawLabel, check_id, not_an_id};
use super::problem::DatasetProblem;

const RECORD_FIELDS: [&str; 4] = ["state", "questions", "id", "group"];
const QUESTION_FIELDS: [&str; 4] = ["type", "instructions", "criteria", "label"];

fn dataset_error(problem: DatasetProblem) -> Error {
    Error::Dataset(problem)
}

/// The text of what a parser said, without its generic prefix.
fn detail(problem: &JsonProblem) -> String {
    let text = problem.to_string();
    text.strip_prefix("invalid JSON: ")
        .map_or_else(|| text.clone(), str::to_string)
}

/// A label as the JSON gave it.
fn raw_label(j: &Json) -> RawLabel {
    match j {
        Json::Str(s) => RawLabel::Str(s.clone()),
        Json::Bool(b) => RawLabel::Bool(*b),
        Json::Int(n) => RawLabel::Int(*n),
        other => RawLabel::Other(other.to_canonical_string()),
    }
}

fn unknown_fields(pairs: &[(String, Json)], allowed: &[&str]) -> Vec<String> {
    pairs
        .iter()
        .filter(|(k, _)| !allowed.contains(&k.as_str()))
        .map(|(k, _)| k.clone())
        .collect()
}

fn text_field(place: &str, field: &'static str, j: Option<&Json>) -> Result<Option<String>> {
    match j {
        None => Ok(None),
        Some(Json::Str(s)) => check_id(place, field, Some(s)),
        Some(_) => Err(not_an_id(place, field)),
    }
}

/// Read one record from a parsed JSON value. `place` is `file:line` or `record n`.
///
/// # Errors
/// A [`DatasetProblem`], or the state or question error of the domain types wrapped with the
/// place (so it keeps its category).
pub fn record_from_json(json: &Json, place: &str) -> Result<LabelledRecord> {
    let Json::Object(pairs) = json else {
        return Err(dataset_error(DatasetProblem::NotAnObject {
            place: place.to_string(),
            got: json.kind().to_string(),
        }));
    };
    let mut unknown = unknown_fields(pairs, &RECORD_FIELDS);
    if let Some(Json::Object(qs)) = json.get("questions") {
        for (_, q) in qs {
            if let Json::Object(fields) = q {
                unknown.extend(unknown_fields(fields, &QUESTION_FIELDS));
            }
        }
    }
    if !unknown.is_empty() {
        return Err(dataset_error(DatasetProblem::UnsupportedFields {
            place: place.to_string(),
            names: unknown,
        }));
    }
    let (Some(state_json), Some(Json::Object(question_pairs))) =
        (json.get("state"), json.get("questions"))
    else {
        return Err(dataset_error(DatasetProblem::MissingKeys {
            place: place.to_string(),
        }));
    };
    if question_pairs.is_empty() {
        return Err(dataset_error(DatasetProblem::MissingKeys {
            place: place.to_string(),
        }));
    }
    let id = text_field(place, "id", json.get("id"))?;
    let group = text_field(place, "group", json.get("group"))?;
    let no_hook = |_: &str| None;
    let state = value_of(state_json, &no_hook)
        .and_then(State::new)
        .map_err(|e| Error::at(place, e))?;
    let mut items = Vec::with_capacity(question_pairs.len());
    let mut labels = Vec::with_capacity(question_pairs.len());
    for (name, q) in question_pairs {
        items.push((
            name.clone(),
            question_of(name, q, &no_hook).map_err(|e| Error::at(place, e))?,
        ));
        match q.get("label") {
            Some(label) => labels.push((name.clone(), raw_label(label))),
            None => {
                return Err(dataset_error(DatasetProblem::MissingLabel {
                    place: place.to_string(),
                    question: name.clone(),
                }));
            }
        }
    }
    let questions = Questions::new(items).map_err(|e| Error::at(place, e))?;
    LabelledRecord::build(place, state, questions, &labels, id, group)
}

/// Read a whole JSONL file (the bytes), or fail without returning anything.
///
/// # Errors
/// A [`DatasetProblem`] with the file and line, or a state or question error with the same
/// prefix.
pub fn read_jsonl(bytes: &[u8], file: &str) -> Result<Dataset> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(dataset_error(DatasetProblem::ByteOrderMark {
            file: file.to_string(),
        }));
    }
    let mut records = Vec::new();
    for (index, raw_line) in bytes.split(|b| *b == b'\n').enumerate() {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line.iter().all(|b| matches!(b, b' ' | b'\t')) {
            continue;
        }
        let place = format!("{file}:{}", index + 1);
        let json = parse_with_big_ints(line).map_err(|p| {
            dataset_error(DatasetProblem::NotJson {
                place: place.clone(),
                detail: detail(&p),
            })
        })?;
        records.push(record_from_json(&json, &place)?);
    }
    Dataset::new(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Category;
    use crate::evaluation::dataset::Label;
    use proptest::prelude::*;

    const SPEC_LINE: &str = r#"{"state": "My running shoes arrived in the wrong size.", "id": "t-17", "group": "customer-4", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"returns": "Exchanges", "billing": null}, "label": "returns"}, "urgent": {"type": "noul", "instructions": "Is this urgent?", "label": true}, "priority": {"type": "score", "instructions": "How important?", "criteria": ["low", "medium", "high"], "label": 2}}}"#;

    fn read(text: &str) -> Result<Dataset> {
        read_jsonl(text.as_bytes(), "data.jsonl")
    }

    fn error_text(text: &str) -> String {
        read(text).unwrap_err().to_string()
    }

    #[test]
    fn spec_example() {
        let d = read(SPEC_LINE).unwrap();
        let r = &d.records()[0];
        assert_eq!(
            r.labels,
            vec![Label::Choice(0), Label::YesNo(true), Label::Score(2)]
        );
        assert_eq!(r.id.as_deref(), Some("t-17"));
        assert_eq!(r.group.as_deref(), Some("customer-4"));
        assert_eq!(r.place, "data.jsonl:1");
        assert_eq!(r.questions.len(), 3);
    }

    #[test]
    fn kev_example() {
        // the record of the docstring of kev.data.load_records
        let line = r#"{"state":"ticket text","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges","shipping":"Delays"},"label":"billing"},"urgent":{"type":"noul","instructions":"Urgent?","label":false},"priority":{"type":"score","instructions":"Priority?","criteria":["low","medium","high"],"label":2}}}"#;
        let d = read(line).unwrap();
        assert_eq!(
            d.records()[0].labels,
            vec![Label::Choice(0), Label::YesNo(false), Label::Score(2)]
        );
        assert_eq!(d.records()[0].id, None);
    }

    #[test]
    fn blank_lines_and_crlf() {
        let text = format!("\r\n{SPEC_LINE}\r\n   \r\n\t\r\n{SPEC_LINE}\r\n");
        let d = read(&text).unwrap();
        assert_eq!(d.records().len(), 2);
        assert_eq!(d.records()[1].place, "data.jsonl:5");
    }

    #[test]
    fn bom_is_an_error() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(SPEC_LINE.as_bytes());
        assert_eq!(
            read_jsonl(&bytes, "data.jsonl").unwrap_err().to_string(),
            "data.jsonl:1: the file starts with a byte order mark; save it as UTF-8 without BOM"
        );
    }

    #[test]
    fn id_and_group_read() {
        let d = read(SPEC_LINE).unwrap();
        assert_eq!(d.records()[0].id.as_deref(), Some("t-17"));
        let no_id = SPEC_LINE.replace(r#""id": "t-17", "group": "customer-4", "#, "");
        let d = read(&no_id).unwrap();
        assert_eq!(d.records()[0].id, None);
        assert_eq!(d.groups(), 1);
        // wrong types
        let bad = SPEC_LINE.replace(r#""id": "t-17""#, r#""id": 17"#);
        assert_eq!(
            error_text(&bad),
            "data.jsonl:1: \"id\" must be a string of at most 200 characters"
        );
    }

    #[test]
    fn errors_name_file_and_line() {
        let text = format!("{SPEC_LINE}\n{{\"state\": \"x\", \"questions\": {{}}}}\n");
        assert_eq!(
            error_text(&text),
            "data.jsonl:2: a record needs \"state\" and a non-empty \"questions\" object"
        );
        assert_eq!(
            error_text(&format!("{SPEC_LINE}\n[1,2]")),
            "data.jsonl:2: a record must be a JSON object, got an array"
        );
        assert_eq!(
            error_text(&format!("{SPEC_LINE}\n\n{{\"state\": \"x\"}}")),
            "data.jsonl:3: a record needs \"state\" and a non-empty \"questions\" object"
        );
    }

    #[test]
    fn strict_json_per_line() {
        for (line, fragment) in [
            (r#"{"state": NaN}"#, "expected a value at column 11"),
            (r#"{"state": Infinity}"#, "expected a value"),
            (r#"{"state": "x",}"#, "expected a string key"),
            (
                r#"{"state": "x", "state": "y"}"#,
                "the key \"state\" appears more than once",
            ),
            (
                r#"{"state": {"a": 1, "a": 2}}"#,
                "the key \"a\" appears more than once",
            ),
            (r#"{"state": "tru"#, "unterminated string"),
            (r#"{"state": "x""#, "expected ',' or '}'"),
            ("this is not json", "expected a value at column 1"),
        ] {
            let text = error_text(&format!("{SPEC_LINE}\n{line}"));
            assert!(
                text.starts_with("data.jsonl:2: not valid JSON: ") && text.contains(fragment),
                "{line} -> {text}"
            );
        }
    }

    #[test]
    fn big_integers_in_the_state_are_kept() {
        let line = r#"{"state": {"id": 123456789012345678901234567890, "n": 7}, "questions": {"u": {"type": "noul", "instructions": "Ok?", "label": true}}}"#;
        let d = read(line).unwrap();
        assert!(
            d.records()[0]
                .state
                .canonical_text()
                .contains("123456789012345678901234567890")
        );
    }

    #[test]
    fn unknown_fields_listed() {
        let with_meta = SPEC_LINE.replacen('{', r#"{"_meta": {"src": "x"}, "#, 1);
        let text = error_text(&with_meta);
        assert!(
            text.starts_with("data.jsonl:1: unsupported fields \"_meta\"; supported fields are"),
            "{text}"
        );
        let in_question = SPEC_LINE.replace(
            r#""label": true"#,
            r#""label": true, "target": 0.7, "src": "b""#,
        );
        let text = error_text(&in_question);
        assert!(
            text.contains("unsupported fields \"target\", \"src\""),
            "{text}"
        );
        let model = SPEC_LINE.replacen('{', r#"{"model": "kev", "#, 1);
        assert!(error_text(&model).contains("unsupported fields \"model\""));
    }

    #[test]
    fn record_without_state_or_questions() {
        for line in [
            r#"{"questions": {"u": {"type": "noul", "instructions": "Ok?", "label": true}}}"#,
            r#"{"state": "x"}"#,
            r#"{"state": "x", "questions": []}"#,
            r#"{"state": "x", "questions": {}}"#,
        ] {
            assert_eq!(
                error_text(line),
                "data.jsonl:1: a record needs \"state\" and a non-empty \"questions\" object",
                "{line}"
            );
        }
    }

    #[test]
    fn missing_label_and_wrong_labels_with_the_line() {
        let no_label = SPEC_LINE.replace(r#", "label": true"#, "");
        assert_eq!(
            error_text(&no_label),
            "data.jsonl:1: question \"urgent\" has no label"
        );
        let wrong = SPEC_LINE.replace(r#""label": "returns""#, r#""label": "Returns""#);
        assert_eq!(
            error_text(&wrong),
            "data.jsonl:1: question \"team\": label \"Returns\" is not one of the options [\"returns\", \"billing\"]"
        );
        let float_label = SPEC_LINE.replace(r#""label": 2"#, r#""label": 2.0"#);
        assert!(error_text(&float_label).ends_with("must be an integer from 0 to 2, got 2.0"));
    }

    #[test]
    fn domain_errors_keep_their_category_with_prefix() {
        let bad_state = SPEC_LINE.replace(
            r#""state": "My running shoes arrived in the wrong size.""#,
            r#""state": 42"#,
        );
        let e = read(&bad_state).unwrap_err();
        assert_eq!(e.category(), Category::State);
        assert!(e.to_string().starts_with("data.jsonl:1: "), "{e}");
        let bad_question = SPEC_LINE.replace(
            r#""instructions": "Is this urgent?""#,
            r#""instructions": """#,
        );
        let e = read(&bad_question).unwrap_err();
        assert_eq!(e.category(), Category::Question);
        assert!(e.to_string().starts_with("data.jsonl:1: "), "{e}");
        let unknown_noul_key = SPEC_LINE.replace(
            r#""instructions": "Is this urgent?", "label": true"#,
            r#""instructions": "Is this urgent?", "criteria": {"maybe": "x"}, "label": true"#,
        );
        let e = read(&unknown_noul_key).unwrap_err();
        assert_eq!(e.category(), Category::Question);
        assert!(e.to_string().contains("unknown key \"maybe\""));
    }

    #[test]
    fn nothing_partial_on_error() {
        let text = format!("{SPEC_LINE}\n{SPEC_LINE}\n{{broken");
        assert!(read(&text).is_err());
        assert_eq!(
            error_text(""),
            "dataset is empty; it needs at least one record"
        );
        assert_eq!(
            error_text("\n\n  \n"),
            "dataset is empty; it needs at least one record"
        );
    }

    #[test]
    fn same_name_with_two_types_across_lines() {
        let other = SPEC_LINE.replace(r#""urgent": {"type": "noul", "instructions": "Is this urgent?", "label": true}"#, r#""urgent": {"type": "score", "instructions": "How?", "criteria": ["a", "b"], "label": 1}"#);
        let text = error_text(&format!("{SPEC_LINE}\n{other}"));
        assert!(text.starts_with("question name \"urgent\" is a yes_no question at data.jsonl:1 and a score question at data.jsonl:2;"), "{text}");
    }

    proptest! {
        #[test]
        fn fuzz_lines_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
            let _ = read_jsonl(&bytes, "f.jsonl");
        }

        #[test]
        fn fuzz_mutated_records_never_panic(cut in 0usize..SPEC_LINE.len(), byte in any::<u8>()) {
            let mut bytes = SPEC_LINE.as_bytes().to_vec();
            if let Some(slot) = bytes.get_mut(cut) { *slot = byte; }
            let _ = read_jsonl(&bytes, "f.jsonl");
            bytes.truncate(cut);
            let _ = read_jsonl(&bytes, "f.jsonl");
        }
    }
}
