//! Why an argument of `evaluate`, `calibrate` or `split_dataset` is refused, and why a dataset is
//! (spec 017 §1.3, §2, §3). The messages are part of the public contract.

use std::fmt;

/// An argument is outside its range. Maps to `ValueError`.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum ArgumentProblem {
    /// `folds` is not an integer in `2..=50`. `got` is the text of what was given.
    Folds { got: String },
    /// Fewer groups than folds.
    MoreFoldsThanGroups { groups: usize, folds: usize },
    /// `test_fraction` is not a number strictly between 0 and 1.
    TestFraction { got: f64 },
    /// `test_fraction` leaves one side of the split empty.
    EmptySide {
        fraction: f64,
        side: &'static str,
        groups: usize,
    },
    /// `bootstrap_samples` is not an integer in `0..=100000`.
    BootstrapSamples { got: String },
    /// `seed` is not an integer in `0..2**64`.
    Seed { got: String },
    /// `question_types` has something other than `choice`, `score` or `yes_no`.
    QuestionType { got: String },
    /// `question_types` is empty.
    NoQuestionTypes,
    /// The dataset has no question of a selected type.
    NothingToCalibrate { types: String },
}

/// A dataset (file or Python records) that cannot be used. Maps to `InvalidDatasetError`.
///
/// `place` is `file:line` or `record n`; question names and labels are shown as JSON text.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum DatasetProblem {
    /// No record at all.
    Empty,
    /// A byte-order mark at the start of the file.
    ByteOrderMark { file: String },
    /// A line that is not valid JSON.
    NotJson { place: String, detail: String },
    /// A line that is JSON but not an object.
    NotAnObject { place: String, got: String },
    /// Fields the format does not have (of the record, or of a question).
    UnsupportedFields { place: String, names: Vec<String> },
    /// `state` or `questions` missing, or `questions` empty.
    MissingKeys { place: String },
    /// A question without `label`.
    MissingLabel { place: String, question: String },
    /// A label for a question that does not exist (Python form).
    LabelForMissingQuestion { place: String, names: Vec<String> },
    /// A Choice label that is not one of the keys.
    ChoiceLabel {
        place: String,
        question: String,
        label: String,
        keys: Vec<String>,
    },
    /// A YesNo label that is not a `bool`.
    YesNoLabel {
        place: String,
        question: String,
        value: String,
    },
    /// A Score label that is not an integer in `0..levels`.
    ScoreLabel {
        place: String,
        question: String,
        levels: usize,
        value: String,
    },
    /// `id` or `group` not a string, or longer than 200 characters.
    IdOrGroup { place: String, field: &'static str },
    /// One name used for two kinds of question.
    NameHasTwoTypes {
        question: String,
        first_kind: &'static str,
        first_place: String,
        second_kind: &'static str,
        second_place: String,
    },
}

/// The most characters of `id` and `group`.
pub const MAX_ID_CHARS: usize = 200;

fn quoted_list(names: &[String]) -> String {
    names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn keys_text(keys: &[String]) -> String {
    let shown: Vec<String> = keys.iter().take(10).map(|k| format!("\"{k}\"")).collect();
    if keys.len() > 10 {
        format!("[{}, ... ({} options)]", shown.join(", "), keys.len())
    } else {
        format!("[{}]", shown.join(", "))
    }
}

impl fmt::Display for DatasetProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetProblem::Empty => {
                write!(f, "dataset is empty; it needs at least one record")
            }
            DatasetProblem::ByteOrderMark { file } => write!(
                f,
                "{file}:1: the file starts with a byte order mark; save it as UTF-8 without BOM"
            ),
            DatasetProblem::NotJson { place, detail } => {
                write!(f, "{place}: not valid JSON: {detail}")
            }
            DatasetProblem::NotAnObject { place, got } => {
                write!(f, "{place}: a record must be a JSON object, got {got}")
            }
            DatasetProblem::UnsupportedFields { place, names } => write!(
                f,
                "{place}: unsupported fields {}; supported fields are \"state\", \"questions\", \"id\" and \"group\" (per question: \"type\", \"instructions\", \"criteria\" and \"label\")",
                quoted_list(names)
            ),
            DatasetProblem::MissingKeys { place } => write!(
                f,
                "{place}: a record needs \"state\" and a non-empty \"questions\" object"
            ),
            DatasetProblem::MissingLabel { place, question } => {
                write!(f, "{place}: question \"{question}\" has no label")
            }
            DatasetProblem::LabelForMissingQuestion { place, names } => write!(
                f,
                "{place}: labels has {} but questions has no such question",
                quoted_list(names)
            ),
            DatasetProblem::ChoiceLabel {
                place,
                question,
                label,
                keys,
            } => write!(
                f,
                "{place}: question \"{question}\": label {label} is not one of the options {}",
                keys_text(keys)
            ),
            DatasetProblem::YesNoLabel {
                place,
                question,
                value,
            } => write!(
                f,
                "{place}: question \"{question}\": the label of a yes/no question must be true or false, got {value}"
            ),
            DatasetProblem::ScoreLabel {
                place,
                question,
                levels,
                value,
            } => write!(
                f,
                "{place}: question \"{question}\": the label of a score question must be an integer from 0 to {}, got {value}",
                levels.saturating_sub(1)
            ),
            DatasetProblem::IdOrGroup { place, field } => write!(
                f,
                "{place}: \"{field}\" must be a string of at most {MAX_ID_CHARS} characters"
            ),
            DatasetProblem::NameHasTwoTypes {
                question,
                first_kind,
                first_place,
                second_kind,
                second_place,
            } => write!(
                f,
                "question name \"{question}\" is a {first_kind} question at {first_place} and a {second_kind} question at {second_place}; a name must keep one type across the dataset"
            ),
        }
    }
}

/// The largest `bootstrap_samples`: a typo like `10**9` must not hang the call.
pub const MAX_BOOTSTRAP_SAMPLES: usize = 100_000;

impl fmt::Display for ArgumentProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgumentProblem::Folds { got } => {
                write!(f, "folds must be an integer between 2 and 50, got {got}")
            }
            ArgumentProblem::MoreFoldsThanGroups { groups, folds } => write!(
                f,
                "dataset has {groups} groups but folds={folds}; use more records or fewer folds"
            ),
            ArgumentProblem::TestFraction { got } => write!(
                f,
                "test_fraction must be a number strictly between 0 and 1, got {got:?}"
            ),
            ArgumentProblem::EmptySide {
                fraction,
                side,
                groups,
            } => write!(
                f,
                "test_fraction={fraction:?} leaves the {side} set empty (the dataset has {groups} groups); use more records or another fraction"
            ),
            ArgumentProblem::BootstrapSamples { got } => write!(
                f,
                "bootstrap_samples must be an integer between 0 and {MAX_BOOTSTRAP_SAMPLES}, got {got}"
            ),
            ArgumentProblem::Seed { got } => {
                write!(
                    f,
                    "seed must be an integer with 0 <= seed < 2**64, got {got}"
                )
            }
            ArgumentProblem::QuestionType { got } => write!(
                f,
                "question_types must contain only \"choice\", \"score\" or \"yes_no\", got {got}"
            ),
            ArgumentProblem::NoQuestionTypes => write!(
                f,
                "question_types is empty; use None for all types or name at least one"
            ),
            ArgumentProblem::NothingToCalibrate { types } => {
                write!(f, "the dataset has no {types} questions to calibrate on")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argument_messages() {
        let cases: Vec<(ArgumentProblem, &str)> = vec![
            (
                ArgumentProblem::Folds { got: "1".into() },
                "folds must be an integer between 2 and 50, got 1",
            ),
            (
                ArgumentProblem::MoreFoldsThanGroups {
                    groups: 3,
                    folds: 5,
                },
                "dataset has 3 groups but folds=5; use more records or fewer folds",
            ),
            (
                ArgumentProblem::TestFraction { got: 1.0 },
                "test_fraction must be a number strictly between 0 and 1, got 1.0",
            ),
            (
                ArgumentProblem::EmptySide {
                    fraction: 0.01,
                    side: "test",
                    groups: 4,
                },
                "test_fraction=0.01 leaves the test set empty (the dataset has 4 groups); use more records or another fraction",
            ),
            (
                ArgumentProblem::BootstrapSamples { got: "-1".into() },
                "bootstrap_samples must be an integer between 0 and 100000, got -1",
            ),
            (
                ArgumentProblem::Seed { got: "-3".into() },
                "seed must be an integer with 0 <= seed < 2**64, got -3",
            ),
            (
                ArgumentProblem::QuestionType {
                    got: "'text'".into(),
                },
                "question_types must contain only \"choice\", \"score\" or \"yes_no\", got 'text'",
            ),
            (
                ArgumentProblem::NothingToCalibrate {
                    types: "yes_no".into(),
                },
                "the dataset has no yes_no questions to calibrate on",
            ),
        ];
        for (problem, expected) in cases {
            assert_eq!(problem.to_string(), expected);
        }
    }
}

#[cfg(test)]
mod dataset_tests {
    use super::*;

    fn p(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn every_message_matches_the_table() {
        let cases: Vec<(DatasetProblem, &str)> = vec![
            (
                DatasetProblem::Empty,
                "dataset is empty; it needs at least one record",
            ),
            (
                DatasetProblem::ByteOrderMark { file: p("d.jsonl") },
                "d.jsonl:1: the file starts with a byte order mark; save it as UTF-8 without BOM",
            ),
            (
                DatasetProblem::NotJson {
                    place: p("d.jsonl:3"),
                    detail: p("expected value at column 5"),
                },
                "d.jsonl:3: not valid JSON: expected value at column 5",
            ),
            (
                DatasetProblem::NotAnObject {
                    place: p("record 2"),
                    got: p("an array"),
                },
                "record 2: a record must be a JSON object, got an array",
            ),
            (
                DatasetProblem::UnsupportedFields {
                    place: p("d.jsonl:41"),
                    names: vec![p("_meta"), p("src")],
                },
                "d.jsonl:41: unsupported fields \"_meta\", \"src\"; supported fields are \"state\", \"questions\", \"id\" and \"group\" (per question: \"type\", \"instructions\", \"criteria\" and \"label\")",
            ),
            (
                DatasetProblem::MissingKeys {
                    place: p("d.jsonl:2"),
                },
                "d.jsonl:2: a record needs \"state\" and a non-empty \"questions\" object",
            ),
            (
                DatasetProblem::MissingLabel {
                    place: p("record 0"),
                    question: p("team"),
                },
                "record 0: question \"team\" has no label",
            ),
            (
                DatasetProblem::LabelForMissingQuestion {
                    place: p("record 0"),
                    names: vec![p("x")],
                },
                "record 0: labels has \"x\" but questions has no such question",
            ),
            (
                DatasetProblem::ChoiceLabel {
                    place: p("d.jsonl:5"),
                    question: p("team"),
                    label: p("\"Returns\""),
                    keys: vec![p("returns"), p("billing")],
                },
                "d.jsonl:5: question \"team\": label \"Returns\" is not one of the options [\"returns\", \"billing\"]",
            ),
            (
                DatasetProblem::YesNoLabel {
                    place: p("d.jsonl:5"),
                    question: p("urgent"),
                    value: p("1"),
                },
                "d.jsonl:5: question \"urgent\": the label of a yes/no question must be true or false, got 1",
            ),
            (
                DatasetProblem::ScoreLabel {
                    place: p("d.jsonl:5"),
                    question: p("priority"),
                    levels: 3,
                    value: p("2.0"),
                },
                "d.jsonl:5: question \"priority\": the label of a score question must be an integer from 0 to 2, got 2.0",
            ),
            (
                DatasetProblem::IdOrGroup {
                    place: p("record 1"),
                    field: "group",
                },
                "record 1: \"group\" must be a string of at most 200 characters",
            ),
            (
                DatasetProblem::NameHasTwoTypes {
                    question: p("q"),
                    first_kind: "choice",
                    first_place: p("d.jsonl:1"),
                    second_kind: "yes_no",
                    second_place: p("d.jsonl:9"),
                },
                "question name \"q\" is a choice question at d.jsonl:1 and a yes_no question at d.jsonl:9; a name must keep one type across the dataset",
            ),
        ];
        for (problem, expected) in cases {
            assert_eq!(problem.to_string(), expected);
        }
    }

    #[test]
    fn long_key_lists_are_cut() {
        let keys: Vec<String> = (0..12).map(|i| format!("k{i}")).collect();
        let text = keys_text(&keys);
        assert!(text.starts_with("[\"k0\", \"k1\""));
        assert!(text.ends_with("\"k9\", ... (12 options)]"));
    }
}
