//! A labelled dataset: records whose labels have been checked against their questions
//! (spec 017 §1). Nothing is skipped and nothing is interpreted: a label is exact or an error.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::schema::{Question, QuestionKind, Questions, State};

use super::fingerprint::{dataset_fingerprint, kind_name, record_fingerprint};
use super::problem::{DatasetProblem, MAX_ID_CHARS};

/// A label as it was given, before it is checked against its question.
#[derive(Debug, Clone, PartialEq)]
pub enum RawLabel {
    /// A string (the key of a Choice option).
    Str(String),
    /// A boolean (the answer of a YesNo).
    Bool(bool),
    /// An integer (the level of a Score).
    Int(i128),
    /// Anything else, already shown as text for the message (`2.0`, `null`, `[1]`).
    Other(String),
}

impl RawLabel {
    fn shown(&self) -> String {
        match self {
            RawLabel::Str(s) => serde_json::to_string(s).unwrap_or_default(),
            RawLabel::Bool(b) => b.to_string(),
            RawLabel::Int(n) => n.to_string(),
            RawLabel::Other(text) => text.clone(),
        }
    }
}

/// A label that fits its question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    /// The index of the right option of a Choice.
    Choice(usize),
    /// The right answer of a YesNo.
    YesNo(bool),
    /// The index of the right level of a Score.
    Score(usize),
}

/// A record: a state, questions and one label per question.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelledRecord {
    /// Where it came from, for messages: `file:line` or `record n`.
    pub place: String,
    /// The state.
    pub state: State,
    /// The questions, in order.
    pub questions: Questions,
    /// One label per question, in the same order.
    pub labels: Vec<Label>,
    /// The optional `id`.
    pub id: Option<String>,
    /// The optional `group`: records with the same one never end up apart.
    pub group: Option<String>,
    /// The fingerprint of state and questions (not the labels).
    pub fingerprint: u64,
}

/// A validated `id` or `group`: a string of at most 200 characters.
///
/// # Errors
/// `DatasetProblem::IdOrGroup`.
pub fn check_id(place: &str, field: &'static str, value: Option<&str>) -> Result<Option<String>> {
    match value {
        None => Ok(None),
        Some(v) if v.chars().count() <= MAX_ID_CHARS => Ok(Some(v.to_string())),
        Some(_) => Err(not_an_id(place, field)),
    }
}

/// The error for an `id` or `group` that is not a string.
pub fn not_an_id(place: &str, field: &'static str) -> Error {
    Error::Dataset(DatasetProblem::IdOrGroup {
        place: place.to_string(),
        field,
    })
}

fn resolve(place: &str, name: &str, question: &Question, raw: &RawLabel) -> Result<Label> {
    let fail_choice = |keys: Vec<String>| {
        Error::Dataset(DatasetProblem::ChoiceLabel {
            place: place.to_string(),
            question: name.to_string(),
            label: raw.shown(),
            keys,
        })
    };
    match question {
        Question::Choice(choice) => {
            let keys: Vec<String> = choice.options().iter().map(|o| o.key.clone()).collect();
            match raw {
                RawLabel::Str(s) => keys
                    .iter()
                    .position(|k| k == s)
                    .map(Label::Choice)
                    .ok_or_else(|| fail_choice(keys)),
                _ => Err(fail_choice(keys)),
            }
        }
        Question::YesNo(_) => match raw {
            RawLabel::Bool(b) => Ok(Label::YesNo(*b)),
            _ => Err(Error::Dataset(DatasetProblem::YesNoLabel {
                place: place.to_string(),
                question: name.to_string(),
                value: raw.shown(),
            })),
        },
        Question::Score(score) => {
            let levels = score.levels().len();
            match raw {
                RawLabel::Int(n) if usize::try_from(*n).is_ok_and(|i| i < levels) => {
                    Ok(Label::Score(usize::try_from(*n).unwrap_or(0)))
                }
                _ => Err(Error::Dataset(DatasetProblem::ScoreLabel {
                    place: place.to_string(),
                    question: name.to_string(),
                    levels,
                    value: raw.shown(),
                })),
            }
        }
    }
}

impl LabelledRecord {
    /// Check `raw` labels (by question name) against `questions` and build the record.
    ///
    /// # Errors
    /// A label for a question that does not exist, a question without a label, or a label that
    /// does not fit its question (`DatasetProblem`).
    pub fn build(
        place: &str,
        state: State,
        questions: Questions,
        raw: &[(String, RawLabel)],
        id: Option<String>,
        group: Option<String>,
    ) -> Result<Self> {
        let unknown: Vec<String> = raw
            .iter()
            .filter(|(n, _)| !questions.iter().any(|(q, _)| q == n))
            .map(|(n, _)| n.clone())
            .collect();
        if !unknown.is_empty() {
            return Err(Error::Dataset(DatasetProblem::LabelForMissingQuestion {
                place: place.to_string(),
                names: unknown,
            }));
        }
        let mut labels = Vec::with_capacity(questions.len());
        for (name, question) in questions.iter() {
            let Some((_, label)) = raw.iter().find(|(n, _)| n == name) else {
                return Err(Error::Dataset(DatasetProblem::MissingLabel {
                    place: place.to_string(),
                    question: name.to_string(),
                }));
            };
            labels.push(resolve(place, name, question, label)?);
        }
        let fingerprint = record_fingerprint(&state, &questions);
        Ok(LabelledRecord {
            place: place.to_string(),
            state,
            questions,
            labels,
            id,
            group,
            fingerprint,
        })
    }
}

/// A validated dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    records: Vec<LabelledRecord>,
    /// The group index of each record (groups numbered by first appearance).
    group_of: Vec<usize>,
    groups: usize,
    /// Each question name with its kind, in order of first appearance.
    names: Vec<(String, QuestionKind)>,
    fingerprint: u64,
}

impl Dataset {
    /// Check the records together: at least one, and every name keeps one type.
    ///
    /// # Errors
    /// `DatasetProblem::Empty` or `DatasetProblem::NameHasTwoTypes`.
    pub fn new(records: Vec<LabelledRecord>) -> Result<Self> {
        if records.is_empty() {
            return Err(Error::Dataset(DatasetProblem::Empty));
        }
        let mut names: Vec<(String, QuestionKind)> = Vec::new();
        let mut first_seen: HashMap<String, (QuestionKind, String)> = HashMap::new();
        for record in &records {
            for (name, question) in record.questions.iter() {
                let kind = question.kind();
                match first_seen.get(name) {
                    None => {
                        first_seen.insert(name.to_string(), (kind, record.place.clone()));
                        names.push((name.to_string(), kind));
                    }
                    Some((first, place)) if *first != kind => {
                        return Err(Error::Dataset(DatasetProblem::NameHasTwoTypes {
                            question: name.to_string(),
                            first_kind: kind_name(*first),
                            first_place: place.clone(),
                            second_kind: kind_name(kind),
                            second_place: record.place.clone(),
                        }));
                    }
                    Some(_) => {}
                }
            }
        }
        let mut by_name: HashMap<&str, usize> = HashMap::new();
        let mut groups = 0usize;
        let group_of: Vec<usize> = records
            .iter()
            .map(|r| match &r.group {
                Some(g) => *by_name.entry(g.as_str()).or_insert_with(|| {
                    groups += 1;
                    groups - 1
                }),
                None => {
                    groups += 1;
                    groups - 1
                }
            })
            .collect();
        let fingerprint =
            dataset_fingerprint(&records.iter().map(|r| r.fingerprint).collect::<Vec<_>>());
        Ok(Dataset {
            records,
            group_of,
            groups,
            names,
            fingerprint,
        })
    }

    /// The records, in order.
    pub fn records(&self) -> &[LabelledRecord] {
        &self.records
    }

    /// The group index of each record.
    pub fn group_of(&self) -> &[usize] {
        &self.group_of
    }

    /// Number of groups (a record without `group` is a group of its own).
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Records per group, indexed by group.
    pub fn group_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.groups];
        for g in &self.group_of {
            if let Some(s) = sizes.get_mut(*g) {
                *s += 1;
            }
        }
        sizes
    }

    /// Question names with their kind, in order of first appearance.
    pub fn names(&self) -> &[(String, QuestionKind)] {
        &self.names
    }

    /// The fingerprint of the whole dataset.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Total number of questions over all records.
    pub fn question_count(&self) -> usize {
        self.records.iter().map(|r| r.questions.len()).sum()
    }

    /// A dataset made of some of the records (same order), keeping each one's place.
    ///
    /// # Errors
    /// `DatasetProblem::Empty` if `keep` selects nothing.
    pub fn select(&self, keep: impl Fn(usize) -> bool) -> Result<Dataset> {
        Dataset::new(
            self.records
                .iter()
                .enumerate()
                .filter(|(i, _)| keep(*i))
                .map(|(_, r)| r.clone())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Choice, ChoiceOption, Score, StateValue, TextEntry, YesNo};

    fn instr(s: &str) -> TextEntry {
        TextEntry::text(s).unwrap()
    }

    fn state(text: &str) -> State {
        State::new(StateValue::string(text)).unwrap()
    }

    fn choice_q() -> Question {
        Question::from(
            Choice::new(
                instr("Which team?"),
                vec![ChoiceOption::key("returns"), ChoiceOption::key("billing")],
            )
            .unwrap(),
        )
    }

    fn yes_no_q() -> Question {
        Question::from(YesNo::new(instr("Urgent?"), None, None).unwrap())
    }

    fn score_q() -> Question {
        Question::from(
            Score::new(
                instr("How much?"),
                vec![instr("low"), instr("mid"), instr("high")],
            )
            .unwrap(),
        )
    }

    fn build(name: &str, q: Question, label: RawLabel) -> Result<LabelledRecord> {
        LabelledRecord::build(
            "record 0",
            state("s"),
            Questions::new(vec![(name, q)]).unwrap(),
            &[(name.to_string(), label)],
            None,
            None,
        )
    }

    fn message(r: Result<LabelledRecord>) -> String {
        r.unwrap_err().to_string()
    }

    #[test]
    fn choice_label_exact_case() {
        let ok = build("team", choice_q(), RawLabel::Str("billing".into())).unwrap();
        assert_eq!(ok.labels, vec![Label::Choice(1)]);
        assert_eq!(
            message(build("team", choice_q(), RawLabel::Str("Billing".into()))),
            "record 0: question \"team\": label \"Billing\" is not one of the options [\"returns\", \"billing\"]"
        );
        assert!(
            message(build("team", choice_q(), RawLabel::Int(1))).contains("label 1 is not one of")
        );
        assert!(
            message(build("team", choice_q(), RawLabel::Bool(true)))
                .contains("label true is not one of")
        );
    }

    #[test]
    fn yes_no_only_bool() {
        assert_eq!(
            build("u", yes_no_q(), RawLabel::Bool(false))
                .unwrap()
                .labels,
            vec![Label::YesNo(false)]
        );
        for (label, shown) in [
            (RawLabel::Int(1), "1"),
            (RawLabel::Str("true".into()), "\"true\""),
            (RawLabel::Other("null".into()), "null"),
        ] {
            assert_eq!(
                message(build("u", yes_no_q(), label)),
                format!(
                    "record 0: question \"u\": the label of a yes/no question must be true or false, got {shown}"
                )
            );
        }
    }

    #[test]
    fn score_only_int_in_range() {
        assert_eq!(
            build("p", score_q(), RawLabel::Int(2)).unwrap().labels,
            vec![Label::Score(2)]
        );
        for (label, shown) in [
            (RawLabel::Other("2.0".into()), "2.0"),
            (RawLabel::Bool(true), "true"),
            (RawLabel::Int(-1), "-1"),
            (RawLabel::Int(3), "3"),
            (RawLabel::Str("2".into()), "\"2\""),
        ] {
            assert_eq!(
                message(build("p", score_q(), label)),
                format!(
                    "record 0: question \"p\": the label of a score question must be an integer from 0 to 2, got {shown}"
                )
            );
        }
    }

    #[test]
    fn missing_label() {
        let r = LabelledRecord::build(
            "record 4",
            state("s"),
            Questions::new(vec![("a", yes_no_q()), ("b", yes_no_q())]).unwrap(),
            &[("a".to_string(), RawLabel::Bool(true))],
            None,
            None,
        );
        assert_eq!(message(r), "record 4: question \"b\" has no label");
    }

    #[test]
    fn label_for_missing_question() {
        let r = LabelledRecord::build(
            "record 1",
            state("s"),
            Questions::new(vec![("a", yes_no_q())]).unwrap(),
            &[
                ("a".to_string(), RawLabel::Bool(true)),
                ("zz".to_string(), RawLabel::Bool(true)),
            ],
            None,
            None,
        );
        assert_eq!(
            message(r),
            "record 1: labels has \"zz\" but questions has no such question"
        );
    }

    #[test]
    fn id_group_types_and_length() {
        assert_eq!(
            check_id("record 0", "id", Some("t-17")).unwrap(),
            Some("t-17".to_string())
        );
        assert_eq!(check_id("record 0", "id", None).unwrap(), None);
        let long = "x".repeat(201);
        assert_eq!(
            check_id("record 3", "group", Some(&long))
                .unwrap_err()
                .to_string(),
            "record 3: \"group\" must be a string of at most 200 characters"
        );
        assert!(check_id("r", "id", Some(&"é".repeat(200))).is_ok());
        assert_eq!(
            not_an_id("record 2", "id").to_string(),
            "record 2: \"id\" must be a string of at most 200 characters"
        );
    }

    fn record(
        place: &str,
        name: &str,
        q: Question,
        label: RawLabel,
        group: Option<&str>,
    ) -> LabelledRecord {
        LabelledRecord::build(
            place,
            state("s"),
            Questions::new(vec![(name, q)]).unwrap(),
            &[(name.to_string(), label)],
            None,
            group.map(str::to_string),
        )
        .unwrap()
    }

    #[test]
    fn same_name_different_types_two_positions() {
        let a = record(
            "d:1",
            "q",
            choice_q(),
            RawLabel::Str("returns".into()),
            None,
        );
        let b = record("d:9", "q", yes_no_q(), RawLabel::Bool(true), None);
        let expected = "question name \"q\" is a choice question at d:1 and a yes_no question at d:9; a name must keep one type across the dataset";
        assert_eq!(
            Dataset::new(vec![a.clone(), b.clone()])
                .unwrap_err()
                .to_string(),
            expected
        );
        // the order of the records decides which position is the first
        let swapped = "question name \"q\" is a yes_no question at d:9 and a choice question at d:1; a name must keep one type across the dataset";
        assert_eq!(Dataset::new(vec![b, a]).unwrap_err().to_string(), swapped);
    }

    #[test]
    fn same_name_same_type_different_options_ok() {
        let a = record(
            "d:1",
            "q",
            choice_q(),
            RawLabel::Str("returns".into()),
            None,
        );
        let other = Question::from(
            Choice::new(
                instr("Another?"),
                vec![ChoiceOption::key("x"), ChoiceOption::key("y")],
            )
            .unwrap(),
        );
        let b = record("d:2", "q", other, RawLabel::Str("y".into()), None);
        let d = Dataset::new(vec![a, b]).unwrap();
        assert_eq!(d.names(), &[("q".to_string(), QuestionKind::Choice)]);
    }

    #[test]
    fn empty_dataset() {
        assert_eq!(
            Dataset::new(vec![]).unwrap_err().to_string(),
            "dataset is empty; it needs at least one record"
        );
    }

    #[test]
    fn groups_are_numbered_by_first_appearance_and_default_to_one_per_record() {
        let r = |g: Option<&str>| record("r", "u", yes_no_q(), RawLabel::Bool(true), g);
        let d = Dataset::new(vec![
            r(Some("a")),
            r(None),
            r(Some("b")),
            r(Some("a")),
            r(None),
        ])
        .unwrap();
        assert_eq!(d.group_of(), &[0, 1, 2, 0, 3]);
        assert_eq!(d.groups(), 4);
        assert_eq!(d.group_sizes(), vec![2, 1, 1, 1]);
        assert_eq!(d.question_count(), 5);
        let subset = d.select(|i| i != 1).unwrap();
        assert_eq!(subset.records().len(), 4);
    }
}
