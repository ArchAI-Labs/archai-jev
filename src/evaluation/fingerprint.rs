//! Fingerprints of records and datasets (spec 017 §3.4).
//!
//! FNV-1a over 64 bits, defined here so the values never change between operating systems,
//! architectures or versions. **A heuristic to warn, not a security hash**: a collision gives a
//! false warning, never a missed one for equal data.

use crate::schema::{QuestionKind, Questions, State};

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// The FNV-1a hash of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Hasher::new();
    h.write(bytes);
    h.finish()
}

/// An incremental FNV-1a.
struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Hasher(OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    /// A field: its length in bytes (little-endian `u64`) and then the bytes, so that `["ab",
    /// "c"]` and `["a", "bc"]` cannot give the same stream.
    fn field(&mut self, bytes: &[u8]) {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// The name of a kind in the fingerprint and in the reports.
pub fn kind_name(kind: QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Choice => "choice",
        QuestionKind::Score => "score",
        QuestionKind::YesNo => "yes_no",
    }
}

/// The fingerprint of a record: the canonical text of the state and, for each question in
/// order, its name, kind, rendered instructions and rendered options. **Not** the labels, `id`
/// or `group`.
pub fn record_fingerprint(state: &State, questions: &Questions) -> u64 {
    let mut h = Hasher::new();
    h.field(state.canonical_text().as_bytes());
    h.write(&(questions.len() as u64).to_le_bytes());
    for (name, question) in questions.iter() {
        h.field(name.as_bytes());
        h.field(kind_name(question.kind()).as_bytes());
        h.field(question.instructions().rendered().as_bytes());
        let options = question.option_texts();
        h.write(&(options.len() as u64).to_le_bytes());
        for text in options {
            h.field(text.as_bytes());
        }
    }
    h.finish()
}

/// The fingerprint of a dataset: FNV-1a of the record fingerprints, in order.
pub fn dataset_fingerprint(records: &[u64]) -> u64 {
    let mut h = Hasher::new();
    for r in records {
        h.write(&r.to_le_bytes());
    }
    h.finish()
}

/// 16 lowercase hex digits.
pub fn hex(fingerprint: u64) -> String {
    format!("{fingerprint:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Choice, ChoiceOption, Question, Score, StateValue, TextEntry, YesNo};

    fn instr(s: &str) -> TextEntry {
        TextEntry::text(s).unwrap()
    }

    fn state(text: &str) -> State {
        State::new(StateValue::string(text)).unwrap()
    }

    fn choice(instructions: &str, keys: &[&str]) -> Question {
        Question::from(
            Choice::new(
                instr(instructions),
                keys.iter().map(|k| ChoiceOption::key(*k)).collect(),
            )
            .unwrap(),
        )
    }

    fn questions(items: Vec<(&str, Question)>) -> Questions {
        Questions::new(items).unwrap()
    }

    #[test]
    fn fnv_literals() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    fn three_records() -> Vec<u64> {
        let a = record_fingerprint(
            &state("hello"),
            &questions(vec![("team", choice("Which team?", &["a", "b"]))]),
        );
        let b = record_fingerprint(
            &state("world"),
            &questions(vec![
                (
                    "urgent",
                    Question::from(YesNo::new(instr("Urgent?"), None, None).unwrap()),
                ),
                (
                    "priority",
                    Question::from(
                        Score::new(instr("How much?"), vec![instr("low"), instr("high")]).unwrap(),
                    ),
                ),
            ]),
        );
        let c = record_fingerprint(
            &state(""),
            &questions(vec![("q", choice("Pick", &["x", "y", "z"]))]),
        );
        vec![a, b, c]
    }

    #[test]
    fn record_fingerprint_literals_3_records() {
        // Cross-checked with an independent Python implementation of the definition
        // (spec 017 §3.4: fields prefixed by their length, little-endian u64).
        let got = three_records();
        assert_eq!(
            got.iter().map(|f| hex(*f)).collect::<Vec<_>>(),
            vec!["974586864a0a20ae", "1f01ca23805c209b", "dce9275b2e2c990b"]
        );
        assert_eq!(hex(dataset_fingerprint(&got)), "eb9058e322747560");
    }

    #[test]
    fn ignores_labels_id_group() {
        // They are not inputs of the function at all; the same state and questions give the
        // same fingerprint however the record is labelled or grouped.
        let q = questions(vec![("team", choice("Which team?", &["a", "b"]))]);
        assert_eq!(
            record_fingerprint(&state("hello"), &q),
            record_fingerprint(&state("hello"), &q)
        );
    }

    #[test]
    fn changes_with_state_instruction_option_and_question_order() {
        let base_q = questions(vec![("team", choice("Which team?", &["a", "b"]))]);
        let base = record_fingerprint(&state("hello"), &base_q);
        assert_ne!(base, record_fingerprint(&state("hellp"), &base_q));
        let other_instruction = questions(vec![("team", choice("Which squad?", &["a", "b"]))]);
        assert_ne!(
            base,
            record_fingerprint(&state("hello"), &other_instruction)
        );
        let other_option = questions(vec![("team", choice("Which team?", &["a", "c"]))]);
        assert_ne!(base, record_fingerprint(&state("hello"), &other_option));
        let other_name = questions(vec![("squad", choice("Which team?", &["a", "b"]))]);
        assert_ne!(base, record_fingerprint(&state("hello"), &other_name));
        let two = |first: &str, second: &str| {
            questions(vec![
                (first, choice("One", &["a", "b"])),
                (second, choice("Two", &["a", "b"])),
            ])
        };
        assert_ne!(
            record_fingerprint(&state("s"), &two("x", "y")),
            record_fingerprint(&state("s"), &two("y", "x"))
        );
    }

    #[test]
    fn the_length_prefix_removes_ambiguity() {
        let split = |a: &str, b: &str| {
            record_fingerprint(&state("s"), &questions(vec![("q", choice(a, &[b, "z"]))]))
        };
        assert_ne!(split("ab", "c"), split("a", "bc"));
    }
}
