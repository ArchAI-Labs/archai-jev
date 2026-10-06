//! Questions: `Choice`, `Score` and `YesNo`, validated when built, and the ordered set of
//! named questions of a request.

use std::collections::HashSet;

use super::render::{is_blank, is_py_space};
use super::state::TextEntry;
use crate::error::{Error, NameProblem, OptionKind, Result};

/// Maximum rendered size of a question (instructions plus all option texts), in bytes.
pub const MAX_QUESTION_BYTES: usize = 262_144;
/// Maximum number of questions per request.
pub const MAX_QUESTIONS: usize = 128;
/// Maximum length of a question name, in characters.
pub const MAX_NAME_CHARS: usize = 128;
/// Maximum number of options (Choice) or levels (Score).
pub const MAX_OPTIONS: usize = 255;

/// The three kinds of question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// Pick one of several options.
    Choice,
    /// Pick a level on an ordered scale.
    Score,
    /// A yes/no question (`Noul` in TypeSafe).
    YesNo,
}

/// One option of a [`Choice`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceOption {
    /// The key returned as the answer.
    pub key: String,
    /// Optional description the model reads after the key.
    pub description: Option<TextEntry>,
}

impl ChoiceOption {
    /// An option with a key and no description.
    pub fn key(key: impl Into<String>) -> Self {
        ChoiceOption {
            key: key.into(),
            description: None,
        }
    }

    /// An option with a key and a plain-text description.
    ///
    /// # Errors
    /// [`Error::StateTooLarge`] if the description exceeds the size limit.
    pub fn described(key: impl Into<String>, description: &str) -> Result<Self> {
        Ok(ChoiceOption {
            key: key.into(),
            description: Some(TextEntry::text(description)?),
        })
    }
}

/// Description present for Kev: not `None` and not the empty string.
fn present(d: &Option<TextEntry>) -> Option<&TextEntry> {
    d.as_ref().filter(|t| !t.is_empty_string())
}

/// Check instructions and total size, shared by the three kinds.
fn check_common(instructions: &TextEntry, texts: &[String]) -> Result<()> {
    if is_blank(instructions.rendered()) {
        return Err(Error::EmptyInstructions { question: None });
    }
    let total: usize = instructions.rendered().len() + texts.iter().map(String::len).sum::<usize>();
    if total > MAX_QUESTION_BYTES {
        return Err(Error::QuestionTooLarge {
            question: None,
            got: total,
        });
    }
    Ok(())
}

/// Reject two options whose rendered texts are identical.
fn check_distinguishable(texts: &[String]) -> Result<()> {
    let mut first_seen: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::with_capacity(texts.len());
    for (j, text) in texts.iter().enumerate() {
        if let Some(&i) = first_seen.get(text.as_str()) {
            return Err(Error::AmbiguousOptions {
                question: None,
                i,
                j,
                text: text.clone(),
            });
        }
        first_seen.insert(text.as_str(), j);
    }
    Ok(())
}

/// A question with several options; the answer is one key.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    instructions: TextEntry,
    options: Vec<ChoiceOption>,
    texts: Vec<String>,
}

impl Choice {
    /// Build and validate a Choice.
    ///
    /// # Errors
    /// `EmptyInstructions`, `InvalidOptionCount`, `EmptyOptionKey`, `DuplicateOptionKey`,
    /// `AmbiguousOptions` or `QuestionTooLarge`.
    pub fn new(instructions: TextEntry, options: Vec<ChoiceOption>) -> Result<Self> {
        if is_blank(instructions.rendered()) {
            return Err(Error::EmptyInstructions { question: None });
        }
        if options.is_empty() || options.len() > MAX_OPTIONS {
            return Err(Error::InvalidOptionCount {
                kind: OptionKind::Choice,
                question: None,
                got: options.len(),
            });
        }
        let mut seen: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::with_capacity(options.len());
        for (i, opt) in options.iter().enumerate() {
            if is_blank(&opt.key) {
                return Err(Error::EmptyOptionKey {
                    question: None,
                    index: i,
                });
            }
            if let Some(&first) = seen.get(opt.key.as_str()) {
                return Err(Error::DuplicateOptionKey {
                    question: None,
                    key: opt.key.clone(),
                    i: first,
                    j: i,
                });
            }
            seen.insert(opt.key.as_str(), i);
        }
        let texts: Vec<String> = options
            .iter()
            .map(|o| match present(&o.description) {
                None => o.key.clone(),
                Some(d) => format!("{}: {}", o.key, d.rendered()),
            })
            .collect();
        check_common(&instructions, &texts)?;
        check_distinguishable(&texts)?;
        Ok(Choice {
            instructions,
            options,
            texts,
        })
    }

    /// The options in order.
    pub fn options(&self) -> &[ChoiceOption] {
        &self.options
    }
}

/// A question answered on an ordered scale of levels.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    instructions: TextEntry,
    levels: Vec<TextEntry>,
    texts: Vec<String>,
}

impl Score {
    /// Build and validate a Score, levels from lowest to highest.
    ///
    /// # Errors
    /// `EmptyInstructions`, `InvalidOptionCount`, `EmptyLevel`, `AmbiguousOptions` or
    /// `QuestionTooLarge`.
    pub fn new(instructions: TextEntry, levels: Vec<TextEntry>) -> Result<Self> {
        if is_blank(instructions.rendered()) {
            return Err(Error::EmptyInstructions { question: None });
        }
        if levels.is_empty() || levels.len() > MAX_OPTIONS {
            return Err(Error::InvalidOptionCount {
                kind: OptionKind::Score,
                question: None,
                got: levels.len(),
            });
        }
        for (i, level) in levels.iter().enumerate() {
            if is_blank(level.rendered()) {
                return Err(Error::EmptyLevel {
                    question: None,
                    index: i,
                });
            }
        }
        let texts: Vec<String> = levels.iter().map(|l| l.rendered().to_string()).collect();
        check_common(&instructions, &texts)?;
        check_distinguishable(&texts)?;
        Ok(Score {
            instructions,
            levels,
            texts,
        })
    }

    /// The levels in order, lowest first.
    pub fn levels(&self) -> &[TextEntry] {
        &self.levels
    }
}

/// A yes/no question. Option 0 is "no", option 1 is "yes".
#[derive(Debug, Clone, PartialEq)]
pub struct YesNo {
    instructions: TextEntry,
    true_description: Option<TextEntry>,
    false_description: Option<TextEntry>,
    texts: Vec<String>,
}

impl YesNo {
    /// Build and validate a YesNo with optional descriptions of the two answers.
    ///
    /// # Errors
    /// `EmptyInstructions` or `QuestionTooLarge`.
    pub fn new(
        instructions: TextEntry,
        true_description: Option<TextEntry>,
        false_description: Option<TextEntry>,
    ) -> Result<Self> {
        let label = |name: &str, d: &Option<TextEntry>| match present(d) {
            None => name.to_string(),
            Some(d) => format!("{name}: {}", d.rendered()),
        };
        let texts = vec![
            label("no", &false_description),
            label("yes", &true_description),
        ];
        check_common(&instructions, &texts)?;
        Ok(YesNo {
            instructions,
            true_description,
            false_description,
            texts,
        })
    }

    /// Description of the "yes" answer, if any.
    pub fn true_description(&self) -> Option<&TextEntry> {
        self.true_description.as_ref()
    }

    /// Description of the "no" answer, if any.
    pub fn false_description(&self) -> Option<&TextEntry> {
        self.false_description.as_ref()
    }
}

/// Any of the three questions.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// See [`Choice`].
    Choice(Choice),
    /// See [`Score`].
    Score(Score),
    /// See [`YesNo`].
    YesNo(YesNo),
}

impl From<Choice> for Question {
    fn from(q: Choice) -> Self {
        Question::Choice(q)
    }
}
impl From<Score> for Question {
    fn from(q: Score) -> Self {
        Question::Score(q)
    }
}
impl From<YesNo> for Question {
    fn from(q: YesNo) -> Self {
        Question::YesNo(q)
    }
}

impl Question {
    /// Which kind of question this is.
    pub fn kind(&self) -> QuestionKind {
        match self {
            Question::Choice(_) => QuestionKind::Choice,
            Question::Score(_) => QuestionKind::Score,
            Question::YesNo(_) => QuestionKind::YesNo,
        }
    }

    /// The instructions entry, verbatim.
    pub fn instructions(&self) -> &TextEntry {
        match self {
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
            Question::YesNo(q) => &q.instructions,
        }
    }

    /// The text of each option in the order of Kev's template; the single source for any scorer.
    pub fn option_texts(&self) -> &[String] {
        match self {
            Question::Choice(q) => &q.texts,
            Question::Score(q) => &q.texts,
            Question::YesNo(q) => &q.texts,
        }
    }

    /// Number of options (levels for Score, always 2 for YesNo).
    pub fn option_count(&self) -> usize {
        self.option_texts().len()
    }
}

/// The ordered, named questions of one request.
#[derive(Debug, Clone, PartialEq)]
pub struct Questions {
    items: Vec<(String, Question)>,
}

impl Questions {
    /// Validate names and build the set, keeping insertion order.
    ///
    /// # Errors
    /// `NoQuestions`, `TooManyQuestions`, `InvalidQuestionName`, `DuplicateQuestionName`.
    pub fn new<N: Into<String>>(items: impl IntoIterator<Item = (N, Question)>) -> Result<Self> {
        let items: Vec<(String, Question)> =
            items.into_iter().map(|(n, q)| (n.into(), q)).collect();
        if items.is_empty() {
            return Err(Error::NoQuestions);
        }
        if items.len() > MAX_QUESTIONS {
            return Err(Error::TooManyQuestions { got: items.len() });
        }
        let mut seen: HashSet<&str> = HashSet::with_capacity(items.len());
        for (name, _) in &items {
            check_name(name)?;
            if !seen.insert(name.as_str()) {
                return Err(Error::DuplicateQuestionName { name: name.clone() });
            }
        }
        Ok(Questions { items })
    }

    /// The questions in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Question)> {
        self.items.iter().map(|(n, q)| (n.as_str(), q))
    }

    /// Number of questions.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Always false: a `Questions` has at least one question.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidQuestionName(NameProblem::Empty));
    }
    let chars = name.chars().count();
    if chars > MAX_NAME_CHARS {
        return Err(Error::InvalidQuestionName(NameProblem::TooLong {
            name: name.to_string(),
            got: chars,
        }));
    }
    for (index, c) in name.chars().enumerate() {
        let cp = c as u32;
        if cp <= 0x1f || (0x7f..=0x9f).contains(&cp) {
            return Err(Error::InvalidQuestionName(NameProblem::Control {
                name: name.to_string(),
                code_point: cp,
                index,
            }));
        }
    }
    let edge = name.chars().next().is_some_and(is_py_space)
        || name.chars().next_back().is_some_and(is_py_space);
    if edge {
        return Err(Error::InvalidQuestionName(NameProblem::EdgeWhitespace {
            name: name.to_string(),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::value::StateValue;

    fn instr(s: &str) -> TextEntry {
        TextEntry::text(s).unwrap()
    }

    fn keys(ks: &[&str]) -> Vec<ChoiceOption> {
        ks.iter().map(|k| ChoiceOption::key(*k)).collect()
    }

    #[test]
    fn choice_option_count_bounds() {
        let many: Vec<String> = (0..255).map(|i| format!("k{i}")).collect();
        let opts = |n: usize| -> Vec<ChoiceOption> {
            (0..n).map(|i| ChoiceOption::key(format!("k{i}"))).collect()
        };
        assert_eq!(many.len(), 255);
        assert!(Choice::new(instr("q?"), opts(1)).is_ok());
        assert!(Choice::new(instr("q?"), opts(255)).is_ok());
        for n in [0, 256] {
            let err = Choice::new(instr("q?"), opts(n)).unwrap_err();
            assert_eq!(
                err,
                Error::InvalidOptionCount {
                    kind: OptionKind::Choice,
                    question: None,
                    got: n
                }
            );
        }
        assert_eq!(
            Choice::new(instr("q?"), opts(256)).unwrap_err().to_string(),
            "choice has 256 options; it needs between 1 and 255"
        );
    }

    #[test]
    fn score_level_count_bounds_and_order() {
        let lv = |n: usize| -> Vec<TextEntry> { (0..n).map(|i| instr(&format!("l{i}"))).collect() };
        assert!(Score::new(instr("q?"), lv(1)).is_ok());
        assert!(Score::new(instr("q?"), lv(255)).is_ok());
        assert!(Score::new(instr("q?"), lv(0)).is_err());
        let err = Score::new(instr("q?"), lv(256)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "score has 256 levels; it needs between 1 and 255"
        );
        let s = Score::new(instr("q?"), vec![instr("low"), instr("high")]).unwrap();
        assert_eq!(Question::from(s).option_texts(), ["low", "high"]);
    }

    #[test]
    fn blank_keys_levels_and_instructions() {
        for key in ["", "  ", "\u{a0}", "\u{1c}"] {
            let err = Choice::new(instr("q?"), keys(&["a", key])).unwrap_err();
            assert_eq!(
                err,
                Error::EmptyOptionKey {
                    question: None,
                    index: 1
                }
            );
        }
        for text in ["", " ", "\u{a0}"] {
            assert!(matches!(
                Choice::new(instr(text), keys(&["a"])),
                Err(Error::EmptyInstructions { .. })
            ));
            assert!(matches!(
                Score::new(instr(text), vec![instr("l")]),
                Err(Error::EmptyInstructions { .. })
            ));
            assert!(matches!(
                YesNo::new(instr(text), None, None),
                Err(Error::EmptyInstructions { .. })
            ));
            assert_eq!(
                Score::new(instr("q?"), vec![instr("a"), instr(text)]).unwrap_err(),
                Error::EmptyLevel {
                    question: None,
                    index: 1
                }
            );
        }
        assert_eq!(
            Choice::new(instr(" ok "), keys(&["a"]))
                .unwrap()
                .instructions
                .rendered(),
            " ok "
        );
    }

    #[test]
    fn duplicate_keys_report_both_positions() {
        let err = Choice::new(instr("q?"), keys(&["a", "b", "a"])).unwrap_err();
        assert_eq!(
            err,
            Error::DuplicateOptionKey {
                question: None,
                key: "a".into(),
                i: 0,
                j: 2
            }
        );
        assert!(err.to_string().contains("options 0 and 2"));
    }

    #[test]
    fn option_texts_of_the_three_kinds() {
        let c = Choice::new(
            instr("q?"),
            vec![
                ChoiceOption::described("a", "desc").unwrap(),
                ChoiceOption::key("b"),
                ChoiceOption::described("c", "").unwrap(),
                ChoiceOption::described("d", "  ").unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            Question::from(c).option_texts(),
            ["a: desc", "b", "c", "d:   "]
        );
        let y = |t, f| YesNo::new(instr("q?"), t, f).unwrap();
        assert_eq!(Question::from(y(None, None)).option_texts(), ["no", "yes"]);
        assert_eq!(
            Question::from(y(Some(instr("T")), Some(instr("F")))).option_texts(),
            ["no: F", "yes: T"]
        );
        assert_eq!(
            Question::from(y(Some(instr("T")), None)).option_texts(),
            ["no", "yes: T"]
        );
        assert_eq!(
            Question::from(y(Some(instr("")), None)).option_texts(),
            ["no", "yes"]
        );
    }

    #[test]
    fn structured_entries_use_rendered_text_and_empty_object_is_present() {
        let obj =
            TextEntry::new("d", StateValue::object([("what", StateValue::string("x"))])).unwrap();
        let empty_obj = TextEntry::new("d", StateValue::object::<String>([])).unwrap();
        let c = Choice::new(
            instr("q?"),
            vec![
                ChoiceOption {
                    key: "a".into(),
                    description: Some(obj),
                },
                ChoiceOption {
                    key: "b".into(),
                    description: Some(empty_obj),
                },
            ],
        )
        .unwrap();
        assert_eq!(Question::from(c).option_texts(), ["a: what: x", "b: "]);
        let blank_instr = TextEntry::new("i", StateValue::object::<String>([])).unwrap();
        assert!(matches!(
            Choice::new(blank_instr, keys(&["a"])),
            Err(Error::EmptyInstructions { .. })
        ));
    }

    #[test]
    fn ambiguous_options() {
        // key "a: b" against key "a" + description "b"
        let err = Choice::new(
            instr("q?"),
            vec![
                ChoiceOption::key("a: b"),
                ChoiceOption::described("a", "b").unwrap(),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::AmbiguousOptions {
                question: None,
                i: 0,
                j: 1,
                text: "a: b".into()
            }
        );
        assert!(
            Choice::new(
                instr("q?"),
                vec![
                    ChoiceOption::key("a: b"),
                    ChoiceOption::described("a", "c").unwrap()
                ]
            )
            .is_ok()
        );
        assert!(matches!(
            Score::new(instr("q?"), vec![instr("x"), instr("y"), instr("x")]),
            Err(Error::AmbiguousOptions { i: 0, j: 2, .. })
        ));
    }

    #[test]
    fn question_size_limit_is_exact() {
        // instructions + the single option "k" = MAX bytes: accepted; one more: rejected
        let ok = "x".repeat(MAX_QUESTION_BYTES - 1);
        assert!(Choice::new(instr(&ok), keys(&["k"])).is_ok());
        let too = "x".repeat(MAX_QUESTION_BYTES);
        let err = Choice::new(instr(&too), keys(&["k"])).unwrap_err();
        assert_eq!(
            err,
            Error::QuestionTooLarge {
                question: None,
                got: MAX_QUESTION_BYTES + 1
            }
        );
    }

    fn q() -> Question {
        Choice::new(instr("q?"), keys(&["a"])).unwrap().into()
    }

    #[test]
    fn question_names() {
        for name in ["team", "équipe ✓", "with inner  spaces", "a"] {
            assert!(Questions::new([(name, q())]).is_ok(), "{name}");
        }
        assert!(Questions::new([("x".repeat(128), q())]).is_ok());
        let bad: Vec<(String, &str)> = vec![
            (String::new(), "must not be empty"),
            ("x".repeat(129), "129 characters"),
            ("a\u{0}b".into(), "U+0000 at position 1"),
            ("a\u{7f}".into(), "U+007F"),
            ("a\u{85}".into(), "U+0085"),
            (" a".into(), "leading or trailing whitespace"),
            ("a\u{a0}".into(), "leading or trailing whitespace"),
        ];
        for (name, want) in bad {
            let err = Questions::new([(name.clone(), q())]).unwrap_err();
            assert!(matches!(err, Error::InvalidQuestionName(_)), "{name:?}");
            assert!(err.to_string().contains(want), "{err}");
        }
    }

    #[test]
    fn question_set_rules() {
        assert_eq!(
            Questions::new(Vec::<(String, Question)>::new()).unwrap_err(),
            Error::NoQuestions
        );
        let many = |n: usize| -> Vec<(String, Question)> {
            (0..n).map(|i| (format!("q{i}"), q())).collect()
        };
        assert!(Questions::new(many(128)).is_ok());
        assert_eq!(
            Questions::new(many(129)).unwrap_err(),
            Error::TooManyQuestions { got: 129 }
        );
        assert_eq!(
            Questions::new([("a", q()), ("b", q()), ("a", q())]).unwrap_err(),
            Error::DuplicateQuestionName { name: "a".into() }
        );
        let qs = Questions::new([("z", q()), ("a", q())]).unwrap();
        assert_eq!(qs.iter().map(|(n, _)| n).collect::<Vec<_>>(), ["z", "a"]);
    }

    #[test]
    fn display_with_and_without_a_question_name() {
        let e = Error::InvalidOptionCount {
            kind: OptionKind::Choice,
            question: Some("team".into()),
            got: 0,
        };
        assert_eq!(
            e.to_string(),
            "choice \"team\" has 0 options; it needs between 1 and 255"
        );
        let e = Error::EmptyInstructions {
            question: Some("team".into()),
        };
        assert!(
            e.to_string()
                .starts_with("instructions of question \"team\" are empty or blank")
        );
    }
}
