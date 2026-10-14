//! The `decision-index-chatml` v1 template (spec 010): the prompt of the Decision Index
//! `transformers` engine inside the ChatML chat template of Qwen2.5.
//!
//! ```text
//! <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! with `body` = `State:`, the state, blank line, `Question:`, the instructions, blank line,
//! `Options:`, one `key: description` line per option, blank line, `Reply with exactly one option
//! key.`. Each question is **one row** (the whole prompt): there is no separate state. The answer
//! is the key of one option, written after the prompt, so the branch also carries the token ids
//! of every key (the head scores them, [`crate::heads::KeyLogProbHead`]).
//!
//! The text of the reference is `json.dumps(x, ensure_ascii=False, separators=(",", ":"))` for
//! anything that is not a string; [`di_text`] writes the same bytes, floats and escapes included.
//! The user's text is sanitised first (`<|name|>` neutralised, D36), which the reference does not
//! do: the one declared difference.

use super::chatml::prefix_cut;
use super::limits::{RequestLimits, check_distinct};
use super::prompt::{Branch, Cut, CutKind, Prompt};
use super::sanitize::sanitize;
use super::template::Template;
use super::tokenize::PromptTokenizer;
use super::unsupported::Unsupported;
use crate::error::{Error, Result};
use crate::schema::{Question, Questions, State, StateValue};

const ROLES: [&str; 2] = ["im_start", "im_end"];

/// The system message of the reference engine.
pub const SYSTEM_PROMPT: &str = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain.";

/// The `decision-index-chatml` template, version 1.
pub struct DecisionIndexChatMl {
    max_key_tokens: usize,
}

/// The instance the template table hands out (only for the consistency checks).
pub static DECISION_INDEX_CHATML_V1: DecisionIndexChatMl = DecisionIndexChatMl {
    max_key_tokens: usize::MAX,
};

impl DecisionIndexChatMl {
    /// The template of a model whose head scores keys of at most `max_key_tokens` tokens.
    pub fn new(max_key_tokens: usize) -> Self {
        DecisionIndexChatMl { max_key_tokens }
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))` of Python.
pub fn di_json(value: &StateValue) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

/// What the reference engine writes for a state, an instruction or a description: a string as it
/// is, anything else as compact JSON.
pub fn di_text(value: &StateValue) -> String {
    match value {
        StateValue::Str(s) => s.clone(),
        other => di_json(other),
    }
}

fn write_json(value: &StateValue, out: &mut String) {
    match value {
        StateValue::Null => out.push_str("null"),
        StateValue::Bool(true) => out.push_str("true"),
        StateValue::Bool(false) => out.push_str("false"),
        StateValue::Int(text) => out.push_str(text),
        StateValue::Float(x) => out.push_str(&crate::schema::render::py_float_repr(*x)),
        StateValue::Str(s) => write_json_string(s, out),
        StateValue::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        StateValue::Object(pairs) => {
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                write_json(item, out);
            }
            out.push('}');
        }
    }
}

/// A JSON string with Python's escapes for `ensure_ascii=False`: the quote, the backslash and the
/// control characters (`\n \r \t \b \f` by name, the others as `\u00xx`); everything else, DEL and
/// non-ASCII included, as it is.
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The state as the reference writes it; an empty one (`""`, `null`, `{}`, `[]`) is `(empty)`.
fn state_text(state: &State) -> String {
    let empty = match state.value() {
        StateValue::Null => true,
        StateValue::Str(s) => s.is_empty(),
        StateValue::List(items) => items.is_empty(),
        StateValue::Object(pairs) => pairs.is_empty(),
        _ => false,
    };
    if empty {
        "(empty)".to_string()
    } else {
        di_text(state.value())
    }
}

/// The options of a question as `(key, description)`: the keys and descriptions of a `Choice`
/// (`None` = the key stands for itself), or the fixed `false`/`true` of a yes/no question.
fn options_of(question: &Question) -> Vec<(String, Option<String>)> {
    match question {
        Question::Choice(choice) => choice
            .options()
            .iter()
            .map(|o| {
                (
                    o.key.clone(),
                    o.description.as_ref().map(|d| di_text(d.value())),
                )
            })
            .collect(),
        _ => vec![
            ("false".to_string(), Some("False".to_string())),
            ("true".to_string(), Some("True".to_string())),
        ],
    }
}

/// The user message of one question; `state` is the text of the state, already sanitised.
fn body(
    state: &str,
    question: &Question,
    keys: &[String],
    descriptions: &[Option<String>],
) -> String {
    let mut lines: Vec<String> = vec![
        "State:".to_string(),
        state.to_string(),
        String::new(),
        "Question:".to_string(),
        sanitize(&di_text(question.instructions().value())).into_owned(),
        String::new(),
        "Options:".to_string(),
    ];
    for (key, description) in keys.iter().zip(descriptions) {
        match description {
            None => lines.push(format!("{key}: {key}")),
            Some(d) => lines.push(format!("{key}: {}", sanitize(d))),
        }
    }
    lines.push(String::new());
    lines.push("Reply with exactly one option key.".to_string());
    lines.join("\n")
}

impl Template for DecisionIndexChatMl {
    fn id(&self) -> &'static str {
        "decision-index-chatml"
    }

    fn version(&self) -> u64 {
        1
    }

    fn roles(&self) -> &'static [&'static str] {
        &ROLES
    }

    fn build(
        &self,
        state: &State,
        questions: &Questions,
        tokenizer: &PromptTokenizer,
        limits: &RequestLimits<'_>,
    ) -> Result<Prompt> {
        let [im_start, im_end] = tokenizer.roles().require(ROLES)?;
        let mut branches = Vec::with_capacity(questions.len());
        for (name, question) in questions.iter() {
            limits.check_question(name, question)?;
            let options = options_of(question);
            // the keys are user text too: neutralised before they are tokenized and written
            let keys: Vec<String> = options
                .iter()
                .map(|(k, _)| sanitize(k).into_owned())
                .collect();
            let descriptions: Vec<Option<String>> = options.into_iter().map(|(_, d)| d).collect();

            let mut key_ids = Vec::with_capacity(keys.len());
            for (j, key) in keys.iter().enumerate() {
                let ids = tokenizer.encode_raw(key, &format!("key {j} of question \"{name}\""))?;
                if ids.is_empty() {
                    return Err(Error::Unsupported(Unsupported::KeyWithoutTokens {
                        question: name.to_string(),
                        option: j,
                        key: key.clone(),
                    }));
                }
                if ids.len() > self.max_key_tokens {
                    return Err(Error::Unsupported(Unsupported::KeyTooLong {
                        question: name.to_string(),
                        option: j,
                        key: key.clone(),
                        max: self.max_key_tokens as u64,
                        got: ids.len() as u64,
                    }));
                }
                key_ids.push(ids);
            }
            check_distinct(name, &key_ids, &keys)?;

            let mut ids = vec![im_start];
            ids.extend(tokenizer.encode_raw(&format!("system\n{SYSTEM_PROMPT}"), "system prompt")?);
            ids.push(im_end);
            ids.extend(tokenizer.encode_raw("\n", "newline")?);
            ids.push(im_start);
            let user_start = ids.len();
            let state_piece = sanitize(&state_text(state)).into_owned();
            let user_ids = tokenizer.encode_raw(
                &format!(
                    "user\n{}",
                    body(&state_piece, question, &keys, &descriptions)
                ),
                &format!("text of question \"{name}\""),
            )?;
            // Cut points, only where the text before the cut, tokenized alone, gives the first
            // tokens of the whole user message (a token that merges across the boundary drops the
            // cut for this row): after `State:\n` (fixed, the same for every request) and after
            // the state and its blank line (the state, shared by the questions about it).
            let mut cuts = Vec::new();
            let head = tokenizer.encode_raw("user\nState:\n", "head of the user text")?;
            if let Some(pos) = prefix_cut(&user_ids, &head, user_start) {
                cuts.push(Cut {
                    pos,
                    kind: CutKind::Fixed,
                });
            }
            let upto = tokenizer.encode_raw(
                &format!("user\nState:\n{state_piece}\n\n"),
                "state of the user text",
            )?;
            if let Some(pos) = prefix_cut(&user_ids, &upto, user_start)
                && cuts.last().is_none_or(|last| pos > last.pos)
            {
                cuts.push(Cut {
                    pos,
                    kind: CutKind::Content,
                });
            }
            ids.extend(user_ids);
            ids.push(im_end);
            ids.extend(tokenizer.encode_raw("\n", "newline")?);
            ids.push(im_start);
            ids.extend(tokenizer.encode_raw("assistant\n", "answer prefix")?);

            // the longest key is written after the prompt: it has to fit too (as in the reference)
            let longest = key_ids.iter().map(Vec::len).max().unwrap_or(0);
            let row = ids.len() + longest;
            // the part of the row that is the state (with its blank line), for the message
            let state_len = upto.len().saturating_sub(head.len()).min(row);
            limits.check_row(name, state_len, row - state_len)?;
            let decide = ids.len() - 1;
            branches.push(Branch::with_cuts(ids, decide, Vec::new(), cuts).with_keys(key_ids));
        }
        Ok(Prompt::new(Vec::new(), branches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::limits::NoLimits;
    use crate::prompt::tokenize::testkit;
    use crate::schema::{Choice, ChoiceOption, TextEntry, YesNo};

    fn tok() -> PromptTokenizer {
        testkit::byte_level(&[("im_start", "<|im_start|>"), ("im_end", "<|im_end|>")])
    }

    fn st(v: StateValue) -> State {
        State::new(v).unwrap()
    }

    // ---- the JSON of the reference, byte for byte --------------------------------------------
    // Expected strings come from Python: json.dumps(x, ensure_ascii=False, separators=(",", ":"))

    #[test]
    fn json_objects_strings_and_literals_equal_pythons() {
        let v = StateValue::object([
            ("message", StateValue::string("my card, \"x\" \\ done")),
            (
                "n",
                StateValue::list([
                    StateValue::int(1),
                    StateValue::Float(2.5),
                    StateValue::Bool(true),
                    StateValue::Null,
                ]),
            ),
            (
                "\u{e9}",
                StateValue::object([("k", StateValue::string(""))]),
            ),
        ]);
        assert_eq!(
            di_json(&v),
            "{\"message\":\"my card, \\\"x\\\" \\\\ done\",\"n\":[1,2.5,true,null],\"\u{e9}\":{\"k\":\"\"}}"
        );
    }

    #[test]
    fn json_escapes_controls_by_name_or_u00xx_and_leaves_del_and_unicode_alone() {
        let v = StateValue::object([(
            "a",
            StateValue::string("\u{1}\u{1f}\n\r\t\u{8}\u{c}\u{7f} \u{e9}\u{65e5}\u{672c}\u{2028}"),
        )]);
        assert_eq!(
            di_json(&v),
            "{\"a\":\"\\u0001\\u001f\\n\\r\\t\\b\\f\u{7f} \u{e9}\u{65e5}\u{672c}\u{2028}\"}"
        );
    }

    #[test]
    fn json_floats_and_integers_equal_pythons_repr() {
        let floats = [
            1.0,
            0.1,
            1e22,
            1e-5,
            123_456_789_012_345_680.0,
            -0.0,
            1.5e300,
            100.0,
            1e16,
            12_345_678.9,
        ];
        let v = StateValue::list(floats.map(StateValue::Float));
        assert_eq!(
            di_json(&v),
            "[1.0,0.1,1e+22,1e-05,1.2345678901234568e+17,-0.0,1.5e+300,100.0,1e+16,12345678.9]"
        );
        let big = StateValue::list([
            StateValue::int_text("12345678901234567890123").unwrap(),
            StateValue::int(-7),
            StateValue::int(0),
        ]);
        assert_eq!(di_json(&big), "[12345678901234567890123,-7,0]");
        let nest = StateValue::object([
            ("a", StateValue::list([])),
            ("b", StateValue::object::<String>([])),
            ("c", StateValue::list([StateValue::list([])])),
        ]);
        assert_eq!(di_json(&nest), "{\"a\":[],\"b\":{},\"c\":[[]]}");
    }

    #[test]
    fn a_string_is_written_as_it_is_and_the_empty_state_is_empty() {
        assert_eq!(
            di_text(&StateValue::string("a \"quoted\" text")),
            "a \"quoted\" text"
        );
        assert_eq!(state_text(&st(StateValue::string(""))), "(empty)");
        assert_eq!(state_text(&st(StateValue::list([]))), "(empty)");
        assert_eq!(state_text(&st(StateValue::object::<String>([]))), "(empty)");
        assert_eq!(state_text(&st(StateValue::string("x"))), "x");
        assert_eq!(
            state_text(&st(StateValue::list([StateValue::string("x")]))),
            "[\"x\"]"
        );
    }

    // ---- the prompt, built from a literal written in the test ------------------------------------

    fn expected_ids(tk: &PromptTokenizer, body: &str) -> Vec<u32> {
        let [s, e] = tk.roles().require(ROLES).unwrap();
        let enc = |t: &str| tk.encode_raw(t, "t").unwrap();
        let mut ids = vec![s];
        ids.extend(enc(&format!("system\n{SYSTEM_PROMPT}")));
        ids.push(e);
        ids.extend(enc("\n"));
        ids.push(s);
        ids.extend(enc(&format!("user\n{body}")));
        ids.push(e);
        ids.extend(enc("\n"));
        ids.push(s);
        ids.extend(enc("assistant\n"));
        ids
    }

    fn build(state: &State, questions: &Questions, max_key_tokens: usize) -> Result<Prompt> {
        let tk = tok();
        let types = ["choice", "noul"];
        let limits = RequestLimits {
            model: "m",
            max_context: 10_000,
            question_types: &types,
            support: &NoLimits,
        };
        DecisionIndexChatMl::new(max_key_tokens).build(state, questions, &tk, &limits)
    }

    fn one(name: &str, q: Question) -> Questions {
        Questions::new([(name, q)]).unwrap()
    }

    fn choice(instr: &str, options: Vec<ChoiceOption>) -> Question {
        Choice::new(TextEntry::text(instr).unwrap(), options)
            .unwrap()
            .into()
    }

    #[test]
    fn a_choice_with_a_dict_state_is_the_documented_prompt() {
        let tk = tok();
        let state = st(StateValue::object([(
            "message",
            StateValue::string("my card still hasn't arrived after two weeks"),
        )]));
        let q = choice(
            "Route the customer request.",
            vec![
                ChoiceOption::described("card_arrival", "Card not delivered yet").unwrap(),
                ChoiceOption::described("top_up", "Top up problem").unwrap(),
                ChoiceOption::key("lost_card"),
            ],
        );
        let prompt = build(&state, &one("route", q), 16).unwrap();
        let body = "State:\n{\"message\":\"my card still hasn't arrived after two weeks\"}\n\nQuestion:\nRoute the customer request.\n\nOptions:\ncard_arrival: Card not delivered yet\ntop_up: Top up problem\nlost_card: lost_card\n\nReply with exactly one option key.";
        let branch = &prompt.branches()[0];
        assert_eq!(branch.ids(), expected_ids(&tk, body).as_slice());
        assert_eq!(branch.decide(), branch.len() - 1);
        assert!(prompt.state().is_empty());
        let keys: Vec<Vec<u32>> = ["card_arrival", "top_up", "lost_card"]
            .iter()
            .map(|k| tk.encode_raw(k, "k").unwrap())
            .collect();
        assert_eq!(branch.keys(), keys.as_slice());
    }

    #[test]
    fn a_yes_no_question_has_the_fixed_false_true_options_and_ignores_its_descriptions() {
        let tk = tok();
        let described = YesNo::new(
            TextEntry::text("Is the review positive?").unwrap(),
            Some(TextEntry::text("sure it is").unwrap()),
            Some(TextEntry::text("no way").unwrap()),
        )
        .unwrap();
        let plain = YesNo::new(
            TextEntry::text("Is the review positive?").unwrap(),
            None,
            None,
        )
        .unwrap();
        let state = st(StateValue::string("The movie was great"));
        let body = "State:\nThe movie was great\n\nQuestion:\nIs the review positive?\n\nOptions:\nfalse: False\ntrue: True\n\nReply with exactly one option key.";
        for q in [described, plain] {
            let prompt = build(&state, &one("q", q.into()), 8).unwrap();
            assert_eq!(
                prompt.branches()[0].ids(),
                expected_ids(&tk, body).as_slice()
            );
            let keys = [
                tk.encode_raw("false", "k").unwrap(),
                tk.encode_raw("true", "k").unwrap(),
            ];
            assert_eq!(prompt.branches()[0].keys(), keys.as_slice());
        }
    }

    #[test]
    fn structured_instructions_and_descriptions_are_compact_json_and_an_empty_state_is_written() {
        let tk = tok();
        let instructions = TextEntry::new(
            "instructions",
            StateValue::object([
                ("task", StateValue::string("route")),
                ("max", StateValue::int(2)),
            ]),
        )
        .unwrap();
        let option = ChoiceOption {
            key: "a".into(),
            description: Some(
                TextEntry::new("d", StateValue::list([StateValue::string("x")])).unwrap(),
            ),
        };
        let q = Choice::new(instructions, vec![option, ChoiceOption::key("b")]).unwrap();
        let prompt = build(&st(StateValue::string("")), &one("q", q.into()), 8).unwrap();
        let body = "State:\n(empty)\n\nQuestion:\n{\"task\":\"route\",\"max\":2}\n\nOptions:\na: [\"x\"]\nb: b\n\nReply with exactly one option key.";
        assert_eq!(
            prompt.branches()[0].ids(),
            expected_ids(&tk, body).as_slice()
        );
    }

    #[test]
    fn every_question_is_its_own_row_with_no_shared_state() {
        let state = st(StateValue::string("same state"));
        let qs = Questions::new([
            (
                "a",
                choice(
                    "First?",
                    vec![ChoiceOption::key("x"), ChoiceOption::key("y")],
                ),
            ),
            (
                "b",
                choice(
                    "Second?",
                    vec![ChoiceOption::key("p"), ChoiceOption::key("q")],
                ),
            ),
        ])
        .unwrap();
        let prompt = build(&state, &qs, 8).unwrap();
        assert!(prompt.state().is_empty());
        assert_eq!(prompt.branches().len(), 2);
        assert_ne!(prompt.branches()[0].ids(), prompt.branches()[1].ids());
    }

    #[test]
    fn the_cuts_are_after_the_state_header_and_after_the_state() {
        let tk = tok();
        let st_text = "some state to share";
        let pick = || {
            choice(
                "Pick one.",
                vec![ChoiceOption::key("a"), ChoiceOption::key("b")],
            )
        };
        let prompt = build(&st(StateValue::string(st_text)), &one("q", pick()), 16).unwrap();
        let branch = &prompt.branches()[0];
        // the row is im_start + system + im_end + newline + im_start, then the user message
        let enc = |t: &str| tk.encode_raw(t, "t").unwrap().len();
        let before_user = 1 + enc(&format!("system\n{SYSTEM_PROMPT}")) + 1 + enc("\n") + 1;
        let cuts: Vec<(usize, CutKind)> = branch.cuts().iter().map(|c| (c.pos, c.kind)).collect();
        assert_eq!(
            cuts,
            vec![
                (before_user + enc("user\nState:\n"), CutKind::Fixed),
                (
                    before_user + enc(&format!("user\nState:\n{st_text}\n\n")),
                    CutKind::Content
                ),
            ]
        );
        // two questions about the same state share every token up to the end of the state
        let other = choice(
            "Another one?",
            vec![ChoiceOption::key("x"), ChoiceOption::key("y")],
        );
        let both = build(
            &st(StateValue::string(st_text)),
            &Questions::new([("a", pick()), ("b", other)]).unwrap(),
            16,
        )
        .unwrap();
        let (a, b) = (&both.branches()[0], &both.branches()[1]);
        let end = a.cuts()[1].pos;
        assert_eq!(a.ids()[..end], b.ids()[..end]);
    }

    // ---- what is refused, before any forward pass ------------------------------------------------

    fn unsupported(r: Result<Prompt>) -> Unsupported {
        match r {
            Err(Error::Unsupported(u)) => u,
            other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn a_key_of_more_tokens_than_the_head_scores_is_refused_with_the_limit() {
        let q = choice(
            "Pick",
            vec![ChoiceOption::key("ab"), ChoiceOption::key("abcde")],
        );
        let u = unsupported(build(&st(StateValue::string("s")), &one("q", q), 4));
        assert_eq!(
            u,
            Unsupported::KeyTooLong {
                question: "q".into(),
                option: 1,
                key: "abcde".into(),
                max: 4,
                got: 5
            }
        );
        assert!(u.to_string().contains("5 tokens long") && u.to_string().contains("at most 4"));
    }

    #[test]
    fn keys_that_are_the_same_tokens_are_refused() {
        // "é" and "e" + U+0301 are one character after NFC: the same tokens
        let q = choice(
            "Pick",
            vec![ChoiceOption::key("\u{e9}"), ChoiceOption::key("e\u{301}")],
        );
        let u = unsupported(build(&st(StateValue::string("s")), &one("q", q), 8));
        assert!(
            matches!(u, Unsupported::IndistinguishableOptions { i: 0, j: 1, .. }),
            "{u}"
        );
    }

    #[test]
    fn a_row_with_its_longest_key_that_does_not_fit_is_refused_never_truncated() {
        let tk = tok();
        let types = ["choice", "noul"];
        let q = choice(
            "Pick",
            vec![ChoiceOption::key("ab"), ChoiceOption::key("cdef")],
        );
        let qs = one("q", q);
        let state = st(StateValue::string("s"));
        let probe = build(&state, &qs, 8).unwrap();
        let fits = probe.branches()[0].len() as u64 + 4; // the prompt plus the longest key
        let build_with = |max_context: u64| {
            let limits = RequestLimits {
                model: "m",
                max_context,
                question_types: &types,
                support: &NoLimits,
            };
            DecisionIndexChatMl::new(8).build(&state, &qs, &tk, &limits)
        };
        assert!(build_with(fits).is_ok());
        let u = unsupported(build_with(fits - 1));
        assert!(
            matches!(u, Unsupported::RowTooLong { max, .. } if max == fits - 1),
            "{u}"
        );
    }

    #[test]
    fn the_refusal_of_a_long_row_says_how_long_the_state_is() {
        let tk = tok();
        let types = ["choice", "noul"];
        let limits = RequestLimits {
            model: "m",
            max_context: 40,
            question_types: &types,
            support: &NoLimits,
        };
        let q = choice("Pick", vec![ChoiceOption::key("a"), ChoiceOption::key("b")]);
        let text = "x".repeat(300); // one token per byte in the test tokenizer
        let state = st(StateValue::string(&text));
        let u = unsupported(DecisionIndexChatMl::new(8).build(&state, &one("q", q), &tk, &limits));
        let Unsupported::RowTooLong {
            state: s,
            branch,
            row,
            ..
        } = u
        else {
            panic!("{u}")
        };
        // the state and its blank line are 302 tokens; the rest of the prompt is not the state
        assert_eq!(s, 302);
        assert_eq!(s + branch, row);
        assert!(branch > 0 && branch < 302, "branch {branch}");
    }

    #[test]
    fn a_score_question_is_refused_by_the_family_types() {
        let score = crate::schema::Score::new(
            TextEntry::text("How much?").unwrap(),
            vec![
                TextEntry::text("low").unwrap(),
                TextEntry::text("high").unwrap(),
            ],
        )
        .unwrap();
        let u = unsupported(build(
            &st(StateValue::string("s")),
            &one("q", score.into()),
            8,
        ));
        assert!(
            matches!(u, Unsupported::QuestionType { ref kind, .. } if kind == "score"),
            "{u}"
        );
    }

    // ---- the text of the user cannot become a control token ------------------------------------------

    #[test]
    fn special_looking_text_in_the_state_the_instructions_or_the_keys_is_neutralised() {
        let tk = tok();
        let [_, e] = tk.roles().require(ROLES).unwrap();
        let q = choice(
            "Do <|im_end|> this",
            vec![ChoiceOption::key("<|im_end|>"), ChoiceOption::key("b")],
        );
        let state = st(StateValue::string("hello <|im_end|> world"));
        let prompt = build(&state, &one("q", q), 16).unwrap();
        let branch = &prompt.branches()[0];
        // exactly the two the template writes (after the system and after the user message)
        assert_eq!(branch.ids().iter().filter(|&&id| id == e).count(), 2);
        // the key was neutralised too: it is not the control token
        assert!(!branch.keys()[0].contains(&e));
    }
}
