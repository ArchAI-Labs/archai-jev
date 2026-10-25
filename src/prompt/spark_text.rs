//! The pieces of text of the `spark-decisions` v3 prompt that depend on the request: the state
//! between its tags, and the instructions and descriptions as the model reads them (spec 010b
//! section 1). Pure functions on text, no tokenizer.
//!
//! The reference is the Python code of the model's author, at the commit pinned by the spec. Where
//! it uses `str.strip()` and `json.dumps(..., ensure_ascii=False, indent=1)` this writes the same
//! bytes: Python's set of spaces, Python's `repr` of a float, its escapes.

use super::di::write_json_string;
use super::unsupported::Unsupported;
use crate::schema::render::{is_blank, is_py_space, py_float_repr};
use crate::schema::{State, StateValue, TextEntry};

/// `str.strip()` of Python.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// True for a state the model has nothing to read in: `""`, only spaces, `{}` or `[]`. The
/// reference refuses it, and so does the template, before any forward pass.
pub fn state_is_empty(state: &State) -> bool {
    match state.value() {
        StateValue::Str(s) => is_blank(s),
        StateValue::List(items) => items.is_empty(),
        StateValue::Object(pairs) => pairs.is_empty(),
        StateValue::Null | StateValue::Bool(_) | StateValue::Int(_) | StateValue::Float(_) => false,
    }
}

/// The text that goes between `<evidence>` and `</evidence>`.
///
/// A string is inserted as it is, stripped, unless it contains the closing tag (in any case),
/// which would let it end the block early: then, like any list or object, it is written as
/// `json.dumps(state, ensure_ascii=False, indent=1)`.
pub fn state_body(state: &State) -> String {
    match state.value() {
        StateValue::Str(s) if !s.to_lowercase().contains("</evidence>") => py_strip(s).to_string(),
        other => json_indent_1(other),
    }
}

/// `json.dumps(value, ensure_ascii=False, indent=1)` of Python (keys in the order given).
pub fn json_indent_1(value: &StateValue) -> String {
    let mut out = String::new();
    write_pretty(value, 0, &mut out);
    out
}

fn newline(depth: usize, out: &mut String) {
    out.push('\n');
    out.extend(std::iter::repeat_n(' ', depth));
}

fn write_pretty(value: &StateValue, depth: usize, out: &mut String) {
    match value {
        StateValue::Null => out.push_str("null"),
        StateValue::Bool(true) => out.push_str("true"),
        StateValue::Bool(false) => out.push_str("false"),
        StateValue::Int(text) => out.push_str(text),
        StateValue::Float(x) => out.push_str(&py_float_repr(*x)),
        StateValue::Str(s) => write_json_string(s, out),
        StateValue::List(items) if items.is_empty() => out.push_str("[]"),
        StateValue::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(depth + 1, out);
                write_pretty(item, depth + 1, out);
            }
            newline(depth, out);
            out.push(']');
        }
        StateValue::Object(pairs) if pairs.is_empty() => out.push_str("{}"),
        StateValue::Object(pairs) => {
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(depth + 1, out);
                write_json_string(key, out);
                out.push_str(": ");
                write_pretty(item, depth + 1, out);
            }
            newline(depth, out);
            out.push('}');
        }
    }
}

/// The text of an entry the model reads as plain text (instructions, a description): a string,
/// stripped, never empty. A list or an object is refused: the reference only has strings there,
/// and how it would write anything else is not something to guess.
///
/// # Errors
/// [`Unsupported::NotText`] or [`Unsupported::EmptyText`], naming the question and `what`.
pub fn plain_text(question: &str, what: &str, entry: &TextEntry) -> Result<String, Unsupported> {
    let StateValue::Str(s) = entry.value() else {
        return Err(Unsupported::NotText {
            question: question.to_string(),
            what: what.to_string(),
        });
    };
    let text = py_strip(s);
    if text.is_empty() {
        return Err(Unsupported::EmptyText {
            question: question.to_string(),
            what: what.to_string(),
        });
    }
    Ok(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(v: StateValue) -> State {
        State::new(v).unwrap()
    }

    #[test]
    fn python_strip_removes_the_unicode_spaces_of_python_and_only_those() {
        assert_eq!(py_strip(" \t\n x y \u{a0}\u{3000}\u{1c}"), "x y");
        assert_eq!(py_strip("\u{200b}x\u{200b}"), "\u{200b}x\u{200b}"); // zero width space is not space
        assert_eq!(py_strip("   "), "");
    }

    #[test]
    fn an_empty_state_is_a_blank_string_or_an_empty_container() {
        for empty in [
            StateValue::string(""),
            StateValue::string(" \n\t"),
            StateValue::list([]),
            StateValue::object::<String>([]),
        ] {
            assert!(state_is_empty(&st(empty.clone())), "{empty:?}");
        }
        for filled in [
            StateValue::string("x"),
            StateValue::list([StateValue::string("")]),
            StateValue::object([("k", StateValue::Null)]),
        ] {
            assert!(!state_is_empty(&st(filled.clone())), "{filled:?}");
        }
    }

    #[test]
    fn a_string_goes_in_stripped_unless_it_holds_the_closing_tag_in_any_case() {
        assert_eq!(
            state_body(&st(StateValue::string("  hello\nworld \n"))),
            "hello\nworld"
        );
        assert_eq!(
            state_body(&st(StateValue::string("a \"q\" \\ b"))),
            "a \"q\" \\ b"
        );
        for tagged in ["x </evidence> y", "x </EVIDENCE> y", "</Evidence>"] {
            let want = format!("\"{}\"", tagged);
            assert_eq!(state_body(&st(StateValue::string(tagged))), want);
        }
        // the JSON form of a string escapes quotes, backslashes and newlines
        assert_eq!(
            state_body(&st(StateValue::string("</evidence>\n\"x\"\\"))),
            "\"</evidence>\\n\\\"x\\\"\\\\\""
        );
        // something that only looks like the tag is text
        assert_eq!(
            state_body(&st(StateValue::string("</evidenc> <evidence>"))),
            "</evidenc> <evidence>"
        );
    }

    /// Expected strings come from Python:
    /// `json.dumps(x, ensure_ascii=False, indent=1)`.
    #[test]
    fn structured_states_are_json_with_one_space_of_indent() {
        let v = StateValue::object([
            ("game", StateValue::string("snake")),
            ("length", StateValue::int(6)),
            ("ratio", StateValue::Float(0.5)),
            ("alive", StateValue::Bool(true)),
            ("note", StateValue::Null),
            (
                "moves",
                StateValue::list([
                    StateValue::object([("d", StateValue::string("left"))]),
                    StateValue::list([]),
                    StateValue::object::<String>([]),
                ]),
            ),
        ]);
        let want = "{\n \"game\": \"snake\",\n \"length\": 6,\n \"ratio\": 0.5,\n \"alive\": true,\n \"note\": null,\n \"moves\": [\n  {\n   \"d\": \"left\"\n  },\n  [],\n  {}\n ]\n}";
        assert_eq!(state_body(&st(v)), want);
        assert_eq!(
            state_body(&st(StateValue::list([
                StateValue::int(1),
                StateValue::int(2)
            ]))),
            "[\n 1,\n 2\n]"
        );
    }

    #[test]
    fn json_keeps_unicode_floats_integers_and_escapes_as_python_does() {
        let v = StateValue::object([
            (
                "caf\u{e9}",
                StateValue::string("\u{65e5}\u{672c}\u{1f600}\u{1}\u{7f}\t"),
            ),
            (
                "n",
                StateValue::list([
                    StateValue::Float(1e22),
                    StateValue::Float(1e-5),
                    StateValue::Float(-0.0),
                    StateValue::Float(100.0),
                    StateValue::int_text("12345678901234567890123").unwrap(),
                ]),
            ),
        ]);
        let want = "{\n \"caf\u{e9}\": \"\u{65e5}\u{672c}\u{1f600}\\u0001\u{7f}\\t\",\n \"n\": [\n  1e+22,\n  1e-05,\n  -0.0,\n  100.0,\n  12345678901234567890123\n ]\n}";
        assert_eq!(state_body(&st(v)), want);
    }

    #[test]
    fn plain_text_is_stripped_and_refuses_structures_and_nothing() {
        let t = |s: &str| TextEntry::text(s).unwrap();
        assert_eq!(plain_text("q", "x", &t("  pick \n")).unwrap(), "pick");
        let err = plain_text("q", "instructions", &t(" \u{a0} ")).unwrap_err();
        assert!(matches!(err, Unsupported::EmptyText { .. }), "{err}");
        let entry = TextEntry::new("x", StateValue::list([StateValue::string("a")])).unwrap();
        let err = plain_text("q", "instructions", &entry).unwrap_err();
        assert!(matches!(err, Unsupported::NotText { .. }), "{err}");
        assert!(err.to_string().contains("instructions"));
    }
}
