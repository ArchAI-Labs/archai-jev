//! A validated state, and the validated "text entries" (instructions, descriptions, levels).

use super::render::render_checked;
use super::value::StateValue;
use crate::error::{Error, Result};

/// Maximum size of the canonical text of a state, in bytes (1 MiB).
pub const MAX_STATE_BYTES: usize = 1_048_576;

/// A validated state: a string, an object or a list, with its canonical text computed once.
///
/// There is no way to build an invalid `State` (parse, don't validate).
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    value: StateValue,
    text: String,
}

impl State {
    /// Validate `value` and render its canonical text.
    ///
    /// # Errors
    /// [`Error::StateType`] for a null, number or boolean root; [`Error::StateTooDeep`],
    /// [`Error::StateTooLarge`], [`Error::NonFiniteNumber`] and [`Error::DuplicateStateKey`]
    /// for the other violations.
    pub fn new(value: StateValue) -> Result<Self> {
        if !matches!(
            value,
            StateValue::Str(_) | StateValue::List(_) | StateValue::Object(_)
        ) {
            return Err(Error::StateType { kind: value.kind() });
        }
        let text = render_checked(&value, MAX_STATE_BYTES, true)?;
        Ok(State { value, text })
    }

    /// The canonical text that goes after the state delimiter of Kev's template.
    pub fn canonical_text(&self) -> &str {
        &self.text
    }

    /// The tree this state was built from.
    pub fn value(&self) -> &StateValue {
        &self.value
    }
}

/// A textual entry of a question (instructions, option description, score level):
/// a string, an object or a list, kept verbatim together with its rendered text.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEntry {
    value: StateValue,
    text: String,
}

impl TextEntry {
    /// Validate `value` as the text entry called `field`.
    ///
    /// # Errors
    /// [`Error::InvalidFieldType`] if the root is null, a number or a boolean; the same limits
    /// as a state apply otherwise.
    pub fn new(field: &str, value: StateValue) -> Result<Self> {
        if !matches!(
            value,
            StateValue::Str(_) | StateValue::List(_) | StateValue::Object(_)
        ) {
            return Err(Error::InvalidFieldType {
                field: field.to_string(),
                got: value.kind().to_string(),
            });
        }
        let text = render_checked(&value, MAX_STATE_BYTES, true)?;
        Ok(TextEntry { value, text })
    }

    /// A plain string entry.
    ///
    /// # Errors
    /// [`Error::StateTooLarge`] if the string exceeds the size limit.
    pub fn text(s: &str) -> Result<Self> {
        Self::new("text", StateValue::string(s))
    }

    /// The text the model reads (Kev's `render` at indentation 0).
    pub fn rendered(&self) -> &str {
        &self.text
    }

    /// The original value, untouched.
    pub fn value(&self) -> &StateValue {
        &self.value
    }

    /// True for the empty string, which Kev treats like an absent description.
    pub(crate) fn is_empty_string(&self) -> bool {
        matches!(&self.value, StateValue::Str(s) if s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_string_object_list_are_accepted_even_empty() {
        assert!(State::new(StateValue::string("")).is_ok());
        assert!(State::new(StateValue::object::<String>([])).is_ok());
        assert!(State::new(StateValue::list([])).is_ok());
    }

    #[test]
    fn roots_null_number_bool_are_rejected_with_kind() {
        for (v, kind) in [
            (StateValue::Null, "null"),
            (StateValue::int(42), "number"),
            (StateValue::Float(1.5), "number"),
            (StateValue::Bool(true), "boolean"),
        ] {
            assert_eq!(State::new(v), Err(Error::StateType { kind }));
        }
        let msg = State::new(StateValue::Null).unwrap_err().to_string();
        assert!(msg.contains("wrap it in an object"), "{msg}");
    }

    fn nested(levels: usize) -> StateValue {
        let mut v = StateValue::list([]);
        for _ in 1..levels {
            v = StateValue::list([v]);
        }
        v
    }

    #[test]
    fn depth_32_ok_33_rejected_with_path() {
        assert!(State::new(nested(32)).is_ok());
        let err = State::new(nested(33)).unwrap_err();
        assert!(matches!(err, Error::StateTooDeep { .. }), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("[0][0]"), "{msg}");
    }

    #[test]
    fn very_deep_state_is_rejected_without_overflow() {
        // 1000 levels: built iteratively, rejected at level 33 before any recursion.
        let err = State::new(nested(1000)).unwrap_err();
        assert!(matches!(err, Error::StateTooDeep { .. }));
    }

    #[test]
    fn size_limit_is_exact() {
        assert!(State::new(StateValue::string("x".repeat(MAX_STATE_BYTES))).is_ok());
        assert_eq!(
            State::new(StateValue::string("x".repeat(MAX_STATE_BYTES + 1))),
            Err(Error::StateTooLarge)
        );
    }

    #[test]
    fn non_finite_floats_are_rejected_with_path() {
        let v = StateValue::object([("a", StateValue::list([StateValue::Float(f64::NAN)]))]);
        let err = State::new(v).unwrap_err();
        assert_eq!(
            err,
            Error::NonFiniteNumber {
                value: "NaN".into(),
                path: "a[0]".into()
            }
        );
        assert!(State::new(StateValue::list([StateValue::Float(f64::INFINITY)])).is_err());
        assert!(State::new(StateValue::list([StateValue::Float(f64::NEG_INFINITY)])).is_err());
    }

    #[test]
    fn duplicate_keys_are_rejected_with_path() {
        let v = StateValue::object([(
            "o",
            StateValue::object([("k", StateValue::Null), ("k", StateValue::Null)]),
        )]);
        assert_eq!(
            State::new(v),
            Err(Error::DuplicateStateKey {
                key: "k".into(),
                path: "o".into()
            })
        );
    }

    #[test]
    fn key_order_is_insertion_order_and_text_is_verbatim() {
        let v = StateValue::object([
            ("b", StateValue::string("1")),
            ("a", StateValue::string("2")),
        ]);
        assert_eq!(State::new(v).unwrap().canonical_text(), "b: 1\na: 2");
        let s = "e\u{301} NFD \u{0} nul \u{7f} <|fim_prefix|> \u{1f600}";
        assert_eq!(
            State::new(StateValue::string(s)).unwrap().canonical_text(),
            s
        );
    }

    #[test]
    fn text_entry_rejects_scalar_roots() {
        let err = TextEntry::new("instructions of question \"q\"", StateValue::int(1)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "instructions of question \"q\" must be a string, an object or an array, got number"
        );
    }
}
