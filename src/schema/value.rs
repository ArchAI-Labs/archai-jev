//! The JSON-like tree used for states and for structured text entries.
//!
//! A tree of our own (not `serde_json::Value`) because object keys must keep their insertion
//! order (it changes the prompt) and integers must keep arbitrary precision.

use crate::error::{Error, Result};

/// One node of a state (or of a structured text entry such as instructions).
#[derive(Debug, Clone, PartialEq)]
pub enum StateValue {
    /// `null`: renders as the empty string.
    Null,
    /// A boolean: renders as `True` / `False`.
    Bool(bool),
    /// An integer of any size, kept as canonical decimal text.
    Int(String),
    /// A float; non-finite values are rejected when the state is built.
    Float(f64),
    /// Unicode text, rendered verbatim.
    Str(String),
    /// An ordered list.
    List(Vec<StateValue>),
    /// An object: pairs in insertion order (duplicates are rejected when the state is built).
    Object(Vec<(String, StateValue)>),
}

impl StateValue {
    /// A string node.
    pub fn string(s: impl Into<String>) -> Self {
        StateValue::Str(s.into())
    }

    /// An integer node from an `i64`.
    pub fn int(n: i64) -> Self {
        StateValue::Int(n.to_string())
    }

    /// An integer node of arbitrary size from decimal text such as `-12345678901234567890123`.
    ///
    /// # Errors
    /// [`Error::InvalidFieldType`] if `text` is not an optional minus sign followed by digits.
    pub fn int_text(text: &str) -> Result<Self> {
        let digits = text.strip_prefix('-').unwrap_or(text);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::InvalidFieldType {
                field: "integer".to_string(),
                got: format!("text {text:?}"),
            });
        }
        let trimmed = digits.trim_start_matches('0');
        if trimmed.is_empty() {
            return Ok(StateValue::Int("0".to_string()));
        }
        let sign = if text.starts_with('-') { "-" } else { "" };
        Ok(StateValue::Int(format!("{sign}{trimmed}")))
    }

    /// An object node from `(key, value)` pairs, keeping their order.
    pub fn object<K: Into<String>>(pairs: impl IntoIterator<Item = (K, StateValue)>) -> Self {
        StateValue::Object(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A list node.
    pub fn list(items: impl IntoIterator<Item = StateValue>) -> Self {
        StateValue::List(items.into_iter().collect())
    }

    /// Human name of the node kind, as used in error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            StateValue::Null => "null",
            StateValue::Bool(_) => "boolean",
            StateValue::Int(_) | StateValue::Float(_) => "number",
            StateValue::Str(_) => "string",
            StateValue::List(_) => "array",
            StateValue::Object(_) => "object",
        }
    }

    pub(crate) fn is_container(&self) -> bool {
        matches!(self, StateValue::List(_) | StateValue::Object(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_text_is_canonical() {
        assert_eq!(
            StateValue::int_text("-0").unwrap(),
            StateValue::Int("0".into())
        );
        assert_eq!(
            StateValue::int_text("007").unwrap(),
            StateValue::Int("7".into())
        );
        assert_eq!(
            StateValue::int_text("-12345678901234567890123").unwrap(),
            StateValue::Int("-12345678901234567890123".into())
        );
        assert!(StateValue::int_text("").is_err());
        assert!(StateValue::int_text("1.5").is_err());
        assert!(StateValue::int_text("-").is_err());
    }
}
