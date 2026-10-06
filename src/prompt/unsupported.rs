//! Why a request is refused before any forward pass (D20 level 3, spec 006a section 6).
//!
//! Every reason says the limit and the value received. Indices are 0-based like the API.

use std::fmt;

use crate::error::quote;

/// A request the model cannot answer correctly; the answer is an error, never a truncation.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// The row (state + question) is longer than the model accepts.
    RowTooLong {
        question: String,
        row: u64,
        state: u64,
        branch: u64,
        max: u64,
    },
    /// The family of the model has no way to answer this kind of question.
    QuestionType {
        question: String,
        kind: String,
        model: String,
        supported: Vec<String>,
    },
    /// The head of the model cannot answer this question (too many options, ...).
    HeadLimit {
        question: String,
        model: String,
        max: u64,
        got: u64,
    },
    /// Two options that differ for the user are the same tokens for the model.
    IndistinguishableOptions {
        question: String,
        i: usize,
        j: usize,
        text: String,
    },
    /// The question is not one of the tasks the model was trained for (spec 006c section 3).
    TaskMismatch {
        question: String,
        model: String,
        detail: String,
    },
    /// The tokenizer failed on a piece of text (an internal fault, not the user's mistake).
    Untokenizable { segment: String, cause: String },
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::RowTooLong {
                question,
                row,
                state,
                branch,
                max,
            } => write!(
                f,
                "question {} needs a row of {row} tokens (state {state} + question {branch}), but the model accepts at most {max}; shorten the state or the question",
                quote(question)
            ),
            Unsupported::QuestionType {
                question,
                kind,
                model,
                supported,
            } => write!(
                f,
                "question {} is a {kind} question, which model {model} cannot answer; it supports: {}",
                quote(question),
                supported.join(", ")
            ),
            Unsupported::HeadLimit {
                question,
                model,
                max,
                got,
            } => write!(
                f,
                "question {}: the head of model {model} supports at most {max} options, got {got}",
                quote(question)
            ),
            Unsupported::IndistinguishableOptions {
                question,
                i,
                j,
                text,
            } => write!(
                f,
                "question {}: options {i} and {j} read exactly the same to the model after normalization (\"{text}\"); make them distinguishable",
                quote(question)
            ),
            Unsupported::TaskMismatch {
                question,
                model,
                detail,
            } => write!(
                f,
                "question {} does not match any task of model {model}; {detail}",
                quote(question)
            ),
            Unsupported::Untokenizable { segment, cause } => {
                write!(f, "cannot tokenize the {segment}: {cause}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_say_the_limit_and_the_value() {
        let row = Unsupported::RowTooLong {
            question: "x".into(),
            row: 20019,
            state: 20001,
            branch: 18,
            max: 8205,
        };
        assert_eq!(
            row.to_string(),
            "question \"x\" needs a row of 20019 tokens (state 20001 + question 18), but the model accepts at most 8205; shorten the state or the question"
        );
        let ty = Unsupported::QuestionType {
            question: "q".into(),
            kind: "score".into(),
            model: "m".into(),
            supported: vec!["choice".into(), "noul".into()],
        };
        assert_eq!(
            ty.to_string(),
            "question \"q\" is a score question, which model m cannot answer; it supports: choice, noul"
        );
        let head = Unsupported::HeadLimit {
            question: "q".into(),
            model: "m".into(),
            max: 4,
            got: 5,
        };
        assert_eq!(
            head.to_string(),
            "question \"q\": the head of model m supports at most 4 options, got 5"
        );
        let dup = Unsupported::IndistinguishableOptions {
            question: "q".into(),
            i: 0,
            j: 2,
            text: "é".into(),
        };
        assert!(
            dup.to_string()
                .contains("options 0 and 2 read exactly the same")
        );
        let tok = Unsupported::Untokenizable {
            segment: "state".into(),
            cause: "boom".into(),
        };
        assert_eq!(tok.to_string(), "cannot tokenize the state: boom");
    }
}
