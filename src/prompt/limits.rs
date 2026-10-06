//! The checks that refuse a request before any forward pass (D20 level 3, spec 006a section 6).
//!
//! Each one says the limit and the value received and returns an `Unsupported` reason. They run
//! per question, in the order of the request, and the first failure wins.

use std::collections::HashMap;

use super::unsupported::Unsupported;
use crate::error::{Error, Result};
use crate::schema::{Question, QuestionKind};

/// What the head of a model can and cannot answer (spec 006b section 8 implements it; 006a only
/// needs the question).
pub trait QuestionSupport: Send + Sync {
    /// `Ok` if the head can answer `question`, otherwise the reason (usually
    /// [`Unsupported::HeadLimit`]).
    ///
    /// # Errors
    /// An [`Unsupported`] reason.
    fn check(
        &self,
        model: &str,
        name: &str,
        question: &Question,
    ) -> std::result::Result<(), Unsupported>;
}

/// A head without limits of its own.
pub struct NoLimits;

impl QuestionSupport for NoLimits {
    fn check(&self, _: &str, _: &str, _: &Question) -> std::result::Result<(), Unsupported> {
        Ok(())
    }
}

impl<F> QuestionSupport for F
where
    F: Fn(&str, &str, &Question) -> std::result::Result<(), Unsupported> + Send + Sync,
{
    fn check(
        &self,
        model: &str,
        name: &str,
        question: &Question,
    ) -> std::result::Result<(), Unsupported> {
        self(model, name, question)
    }
}

/// Everything a template needs to know about the model to refuse what it cannot answer.
pub struct RequestLimits<'a> {
    /// Name of the model, for messages.
    pub model: &'a str,
    /// Most tokens in one row (state + question): `max_context` of the manifest.
    pub max_context: u64,
    /// Question types the family answers: `choice`, `noul`, `score`.
    pub question_types: &'a [&'a str],
    /// The limits of the head.
    pub support: &'a dyn QuestionSupport,
}

/// The name of a question kind in manifests and families.
pub fn kind_name(question: &Question) -> &'static str {
    match question.kind() {
        QuestionKind::Choice => "choice",
        QuestionKind::YesNo => "noul",
        QuestionKind::Score => "score",
    }
}

impl RequestLimits<'_> {
    /// Refuse a question the family or the head cannot answer.
    ///
    /// # Errors
    /// [`Unsupported::QuestionType`] or whatever the head reports.
    pub fn check_question(&self, name: &str, question: &Question) -> Result<()> {
        let kind = kind_name(question);
        if !self.question_types.contains(&kind) {
            return Err(Error::Unsupported(Unsupported::QuestionType {
                question: name.to_string(),
                kind: kind.to_string(),
                model: self.model.to_string(),
                supported: self
                    .question_types
                    .iter()
                    .map(|t| (*t).to_string())
                    .collect(),
            }));
        }
        self.support
            .check(self.model, name, question)
            .map_err(Error::Unsupported)
    }

    /// Refuse a row longer than the model accepts.
    ///
    /// # Errors
    /// [`Unsupported::RowTooLong`].
    pub fn check_row(&self, name: &str, state: usize, branch: usize) -> Result<()> {
        let row = (state as u64).saturating_add(branch as u64);
        if row > self.max_context {
            return Err(Error::Unsupported(Unsupported::RowTooLong {
                question: name.to_string(),
                row,
                state: state as u64,
                branch: branch as u64,
                max: self.max_context,
            }));
        }
        Ok(())
    }
}

/// Refuse two options that are the same tokens for the model although the user wrote them
/// differently (`é` and `e` + U+0301, `<|x|>` and `<¦x¦>`). `ids[j]` are the tokens of option `j`.
///
/// # Errors
/// [`Unsupported::IndistinguishableOptions`] for the first pair, by the later index.
pub fn check_distinct(name: &str, ids: &[Vec<u32>], texts: &[String]) -> Result<()> {
    let mut seen: HashMap<&[u32], usize> = HashMap::with_capacity(ids.len());
    for (j, tokens) in ids.iter().enumerate() {
        if let Some(&i) = seen.get(tokens.as_slice()) {
            return Err(Error::Unsupported(Unsupported::IndistinguishableOptions {
                question: name.to_string(),
                i,
                j,
                text: texts.get(j).cloned().unwrap_or_default(),
            }));
        }
        seen.insert(tokens.as_slice(), j);
    }
    Ok(())
}
