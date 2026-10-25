//! The key log-probability head (spec 010): the model answers by writing the **key** of an option
//! (one token or several) after the prompt, and the score of an option is the sum of the
//! log-probabilities of its key tokens, normalised over the whole vocabulary. This is the scoring
//! of the Decision Index `transformers` engine, the reference of the format.
//!
//! The keys are in the [`Branch`] (the template tokenizes them); the engine does the work
//! ([`OutputSpec::KeyLogProbs`]). The head returns the sums as the raw logits: the temperature is
//! applied by `ask`, in one place.

use super::Head;
use crate::engine::{Output, OutputSpec};
use crate::error::{Error, Result};
use crate::prompt::{Branch, QuestionSupport, Unsupported};
use crate::schema::{Question, QuestionKind};

/// The key log-probability head, with the limits the manifest declares.
#[derive(Debug, Clone)]
pub struct KeyLogProbHead {
    max_options: usize,
    max_key_tokens: usize,
}

impl KeyLogProbHead {
    /// A head that answers questions of up to `max_options` options whose keys have at most
    /// `max_key_tokens` tokens.
    pub fn new(max_options: usize, max_key_tokens: usize) -> Self {
        KeyLogProbHead {
            max_options,
            max_key_tokens,
        }
    }

    /// The most options of a question.
    pub fn max_options(&self) -> usize {
        self.max_options
    }

    /// The most tokens of an option key.
    pub fn max_key_tokens(&self) -> usize {
        self.max_key_tokens
    }
}

impl QuestionSupport for KeyLogProbHead {
    fn check(
        &self,
        model: &str,
        name: &str,
        question: &Question,
    ) -> std::result::Result<(), Unsupported> {
        match question.kind() {
            QuestionKind::Score => Err(Unsupported::QuestionType {
                question: name.to_string(),
                kind: "score".to_string(),
                model: model.to_string(),
                supported: vec!["choice".to_string(), "noul".to_string()],
            }),
            // a yes/no question has two fixed options, whatever the limit
            QuestionKind::Choice if question.option_count() > self.max_options => {
                Err(Unsupported::HeadLimit {
                    question: name.to_string(),
                    model: model.to_string(),
                    max: self.max_options as u64,
                    got: question.option_count() as u64,
                })
            }
            _ => Ok(()),
        }
    }
}

impl Head for KeyLogProbHead {
    fn kind(&self) -> &'static str {
        "key_logprob"
    }

    fn outputs(&self, _question: &Question, branch: &Branch) -> Vec<OutputSpec> {
        vec![OutputSpec::KeyLogProbs {
            position: branch.decide(),
            keys: branch.keys().to_vec(),
        }]
    }

    fn score(&self, question: &Question, outputs: &[Output]) -> Result<Vec<f64>> {
        let Some(Output::KeyLogProbs(sums)) = outputs.first() else {
            return Err(Error::Inference {
                detail: "the key log-probability head reads key log-probabilities".to_string(),
            });
        };
        if sums.len() != question.option_count() {
            return Err(Error::Inference {
                detail: format!(
                    "the key log-probability head got {} scores for {} options",
                    sums.len(),
                    question.option_count()
                ),
            });
        }
        Ok(sums.clone())
    }
}
