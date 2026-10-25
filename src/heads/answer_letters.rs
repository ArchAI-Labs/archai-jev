//! The answer-letters head (spec 010b): the model answers a multiple-choice question with the
//! letter of an option, and the answer is the logit of the letters `A`, `B`, ... at the **last
//! position** of the prompt, **restricted** to the letters of the question.
//!
//! The ids come from the manifest (checked against the tokenizer at load time). The head returns
//! the raw logit of the first *K* letters for a question of *K* options; `ask` applies the
//! temperature and the softmax over those *K* values only, in one place (006c). A softmax over
//! the whole vocabulary would be a different number: the letters have little mass there.
//!
//! The options are positional: option *i* of the question is read at letter *i*. A yes/no
//! question is two options, `false` then `true` (the order of the answers: `no`, `yes`).

use super::Head;
use crate::engine::{Output, OutputSpec};
use crate::error::{Error, Result};
use crate::models::manifest::Target;
use crate::prompt::{Branch, QuestionSupport, Unsupported};
use crate::schema::{Question, QuestionKind};

/// The answer-letters head.
#[derive(Debug, Clone)]
pub struct AnswerLettersHead {
    letters: Vec<u32>,
    max_options: usize,
}

impl AnswerLettersHead {
    /// From the manifest: the ids of `A`, `B`, ... in order, and the most options of a question.
    pub fn new(letters: &[Target], max_options: usize) -> Self {
        AnswerLettersHead {
            letters: letters.iter().map(|t| t.id).collect(),
            max_options,
        }
    }

    /// How many options a question can have.
    pub fn max_options(&self) -> usize {
        self.max_options.min(self.letters.len())
    }

    /// The ids to read for `question`: the first *K* letters.
    fn ids_for(&self, question: &Question) -> Option<Vec<u32>> {
        match question.kind() {
            QuestionKind::Choice | QuestionKind::YesNo => self
                .letters
                .get(..question.option_count())
                .map(<[u32]>::to_vec),
            QuestionKind::Score => None,
        }
    }
}

impl QuestionSupport for AnswerLettersHead {
    fn check(
        &self,
        model: &str,
        name: &str,
        question: &Question,
    ) -> std::result::Result<(), Unsupported> {
        if question.kind() == QuestionKind::Score {
            return Err(Unsupported::QuestionType {
                question: name.to_string(),
                kind: "score".to_string(),
                model: model.to_string(),
                supported: vec!["choice".to_string(), "noul".to_string()],
            });
        }
        let got = question.option_count();
        // the declared limit first; a letter without an id is a limit of its own (the manifest
        // keeps `max_options` within the letters, this is the defence if it ever does not)
        for max in [self.max_options, self.letters.len()] {
            if got > max {
                return Err(Unsupported::HeadLimit {
                    question: name.to_string(),
                    model: model.to_string(),
                    max: max as u64,
                    got: got as u64,
                });
            }
        }
        Ok(())
    }
}

impl Head for AnswerLettersHead {
    fn kind(&self) -> &'static str {
        "answer_letters"
    }

    fn outputs(&self, question: &Question, branch: &Branch) -> Vec<OutputSpec> {
        vec![OutputSpec::Logits {
            position: branch.decide(),
            ids: self.ids_for(question).unwrap_or_default(),
        }]
    }

    fn score(&self, question: &Question, outputs: &[Output]) -> Result<Vec<f64>> {
        let Some(Output::Logits(values)) = outputs.first() else {
            return Err(Error::Inference {
                detail: "the answer-letters head reads logits".to_string(),
            });
        };
        if values.len() != question.option_count() {
            return Err(Error::Inference {
                detail: format!(
                    "the answer-letters head got {} logits for {} options",
                    values.len(),
                    question.option_count()
                ),
            });
        }
        Ok(values.iter().map(|v| f64::from(*v)).collect())
    }
}
