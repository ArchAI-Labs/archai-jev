//! The letters head (spec 006c section 5): the model answers with one token after `Option:`; the
//! answer is the logit of the token of each option, **restricted** to those ids.
//!
//! The ids come from the manifest. In the full vocabulary these tokens have almost no mass (the
//! model was trained on the restricted softmax, spike S5 section 4), so there is no "mass on the
//! targets" check and the ids and the rule are what makes the numbers valid.

use super::Head;
use crate::engine::{Output, OutputSpec};
use crate::error::{Error, Result};
use crate::models::manifest::Target;
use crate::prompt::{Branch, QuestionSupport, Unsupported};
use crate::schema::{Question, QuestionKind};

/// The letters head.
#[derive(Debug, Clone)]
pub struct LettersHead {
    choice: Vec<u32>,
    no_yes: Option<(u32, u32)>,
}

impl LettersHead {
    /// From the manifest: the answer letters of a Choice, in option order, and `(no, yes)`.
    pub fn new(choice_targets: &[Target], yes_no: Option<&(Target, Target)>) -> Self {
        LettersHead {
            choice: choice_targets.iter().map(|t| t.id).collect(),
            no_yes: yes_no.map(|(no, yes)| (no.id, yes.id)),
        }
    }

    /// How many answer letters a Choice can have.
    pub fn max_options(&self) -> usize {
        self.choice.len()
    }

    fn ids_for(&self, question: &Question) -> Option<Vec<u32>> {
        match question.kind() {
            QuestionKind::Choice => self
                .choice
                .get(..question.option_count())
                .map(<[u32]>::to_vec),
            QuestionKind::YesNo => self.no_yes.map(|(no, yes)| vec![no, yes]),
            QuestionKind::Score => None,
        }
    }
}

impl QuestionSupport for LettersHead {
    fn check(
        &self,
        model: &str,
        name: &str,
        question: &Question,
    ) -> std::result::Result<(), Unsupported> {
        match question.kind() {
            QuestionKind::Choice if question.option_count() > self.choice.len() => {
                Err(Unsupported::HeadLimit {
                    question: name.to_string(),
                    model: model.to_string(),
                    max: self.choice.len() as u64,
                    got: question.option_count() as u64,
                })
            }
            QuestionKind::YesNo if self.no_yes.is_none() => Err(Unsupported::QuestionType {
                question: name.to_string(),
                kind: "noul".to_string(),
                model: model.to_string(),
                supported: vec!["choice".to_string()],
            }),
            QuestionKind::Score => Err(Unsupported::QuestionType {
                question: name.to_string(),
                kind: "score".to_string(),
                model: model.to_string(),
                supported: if self.no_yes.is_some() {
                    vec!["choice".to_string(), "noul".to_string()]
                } else {
                    vec!["choice".to_string()]
                },
            }),
            _ => Ok(()),
        }
    }
}

impl Head for LettersHead {
    fn kind(&self) -> &'static str {
        "letters"
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
                detail: "the letters head reads logits".to_string(),
            });
        };
        if values.len() != question.option_count() {
            return Err(Error::Inference {
                detail: format!(
                    "the letters head got {} logits for {} options",
                    values.len(),
                    question.option_count()
                ),
            });
        }
        Ok(values.iter().map(|v| f64::from(*v)).collect())
    }
}
