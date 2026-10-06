//! A head for tests (feature `testing`): proof that the engine does not know the type of head.
//! It reads **only the hidden state at the last token** of the branch and gives option `j` the
//! logit `j-th value of that vector`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::Head;
use crate::engine::{Output, OutputSpec};
use crate::error::{Error, Result};
use crate::prompt::{Branch, QuestionSupport, Unsupported};
use crate::schema::Question;

/// Reads the last token only.
pub struct LastTokenHead {
    /// Most options it accepts.
    pub max_options: usize,
}

impl QuestionSupport for LastTokenHead {
    fn check(
        &self,
        model: &str,
        name: &str,
        question: &Question,
    ) -> std::result::Result<(), Unsupported> {
        if question.option_count() > self.max_options {
            return Err(Unsupported::HeadLimit {
                question: name.to_string(),
                model: model.to_string(),
                max: self.max_options as u64,
                got: question.option_count() as u64,
            });
        }
        Ok(())
    }
}

impl Head for LastTokenHead {
    fn kind(&self) -> &'static str {
        "last-token"
    }

    fn outputs(&self, _question: &Question, branch: &Branch) -> Vec<OutputSpec> {
        vec![OutputSpec::Hidden {
            position: branch.len() - 1,
        }]
    }

    fn score(&self, question: &Question, outputs: &[Output]) -> Result<Vec<f64>> {
        let Some(Output::Hidden(h)) = outputs.first() else {
            return Err(Error::Inference {
                detail: "no hidden state".to_string(),
            });
        };
        Ok(h.iter()
            .take(question.option_count())
            .map(|v| f64::from(*v))
            .collect())
    }
}
