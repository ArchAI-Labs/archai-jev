//! Decision heads (D16, spec 006b section 8): what turns what the model emits into **one raw
//! logit per option**. The engine knows nothing about a head's type: it runs the outputs the head
//! asks for. Kev's pointer head reads hidden states; the letters head of 006c reads logits.
//!
//! Heads return **raw** logits. The temperature is applied by `ask` through the calibration the
//! model was loaded with, in one place.

pub mod letters;
pub mod pointer;
#[cfg(any(test, feature = "testing"))]
pub mod test_head;

use crate::engine::{Output, OutputSpec};
use crate::error::Result;
use crate::prompt::{Branch, QuestionSupport};
use crate::schema::Question;

pub use letters::LettersHead;
pub use pointer::PointerHead;

/// A decision head.
pub trait Head: QuestionSupport {
    /// The kind written in manifests (`pointer`, `letters`).
    fn kind(&self) -> &'static str;

    /// What to read from the row of `question`, whose branch is `branch`.
    fn outputs(&self, question: &Question, branch: &Branch) -> Vec<OutputSpec>;

    /// One raw logit per option, in option order, from the numbers the engine returned for
    /// [`Head::outputs`] (same order).
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the numbers do not have the shape the head asked for.
    fn score(&self, question: &Question, outputs: &[Output]) -> Result<Vec<f64>>;
}
