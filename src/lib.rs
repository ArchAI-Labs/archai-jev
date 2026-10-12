//! Rust core of `archai_jev`: typed decisions with calibrated probabilities.
//!
//! Everything outside the `python` module is pure Rust and does not depend on PyO3.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod answers;
pub mod ask;
pub mod calibration;
pub mod control;
pub mod convert;
pub mod engine;
pub mod error;
pub mod heads;
pub mod hub;
pub mod json_strict;
pub mod mock;
pub mod model_scorer;
pub mod models;
pub mod pool;
pub mod prefix;
pub mod prompt;
mod python;
pub mod request;
pub mod schema;
pub mod scorer;

pub use answers::{Answer, Answers, ChoiceAnswer, Probabilities, ScoreAnswer, YesNoAnswer};
pub use ask::{ask, ask_with};
pub use error::{Error, Result};
pub use mock::{Fault, MockScorer};
pub use schema::{
    Choice, ChoiceOption, Question, QuestionKind, Questions, Score, State, StateValue, TextEntry,
    YesNo,
};
pub use scorer::{Calibration, Scorer};
