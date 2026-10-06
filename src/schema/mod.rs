//! Domain types: states, text entries and questions, all validated when built.

pub mod question;
pub mod render;
pub mod state;
pub mod value;

pub use question::{
    Choice, ChoiceOption, MAX_NAME_CHARS, MAX_OPTIONS, MAX_QUESTION_BYTES, MAX_QUESTIONS, Question,
    QuestionKind, Questions, Score, YesNo,
};
pub use render::{MAX_DEPTH, is_blank, is_py_space, py_float_repr};
pub use state::{MAX_STATE_BYTES, State, TextEntry};
pub use value::StateValue;
