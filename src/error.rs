//! Error type shared by the whole core.

use std::fmt;

use crate::models::failures::{ArgFailure, DownloadFailure, VerifyFailure};
use crate::models::incompat::Incompat;
use crate::prompt::Unsupported;

/// Every failure the core can report. Each variant maps to a typed Python exception
/// in the binding layer (see `python::errors`); the match there is exhaustive on purpose.
/// Fields are named after what the message says about them, so they are not documented one by one.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A required sequence was empty.
    #[error("{name} must not be empty")]
    Empty { name: &'static str },

    /// A sequence contained NaN or infinity.
    #[error("{name} must contain only finite numbers")]
    NonFinite { name: &'static str },

    /// The softmax temperature was not a finite number greater than zero.
    #[error("temperature must be a finite number > 0, got {0}")]
    InvalidTemperature(f64),

    // ---- State (category S) ----
    /// The state root is null, a number or a boolean.
    #[error(
        "state must be a string, an object or an array, got {kind}; to pass a single value, wrap it in an object, e.g. {{\"value\": 42}}"
    )]
    StateType { kind: &'static str },

    /// More than 32 nested containers.
    #[error(
        "state is nested more than 32 levels deep at {path}; flatten it or pass only the fields the questions need"
    )]
    StateTooDeep { path: String },

    /// The canonical text exceeds 1 MiB.
    #[error(
        "rendered state exceeds the maximum of 1048576 bytes; shorten it or pass only the fields the questions need"
    )]
    StateTooLarge,

    /// A float that is NaN or infinite.
    #[error(
        "state contains the non-finite number {value} at {path}; JSON cannot represent NaN or infinity"
    )]
    NonFiniteNumber { value: String, path: String },

    /// An object with a repeated key.
    #[error("state object has the key \"{key}\" more than once at {path}; keys must be unique")]
    DuplicateStateKey { key: String, path: String },

    /// A Python value that cannot become a state value (emitted by the binding's conversion).
    #[error("{message}")]
    StateConversion { message: String },

    // ---- Questions (category Q) ----
    /// No question was given.
    #[error("at least one question is required, got none")]
    NoQuestions,

    /// More than 128 questions.
    #[error(
        "{got} questions given; the maximum per request is 128; split them into several requests"
    )]
    TooManyQuestions { got: usize },

    /// A question name that is empty, too long, has control characters or edge whitespace.
    #[error("{0}")]
    InvalidQuestionName(NameProblem),

    /// A question name used twice.
    #[error(
        "question name {} is used more than once; names must be unique within a request",
        quote(name)
    )]
    DuplicateQuestionName { name: String },

    /// Instructions whose rendered text is empty or blank.
    #[error(
        "instructions {}are empty or blank; describe what the model has to decide, e.g. \"Is the customer angry?\"",
        of_question(question)
    )]
    EmptyInstructions { question: Option<String> },

    /// A Choice or Score with a number of options outside 1..=255.
    #[error("{} has {got} {}; it needs between 1 and 255", named(kind.noun(), question), kind.plural())]
    InvalidOptionCount {
        kind: OptionKind,
        question: Option<String>,
        got: usize,
    },

    /// A Choice option whose key is empty or blank.
    #[error(
        "{}: the key of option {index} is empty or blank; every option needs a key",
        named("choice", question)
    )]
    EmptyOptionKey {
        question: Option<String>,
        index: usize,
    },

    /// A Choice key used twice.
    #[error(
        "{}: the key \"{key}\" appears at options {i} and {j}; keys must be unique",
        named("choice", question)
    )]
    DuplicateOptionKey {
        question: Option<String>,
        key: String,
        i: usize,
        j: usize,
    },

    /// A Score level whose rendered text is empty or blank.
    #[error(
        "{}: level {index} is empty or blank; every level needs a text",
        named("score", question)
    )]
    EmptyLevel {
        question: Option<String>,
        index: usize,
    },

    /// Two options that read exactly the same to the model.
    #[error(
        "{}: options {i} and {j} would read exactly the same to the model (\"{text}\"); make them distinguishable",
        named("question", question)
    )]
    AmbiguousOptions {
        question: Option<String>,
        i: usize,
        j: usize,
        text: String,
    },

    /// Instructions plus options exceed 262144 bytes.
    #[error(
        "{} is {got} bytes long (instructions plus options); the maximum is 262144",
        named("question", question)
    )]
    QuestionTooLarge {
        question: Option<String>,
        got: usize,
    },

    /// Emitted by the conversion layers (Python, JSON), never by the core types themselves.
    #[error("{field} must be a string, an object or an array, got {got}")]
    InvalidFieldType { field: String, got: String },

    /// A Python value that cannot become a text entry of a question (emitted by the binding).
    #[error("{message}")]
    EntryConversion { message: String },

    // ---- Model loading (spec 005) ----
    /// A checkpoint or its manifest cannot be loaded.
    #[error("{0}")]
    IncompatibleModel(Incompat),

    /// The self-check vectors of a model disagreed with its output.
    #[error("{0}")]
    ModelVerification(VerifyFailure),

    /// A model file could not be fetched or found.
    #[error("{0}")]
    ModelDownload(DownloadFailure),

    /// A wrong argument to `from_pretrained`.
    #[error("{0}")]
    ModelArgument(ArgFailure),

    /// A request the model cannot answer correctly (context too long, unsupported question).
    #[error("{0}")]
    Unsupported(Unsupported),

    /// A feature that this build does not have yet (no inference engine, no default model).
    #[error("{detail}")]
    NotAvailable { detail: String },

    /// The user interrupted a long operation (Ctrl-C).
    #[error("the operation was cancelled")]
    Cancelled,

    // ---- Numerical (category N) ----
    /// The scorer returned NaN or infinity.
    #[error(
        "scorer returned a non-finite logit ({value}) for option {index} of question {}; refusing to compute probabilities",
        quote(question)
    )]
    NonFiniteLogit {
        question: String,
        index: usize,
        value: f64,
    },

    /// A hidden state (or logit) read from the model is NaN or infinite.
    #[error(
        "the model produced a non-finite hidden state (value {value}) at position {position} of the row for question {}; refusing to compute probabilities",
        quote(question)
    )]
    NonFiniteHidden {
        question: String,
        position: usize,
        value: f64,
    },

    /// The inference engine could not run (decoding error, memory, unsupported CPU).
    #[error("the inference engine failed: {detail}")]
    Inference { detail: String },

    /// The scorer returned the wrong number of logits.
    #[error("{0}")]
    ScorerOutputShape(ShapeProblem),

    /// Probabilities that are not a valid distribution.
    #[error(
        "probabilities for question {} are not a valid distribution: {reason}",
        quote(question)
    )]
    InvalidDistribution {
        question: String,
        reason: DistributionProblem,
    },
}

/// Result alias used across the core.
pub type Result<T> = std::result::Result<T, Error>;

/// Which kind of option list a count error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    /// A Choice question (options).
    Choice,
    /// A Score question (levels).
    Score,
}

impl OptionKind {
    fn noun(self) -> &'static str {
        match self {
            OptionKind::Choice => "choice",
            OptionKind::Score => "score",
        }
    }
    fn plural(self) -> &'static str {
        match self {
            OptionKind::Choice => "options",
            OptionKind::Score => "levels",
        }
    }
}

/// Why a question name was rejected.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameProblem {
    /// The name is empty.
    Empty,
    /// The name is longer than 128 characters.
    TooLong { name: String, got: usize },
    /// The name contains a control character.
    Control {
        name: String,
        code_point: u32,
        index: usize,
    },
    /// The name starts or ends with whitespace.
    EdgeWhitespace { name: String },
}

impl fmt::Display for NameProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameProblem::Empty => write!(f, "question name must not be empty"),
            NameProblem::TooLong { name, got } => write!(
                f,
                "question name {} is {got} characters long; the maximum is 128",
                quote(name)
            ),
            NameProblem::Control {
                name,
                code_point,
                index,
            } => write!(
                f,
                "question name {} contains the control character U+{code_point:04X} at position {index}; remove it",
                quote(name)
            ),
            NameProblem::EdgeWhitespace { name } => write!(
                f,
                "question name {} has leading or trailing whitespace; remove it",
                quote(name)
            ),
        }
    }
}

/// Which shape check failed on the scorer output.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeProblem {
    /// A question got the wrong number of logits.
    Options {
        question: String,
        got: usize,
        expected: usize,
    },
    /// The scorer answered for the wrong number of questions.
    Questions { got: usize, expected: usize },
}

impl fmt::Display for ShapeProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeProblem::Options {
                question,
                got,
                expected,
            } => write!(
                f,
                "scorer returned {got} logits for question {}, expected {expected} (one per option)",
                quote(question)
            ),
            ShapeProblem::Questions { got, expected } => write!(
                f,
                "scorer returned logits for {got} questions, expected {expected}"
            ),
        }
    }
}

/// Why a probability vector is not a valid distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionProblem {
    /// No values at all.
    Empty,
    /// A NaN or infinite value.
    NotFinite,
    /// A value below 0 or above 1.
    OutOfRange(f64),
    /// The values do not sum to 1.
    BadSum(f64),
}

impl fmt::Display for DistributionProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionProblem::Empty => write!(f, "is empty"),
            DistributionProblem::NotFinite => write!(f, "contains NaN or infinity"),
            DistributionProblem::OutOfRange(p) => write!(f, "value {p} is outside [0, 1]"),
            DistributionProblem::BadSum(s) => write!(f, "sum is {s}, expected 1 within 1e-9"),
        }
    }
}

/// A question name between quotes, cut to 64 characters.
pub(crate) fn quote(name: &str) -> String {
    let short: String = name.chars().take(64).collect();
    if name.chars().count() > 64 {
        format!("\"{short}...\"")
    } else {
        format!("\"{short}\"")
    }
}

/// `choice "q"` with a name, plain `choice` without.
fn named(kind: &str, question: &Option<String>) -> String {
    match question {
        Some(q) => format!("{kind} {}", quote(q)),
        None => kind.to_string(),
    }
}

/// `of question "q" ` with a name, nothing without (for "instructions ... are").
fn of_question(question: &Option<String>) -> String {
    match question {
        Some(q) => format!("of question {} ", quote(q)),
        None => String::new(),
    }
}

/// Which Python exception family an [`Error`] belongs to (the binding maps these to classes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Plain `ValueError` (invalid scalar arguments such as a temperature).
    Value,
    /// `InvalidStateError`.
    State,
    /// `InvalidQuestionError`.
    Question,
    /// `NumericalError`.
    Numerical,
    /// `UnsupportedRequestError`.
    Unsupported,
    /// `InferenceError`.
    Inference,
    /// `IncompatibleModelError`.
    Incompatible,
    /// `ModelVerificationError`.
    Verification,
    /// `ModelDownloadError`.
    Download,
    /// Plain `ValueError` from a wrong `from_pretrained` argument.
    Argument,
    /// `FileNotFoundError`.
    PathNotFound,
    /// `KeyboardInterrupt`.
    Cancelled,
    /// `NotImplementedError`.
    NotImplemented,
}

impl Error {
    /// The exception family of this error. Exhaustive on purpose: a new variant must choose one.
    pub fn category(&self) -> Category {
        match self {
            Error::Empty { .. } | Error::NonFinite { .. } | Error::InvalidTemperature(_) => {
                Category::Value
            }
            Error::StateType { .. }
            | Error::StateTooDeep { .. }
            | Error::StateTooLarge
            | Error::NonFiniteNumber { .. }
            | Error::DuplicateStateKey { .. }
            | Error::StateConversion { .. } => Category::State,
            Error::NoQuestions
            | Error::TooManyQuestions { .. }
            | Error::InvalidQuestionName(_)
            | Error::DuplicateQuestionName { .. }
            | Error::EmptyInstructions { .. }
            | Error::InvalidOptionCount { .. }
            | Error::EmptyOptionKey { .. }
            | Error::DuplicateOptionKey { .. }
            | Error::EmptyLevel { .. }
            | Error::AmbiguousOptions { .. }
            | Error::QuestionTooLarge { .. }
            | Error::InvalidFieldType { .. }
            | Error::EntryConversion { .. } => Category::Question,
            Error::Cancelled => Category::Cancelled,
            Error::NotAvailable { .. } => Category::NotImplemented,
            Error::Unsupported(_) => Category::Unsupported,
            Error::IncompatibleModel(_) => Category::Incompatible,
            Error::ModelVerification(_) => Category::Verification,
            Error::ModelDownload(_) => Category::Download,
            Error::ModelArgument(ArgFailure::PathNotFound { .. }) => Category::PathNotFound,
            Error::ModelArgument(_) => Category::Argument,
            Error::Inference { .. } => Category::Inference,
            Error::NonFiniteLogit { .. }
            | Error::NonFiniteHidden { .. }
            | Error::ScorerOutputShape(_)
            | Error::InvalidDistribution { .. } => Category::Numerical,
        }
    }
}
