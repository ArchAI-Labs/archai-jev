//! Why the input of a calibration function is rejected, and why a calibration is refused.
//!
//! The messages are part of the public contract (spec 004 §2.3 and §4): they name the row and
//! the position (0-based) and say what to do.

use std::fmt;

/// An argument of a calibration function is not valid. Maps to `ValueError`.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum RowsProblem {
    /// `rows` and `labels` have different lengths.
    LengthMismatch {
        name: &'static str,
        rows: usize,
        labels: usize,
    },
    /// A row of logits with one option cannot be used to fit a temperature.
    SingleOption { row: usize },
    /// A row has no option at all.
    EmptyRow { name: &'static str, row: usize },
    /// A row of a yes/no batch does not have exactly two logits.
    NotAPair { row: usize, got: usize },
    /// The logits of a row span more than the largest float, so the fit cannot be computed.
    RangeTooLarge { row: usize },
    /// `1 / lower` is not a finite number.
    LowerBoundTooSmall { lower: f64 },
    /// A logit is NaN or infinite.
    NonFiniteValue {
        name: &'static str,
        row: usize,
        option: usize,
        value: f64,
    },
    /// A label does not name an option of its row.
    LabelOutOfRange {
        row: usize,
        label: i64,
        options: usize,
    },
    /// A row of probabilities contains NaN or infinity.
    ProbabilityNotFinite { row: usize },
    /// A probability is below 0 or above 1.
    ProbabilityOutOfRange {
        row: usize,
        option: usize,
        value: f64,
    },
    /// The probabilities of a row do not sum to 1 within the tolerance.
    ProbabilitySum {
        row: usize,
        sum: f64,
        tolerance: f64,
    },
    /// `n_bins` is not an integer in `1..=1000`. `got` is the text of what was given.
    InvalidBins { got: String },
    /// `bounds` do not satisfy `0 < lower < upper < inf`.
    InvalidBounds { lower: f64, upper: f64 },
    /// `sum_tolerance` is not a number in `[0, 0.05]`.
    InvalidSumTolerance { got: f64 },
}

impl fmt::Display for RowsProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RowsProblem::LengthMismatch { name, rows, labels } => write!(
                f,
                "{name} has {rows} rows but labels has {labels}; they must have the same length"
            ),
            RowsProblem::SingleOption { row } => write!(
                f,
                "logits row {row} has a single option; it carries no calibration information, remove it"
            ),
            RowsProblem::EmptyRow { name, row } => write!(
                f,
                "{name} row {row} is empty; every row needs at least one option"
            ),
            RowsProblem::RangeTooLarge { row } => write!(
                f,
                "logits row {row} spans more than the largest float number (the difference between its largest and smallest logit overflows); scale the logits down"
            ),
            RowsProblem::LowerBoundTooSmall { lower } => write!(
                f,
                "the lower bound {lower:?} of bounds is too small: 1 / lower must be a finite number"
            ),
            RowsProblem::NotAPair { row, got } => write!(
                f,
                "logits row {row} must have exactly 2 options ([no, yes]), got {got}"
            ),
            RowsProblem::NonFiniteValue {
                name,
                row,
                option,
                value,
            } => write!(
                f,
                "{name} row {row} has the non-finite value {value} at option {option}"
            ),
            RowsProblem::LabelOutOfRange {
                row,
                label,
                options,
            } => write!(
                f,
                "label of row {row} is {label} but the row has {options} options (valid labels are 0..{})",
                options.saturating_sub(1)
            ),
            RowsProblem::ProbabilityNotFinite { row } => {
                write!(f, "probabilities row {row}: contains NaN or infinity")
            }
            RowsProblem::ProbabilityOutOfRange { row, option, value } => write!(
                f,
                "probabilities row {row}: value {value:?} at option {option} is outside [0, 1]"
            ),
            RowsProblem::ProbabilitySum {
                row,
                sum,
                tolerance,
            } => write!(
                f,
                "probabilities row {row}: sums to {sum:?}, expected 1 within {tolerance:?}"
            ),
            RowsProblem::InvalidBins { got } => {
                write!(f, "n_bins must be an integer between 1 and 1000, got {got}")
            }
            RowsProblem::InvalidBounds { lower, upper } => write!(
                f,
                "bounds must satisfy 0 < lower < upper < inf, got ({lower:?}, {upper:?})"
            ),
            RowsProblem::InvalidSumTolerance { got } => {
                write!(
                    f,
                    "sum_tolerance must be a number in [0, 0.05], got {got:?}"
                )
            }
        }
    }
}

/// The data are valid but do not determine a calibration. Maps to `CalibrationError`.
///
/// Using a temperature at the edge of the bounds would make the probabilities look calibrated
/// without any basis (D20), so no result is returned.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum NoCalibration {
    /// Every row has all its logits equal: the loss does not depend on the temperature.
    Uniform,
    /// The best temperature is below the lower bound.
    BelowLower { lower: f64 },
    /// The best temperature is above the upper bound.
    AboveUpper { upper: f64 },
    /// The search did not converge. A guard that the tests prove unreachable.
    DidNotConverge,
}

impl fmt::Display for NoCalibration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoCalibration::Uniform => write!(
                f,
                "cannot fit a temperature: the logits do not discriminate between options (every row is uniform), so every temperature gives the same loss"
            ),
            NoCalibration::BelowLower { lower } => write!(
                f,
                "the best temperature is below the lower bound {lower:?}: the labels always match the top option, so the loss keeps decreasing as the temperature goes to 0; provide more examples that include mistakes"
            ),
            NoCalibration::AboveUpper { upper } => write!(
                f,
                "the best temperature is above the upper bound {upper:?}: the logits carry no usable information about the labels (or point the wrong way), so calibrating them would be meaningless; check the labels and the model"
            ),
            NoCalibration::DidNotConverge => write!(
                f,
                "the search for the temperature did not converge; this is a bug, please report it"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_message_matches_the_table() {
        let cases: Vec<(RowsProblem, &str)> = vec![
            (
                RowsProblem::LengthMismatch {
                    name: "logits",
                    rows: 3,
                    labels: 2,
                },
                "logits has 3 rows but labels has 2; they must have the same length",
            ),
            (
                RowsProblem::SingleOption { row: 4 },
                "logits row 4 has a single option; it carries no calibration information, remove it",
            ),
            (
                RowsProblem::EmptyRow {
                    name: "probabilities",
                    row: 1,
                },
                "probabilities row 1 is empty; every row needs at least one option",
            ),
            (
                RowsProblem::NonFiniteValue {
                    name: "logits",
                    row: 2,
                    option: 1,
                    value: f64::NAN,
                },
                "logits row 2 has the non-finite value NaN at option 1",
            ),
            (
                RowsProblem::LabelOutOfRange {
                    row: 0,
                    label: 3,
                    options: 3,
                },
                "label of row 0 is 3 but the row has 3 options (valid labels are 0..2)",
            ),
            (
                RowsProblem::ProbabilityNotFinite { row: 5 },
                "probabilities row 5: contains NaN or infinity",
            ),
            (
                RowsProblem::ProbabilityOutOfRange {
                    row: 1,
                    option: 2,
                    value: 1.5,
                },
                "probabilities row 1: value 1.5 at option 2 is outside [0, 1]",
            ),
            (
                RowsProblem::ProbabilitySum {
                    row: 0,
                    sum: 0.5,
                    tolerance: 1e-6,
                },
                "probabilities row 0: sums to 0.5, expected 1 within 1e-6",
            ),
            (
                RowsProblem::InvalidBins { got: "0".into() },
                "n_bins must be an integer between 1 and 1000, got 0",
            ),
            (
                RowsProblem::InvalidBounds {
                    lower: 2.0,
                    upper: 1.0,
                },
                "bounds must satisfy 0 < lower < upper < inf, got (2.0, 1.0)",
            ),
            (
                RowsProblem::InvalidSumTolerance { got: 0.5 },
                "sum_tolerance must be a number in [0, 0.05], got 0.5",
            ),
        ];
        for (problem, expected) in cases {
            assert_eq!(problem.to_string(), expected);
        }
    }

    #[test]
    fn calibration_refusals_say_the_bound_and_what_to_do() {
        assert!(NoCalibration::Uniform.to_string().starts_with(
            "cannot fit a temperature: the logits do not discriminate between options (every row is uniform)"
        ));
        assert_eq!(
            NoCalibration::BelowLower { lower: 0.01 }.to_string(),
            "the best temperature is below the lower bound 0.01: the labels always match the top option, so the loss keeps decreasing as the temperature goes to 0; provide more examples that include mistakes"
        );
        assert_eq!(
            NoCalibration::AboveUpper { upper: 100.0 }.to_string(),
            "the best temperature is above the upper bound 100.0: the logits carry no usable information about the labels (or point the wrong way), so calibrating them would be meaningless; check the labels and the model"
        );
    }
}
