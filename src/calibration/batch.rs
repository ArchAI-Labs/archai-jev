//! The public formulas of `archai_jev.calibration`: validated inputs, one value or one row at a
//! time, always the same functions `ask` uses (spec 004 §2.1).
//!
//! The probabilities are checked strictly (a sum other than 1, in particular 0, is an error) and
//! then handed to the functions of [`super::formulas`], which normalise like Kev: with a sum of
//! `1 ± 1e-6` the result is identical, bit for bit, to the one `ask` gives for the same numbers.

use crate::error::{Error, Result};

use super::formulas::{choice_confidence, score_confidence, score_value};
use super::problem::RowsProblem;
use super::rows::{Rows, SumTolerance, logit_rows, probability_rows};
use super::{check_temperature, softmax_of_valid};

/// `softmax(logits / T)` for every row, rows independent.
///
/// # Errors
/// An invalid temperature, an empty set, an empty row or a non-finite logit.
pub fn softmax_batch(rows: Rows, temperature: f64) -> Result<Vec<Vec<f64>>> {
    check_temperature(temperature)?;
    let rows = logit_rows(rows)?;
    Ok(rows
        .iter()
        .map(|row| softmax_of_valid(row, temperature))
        .collect())
}

/// `P(yes) = sigmoid((logit_yes - logit_no) / T)`, without overflow for huge differences.
///
/// # Errors
/// An invalid temperature or a non-finite logit.
pub fn yes_no_probability(logit_no: f64, logit_yes: f64, temperature: f64) -> Result<f64> {
    check_temperature(temperature)?;
    for (option, value) in [(0, logit_no), (1, logit_yes)] {
        if !value.is_finite() {
            return Err(Error::Rows(RowsProblem::NonFiniteValue {
                name: "logits",
                row: 0,
                option,
                value,
            }));
        }
    }
    Ok(sigmoid((logit_yes - logit_no) / temperature))
}

/// [`yes_no_probability`] for every row `[logit_no, logit_yes]`.
///
/// # Errors
/// As [`yes_no_probability`], and a row that does not have exactly two logits.
pub fn yes_no_batch(rows: Rows, temperature: f64) -> Result<Vec<f64>> {
    check_temperature(temperature)?;
    let rows = logit_rows(rows)?;
    rows.iter()
        .enumerate()
        .map(|(i, row)| match row {
            [no, yes] => Ok(sigmoid((yes - no) / temperature)),
            _ => Err(Error::Rows(RowsProblem::NotAPair {
                row: i,
                got: row.len(),
            })),
        })
        .collect()
}

/// A logistic function that neither overflows nor returns NaN.
fn sigmoid(d: f64) -> f64 {
    if d >= 0.0 {
        1.0 / (1.0 + (-d).exp())
    } else {
        let e = d.exp();
        e / (1.0 + e)
    }
}

/// Confidence of a choice for every row of probabilities.
///
/// # Errors
/// An empty set or a row that is not a valid distribution within `tolerance`.
pub fn choice_confidence_batch(rows: Rows, tolerance: SumTolerance) -> Result<Vec<f64>> {
    Ok(probability_rows(rows, tolerance)?
        .iter()
        .map(choice_confidence)
        .collect())
}

/// Expected level of a score for every row of probabilities.
///
/// # Errors
/// As [`choice_confidence_batch`].
pub fn score_value_batch(rows: Rows, tolerance: SumTolerance) -> Result<Vec<f64>> {
    Ok(probability_rows(rows, tolerance)?
        .iter()
        .map(score_value)
        .collect())
}

/// Confidence of a score for every row of probabilities.
///
/// # Errors
/// As [`choice_confidence_batch`].
pub fn score_confidence_batch(rows: Rows, tolerance: SumTolerance) -> Result<Vec<f64>> {
    Ok(probability_rows(rows, tolerance)?
        .iter()
        .map(score_confidence)
        .collect())
}
