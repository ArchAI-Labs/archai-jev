//! Rows of logits or probabilities and their labels, validated once (spec 004 §3.3).
//!
//! The rows may have different lengths (questions with different numbers of options), so they
//! live in one flat buffer with the offset of every row: one allocation, and the loops of the
//! fit read it in order.
//!
//! The checks run in a fixed order, so the first error is predictable: the arguments, the empty
//! set, the lengths, then row by row (empty, single option, non-finite, range, sum) and the
//! label of that row.

use crate::error::{Error, Result};

use super::problem::RowsProblem;

/// Neumaier's compensated sum: the error does not grow with the number of terms, so the result
/// does not depend on how the rows are ordered or split.
#[derive(Debug, Clone, Copy, Default)]
pub struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    /// An empty sum.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one term. A sum that reaches infinity stays infinite (no NaN from the compensation).
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if !t.is_finite() {
            self.sum = t;
            return;
        }
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// The sum of the terms added so far.
    pub fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// Rows of numbers of different lengths in one flat buffer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Rows {
    values: Vec<f64>,
    /// `offsets[i]..offsets[i + 1]` is row `i`; always starts with 0 and never decreases.
    offsets: Vec<usize>,
}

impl Rows {
    /// Rows copied from slices.
    pub fn from_slices<R: AsRef<[f64]>>(rows: &[R]) -> Self {
        let mut builder = RowsBuilder::new();
        for row in rows {
            for v in row.as_ref() {
                builder.push(*v);
            }
            builder.end_row();
        }
        builder.finish()
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i` (an empty slice if `i` is out of range).
    pub fn row(&self, i: usize) -> &[f64] {
        match (self.offsets.get(i), self.offsets.get(i + 1)) {
            (Some(&start), Some(&end)) => self.values.get(start..end).unwrap_or(&[]),
            _ => &[],
        }
    }

    /// The rows in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f64]> + DoubleEndedIterator {
        (0..self.len()).map(|i| self.row(i))
    }

    /// Every number of every row, in order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Builds [`Rows`]: the offsets are right by construction.
#[derive(Debug, Clone)]
pub struct RowsBuilder {
    values: Vec<f64>,
    offsets: Vec<usize>,
}

impl Default for RowsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RowsBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// Reserve room for `rows` rows and `values` numbers.
    pub fn with_capacity(rows: usize, values: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        RowsBuilder {
            values: Vec::with_capacity(values),
            offsets,
        }
    }

    /// Add a number to the current row.
    pub fn push(&mut self, value: f64) {
        self.values.push(value);
    }

    /// Close the current row (it may be empty: validation rejects it with its index).
    pub fn end_row(&mut self) {
        self.offsets.push(self.values.len());
    }

    /// The rows built so far.
    pub fn finish(self) -> Rows {
        Rows {
            values: self.values,
            offsets: self.offsets,
        }
    }
}

/// How far from 1 the sum of a row of probabilities may be (default `1e-6`, at most `0.05`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumTolerance(f64);

impl Default for SumTolerance {
    fn default() -> Self {
        SumTolerance(1e-6)
    }
}

impl SumTolerance {
    /// The largest accepted tolerance: probabilities rounded to 4 decimals with 255 options.
    pub const MAX: f64 = 0.05;

    /// # Errors
    /// `RowsProblem::InvalidSumTolerance` unless `x` is a number in `[0, 0.05]`.
    pub fn new(x: f64) -> Result<Self> {
        if x.is_finite() && (0.0..=Self::MAX).contains(&x) {
            Ok(SumTolerance(x))
        } else {
            Err(Error::Rows(RowsProblem::InvalidSumTolerance { got: x }))
        }
    }

    /// The value.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Validated rows and their labels (the index of the right option in each row).
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    /// The rows.
    pub rows: Rows,
    /// One label per row, in `0..len(row)`.
    pub labels: Vec<usize>,
}

fn empty_set(name: &'static str) -> Error {
    Error::Empty { name }
}

fn check_logit_row(name: &'static str, i: usize, row: &[f64]) -> Result<()> {
    if row.is_empty() {
        return Err(Error::Rows(RowsProblem::EmptyRow { name, row: i }));
    }
    if let Some((option, &value)) = row.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(Error::Rows(RowsProblem::NonFiniteValue {
            name,
            row: i,
            option,
            value,
        }));
    }
    Ok(())
}

fn check_probability_row(i: usize, row: &[f64], tolerance: SumTolerance) -> Result<()> {
    if row.is_empty() {
        return Err(Error::Rows(RowsProblem::EmptyRow {
            name: "probabilities",
            row: i,
        }));
    }
    if row.iter().any(|v| !v.is_finite()) {
        return Err(Error::Rows(RowsProblem::ProbabilityNotFinite { row: i }));
    }
    if let Some((option, &value)) = row
        .iter()
        .enumerate()
        .find(|(_, v)| !(0.0..=1.0).contains(*v))
    {
        return Err(Error::Rows(RowsProblem::ProbabilityOutOfRange {
            row: i,
            option,
            value,
        }));
    }
    let mut sum = Neumaier::new();
    for v in row {
        sum.add(*v);
    }
    let sum = sum.total();
    if (sum - 1.0).abs() > tolerance.get() {
        return Err(Error::Rows(RowsProblem::ProbabilitySum {
            row: i,
            sum,
            tolerance: tolerance.get(),
        }));
    }
    Ok(())
}

fn check_label(i: usize, label: i64, options: usize) -> Result<usize> {
    usize::try_from(label)
        .ok()
        .filter(|l| *l < options)
        .ok_or(Error::Rows(RowsProblem::LabelOutOfRange {
            row: i,
            label,
            options,
        }))
}

fn check_lengths(name: &'static str, rows: &Rows, labels: &[i64]) -> Result<()> {
    if rows.is_empty() {
        return Err(empty_set(name));
    }
    if rows.len() != labels.len() {
        return Err(Error::Rows(RowsProblem::LengthMismatch {
            name,
            rows: rows.len(),
            labels: labels.len(),
        }));
    }
    Ok(())
}

/// Rows of logits without labels (for the batch softmax).
///
/// # Errors
/// An empty set, an empty row or a non-finite logit.
pub fn logit_rows(rows: Rows) -> Result<Rows> {
    if rows.is_empty() {
        return Err(empty_set("logits"));
    }
    for (i, row) in rows.iter().enumerate() {
        check_logit_row("logits", i, row)?;
    }
    Ok(rows)
}

/// Rows of probabilities without labels (for the batch formulas).
///
/// # Errors
/// An empty set, or a row that is empty, not finite, outside `[0, 1]` or not summing to 1.
pub fn probability_rows(rows: Rows, tolerance: SumTolerance) -> Result<Rows> {
    if rows.is_empty() {
        return Err(empty_set("probabilities"));
    }
    for (i, row) in rows.iter().enumerate() {
        check_probability_row(i, row, tolerance)?;
    }
    Ok(rows)
}

/// Rows of logits with labels. With `require_two_options` a single-option row is rejected
/// (it carries no calibration information).
///
/// # Errors
/// An empty set, different lengths, an invalid row or a label outside its row.
pub fn labelled_logits(rows: Rows, labels: &[i64], require_two_options: bool) -> Result<Dataset> {
    check_lengths("logits", &rows, labels)?;
    let mut checked = Vec::with_capacity(labels.len());
    for (i, (row, &label)) in rows.iter().zip(labels).enumerate() {
        check_logit_row("logits", i, row)?;
        if require_two_options && row.len() == 1 {
            return Err(Error::Rows(RowsProblem::SingleOption { row: i }));
        }
        checked.push(check_label(i, label, row.len())?);
    }
    Ok(Dataset {
        rows,
        labels: checked,
    })
}

/// Rows of probabilities with labels.
///
/// # Errors
/// An empty set, different lengths, an invalid row or a label outside its row.
pub fn labelled_probabilities(
    rows: Rows,
    labels: &[i64],
    tolerance: SumTolerance,
) -> Result<Dataset> {
    check_lengths("probabilities", &rows, labels)?;
    let mut checked = Vec::with_capacity(labels.len());
    for (i, (row, &label)) in rows.iter().zip(labels).enumerate() {
        check_probability_row(i, row, tolerance)?;
        checked.push(check_label(i, label, row.len())?);
    }
    Ok(Dataset {
        rows,
        labels: checked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(e: Error) -> String {
        e.to_string()
    }

    fn rows(r: &[&[f64]]) -> Rows {
        Rows::from_slices(r)
    }

    #[test]
    fn rows_keep_their_lengths() {
        let r = rows(&[&[1.0, 2.0], &[3.0], &[]]);
        assert_eq!(r.len(), 3);
        assert_eq!(r.row(0), &[1.0, 2.0]);
        assert_eq!(r.row(1), &[3.0]);
        assert!(r.row(2).is_empty());
        assert!(r.row(9).is_empty());
        assert_eq!(r.values().len(), 3);
    }

    #[test]
    fn empty_set() {
        let e = labelled_logits(Rows::default(), &[], true).unwrap_err();
        assert_eq!(msg(e), "logits must not be empty");
        let e = probability_rows(Rows::default(), SumTolerance::default()).unwrap_err();
        assert_eq!(msg(e), "probabilities must not be empty");
    }

    #[test]
    fn length_mismatch() {
        let e = labelled_logits(Rows::from_slices(&[[0.0, 1.0]; 3]), &[0, 1], true).unwrap_err();
        assert_eq!(
            msg(e),
            "logits has 3 rows but labels has 2; they must have the same length"
        );
    }

    #[test]
    fn single_option_row_for_fit() {
        let e = labelled_logits(rows(&[&[0.0, 1.0], &[2.0]]), &[0, 0], true).unwrap_err();
        assert_eq!(
            msg(e),
            "logits row 1 has a single option; it carries no calibration information, remove it"
        );
        assert!(labelled_logits(rows(&[&[2.0]]), &[0], false).is_ok());
    }

    #[test]
    fn empty_row() {
        let e = labelled_logits(rows(&[&[0.0, 1.0], &[]]), &[0, 0], true).unwrap_err();
        assert_eq!(
            msg(e),
            "logits row 1 is empty; every row needs at least one option"
        );
        let e = probability_rows(rows(&[&[]]), SumTolerance::default()).unwrap_err();
        assert_eq!(
            msg(e),
            "probabilities row 0 is empty; every row needs at least one option"
        );
    }

    #[test]
    fn non_finite_value_names_row_and_option() {
        let e = logit_rows(rows(&[&[0.0, 1.0], &[0.0, f64::INFINITY]])).unwrap_err();
        assert_eq!(
            msg(e),
            "logits row 1 has the non-finite value inf at option 1"
        );
        let e = logit_rows(rows(&[&[f64::NAN, 1.0]])).unwrap_err();
        assert_eq!(
            msg(e),
            "logits row 0 has the non-finite value NaN at option 0"
        );
    }

    #[test]
    fn label_out_of_range() {
        for (label, text) in [
            (
                2_i64,
                "label of row 1 is 2 but the row has 2 options (valid labels are 0..1)",
            ),
            (
                -1,
                "label of row 1 is -1 but the row has 2 options (valid labels are 0..1)",
            ),
        ] {
            let e =
                labelled_logits(rows(&[&[0.0, 1.0], &[0.0, 1.0]]), &[0, label], true).unwrap_err();
            assert_eq!(msg(e), text);
        }
    }

    #[test]
    fn bad_probabilities_all_reasons() {
        let tol = SumTolerance::default();
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (
                vec![0.5, f64::NAN],
                "probabilities row 0: contains NaN or infinity",
            ),
            (
                vec![0.5, f64::INFINITY],
                "probabilities row 0: contains NaN or infinity",
            ),
            (
                vec![1.5, -0.5],
                "probabilities row 0: value 1.5 at option 0 is outside [0, 1]",
            ),
            (
                vec![0.5, -0.25, 0.75],
                "probabilities row 0: value -0.25 at option 1 is outside [0, 1]",
            ),
            (
                vec![0.5, 0.2],
                "probabilities row 0: sums to 0.7, expected 1 within 1e-6",
            ),
        ];
        for (row, text) in cases {
            let e = probability_rows(rows(&[&row]), tol).unwrap_err();
            assert_eq!(msg(e), text);
        }
    }

    #[test]
    fn zero_sum_is_an_error() {
        let e = probability_rows(rows(&[&[0.0, 0.0, 0.0]]), SumTolerance::default()).unwrap_err();
        assert_eq!(
            msg(e),
            "probabilities row 0: sums to 0.0, expected 1 within 1e-6"
        );
    }

    #[test]
    fn sum_tolerance_bounds() {
        assert_eq!(SumTolerance::default().get(), 1e-6);
        assert!(SumTolerance::new(0.0).is_ok());
        assert!(SumTolerance::new(0.05).is_ok());
        for bad in [-1e-9, 0.0500001, f64::NAN, f64::INFINITY] {
            let e = SumTolerance::new(bad).unwrap_err();
            assert!(msg(e).starts_with("sum_tolerance must be a number in [0, 0.05], got "));
        }
    }

    #[test]
    fn rounded_255_options_with_tolerance() {
        // 255 probabilities rounded to 4 decimals: the sum is off by up to 255 * 5e-5.
        let exact = 1.0 / 255.0;
        let rounded = (exact * 1e4_f64).round() / 1e4;
        let row = vec![rounded; 255];
        let off = (row.iter().sum::<f64>() - 1.0).abs();
        assert!(off > 1e-3 && off < 0.02, "{off}");
        assert!(probability_rows(rows(&[&row]), SumTolerance::default()).is_err());
        assert!(probability_rows(rows(&[&row]), SumTolerance::new(0.02).unwrap()).is_ok());
    }

    #[test]
    fn compensated_sum_beats_the_plain_one() {
        let mut s = Neumaier::new();
        s.add(1.0);
        for _ in 0..1_000_000 {
            s.add(1e-16);
        }
        assert!((s.total() - (1.0 + 1e-10)).abs() < 1e-15);
        let plain: f64 = std::iter::once(1.0)
            .chain(std::iter::repeat_n(1e-16, 1_000_000))
            .sum();
        assert_eq!(plain, 1.0);
    }
}
