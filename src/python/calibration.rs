//! Binding of the calibration functions (spec 004). Thin: it converts the Python arguments once,
//! releases the GIL for the calculation and maps the errors. The public functions of
//! `archai_jev.calibration` are Python wrappers over these private `_core` functions, which
//! return plain numbers, lists and tuples.

use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::calibration::fit::{check_bounds, fit_temperature};
use crate::calibration::problem::RowsProblem;
use crate::calibration::rows::{SumTolerance, labelled_logits, labelled_probabilities};
use crate::calibration::{batch, check_temperature};
use crate::error::{Error, Result};
use crate::metrics;

use super::rows::{describe, is_integer_like, read_labels, read_row, read_rows};

pub(crate) type BinTuple = (f64, f64, usize, Option<f64>, Option<f64>);
pub(crate) type ReportTuple = (usize, f64, f64, f64, f64, f64, f64, Vec<BinTuple>);

pub(crate) fn parse_bins(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    let invalid = || Error::Rows(RowsProblem::InvalidBins { got: describe(obj) });
    if obj.is_instance_of::<PyBool>() || !is_integer_like(obj)? {
        return Err(invalid().into());
    }
    match obj.extract::<i64>() {
        Ok(n) => Ok(metrics::check_bins(n)?),
        Err(_) => Err(invalid().into()),
    }
}

/// `softmax(logits / T)` for every row.
#[pyfunction]
#[pyo3(name = "_softmax_batch", signature = (rows, temperature = 1.0))]
pub fn softmax_batch(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    temperature: f64,
) -> PyResult<Vec<Vec<f64>>> {
    check_temperature(temperature)?;
    let rows = read_rows(rows, "logits")?;
    Ok(py.detach(|| batch::softmax_batch(rows, temperature))?)
}

/// `sigmoid((logit_yes - logit_no) / T)`.
#[pyfunction]
#[pyo3(name = "_yes_no_probability", signature = (logit_no, logit_yes, temperature = 1.0))]
pub fn yes_no_probability(logit_no: f64, logit_yes: f64, temperature: f64) -> PyResult<f64> {
    Ok(batch::yes_no_probability(logit_no, logit_yes, temperature)?)
}

/// [`yes_no_probability`] for every row `[logit_no, logit_yes]`.
#[pyfunction]
#[pyo3(name = "_yes_no_probability_batch", signature = (rows, temperature = 1.0))]
pub fn yes_no_probability_batch(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    temperature: f64,
) -> PyResult<Vec<f64>> {
    check_temperature(temperature)?;
    let rows = read_rows(rows, "logits")?;
    Ok(py.detach(|| batch::yes_no_batch(rows, temperature))?)
}

type RowFormula = fn(crate::calibration::rows::Rows, SumTolerance) -> Result<Vec<f64>>;

fn one_row(
    py: Python<'_>,
    probabilities: &Bound<'_, PyAny>,
    sum_tolerance: f64,
    formula: RowFormula,
) -> PyResult<f64> {
    let tolerance = SumTolerance::new(sum_tolerance)?;
    let rows = read_row(probabilities, "probabilities")?;
    let values = py.detach(|| formula(rows, tolerance))?;
    values.first().copied().ok_or_else(|| {
        Error::Empty {
            name: "probabilities",
        }
        .into()
    })
}

fn many_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    sum_tolerance: f64,
    formula: RowFormula,
) -> PyResult<Vec<f64>> {
    let tolerance = SumTolerance::new(sum_tolerance)?;
    let rows = read_rows(rows, "probabilities")?;
    Ok(py.detach(|| formula(rows, tolerance))?)
}

macro_rules! formula_functions {
    ($($single:ident / $batched:ident => $core:path, $name:literal, $batch_name:literal;)+) => {$(
        #[doc = concat!("`", $name, "` of one row of probabilities.")]
        #[pyfunction]
        #[pyo3(name = $name, signature = (probabilities, sum_tolerance = 1e-6))]
        pub fn $single(
            py: Python<'_>,
            probabilities: &Bound<'_, PyAny>,
            sum_tolerance: f64,
        ) -> PyResult<f64> {
            one_row(py, probabilities, sum_tolerance, $core)
        }

        #[doc = concat!("`", $name, "` of every row of probabilities.")]
        #[pyfunction]
        #[pyo3(name = $batch_name, signature = (rows, sum_tolerance = 1e-6))]
        pub fn $batched(
            py: Python<'_>,
            rows: &Bound<'_, PyAny>,
            sum_tolerance: f64,
        ) -> PyResult<Vec<f64>> {
            many_rows(py, rows, sum_tolerance, $core)
        }
    )+};
}

formula_functions! {
    choice_confidence / choice_confidence_batch => batch::choice_confidence_batch, "_choice_confidence", "_choice_confidence_batch";
    score_value / score_value_batch => batch::score_value_batch, "_score_value", "_score_value_batch";
    score_confidence / score_confidence_batch => batch::score_confidence_batch, "_score_confidence", "_score_confidence_batch";
}

/// The temperature that minimises the log-loss: `(T, loss_before, loss_after, n, iterations)`.
#[pyfunction]
#[pyo3(name = "_fit_temperature")]
pub fn fit_temperature_py(
    py: Python<'_>,
    logits: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    bounds: (f64, f64),
) -> PyResult<(f64, f64, f64, usize, usize)> {
    check_bounds(bounds)?;
    let rows = read_rows(logits, "logits")?;
    let labels = read_labels(labels)?;
    let fit = py.detach(|| {
        let data = labelled_logits(rows, &labels, true)?;
        fit_temperature(&data, bounds)
    })?;
    Ok((
        fit.temperature,
        fit.loss_before,
        fit.loss_after,
        fit.n,
        fit.iterations,
    ))
}

/// A report as the plain tuple the Python layer turns into a `CalibrationReport`.
pub(crate) fn report_tuple(report: &crate::metrics::CalibrationReport) -> ReportTuple {
    let bins = report
        .bins
        .iter()
        .map(|b| (b.lower, b.upper, b.count, b.mean_confidence, b.accuracy))
        .collect();
    (
        report.n,
        report.accuracy,
        report.mean_confidence,
        report.ece,
        report.mce,
        report.brier,
        report.log_loss,
        bins,
    )
}

/// Every metric in one pass: `(n, accuracy, mean_confidence, ece, mce, brier, log_loss, bins)`.
#[pyfunction]
#[pyo3(name = "_calibration_report")]
pub fn calibration_report(
    py: Python<'_>,
    probabilities: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    n_bins: &Bound<'_, PyAny>,
    sum_tolerance: f64,
) -> PyResult<ReportTuple> {
    let n_bins = parse_bins(n_bins)?;
    let tolerance = SumTolerance::new(sum_tolerance)?;
    let rows = read_rows(probabilities, "probabilities")?;
    let labels = read_labels(labels)?;
    let report = py.detach(|| {
        let data = labelled_probabilities(rows, &labels, tolerance)?;
        Ok::<_, Error>(metrics::calibration_report(&data, n_bins))
    })?;
    Ok(report_tuple(&report))
}

/// Mean log-loss of `softmax(logits / T)` at the labels.
#[pyfunction]
#[pyo3(name = "_log_loss_from_logits", signature = (logits, labels, temperature = 1.0))]
pub fn log_loss_from_logits(
    py: Python<'_>,
    logits: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    temperature: f64,
) -> PyResult<f64> {
    check_temperature(temperature)?;
    let rows = read_rows(logits, "logits")?;
    let labels = read_labels(labels)?;
    Ok(py.detach(|| metrics::log_loss_from_logits(rows, &labels, temperature))?)
}
