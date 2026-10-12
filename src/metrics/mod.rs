//! Calibration metrics over rows of probabilities or logits (spec 004 §3).
//!
//! Every metric is a pure function of its inputs. The definitions are written in full because
//! ECE and Brier have several variants in the wild:
//!
//! - **ECE** is the top-label expected calibration error: the confidence of a row is its largest
//!   probability (first maximum), the row is correct when that option is the label, the bins are
//!   uniform on `[0, 1]` and closed on the right, and `ECE = sum_b (n_b / N) |acc_b - conf_b|`
//!   over the non-empty bins.
//! - **Brier** is the original multi-class score: the mean over rows of `sum_k (p_k - y_k)^2`
//!   with `y` one-hot, in `[0, 2]`. With two classes it is twice the usual binary Brier score.
//! - **Log-loss** is the mean of `-ln p_y`, natural logarithm, with no clipping: a zero
//!   probability on the label gives `inf`.
//!
//! All sums are compensated, so the results do not depend on the order of the rows.

mod bins;

use crate::calibration::check_temperature;
use crate::calibration::formulas::first_argmax;
use crate::calibration::problem::RowsProblem;
use crate::calibration::rows::{Dataset, Neumaier, Rows, labelled_logits};
use crate::error::{Error, Result};

pub(crate) use bins::bin_index;

#[cfg(test)]
mod tests;

/// The largest number of bins.
pub const MAX_BINS: usize = 1000;

/// One bin of the reliability diagram.
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityBin {
    /// Lower edge (exclusive, except for the first bin which also holds 0).
    pub lower: f64,
    /// Upper edge (inclusive).
    pub upper: f64,
    /// Rows whose confidence is in the bin.
    pub count: usize,
    /// Mean confidence of those rows (`None` for an empty bin).
    pub mean_confidence: Option<f64>,
    /// Fraction of those rows that are correct (`None` for an empty bin).
    pub accuracy: Option<f64>,
}

impl ReliabilityBin {
    /// `|accuracy - mean_confidence|`, `None` for an empty bin.
    pub fn gap(&self) -> Option<f64> {
        Some((self.accuracy? - self.mean_confidence?).abs())
    }
}

/// Every metric of a set of labelled probabilities, in one pass.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReport {
    /// Number of rows.
    pub n: usize,
    /// Fraction of rows whose top option is the label.
    pub accuracy: f64,
    /// Mean confidence (largest probability).
    pub mean_confidence: f64,
    /// Expected calibration error.
    pub ece: f64,
    /// Maximum calibration error: the largest gap over the non-empty bins.
    pub mce: f64,
    /// Multi-class Brier score.
    pub brier: f64,
    /// Mean natural log-loss.
    pub log_loss: f64,
    /// The bins, empty ones included.
    pub bins: Vec<ReliabilityBin>,
}

/// # Errors
/// `RowsProblem::InvalidBins` unless `n` is an integer in `1..=1000`.
pub fn check_bins(n: i64) -> Result<usize> {
    usize::try_from(n)
        .ok()
        .filter(|n| (1..=MAX_BINS).contains(n))
        .ok_or_else(|| Error::Rows(RowsProblem::InvalidBins { got: n.to_string() }))
}

/// All the metrics of validated rows of probabilities, in one pass.
pub fn calibration_report(data: &Dataset, n_bins: usize) -> CalibrationReport {
    struct Acc {
        count: usize,
        correct: usize,
        confidence: Neumaier,
    }
    let n_bins = n_bins.max(1);
    let mut acc: Vec<Acc> = (0..n_bins)
        .map(|_| Acc {
            count: 0,
            correct: 0,
            confidence: Neumaier::new(),
        })
        .collect();
    let (mut correct, mut confidence) = (0usize, Neumaier::new());
    let (mut brier, mut loss) = (Neumaier::new(), Neumaier::new());
    for (row, &label) in data.rows.iter().zip(&data.labels) {
        let top = first_argmax(row);
        let c = row.get(top).copied().unwrap_or(0.0);
        let hit = top == label;
        if let Some(a) = acc.get_mut(bin_index(c, n_bins)) {
            a.count += 1;
            a.correct += usize::from(hit);
            a.confidence.add(c);
        }
        correct += usize::from(hit);
        confidence.add(c);
        let mut squares = 0.0;
        for (k, p) in row.iter().enumerate() {
            let y = if k == label { 1.0 } else { 0.0 };
            squares += (p - y) * (p - y);
        }
        brier.add(squares);
        loss.add(0.0 - row.get(label).copied().unwrap_or(0.0).ln());
    }
    let n = data.rows.len();
    let nf = n.max(1) as f64;
    let (mut ece, mut mce) = (Neumaier::new(), 0.0_f64);
    let bins: Vec<ReliabilityBin> = acc
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let (lower, upper) = (i as f64 / n_bins as f64, (i + 1) as f64 / n_bins as f64);
            if a.count == 0 {
                return ReliabilityBin {
                    lower,
                    upper,
                    count: 0,
                    mean_confidence: None,
                    accuracy: None,
                };
            }
            let count = a.count as f64;
            let bin = ReliabilityBin {
                lower,
                upper,
                count: a.count,
                mean_confidence: Some(a.confidence.total() / count),
                accuracy: Some(a.correct as f64 / count),
            };
            let gap = bin.gap().unwrap_or(0.0);
            ece.add(count / nf * gap);
            mce = mce.max(gap);
            bin
        })
        .collect();
    CalibrationReport {
        n,
        accuracy: correct as f64 / nf,
        mean_confidence: confidence.total() / nf,
        ece: ece.total(),
        mce,
        brier: brier.total() / nf,
        log_loss: loss.total() / nf,
        bins,
    }
}

/// Mean natural log-loss of `softmax(logits / temperature)` at the labelled option, computed
/// with `logsumexp` so that nothing underflows. Validates its arguments.
///
/// # Errors
/// An invalid temperature, an empty set, different lengths, an invalid row or a bad label.
pub fn log_loss_from_logits(rows: Rows, labels: &[i64], temperature: f64) -> Result<f64> {
    check_temperature(temperature)?;
    let data = labelled_logits(rows, labels, false)?;
    Ok(mean_logit_loss(&data.rows, &data.labels, temperature))
}

/// The loss of already validated data (also what `fit_temperature` minimises).
pub(crate) fn mean_logit_loss(rows: &Rows, labels: &[usize], temperature: f64) -> f64 {
    let mut total = Neumaier::new();
    for (row, &label) in rows.iter().zip(labels) {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut z = Neumaier::new();
        for v in row {
            z.add(((v - max) / temperature).exp());
        }
        let target = row.get(label).or(row.first()).copied().unwrap_or(max);
        total.add(z.total().ln() - (target - max) / temperature);
    }
    total.total() / rows.len().max(1) as f64
}
