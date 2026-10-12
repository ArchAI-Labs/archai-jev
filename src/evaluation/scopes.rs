//! The three kinds of scope an evaluation reports on (all rows, one kind of question, one
//! question name), each with its metrics, its extras and, on request, its intervals.

use crate::calibration::rows::{Neumaier, RowsBuilder, SumTolerance, labelled_probabilities};
use crate::error::Result;
use crate::metrics::{CalibrationReport, calibration_report};

use super::bootstrap::intervals;

/// What only some kinds of question have.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Extras {
    /// Nothing more (all rows together, or a Choice).
    None,
    /// A Score: the mean of `|expected level − label|`.
    Score { expected_level_mae: f64 },
    /// A YesNo: the share of `true` labels and the mean probability of "yes".
    YesNo {
        positive_rate: f64,
        mean_probability: f64,
    },
}

/// The report of one scope.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeReport {
    /// `n`, accuracy, mean confidence, ECE, MCE, Brier, log-loss and the bins (spec 004).
    pub report: CalibrationReport,
    /// The extras of the kind of question.
    pub extras: Extras,
    /// 95 % intervals of `accuracy, ece, brier, log_loss`, if the bootstrap was asked for.
    pub ci95: Option<[(f64, f64); 4]>,
}

/// Rows being collected for a scope.
#[derive(Debug, Default)]
pub struct ScopeAcc {
    rows: RowsBuilder,
    labels: Vec<i64>,
    groups: Vec<usize>,
    score_error: Neumaier,
    positives: usize,
    yes_probability: Neumaier,
}

impl ScopeAcc {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// One row of probabilities with its label and the group of its record.
    pub fn push(&mut self, row: &[f64], label: usize, group: usize) {
        for p in row {
            self.rows.push(*p);
        }
        self.rows.end_row();
        self.labels.push(i64::try_from(label).unwrap_or(i64::MAX));
        self.groups.push(group);
    }

    /// The error of a Score row: `|expected level − label|`.
    pub fn push_score_error(&mut self, error: f64) {
        self.score_error.add(error);
    }

    /// A YesNo row: the probability of "yes" and whether the label is "yes".
    pub fn push_yes_no(&mut self, probability: f64, positive: bool) {
        self.yes_probability.add(probability);
        self.positives += usize::from(positive);
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether no row was pushed.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// The metrics (and intervals) of the collected rows.
    ///
    /// # Errors
    /// Only if a row is not a distribution (an internal invariant of the caller).
    pub fn finish(
        self,
        kind: ExtrasKind,
        n_bins: usize,
        bootstrap_samples: usize,
        seed: u64,
    ) -> Result<ScopeReport> {
        let n = self.labels.len().max(1) as f64;
        let data =
            labelled_probabilities(self.rows.finish(), &self.labels, SumTolerance::default())?;
        let report = calibration_report(&data, n_bins);
        let extras = match kind {
            ExtrasKind::None => Extras::None,
            ExtrasKind::Score => Extras::Score {
                expected_level_mae: self.score_error.total() / n,
            },
            ExtrasKind::YesNo => Extras::YesNo {
                positive_rate: self.positives as f64 / n,
                mean_probability: self.yes_probability.total() / n,
            },
        };
        let ci95 = (bootstrap_samples > 0)
            .then(|| intervals(&data, &self.groups, n_bins, bootstrap_samples, seed));
        Ok(ScopeReport {
            report,
            extras,
            ci95,
        })
    }
}

/// Which extras a scope carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrasKind {
    /// None.
    None,
    /// Score extras.
    Score,
    /// YesNo extras.
    YesNo,
}
