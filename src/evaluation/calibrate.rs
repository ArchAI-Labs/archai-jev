//! `calibrate`: one global temperature for a model, with an honest estimate of what it gains
//! (spec 017 §3).
//!
//! Calibrating and measuring on the same data flatters the result. So besides the temperature
//! (fitted on everything) the study reports three arms: `current` (the model as it is),
//! `in_sample` (the new temperature on the rows that fitted it, optimistic) and `out_of_fold`
//! (each row measured with a temperature fitted on the **other** folds, the honest one). Folds are
//! made of whole groups.

use std::collections::HashMap;

use crate::ask::check_logits;
use crate::calibration::fit::{Prepared, TemperatureFit, check_bounds, fit_prepared, solve};
use crate::calibration::rows::{Dataset as Labelled, Rows, RowsBuilder, labelled_logits};
use crate::calibration::softmax_into;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::metrics::{CalibrationReport, calibration_report};
use crate::schema::QuestionKind;
use crate::scorer::{Calibration, Scorer};

use super::dataset::{Dataset, Label};
use super::evaluate::KIND_ORDER;
use super::groups::assign_folds;
use super::problem::ArgumentProblem;

/// The options of a study.
#[derive(Debug, Clone)]
pub struct StudyOptions {
    /// Number of folds, `2..=50`.
    pub folds: usize,
    /// Seed of the assignment of groups to folds.
    pub seed: u64,
    /// The kinds to calibrate on; `None` for all.
    pub question_types: Option<Vec<QuestionKind>>,
    /// Limits for the temperature (see `fit_temperature`).
    pub bounds: (f64, f64),
}

/// Everything a study computes.
#[derive(Debug, Clone, PartialEq)]
pub struct StudyData {
    /// The new temperature, fitted on all rows (absolute, not relative to the current one).
    pub temperature: f64,
    /// The fit behind it.
    pub fit: TemperatureFit,
    /// The temperature fitted without each fold, in fold order.
    pub fold_temperatures: Vec<f64>,
    /// The rows as the model serves them now.
    pub current: CalibrationReport,
    /// The new temperature on the rows that fitted it.
    pub in_sample: CalibrationReport,
    /// Each row with the temperature of the other folds.
    pub out_of_fold: CalibrationReport,
    /// For each kind present: `(kind, current, out_of_fold)`.
    pub by_type: Vec<(QuestionKind, CalibrationReport, CalibrationReport)>,
    /// Number of rows used (one per selected question).
    pub n: usize,
    /// Number of groups that have rows.
    pub n_groups: usize,
    /// The fingerprint of the whole dataset.
    pub dataset_fingerprint: u64,
    /// The fingerprints of the records that contributed at least one row.
    pub record_fingerprints: Vec<u64>,
    /// The notes of the study.
    pub notes: Vec<String>,
    /// The calibration of the model when the study was made.
    pub calibration: Calibration,
}

/// Fewer rows than this make the estimate unreliable.
pub const SMALL_STUDY: usize = 100;

struct Collected {
    logits: RowsBuilder,
    labels: Vec<usize>,
    kinds: Vec<QuestionKind>,
    /// Group index (compact, by first appearance among the rows) of each row.
    groups: Vec<usize>,
    records: Vec<u64>,
}

fn label_index(label: Label) -> usize {
    match label {
        Label::Choice(i) | Label::Score(i) => i,
        Label::YesNo(b) => usize::from(b),
    }
}

fn report_with(
    rows: &Rows,
    labels: &[usize],
    kinds: &[QuestionKind],
    pick: impl Fn(usize) -> Option<f64>,
    keep: impl Fn(usize) -> bool,
) -> CalibrationReport {
    let mut builder = RowsBuilder::new();
    let mut picked = Vec::new();
    let mut scratch = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let (Some(t), true) = (pick(i), keep(i)) else {
            continue;
        };
        softmax_into(row, t, &mut scratch);
        // The row the model returns: a YesNo is `[1 - p, p]`.
        match (kinds.get(i), scratch.get(1)) {
            (Some(QuestionKind::YesNo), Some(yes)) => {
                builder.push(1.0 - yes);
                builder.push(*yes);
            }
            _ => {
                for p in &scratch {
                    builder.push(*p);
                }
            }
        }
        builder.end_row();
        picked.push(labels.get(i).copied().unwrap_or(0));
    }
    let rows = builder.finish();
    calibration_report(
        &Labelled {
            rows,
            labels: picked,
        },
        10,
    )
}

/// Study the calibration of `scorer` on `dataset`.
///
/// # Errors
/// The errors of `ask` for a record (prefixed with its place), `CalibrationError` when the data
/// do not determine a temperature (with `fold k: ` when it is a fold that cannot), or an
/// argument error (`ArgumentProblem`).
pub fn calibrate(
    scorer: &dyn Scorer,
    dataset: &Dataset,
    options: &StudyOptions,
    control: &dyn Control,
) -> Result<StudyData> {
    let kinds: Vec<QuestionKind> = match &options.question_types {
        None => KIND_ORDER.to_vec(),
        Some(list) if list.is_empty() => {
            return Err(Error::EvalArgument(ArgumentProblem::NoQuestionTypes));
        }
        Some(list) => list.clone(),
    };
    // Without `question_types` every kind present is used; naming a kind the dataset lacks is an
    // error (there would be nothing to calibrate on).
    let missing: Vec<&str> = if options.question_types.is_some() {
        kinds
            .iter()
            .filter(|k| !dataset.names().iter().any(|(_, present)| present == *k))
            .map(|k| super::fingerprint::kind_name(*k))
            .collect()
    } else {
        Vec::new()
    };
    if !missing.is_empty() {
        return Err(Error::EvalArgument(ArgumentProblem::NothingToCalibrate {
            types: missing.join(" or "),
        }));
    }

    // The groups that have rows, and their folds, are known from the dataset alone: a mistake in
    // `folds` is reported before the model is asked anything.
    let mut compact: HashMap<usize, usize> = HashMap::new();
    for (r, record) in dataset.records().iter().enumerate() {
        if record
            .questions
            .iter()
            .any(|(_, q)| kinds.contains(&q.kind()))
        {
            let group = dataset.group_of().get(r).copied().unwrap_or(0);
            let next = compact.len();
            compact.entry(group).or_insert(next);
        }
    }
    let n_groups = compact.len();
    let fold_of_group = assign_folds(n_groups, options.folds, options.seed)?;

    let calibration = scorer.calibration();
    let mut got = Collected {
        logits: RowsBuilder::new(),
        labels: Vec::new(),
        kinds: Vec::new(),
        groups: Vec::new(),
        records: Vec::new(),
    };
    for (r, record) in dataset.records().iter().enumerate() {
        control.checkpoint()?;
        let logits = scorer
            .score_with(&record.state, &record.questions, control)
            .map_err(|e| Error::at(&record.place, e))?;
        check_logits(&record.questions, &logits).map_err(|e| Error::at(&record.place, e))?;
        let group = dataset.group_of().get(r).copied().unwrap_or(0);
        let mut contributed = false;
        for (((_, question), z), label) in record
            .questions
            .iter()
            .zip(&logits)
            .zip(record.labels.iter().copied())
        {
            if !kinds.contains(&question.kind()) {
                continue;
            }
            contributed = true;
            for v in z {
                got.logits.push(*v);
            }
            got.logits.end_row();
            got.labels.push(label_index(label));
            got.kinds.push(question.kind());
            got.groups.push(compact.get(&group).copied().unwrap_or(0));
        }
        if contributed {
            got.records.push(record.fingerprint);
        }
    }

    let rows = got.logits.finish();
    let label_i64: Vec<i64> = got
        .labels
        .iter()
        .map(|l| i64::try_from(*l).unwrap_or(i64::MAX))
        .collect();
    check_bounds(options.bounds)?;
    let all = labelled_logits(rows, &label_i64, true)?;
    let prepared = Prepared::new(&all)?;
    let fit = fit_prepared(&prepared, &all, options.bounds)?;
    let rows = all.rows;

    let fold_of_row: Vec<usize> = got
        .groups
        .iter()
        .map(|g| fold_of_group.get(*g).copied().unwrap_or(0))
        .collect();
    let mut fold_temperatures = Vec::with_capacity(options.folds);
    for k in 0..options.folds {
        let train: Vec<bool> = fold_of_row.iter().map(|f| *f != k).collect();
        let (temperature, _) = solve(&prepared, Some(&train), options.bounds)
            .map_err(|e| Error::at(&format!("fold {k}"), e))?;
        fold_temperatures.push(temperature);
    }

    let current_t = calibration.temperature();
    let everything = |_: usize| true;
    let current = report_with(
        &rows,
        &got.labels,
        &got.kinds,
        |_| Some(current_t),
        everything,
    );
    let in_sample = report_with(
        &rows,
        &got.labels,
        &got.kinds,
        |_| Some(fit.temperature),
        everything,
    );
    let out_of_fold = report_with(
        &rows,
        &got.labels,
        &got.kinds,
        |i| {
            fold_of_row
                .get(i)
                .and_then(|k| fold_temperatures.get(*k))
                .copied()
        },
        everything,
    );
    let present: Vec<QuestionKind> = KIND_ORDER
        .iter()
        .copied()
        .filter(|k| got.kinds.contains(k))
        .collect();
    let mut by_type = Vec::new();
    for kind in &present {
        if present.len() == 1 {
            // one kind only: its reports are the overall ones
            by_type.push((*kind, current.clone(), out_of_fold.clone()));
            continue;
        }
        let of_kind = |i: usize| got.kinds.get(i) == Some(kind);
        by_type.push((
            *kind,
            report_with(&rows, &got.labels, &got.kinds, |_| Some(current_t), of_kind),
            report_with(
                &rows,
                &got.labels,
                &got.kinds,
                |i| {
                    fold_of_row
                        .get(i)
                        .and_then(|k| fold_temperatures.get(*k))
                        .copied()
                },
                of_kind,
            ),
        ));
    }

    let mut notes = Vec::new();
    if out_of_fold.ece >= current.ece {
        notes.push(format!(
            "calibration did not improve the out-of-fold ECE ({:.4} → {:.4}); keep the current temperature",
            current.ece, out_of_fold.ece
        ));
    }
    if got.labels.len() < SMALL_STUDY {
        notes.push(format!(
            "only {} rows were used to estimate the temperature (fewer than {SMALL_STUDY}); the estimate and the out-of-fold comparison are unreliable",
            got.labels.len()
        ));
    }
    Ok(StudyData {
        temperature: fit.temperature,
        fit,
        fold_temperatures,
        current,
        in_sample,
        out_of_fold,
        by_type,
        n: got.labels.len(),
        n_groups,
        dataset_fingerprint: dataset.fingerprint(),
        record_fingerprints: got.records,
        notes,
        calibration,
    })
}
