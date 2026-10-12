//! `evaluate`: how a model does on a labelled dataset (spec 017 §2).
//!
//! The model is asked through the very function `ask` uses, with the same numeric checks; the
//! rows of probabilities are the ones the model **returns** (with its current temperature) and
//! the metrics are those of spec 004. All or nothing: any failure ends the evaluation without
//! partial numbers.

use std::collections::{HashMap, HashSet};

use crate::answers::Answer;
use crate::ask::ask_with;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::schema::QuestionKind;
use crate::scorer::{Calibration, Scorer};

use super::dataset::{Dataset, Label};
use super::notes::notes;
use super::scopes::{Extras, ExtrasKind, ScopeAcc, ScopeReport};

/// The options of an evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Bins of the reliability diagram, `1..=1000`.
    pub n_bins: usize,
    /// Bootstrap resamples; 0 for no intervals.
    pub bootstrap_samples: usize,
    /// Seed of the bootstrap.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            n_bins: 10,
            bootstrap_samples: 0,
            seed: 0,
        }
    }
}

/// Everything an evaluation computes.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationData {
    /// Number of records.
    pub n_records: usize,
    /// Number of questions over all records.
    pub n_questions: usize,
    /// All rows of all kinds.
    pub overall: ScopeReport,
    /// One report per kind present, in the order choice, score, yes_no.
    pub by_type: Vec<(QuestionKind, ScopeReport)>,
    /// One report per question name, in order of first appearance.
    pub by_question: Vec<(String, QuestionKind, ScopeReport)>,
    /// The fingerprint of the dataset.
    pub dataset_fingerprint: u64,
    /// How many records were also used to calibrate the model; `None` without provenance.
    pub calibration_overlap: Option<usize>,
    /// The notes of spec 017 §2.2.
    pub notes: Vec<String>,
    /// The temperature the model had and whether it is a calibration.
    pub calibration: Calibration,
}

/// The kinds in the fixed order of the reports.
pub const KIND_ORDER: [QuestionKind; 3] = [
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::YesNo,
];

fn extras_kind(kind: QuestionKind) -> ExtrasKind {
    match kind {
        QuestionKind::Choice => ExtrasKind::None,
        QuestionKind::Score => ExtrasKind::Score,
        QuestionKind::YesNo => ExtrasKind::YesNo,
    }
}

fn kind_slot(kind: QuestionKind) -> usize {
    KIND_ORDER.iter().position(|k| *k == kind).unwrap_or(0)
}

/// The row of probabilities a model returned for a question, with the label as an index.
/// A YesNo answer is `[1 − p, p]` and the label `true` is index 1.
fn row_of(answer: &Answer, label: Label) -> Option<(Vec<f64>, usize)> {
    match (answer, label) {
        (Answer::Choice(c), Label::Choice(i)) => {
            Some((c.probabilities().iter().map(|(_, p)| *p).collect(), i))
        }
        (Answer::Score(s), Label::Score(i)) => Some((s.probabilities().to_vec(), i)),
        (Answer::YesNo(y), Label::YesNo(b)) => {
            let p = y.probability();
            Some((vec![1.0 - p, p], usize::from(b)))
        }
        _ => None,
    }
}

/// One row into one scope, with the extras of its kind.
fn add_row(
    acc: &mut ScopeAcc,
    row: &[f64],
    index: usize,
    group: usize,
    answer: &Answer,
    label: Label,
) {
    acc.push(row, index, group);
    match (answer, label) {
        (Answer::Score(s), Label::Score(i)) => acc.push_score_error((s.value() - i as f64).abs()),
        (Answer::YesNo(y), Label::YesNo(b)) => acc.push_yes_no(y.probability(), b),
        _ => {}
    }
}

/// Evaluate `scorer` on `dataset`.
///
/// `provenance` holds the fingerprints of the records the model's calibration was fitted on
/// (if it has such a history). `control` is asked between the records.
///
/// # Errors
/// Whatever `ask` reports for a record, prefixed with where the record is (`NumericalError`,
/// `UnsupportedRequestError`, …), or [`Error::Cancelled`]. No partial result.
pub fn evaluate(
    scorer: &dyn Scorer,
    dataset: &Dataset,
    options: &Options,
    control: &dyn Control,
    provenance: Option<&[u64]>,
) -> Result<EvaluationData> {
    let calibration = scorer.calibration();
    let name_slot: HashMap<&str, usize> = dataset
        .names()
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.as_str(), i))
        .collect();
    let mut overall = ScopeAcc::new();
    let mut by_kind: [ScopeAcc; 3] = Default::default();
    let mut by_name: Vec<ScopeAcc> = dataset.names().iter().map(|_| ScopeAcc::new()).collect();

    for (r, record) in dataset.records().iter().enumerate() {
        control.checkpoint()?;
        let answers = ask_with(scorer, &record.state, &record.questions, control)
            .map_err(|e| Error::at(&record.place, e))?;
        let group = dataset.group_of().get(r).copied().unwrap_or(0);
        for (((name, question), (_, answer)), label) in record
            .questions
            .iter()
            .zip(answers.iter())
            .zip(record.labels.iter().copied())
        {
            let Some((row, index)) = row_of(answer, label) else {
                return Err(Error::Inference {
                    detail: format!(
                        "{}: the answer to \"{name}\" does not match its label",
                        record.place
                    ),
                });
            };
            let kind = question.kind();
            let slot = name_slot.get(name).copied().unwrap_or(0);
            add_row(&mut overall, &row, index, group, answer, label);
            if let Some(acc) = by_kind.get_mut(kind_slot(kind)) {
                add_row(acc, &row, index, group, answer, label);
            }
            if let Some(acc) = by_name.get_mut(slot) {
                add_row(acc, &row, index, group, answer, label);
            }
        }
    }

    let finish = |acc: ScopeAcc, kind: ExtrasKind| {
        acc.finish(
            kind,
            options.n_bins,
            options.bootstrap_samples,
            options.seed,
        )
    };
    // The intervals are for the kinds and the questions; the overall scope has none.
    let overall = overall.finish(ExtrasKind::None, options.n_bins, 0, options.seed)?;
    let mut by_type = Vec::new();
    for (kind, acc) in KIND_ORDER.iter().zip(by_kind) {
        if !acc.is_empty() {
            by_type.push((*kind, finish(acc, extras_kind(*kind))?));
        }
    }
    let mut by_question = Vec::with_capacity(by_name.len());
    for ((name, kind), acc) in dataset.names().iter().zip(by_name) {
        by_question.push((name.clone(), *kind, finish(acc, extras_kind(*kind))?));
    }

    let n_records = dataset.records().len();
    let calibration_overlap = provenance.map(|used| {
        let used: HashSet<u64> = used.iter().copied().collect();
        dataset
            .records()
            .iter()
            .filter(|r| used.contains(&r.fingerprint))
            .count()
    });
    let rows_by_kind: Vec<(QuestionKind, usize)> =
        by_type.iter().map(|(k, s)| (*k, s.report.n)).collect();
    let notes = notes(
        overall.report.accuracy,
        overall.report.mean_confidence,
        &rows_by_kind,
        calibration.calibrated(),
        calibration_overlap.map(|k| (k, n_records)),
    );
    Ok(EvaluationData {
        n_records,
        n_questions: dataset.question_count(),
        overall,
        by_type,
        by_question,
        dataset_fingerprint: dataset.fingerprint(),
        calibration_overlap,
        notes,
        calibration,
    })
}

impl Extras {
    /// The extras of a report, if any, as `(name, value)` pairs in the order of the table.
    pub fn pairs(&self) -> Vec<(&'static str, f64)> {
        match self {
            Extras::None => Vec::new(),
            Extras::Score { expected_level_mae } => {
                vec![("expected_level_mae", *expected_level_mae)]
            }
            Extras::YesNo {
                positive_rate,
                mean_probability,
            } => vec![
                ("positive_rate", *positive_rate),
                ("mean_probability", *mean_probability),
            ],
        }
    }
}
