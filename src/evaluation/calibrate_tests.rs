//! Tests of `calibrate`.

use crate::calibration::fit::{DEFAULT_BOUNDS, fit_temperature};
use crate::calibration::rows::{Rows, labelled_logits};
use crate::control::NoControl;
use crate::error::{Category, Error};
use crate::schema::QuestionKind;
use crate::scorer::Calibration;

use super::calibrate::{StudyData, StudyOptions, calibrate};
use super::dataset::RawLabel;
use super::evaluate::{Options, evaluate};
use super::groups::assign_folds;
use super::testing::{TableScorer, choice, dataset, generated, record, score, yes_no};

fn options(folds: usize, seed: u64) -> StudyOptions {
    StudyOptions {
        folds,
        seed,
        question_types: None,
        bounds: DEFAULT_BOUNDS,
    }
}

fn study(scorer: &TableScorer, data: &super::dataset::Dataset, o: &StudyOptions) -> StudyData {
    calibrate(scorer, data, o, &NoControl).unwrap()
}

fn fit_on(logits: &[Vec<f64>], labels: &[i64]) -> f64 {
    let data = labelled_logits(Rows::from_slices(logits), labels, true).unwrap();
    fit_temperature(&data, DEFAULT_BOUNDS).unwrap().temperature
}

fn labels_of(data: &super::dataset::Dataset) -> Vec<i64> {
    data.records()
        .iter()
        .map(|r| match r.labels[0] {
            super::dataset::Label::Choice(i) | super::dataset::Label::Score(i) => i as i64,
            super::dataset::Label::YesNo(b) => i64::from(b),
        })
        .collect()
}

#[test]
fn recovers_temperature_within_5_percent() {
    let (scorer, data, _) = generated(1, 20_000, 4, 3.0, 2.0, 1);
    let s = study(&scorer, &data, &options(5, 0));
    assert!((s.temperature - 2.0).abs() <= 0.1, "{}", s.temperature);
    assert_eq!(s.n, 20_000);
    assert_eq!(s.n_groups, 20_000);
    assert_eq!(s.fold_temperatures.len(), 5);
    assert_eq!(s.record_fingerprints.len(), 20_000);
    assert_eq!(s.dataset_fingerprint, data.fingerprint());
}

#[test]
fn equals_fit_temperature_on_the_same_logits() {
    let (scorer, data, logits) = generated(2, 600, 4, 3.0, 2.0, 3);
    let s = study(&scorer, &data, &options(4, 1));
    let direct = fit_on(&logits, &labels_of(&data));
    assert!((s.temperature - direct).abs() <= 1e-9);
    assert!((s.fit.temperature - direct).abs() <= 1e-9);
    assert_eq!(s.fit.n, 600);
}

#[test]
fn temperature_is_absolute_whatever_the_current_one() {
    let (mut scorer, data, _) = generated(3, 400, 3, 3.0, 2.0, 1);
    let a = study(&scorer, &data, &options(5, 0));
    scorer.calibration = Calibration::new(3.7, true).unwrap();
    let b = study(&scorer, &data, &options(5, 0));
    assert_eq!(a.temperature, b.temperature);
    assert_eq!(a.fold_temperatures, b.fold_temperatures);
    assert_eq!(a.in_sample, b.in_sample);
    assert_ne!(a.current, b.current);
}

fn folds_of_rows(data: &super::dataset::Dataset, folds: usize, seed: u64) -> Vec<usize> {
    let fold_of_group = assign_folds(data.groups(), folds, seed).unwrap();
    data.group_of().iter().map(|g| fold_of_group[*g]).collect()
}

#[test]
fn fold_temperatures_equal_fit_on_other_folds() {
    let (scorer, data, logits) = generated(4, 500, 4, 3.0, 2.0, 2);
    let s = study(&scorer, &data, &options(5, 9));
    let fold_of_row = folds_of_rows(&data, 5, 9);
    let labels = labels_of(&data);
    for (k, got) in s.fold_temperatures.iter().enumerate() {
        let (rows, ys): (Vec<Vec<f64>>, Vec<i64>) = logits
            .iter()
            .zip(&labels)
            .zip(&fold_of_row)
            .filter(|(_, f)| **f != k)
            .map(|((z, y), _)| (z.clone(), *y))
            .unzip();
        assert!((got - fit_on(&rows, &ys)).abs() <= 1e-9, "fold {k}");
    }
}

fn naive_report(
    logits: &[Vec<f64>],
    labels: &[i64],
    temperature: impl Fn(usize) -> f64,
) -> crate::metrics::CalibrationReport {
    let rows: Vec<Vec<f64>> = logits
        .iter()
        .enumerate()
        .map(|(i, z)| crate::calibration::calibrated_softmax(z, temperature(i)).unwrap())
        .collect();
    let d = crate::calibration::rows::labelled_probabilities(
        Rows::from_slices(&rows),
        labels,
        crate::calibration::rows::SumTolerance::default(),
    )
    .unwrap();
    crate::metrics::calibration_report(&d, 10)
}

#[test]
fn out_of_fold_uses_only_its_own_fold_temperature() {
    let (scorer, data, logits) = generated(5, 300, 4, 3.0, 2.0, 2);
    let s = study(&scorer, &data, &options(3, 4));
    let fold_of_row = folds_of_rows(&data, 3, 4);
    let expected = naive_report(&logits, &labels_of(&data), |i| {
        s.fold_temperatures[fold_of_row[i]]
    });
    assert_eq!(s.out_of_fold, expected);
}

#[test]
fn in_sample_uses_global_temperature() {
    let (scorer, data, logits) = generated(6, 300, 4, 3.0, 2.0, 1);
    let s = study(&scorer, &data, &options(5, 0));
    assert_eq!(
        s.in_sample,
        naive_report(&logits, &labels_of(&data), |_| s.temperature)
    );
    assert_eq!(s.current, naive_report(&logits, &labels_of(&data), |_| 1.0));
}

#[test]
fn overfit_dataset_gives_out_of_fold_worse_than_in_sample() {
    // Few rows, whole groups of 5: the temperature of the other folds fits worse than the one
    // fitted on everything. Over many small datasets this must be the rule.
    let mut worse = 0;
    for seed in 0..40 {
        let (scorer, data, _) = generated(100 + seed, 40, 4, 3.0, 2.0, 5);
        let s = study(&scorer, &data, &options(4, seed));
        if s.out_of_fold.log_loss > s.in_sample.log_loss {
            worse += 1;
        }
    }
    assert!(worse >= 36, "out-of-fold was worse in only {worse} of 40");
}

#[test]
fn current_equals_evaluate_overall_exactly() {
    let (mut scorer, data, _) = generated(7, 250, 5, 2.5, 1.5, 1);
    scorer.calibration = Calibration::new(1.3, true).unwrap();
    let s = study(&scorer, &data, &options(5, 0));
    let e = evaluate(&scorer, &data, &Options::default(), &NoControl, None).unwrap();
    assert_eq!(s.current, e.overall.report);
}

fn mixed() -> (TableScorer, super::dataset::Dataset) {
    let mut scorer = TableScorer::new(Calibration::uncalibrated(), vec![]);
    let mut records = Vec::new();
    let mut source = super::testing::Gen::new(11);
    for i in 0..120 {
        let team: Vec<f64> = (0..3).map(|_| 2.0 * source.gaussian()).collect();
        let yn: Vec<f64> = (0..2).map(|_| 2.0 * source.gaussian()).collect();
        let sc: Vec<f64> = (0..3).map(|_| 2.0 * source.gaussian()).collect();
        let mut label = |z: &[f64]| {
            source.categorical(&crate::calibration::calibrated_softmax(z, 1.8).unwrap())
        };
        let (a, b, c) = (label(&team), label(&yn), label(&sc));
        let state = format!("m{i}");
        scorer.table.insert(state.clone(), vec![team, yn, sc]);
        records.push(record(
            &format!("record {i}"),
            &state,
            Some(&format!("g{}", i / 2)),
            vec![
                (
                    "team",
                    choice(&["a", "b", "c"]),
                    RawLabel::Str(["a", "b", "c"][a].into()),
                ),
                ("ok", yes_no(), RawLabel::Bool(b == 1)),
                ("level", score(3), RawLabel::Int(c as i128)),
            ],
        ));
    }
    (scorer, dataset(records))
}

#[test]
fn question_types_filter() {
    let (scorer, data) = mixed();
    let all = study(&scorer, &data, &options(4, 0));
    assert_eq!(all.n, 360);
    assert_eq!(all.by_type.len(), 3);
    let only = StudyOptions {
        question_types: Some(vec![QuestionKind::YesNo]),
        ..options(4, 0)
    };
    let s = study(&scorer, &data, &only);
    assert_eq!(s.n, 120);
    assert_eq!(s.by_type.len(), 1);
    assert_eq!(s.by_type[0].0, QuestionKind::YesNo);
    // the same T as fitting directly on the yes/no logits
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for r in data.records() {
        rows.push(scorer.table[r.state.canonical_text()][1].clone());
        labels.push(match r.labels[1] {
            super::dataset::Label::YesNo(b) => i64::from(b),
            _ => 0,
        });
    }
    assert!((s.temperature - fit_on(&rows, &labels)).abs() <= 1e-9);
}

#[test]
fn bad_question_types() {
    let (scorer, data, _) = generated(8, 50, 3, 2.0, 2.0, 1);
    let only_yes_no = StudyOptions {
        question_types: Some(vec![QuestionKind::YesNo]),
        ..options(2, 0)
    };
    let e = calibrate(&scorer, &data, &only_yes_no, &NoControl).unwrap_err();
    assert_eq!(
        e.to_string(),
        "the dataset has no yes_no questions to calibrate on"
    );
    let empty = StudyOptions {
        question_types: Some(vec![]),
        ..options(2, 0)
    };
    let e = calibrate(&scorer, &data, &empty, &NoControl).unwrap_err();
    assert!(e.to_string().starts_with("question_types is empty"));
    assert_eq!(scorer.calls(), 0);
}

#[test]
fn degenerate_always_argmax_and_uniform() {
    // the label is always the option with the highest logit
    let mut scorer = TableScorer::new(Calibration::uncalibrated(), vec![]);
    let mut records = Vec::new();
    for i in 0..30 {
        let state = format!("a{i}");
        scorer
            .table
            .insert(state.clone(), vec![vec![3.0, 1.0, 0.0]]);
        records.push(record(
            &format!("record {i}"),
            &state,
            None,
            vec![("q", choice(&["x", "y", "z"]), RawLabel::Str("x".into()))],
        ));
    }
    let data = dataset(records);
    let e = calibrate(&scorer, &data, &options(3, 0), &NoControl).unwrap_err();
    assert_eq!(e.category(), Category::Calibration);
    assert!(
        e.to_string()
            .starts_with("the best temperature is below the lower bound 0.01")
    );
    // wider bounds do not accept the edge
    let wide = StudyOptions {
        bounds: (1e-6, 1e9),
        ..options(3, 0)
    };
    assert!(calibrate(&scorer, &data, &wide, &NoControl).is_err());
    // uniform logits
    let mut uniform = TableScorer::new(Calibration::uncalibrated(), vec![]);
    for i in 0..30 {
        uniform
            .table
            .insert(format!("a{i}"), vec![vec![1.0, 1.0, 1.0]]);
    }
    let e = calibrate(&uniform, &data, &options(3, 0), &NoControl).unwrap_err();
    assert!(e.to_string().starts_with("cannot fit a temperature"));
}

#[test]
fn degenerate_training_fold_has_fold_prefix() {
    // Two groups. Group A: the label always matches the top option. Group B: mixed. The fit on
    // everything works, but the fold that holds out B trains on A alone and cannot fit.
    let mut scorer = TableScorer::new(Calibration::uncalibrated(), vec![]);
    let mut records = Vec::new();
    for i in 0..12 {
        let state = format!("d{i}");
        let (group, label, logits) = if i < 6 {
            ("A", "x", vec![3.0, 1.0, 0.0])
        } else if i % 2 == 0 {
            ("B", "x", vec![3.0, 1.0, 0.0])
        } else {
            ("B", "z", vec![3.0, 1.0, 0.0])
        };
        scorer.table.insert(state.clone(), vec![logits]);
        records.push(record(
            &format!("record {i}"),
            &state,
            Some(group),
            vec![("q", choice(&["x", "y", "z"]), RawLabel::Str(label.into()))],
        ));
    }
    let data = dataset(records);
    let e = calibrate(&scorer, &data, &options(2, 0), &NoControl).unwrap_err();
    assert_eq!(e.category(), Category::Calibration);
    let text = e.to_string();
    assert!(
        text.starts_with("fold 0: ") || text.starts_with("fold 1: "),
        "{text}"
    );
    assert!(text.contains("below the lower bound"), "{text}");
}

#[test]
fn folds_range_and_more_folds_than_groups() {
    let (scorer, data, _) = generated(9, 20, 3, 2.0, 2.0, 10);
    for bad in [1usize, 51] {
        let e = calibrate(&scorer, &data, &options(bad, 0), &NoControl).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("folds must be an integer between 2 and 50, got {bad}")
        );
    }
    // 20 records in 2 groups
    let e = calibrate(&scorer, &data, &options(3, 0), &NoControl).unwrap_err();
    assert_eq!(
        e.to_string(),
        "dataset has 2 groups but folds=3; use more records or fewer folds"
    );
    // the check runs before the model is asked anything
    assert_eq!(scorer.calls(), 0);
}

#[test]
fn a_failing_record_names_its_place() {
    let (mut scorer, data, _) = generated(10, 30, 3, 2.0, 2.0, 1);
    scorer.fail_on.insert(
        "s4".into(),
        Error::Inference {
            detail: "boom".into(),
        },
    );
    let e = calibrate(&scorer, &data, &options(3, 0), &NoControl).unwrap_err();
    assert!(e.to_string().starts_with("record 4: "), "{e}");
}

#[test]
fn notes_say_when_it_did_not_help_and_when_it_is_small() {
    // data drawn with T = 1: the current temperature (1) is already right
    let (scorer, data, _) = generated(12, 60, 4, 3.0, 1.0, 1);
    let s = study(&scorer, &data, &options(3, 0));
    assert!(
        s.notes
            .iter()
            .any(|n| n.starts_with("only 60 rows were used"))
    );
    let (scorer, data, _) = generated(13, 3000, 4, 3.0, 1.0, 1);
    let s = study(&scorer, &data, &options(3, 0));
    if s.out_of_fold.ece >= s.current.ece {
        assert!(
            s.notes
                .iter()
                .any(|n| n.starts_with("calibration did not improve the out-of-fold ECE ("))
        );
    }
    assert!(!s.notes.iter().any(|n| n.starts_with("only")));
}
