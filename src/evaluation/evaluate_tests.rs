//! Tests of `evaluate`: exact numbers worked out by hand with a table scorer.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::control::{Control, NoControl};
use crate::error::{Category, Error};
use crate::prompt::Unsupported;
use crate::schema::QuestionKind;
use crate::scorer::Calibration;

use super::dataset::{Dataset, RawLabel};
use super::evaluate::{EvaluationData, Options, evaluate};
use super::scopes::Extras;
use super::testing::{TableScorer, choice, dataset, record, score, yes_no};

const T: f64 = 1.7;

fn softmax(z: &[f64]) -> Vec<f64> {
    let e: Vec<f64> = z.iter().map(|x| (x / T).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn calibration() -> Calibration {
    Calibration::new(T, true).unwrap()
}

fn keys() -> [&'static str; 3] {
    ["returns", "billing", "other"]
}

/// Three records with a choice, a yes/no and a score each.
fn fixture() -> (TableScorer, Dataset) {
    let logits = vec![
        (
            "r0",
            vec![vec![2.0, 1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0, 2.0]],
        ),
        (
            "r1",
            vec![vec![0.0, 3.0, 0.0], vec![2.0, 0.0], vec![1.0, 0.0, 0.0]],
        ),
        (
            "r2",
            vec![vec![1.0, 1.0, 1.5], vec![0.0, 0.0], vec![0.0, 3.0, 0.0]],
        ),
    ];
    let labels = [
        (
            RawLabel::Str("returns".into()),
            RawLabel::Bool(true),
            RawLabel::Int(2),
        ),
        (
            RawLabel::Str("returns".into()),
            RawLabel::Bool(true),
            RawLabel::Int(1),
        ),
        (
            RawLabel::Str("other".into()),
            RawLabel::Bool(false),
            RawLabel::Int(1),
        ),
    ];
    let records = ["r0", "r1", "r2"]
        .iter()
        .zip(labels)
        .map(|(state, (team, urgent, prio))| {
            record(
                state,
                state,
                None,
                vec![
                    ("team", choice(&keys()), team),
                    ("urgent", yes_no(), urgent),
                    ("prio", score(3), prio),
                ],
            )
        })
        .collect();
    (TableScorer::new(calibration(), logits), dataset(records))
}

/// The definitions of spec 004 §3, written naively.
struct Reference {
    n: usize,
    accuracy: f64,
    mean_confidence: f64,
    ece: f64,
    mce: f64,
    brier: f64,
    log_loss: f64,
}

fn reference(rows: &[Vec<f64>], labels: &[usize]) -> Reference {
    let n = rows.len();
    let mut bins: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); 10];
    let (mut hits, mut conf_sum, mut brier, mut loss) = (0.0, 0.0, 0.0, 0.0);
    for (row, &y) in rows.iter().zip(labels) {
        let (top, conf) =
            row.iter().enumerate().fold(
                (0, f64::MIN),
                |best, (i, p)| if *p > best.1 { (i, *p) } else { best },
            );
        let hit = f64::from(u8::from(top == y));
        let bin = (0..10)
            .find(|i| conf <= (*i + 1) as f64 / 10.0)
            .unwrap_or(9);
        bins[bin].0 += 1.0;
        bins[bin].1 += hit;
        bins[bin].2 += conf;
        hits += hit;
        conf_sum += conf;
        brier += row
            .iter()
            .enumerate()
            .map(|(k, p)| (p - if k == y { 1.0 } else { 0.0 }).powi(2))
            .sum::<f64>();
        loss += -row[y].ln();
    }
    let nf = n as f64;
    let gaps: Vec<(f64, f64)> = bins
        .iter()
        .filter(|b| b.0 > 0.0)
        .map(|b| (b.0 / nf, (b.1 / b.0 - b.2 / b.0).abs()))
        .collect();
    Reference {
        n,
        accuracy: hits / nf,
        mean_confidence: conf_sum / nf,
        ece: gaps.iter().map(|(w, g)| w * g).sum(),
        mce: gaps.iter().map(|(_, g)| *g).fold(0.0, f64::max),
        brier: brier / nf,
        log_loss: loss / nf,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

fn check(scope: &super::scopes::ScopeReport, rows: &[Vec<f64>], labels: &[usize]) {
    let want = reference(rows, labels);
    let got = &scope.report;
    assert_eq!(got.n, want.n);
    assert!(close(got.accuracy, want.accuracy), "accuracy");
    assert!(
        close(got.mean_confidence, want.mean_confidence),
        "mean_confidence"
    );
    assert!(close(got.ece, want.ece), "ece {} vs {}", got.ece, want.ece);
    assert!(close(got.mce, want.mce), "mce");
    assert!(close(got.brier, want.brier), "brier");
    assert!(close(got.log_loss, want.log_loss), "log_loss");
}

fn run(scorer: &TableScorer, data: &Dataset) -> EvaluationData {
    evaluate(scorer, data, &Options::default(), &NoControl, None).unwrap()
}

#[test]
fn exact_hand_vectors_all_scopes() {
    let (scorer, data) = fixture();
    let report = run(&scorer, &data);

    let team: Vec<Vec<f64>> = [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.5]]
        .iter()
        .map(|z| softmax(z))
        .collect();
    let team_labels = [0, 0, 2];
    let urgent_p: Vec<f64> = [[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]]
        .iter()
        .map(|z| softmax(z)[1])
        .collect();
    let urgent: Vec<Vec<f64>> = urgent_p.iter().map(|p| vec![1.0 - p, *p]).collect();
    let urgent_labels = [1, 1, 0];
    let prio: Vec<Vec<f64>> = [[0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        .iter()
        .map(|z| softmax(z))
        .collect();
    let prio_labels = [2, 1, 1];

    let all_rows: Vec<Vec<f64>> = [team.clone(), urgent.clone(), prio.clone()].concat();
    let all_labels: Vec<usize> = [
        team_labels.to_vec(),
        urgent_labels.to_vec(),
        prio_labels.to_vec(),
    ]
    .concat();
    check(&report.overall, &all_rows, &all_labels);

    let by_type: Vec<_> = report.by_type.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        by_type,
        vec![
            QuestionKind::Choice,
            QuestionKind::Score,
            QuestionKind::YesNo
        ]
    );
    check(&report.by_type[0].1, &team, &team_labels);
    check(&report.by_type[1].1, &prio, &prio_labels);
    check(&report.by_type[2].1, &urgent, &urgent_labels);
    let names: Vec<_> = report
        .by_question
        .iter()
        .map(|(n, _, _)| n.as_str())
        .collect();
    assert_eq!(names, vec!["team", "urgent", "prio"]);
    check(&report.by_question[0].2, &team, &team_labels);
    check(&report.by_question[1].2, &urgent, &urgent_labels);
    check(&report.by_question[2].2, &prio, &prio_labels);

    // extras
    let mae: f64 = prio
        .iter()
        .zip(prio_labels)
        .map(|(p, y)| {
            (p.iter().enumerate().map(|(i, q)| i as f64 * q).sum::<f64>() - y as f64).abs()
        })
        .sum::<f64>()
        / 3.0;
    assert!(
        matches!(&report.by_type[1].1.extras, Extras::Score { expected_level_mae } if close(*expected_level_mae, mae))
    );
    let Extras::YesNo {
        positive_rate,
        mean_probability,
    } = &report.by_type[2].1.extras
    else {
        panic!("no yes/no extras");
    };
    assert!(close(*positive_rate, 2.0 / 3.0));
    assert!(close(*mean_probability, urgent_p.iter().sum::<f64>() / 3.0));
    assert_eq!(report.by_type[0].1.extras, Extras::None);
    assert_eq!(report.overall.extras, Extras::None);
    assert!(report.overall.ci95.is_none());
}

#[test]
fn structure_order_and_counts() {
    let (scorer, data) = fixture();
    let report = run(&scorer, &data);
    assert_eq!(report.n_records, 3);
    assert_eq!(report.n_questions, 9);
    let sum_types: usize = report.by_type.iter().map(|(_, s)| s.report.n).sum();
    let sum_questions: usize = report.by_question.iter().map(|(_, _, s)| s.report.n).sum();
    assert_eq!(report.overall.report.n, 9);
    assert_eq!(sum_types, 9);
    assert_eq!(sum_questions, 9);
    assert_eq!(report.dataset_fingerprint, data.fingerprint());
    assert!(report.calibration.calibrated());

    // only the kinds that are present, in the fixed order, and names by first appearance
    let only = dataset(vec![record(
        "a",
        "q0",
        None,
        vec![
            ("zeta", yes_no(), RawLabel::Bool(true)),
            ("alpha", choice(&["x", "y"]), RawLabel::Str("y".into())),
        ],
    )]);
    let scorer = TableScorer::new(
        calibration(),
        vec![("q0", vec![vec![0.0, 1.0], vec![0.0, 2.0]])],
    );
    let report = run(&scorer, &only);
    let kinds: Vec<_> = report.by_type.iter().map(|(k, _)| *k).collect();
    assert_eq!(kinds, vec![QuestionKind::Choice, QuestionKind::YesNo]);
    let names: Vec<_> = report
        .by_question
        .iter()
        .map(|(n, _, _)| n.as_str())
        .collect();
    assert_eq!(names, vec!["zeta", "alpha"]);
}

#[test]
fn confidence_is_max_probability_not_the_answer_field() {
    // p = [0.5, 0.3, 0.2]: the answer's `confidence` is (0.5 - 1/3) / (2/3) = 0.25, but the
    // confidence of the ECE is the largest probability, 0.5.
    let z = [0.5_f64.ln() * T, 0.3_f64.ln() * T, 0.2_f64.ln() * T];
    let scorer = TableScorer::new(calibration(), vec![("s", vec![z.to_vec()])]);
    let data = dataset(vec![record(
        "r",
        "s",
        None,
        vec![("team", choice(&keys()), RawLabel::Str("returns".into()))],
    )]);
    let report = run(&scorer, &data);
    assert!(close(report.overall.report.mean_confidence, 0.5));
    assert!(!close(report.overall.report.mean_confidence, 0.25));
}

#[test]
fn numerical_fault_gives_no_report() {
    let (mut scorer, data) = fixture();
    scorer.table.insert(
        "r1".into(),
        vec![vec![0.0, f64::NAN, 0.0], vec![0.0; 2], vec![0.0; 3]],
    );
    let e = evaluate(&scorer, &data, &Options::default(), &NoControl, None).unwrap_err();
    assert_eq!(e.category(), Category::Numerical);
    assert!(e.to_string().starts_with("r1: "), "{e}");
}

#[test]
fn unsupported_has_record_prefix() {
    let (mut scorer, data) = fixture();
    scorer.fail_on.insert(
        "r2".into(),
        Error::Unsupported(Unsupported::HeadLimit {
            question: "team".into(),
            model: "m".into(),
            max: 4,
            got: 5,
        }),
    );
    let e = evaluate(&scorer, &data, &Options::default(), &NoControl, None).unwrap_err();
    assert_eq!(e.category(), Category::Unsupported);
    assert!(e.to_string().starts_with("r2: "), "{e}");
}

#[test]
fn scorer_untouched_after_failure() {
    let (mut scorer, data) = fixture();
    scorer.fail_on.insert(
        "r1".into(),
        Error::Inference {
            detail: "boom".into(),
        },
    );
    assert!(evaluate(&scorer, &data, &Options::default(), &NoControl, None).is_err());
    // two records were asked (r0, then the failing r1); a second evaluation behaves the same
    assert_eq!(scorer.calls(), 2);
    scorer.fail_on.clear();
    let ok = run(&scorer, &data);
    assert_eq!(ok.n_records, 3);
}

#[test]
fn validates_before_scoring_calls_zero() {
    let (scorer, _) = fixture();
    let good = record("a", "r0", None, vec![("u", yes_no(), RawLabel::Bool(true))]);
    let bad = super::dataset::LabelledRecord::build(
        "last",
        super::testing::state("r1"),
        crate::schema::Questions::new(vec![("u", yes_no())]).unwrap(),
        &[("u".to_string(), RawLabel::Int(1))],
        None,
        None,
    );
    assert!(bad.is_err());
    let _ = good;
    assert_eq!(scorer.calls(), 0);
}

struct StopAfter {
    seen: AtomicUsize,
    limit: usize,
}

impl Control for StopAfter {
    fn checkpoint(&self) -> crate::error::Result<()> {
        if self.seen.fetch_add(1, Ordering::SeqCst) >= self.limit {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn control_is_asked_between_records() {
    let (scorer, data) = fixture();
    let control = StopAfter {
        seen: AtomicUsize::new(0),
        limit: 1,
    };
    let e = evaluate(&scorer, &data, &Options::default(), &control, None).unwrap_err();
    assert_eq!(e, Error::Cancelled);
    assert_eq!(scorer.calls(), 1);
}

#[test]
fn overlap_counts_records() {
    let (scorer, data) = fixture();
    let used: Vec<u64> = data
        .records()
        .iter()
        .take(2)
        .map(|r| r.fingerprint)
        .collect();
    let report = evaluate(&scorer, &data, &Options::default(), &NoControl, Some(&used)).unwrap();
    assert_eq!(report.calibration_overlap, Some(2));
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.starts_with("2 of 3 records were also used"))
    );
    let none = evaluate(&scorer, &data, &Options::default(), &NoControl, Some(&[])).unwrap();
    assert_eq!(none.calibration_overlap, Some(0));
    assert!(
        !none
            .notes
            .iter()
            .any(|n| n.contains("also used to calibrate"))
    );
}

#[test]
fn overlap_none_without_provenance() {
    let (scorer, data) = fixture();
    let report = run(&scorer, &data);
    assert_eq!(report.calibration_overlap, None);
}

#[test]
fn notes_follow_the_numbers() {
    let (scorer, data) = fixture();
    let report = run(&scorer, &data);
    // 9 rows: every kind is a small sample (3 rows each)
    assert_eq!(
        report
            .notes
            .iter()
            .filter(|n| n.starts_with("small sample"))
            .count(),
        3
    );
    // calibrated = true: no note about it
    assert!(
        !report
            .notes
            .iter()
            .any(|n| n.contains("no declared calibration"))
    );
    let uncalibrated = TableScorer {
        calibration: Calibration::new(T, false).unwrap(),
        ..TableScorer::new(calibration(), vec![])
    };
    let mut uncalibrated = uncalibrated;
    uncalibrated.table = scorer.table.clone();
    let report = run(&uncalibrated, &data);
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("no declared calibration"))
    );
}

#[test]
fn bootstrap_intervals_are_per_scope_and_repeatable() {
    let (scorer, data) = fixture();
    let options = Options {
        n_bins: 10,
        bootstrap_samples: 50,
        seed: 3,
    };
    let a = evaluate(&scorer, &data, &options, &NoControl, None).unwrap();
    let b = evaluate(&scorer, &data, &options, &NoControl, None).unwrap();
    assert_eq!(a, b);
    assert!(a.overall.ci95.is_none());
    assert!(a.by_type.iter().all(|(_, s)| s.ci95.is_some()));
    assert!(a.by_question.iter().all(|(_, _, s)| s.ci95.is_some()));
}
