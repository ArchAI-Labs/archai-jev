//! Tests of the metrics.

use proptest::prelude::*;

use super::*;
use crate::calibration::calibrated_softmax;
use crate::calibration::rows::{SumTolerance, labelled_probabilities};

fn data(rows: &[&[f64]], labels: &[i64]) -> Dataset {
    labelled_probabilities(Rows::from_slices(rows), labels, SumTolerance::default()).unwrap()
}

fn hand() -> Dataset {
    data(
        &[&[0.9, 0.1], &[0.8, 0.2], &[0.3, 0.7], &[0.6, 0.4]],
        &[0, 1, 1, 0],
    )
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

#[test]
fn hand_vectors_ece_brier_logloss_mce() {
    let d = hand();
    let report = calibration_report(&d, 10);
    assert_eq!(report.n, 4);
    assert!(close(report.ece, 0.4), "{}", report.ece);
    assert!(close(report.brier, 0.45), "{}", report.brier);
    assert!(
        close(report.log_loss, 0.6455747489491624),
        "{}",
        report.log_loss
    );
    assert!(close(report.accuracy, 0.75));
    assert!(close(report.mean_confidence, 0.75));
    assert!(close(report.mce, 0.8), "{}", report.mce);
}

#[test]
fn reliability_ten_bins_six_empty() {
    let bins = calibration_report(&hand(), 10).bins;
    assert_eq!(bins.len(), 10);
    let empty: Vec<_> = bins.iter().filter(|b| b.count == 0).collect();
    assert_eq!(empty.len(), 6);
    assert!(
        empty
            .iter()
            .all(|b| b.accuracy.is_none() && b.mean_confidence.is_none() && b.gap().is_none())
    );
    assert_eq!(bins[5].count, 1);
    assert_eq!(bins[8].count, 1);
    assert!(close(bins[7].gap().unwrap(), 0.8));
    assert_eq!((bins[0].lower, bins[9].upper), (0.0, 1.0));
}

fn uniform_row(k: usize) -> Vec<f64> {
    vec![1.0 / k as f64; k]
}

#[test]
fn bin_edges_closed_on_the_right() {
    // Confidences exactly on a bin edge go to the bin that closes there.
    assert_eq!(bin_index(0.1, 10), 0);
    assert_eq!(bin_index(0.3, 10), 2);
    assert_eq!(bin_index(0.5, 10), 4);
    assert_eq!(bin_index(0.7, 10), 6);
    assert_eq!(bin_index(0.9, 10), 8);
    // Through the public path, with real rows of probabilities.
    let ten = uniform_row(10); // confidence 0.1
    let four = [0.3, 0.3, 0.2, 0.2]; // 0.3
    let two = [0.5, 0.5]; // 0.5
    let d = data(&[&ten, &four, &two], &[0, 0, 0]);
    let counts: Vec<usize> = calibration_report(&d, 10)
        .bins
        .iter()
        .map(|b| b.count)
        .collect();
    assert_eq!(counts, vec![1, 0, 1, 0, 1, 0, 0, 0, 0, 0]);
}

#[test]
fn one_exactly_in_last_bin() {
    assert_eq!(bin_index(1.0, 10), 9);
    let d = data(&[&[1.0, 0.0]], &[0]);
    assert_eq!(calibration_report(&d, 10).bins[9].count, 1);
    assert_eq!(bin_index(1.0, 1), 0);
}

#[test]
fn just_above_edge_next_bin() {
    for edge in [0.1_f64, 0.3, 0.5, 0.9] {
        let above = edge.next_up();
        let below = edge.next_down();
        assert_eq!(bin_index(above, 10), bin_index(edge, 10) + 1, "{edge}");
        assert_eq!(bin_index(below, 10), bin_index(edge, 10), "{edge}");
    }
}

#[test]
fn zero_in_first_bin() {
    assert_eq!(bin_index(0.0, 10), 0);
    assert_eq!(bin_index(1.0 / 255.0, 10), 0);
}

proptest! {
    #[test]
    fn bin_index_follows_the_definition(c in 0.0f64..=1.0, bins in 1usize..=1000) {
        let i = bin_index(c, bins);
        let b = bins as f64;
        prop_assert!(i < bins);
        prop_assert!(c <= (i + 1) as f64 / b || i == bins - 1);
        prop_assert!(i == 0 || c > i as f64 / b);
    }
}

#[test]
fn n_bins_one_is_acc_minus_conf() {
    let report = calibration_report(&hand(), 1);
    assert!(close(
        report.ece,
        (report.accuracy - report.mean_confidence).abs()
    ));
    assert_eq!(report.bins.len(), 1);
}

#[test]
fn brier_two_classes_is_twice_binary() {
    // p(label) = 0.3 for the second class: binary Brier is (0.3 - 1)^2 = 0.49, ours is twice.
    let d = data(&[&[0.7, 0.3]], &[1]);
    assert!(close(calibration_report(&d, 1).brier, 0.98));
    // Worst case: all the mass on the wrong option gives 2.
    assert!(close(
        calibration_report(&data(&[&[1.0, 0.0]], &[1]), 1).brier,
        2.0
    ));
}

#[test]
fn log_loss_zero_probability_is_inf() {
    let d = data(&[&[1.0, 0.0], &[0.5, 0.5]], &[1, 0]);
    assert_eq!(calibration_report(&d, 1).log_loss, f64::INFINITY);
    assert_eq!(calibration_report(&d, 10).log_loss, f64::INFINITY);
}

#[test]
fn logits_log_loss_extreme() {
    let right = log_loss_from_logits(Rows::from_slices(&[[1e6, -1e6]]), &[0], 1.0).unwrap();
    assert!(right.is_finite() && right.abs() < 1e-300);
    let wrong = log_loss_from_logits(Rows::from_slices(&[[1e6, -1e6]]), &[1], 1.0).unwrap();
    assert!(close(wrong, 2e6));
}

#[test]
fn logits_equals_softmax_log_loss() {
    let logits = [[2.0, 1.0, 0.1], [0.3, -1.2, 2.2], [5.0, 4.9, -3.0]];
    let labels = [0, 2, 1];
    let t = 2.3;
    let direct = log_loss_from_logits(Rows::from_slices(&logits), &labels, t).unwrap();
    let probabilities: Vec<Vec<f64>> = logits
        .iter()
        .map(|r| calibrated_softmax(r, t).unwrap())
        .collect();
    let d = labelled_probabilities(
        Rows::from_slices(&probabilities),
        &labels,
        SumTolerance::default(),
    )
    .unwrap();
    assert!((direct - calibration_report(&d, 1).log_loss).abs() <= 1e-9);
}

#[test]
fn logits_arguments_are_checked() {
    let rows = || Rows::from_slices(&[[0.0, 1.0]]);
    assert!(matches!(
        log_loss_from_logits(rows(), &[0], 0.0),
        Err(Error::InvalidTemperature(_))
    ));
    assert!(log_loss_from_logits(Rows::default(), &[], 1.0).is_err());
    // A single-option row is allowed here (its loss is 0).
    assert_eq!(
        log_loss_from_logits(Rows::from_slices(&[[3.0]]), &[0], 1.0).unwrap(),
        0.0
    );
}

#[test]
fn errors_with_index() {
    for bad in [0, -1, 1001, i64::MAX] {
        let e = check_bins(bad).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("n_bins must be an integer between 1 and 1000, got {bad}")
        );
    }
    assert_eq!(check_bins(1).unwrap(), 1);
    assert_eq!(check_bins(1000).unwrap(), 1000);
    let e = labelled_probabilities(
        Rows::from_slices(&[[0.5, 0.5], [0.9, 0.9]]),
        &[0, 0],
        SumTolerance::default(),
    )
    .unwrap_err();
    assert!(e.to_string().starts_with("probabilities row 1: sums to"));
}

fn random_dataset(seed: u64, n: usize, k: usize) -> Dataset {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64
    };
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..n {
        let logits: Vec<f64> = (0..k).map(|_| 6.0 * next() - 3.0).collect();
        rows.push(calibrated_softmax(&logits, 1.0).unwrap());
        labels.push((next() * k as f64) as i64 % k as i64);
    }
    labelled_probabilities(Rows::from_slices(&rows), &labels, SumTolerance::default()).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn invariants(seed in 0u64..u64::MAX, n in 1usize..300, k in 2usize..12, bins in 1usize..40) {
        let d = random_dataset(seed, n, k);
        let r = calibration_report(&d, bins);
        prop_assert!((0.0..=1.0).contains(&r.ece) && (0.0..=1.0).contains(&r.mce));
        prop_assert!((0.0..=2.0).contains(&r.brier) && r.log_loss >= 0.0);
        prop_assert_eq!(r.bins.iter().map(|b| b.count).sum::<usize>(), n);
        let by_bins: f64 = r
            .bins
            .iter()
            .filter_map(|b| b.gap().map(|g| b.count as f64 / n as f64 * g))
            .sum();
        prop_assert!((r.ece - by_bins).abs() <= 1e-12);
        let one = calibration_report(&d, 1);
        prop_assert!((one.ece - (one.accuracy - one.mean_confidence).abs()).abs() <= 1e-12);
        prop_assert_eq!(r.ece, calibration_report(&d, bins).ece);
    }

    #[test]
    fn permuting_rows_changes_nothing(seed in 0u64..u64::MAX, n in 2usize..200) {
        let d = random_dataset(seed, n, 4);
        let mut rows: Vec<Vec<f64>> = d.rows.iter().map(|r| r.to_vec()).collect();
        let mut labels: Vec<i64> = d.labels.iter().map(|l| *l as i64).collect();
        rows.reverse();
        labels.reverse();
        let shuffled = labelled_probabilities(Rows::from_slices(&rows), &labels, SumTolerance::default()).unwrap();
        let (a, b) = (calibration_report(&d, 10), calibration_report(&shuffled, 10));
        prop_assert!((a.ece - b.ece).abs() <= 1e-12 && (a.brier - b.brier).abs() <= 1e-12);
        prop_assert!((a.log_loss - b.log_loss).abs() <= 1e-12);
    }
}

#[test]
fn closed_on_the_left_bins_would_fail_the_edge_test() {
    // Negative control: with bins closed on the left [i/B, (i+1)/B) the confidence 0.5 would be
    // in bin 5, not 4, so `bin_closed_on_the_right` really distinguishes the two conventions.
    let left_closed = |c: f64, b: usize| ((c * b as f64).floor() as usize).min(b - 1);
    assert_eq!(left_closed(0.5, 10), 5);
    assert_eq!(bin_index(0.5, 10), 4);
}
