//! Tests of `fit_temperature`.

use proptest::prelude::*;

use super::calibrated_softmax;
use super::fit::{DEFAULT_BOUNDS, MAX_ITERATIONS, Prepared, TemperatureFit, fit_temperature};
use super::problem::NoCalibration;
use super::rows::{Dataset, Rows, labelled_logits};
use crate::error::Error;
use crate::metrics::mean_logit_loss;

/// SplitMix64: a tiny deterministic generator for the tests.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }

    /// An index drawn from `probabilities`.
    fn categorical(&mut self, probabilities: &[f64]) -> usize {
        let mut u = self.uniform();
        for (i, p) in probabilities.iter().enumerate() {
            if u < *p {
                return i;
            }
            u -= p;
        }
        probabilities.len() - 1
    }
}

/// Gaussian logits with `sigma`, labels drawn from `softmax(z / true_t)`.
fn generate(seed: u64, n: usize, ks: &[usize], sigma: f64, true_t: f64) -> Dataset {
    let mut rng = Rng(seed);
    let mut rows = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let k = ks[i % ks.len()];
        let z: Vec<f64> = (0..k).map(|_| sigma * rng.gaussian()).collect();
        let p = calibrated_softmax(&z, true_t).unwrap();
        labels.push(rng.categorical(&p) as i64);
        rows.push(z);
    }
    labelled_logits(Rows::from_slices(&rows), &labels, true).unwrap()
}

fn dataset(rows: &[&[f64]], labels: &[i64]) -> Dataset {
    labelled_logits(Rows::from_slices(rows), labels, true).unwrap()
}

fn fit(data: &Dataset) -> TemperatureFit {
    fit_temperature(data, DEFAULT_BOUNDS).unwrap()
}

fn refusal(result: crate::error::Result<TemperatureFit>) -> NoCalibration {
    match result.unwrap_err() {
        Error::NoCalibration(reason) => reason,
        other => panic!("expected a calibration refusal, got {other}"),
    }
}

fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * a.abs().max(b.abs())
}

fn analytic() -> Dataset {
    dataset(&[&[1.0, -1.0], &[1.0, -1.0], &[1.0, -1.0]], &[0, 0, 1])
}

#[test]
fn analytic_vector_2_over_ln2() {
    let result = fit(&analytic());
    let expected = 2.0 / std::f64::consts::LN_2;
    assert!(
        close(result.temperature, expected, 1e-9),
        "{}",
        result.temperature
    );
    assert_eq!(result.n, 3);
    assert_eq!(result.bounds, DEFAULT_BOUNDS);
}

#[test]
fn loss_before_after_by_hand() {
    let result = fit(&analytic());
    let by_hand = |t: f64| {
        let right = (1.0 + (-2.0 / t).exp()).ln();
        let wrong = (1.0 + (2.0 / t).exp()).ln();
        (2.0 * right + wrong) / 3.0
    };
    assert!(close(result.loss_before, by_hand(1.0), 1e-12));
    assert!(close(result.loss_after, by_hand(result.temperature), 1e-12));
    assert!(result.loss_after < result.loss_before);
}

#[test]
fn a_grid_search_would_not_pass_the_analytic_vector() {
    // Negative control: Kev's grid (121 points in [0.25, 4]) cannot reach 1e-9 on this vector.
    let data = analytic();
    let best = (0..121)
        .map(|i| 0.25 + (4.0 - 0.25) * i as f64 / 120.0)
        .min_by(|a, b| {
            mean_logit_loss(&data.rows, &data.labels, *a).total_cmp(&mean_logit_loss(
                &data.rows,
                &data.labels,
                *b,
            ))
        })
        .unwrap();
    assert!(!close(best, 2.0 / std::f64::consts::LN_2, 1e-9));
}

fn grid_minimum(data: &Dataset, bounds: (f64, f64)) -> f64 {
    let (lo, hi) = (bounds.0.ln(), bounds.1.ln());
    (0..2000)
        .map(|i| {
            let t = (lo + (hi - lo) * i as f64 / 1999.0).exp();
            mean_logit_loss(&data.rows, &data.labels, t)
        })
        .fold(f64::INFINITY, f64::min)
}

fn check_optimum(data: &Dataset) {
    let result = fit(data);
    assert!(result.loss_after <= grid_minimum(data, DEFAULT_BOUNDS) + 1e-12);
    let prepared = Prepared::new(data).unwrap();
    let (d, _) = prepared.derivative(1.0 / result.temperature);
    assert!(
        d.abs() <= 1e-9 * prepared.scale,
        "derivative {d} at T={}",
        result.temperature
    );
    assert!(result.iterations <= MAX_ITERATIONS);
}

#[test]
fn optimal_vs_grid_2000() {
    // The largest case of the spec: N = 2000 rows of 2..=20 options, logits in [-30, 30].
    let ks: Vec<usize> = (2..=20).collect();
    check_optimum(&generate(7, 2000, &ks, 10.0, 2.0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn derivative_precision_bound(seed in 0u64..u64::MAX, n in 1usize..80, t in 0.4f64..6.0) {
        let ks: Vec<usize> = (2..=20).collect();
        let data = generate(seed, n, &ks, 8.0, t);
        match fit_temperature(&data, DEFAULT_BOUNDS) {
            Ok(_) => check_optimum(&data),
            Err(Error::NoCalibration(_)) => {}
            Err(other) => prop_assert!(false, "unexpected {other}"),
        }
    }

    #[test]
    fn scale_invariance(seed in 0u64..u64::MAX, c in 0.2f64..5.0) {
        let data = generate(seed, 60, &[2, 4, 7], 3.0, 2.0);
        let scaled_rows: Vec<Vec<f64>> =
            data.rows.iter().map(|r| r.iter().map(|z| z * c).collect()).collect();
        let labels: Vec<i64> = data.labels.iter().map(|l| *l as i64).collect();
        let scaled = labelled_logits(Rows::from_slices(&scaled_rows), &labels, true).unwrap();
        if let (Ok(a), Ok(b)) = (
            fit_temperature(&data, DEFAULT_BOUNDS),
            fit_temperature(&scaled, (0.001, 1000.0)),
        ) {
            prop_assert!(close(b.temperature, a.temperature * c, 1e-6));
        }
    }

    #[test]
    fn row_constant_shift_invariance(seed in 0u64..u64::MAX) {
        let data = generate(seed, 60, &[3, 5], 3.0, 2.0);
        let mut rng = Rng(seed ^ 0xABCD);
        let rows: Vec<Vec<f64>> = data
            .rows
            .iter()
            .map(|r| {
                let shift = 50.0 * (rng.uniform() - 0.5);
                r.iter().map(|z| z + shift).collect()
            })
            .collect();
        let labels: Vec<i64> = data.labels.iter().map(|l| *l as i64).collect();
        let shifted = labelled_logits(Rows::from_slices(&rows), &labels, true).unwrap();
        if let (Ok(a), Ok(b)) = (fit_temperature(&data, DEFAULT_BOUNDS), fit_temperature(&shifted, DEFAULT_BOUNDS)) {
            prop_assert!(close(a.temperature, b.temperature, 1e-9));
        }
    }

    #[test]
    fn row_and_column_permutation(seed in 0u64..u64::MAX) {
        let data = generate(seed, 40, &[2, 3, 6], 3.0, 2.0);
        let mut rng = Rng(seed ^ 0x1234);
        // Reverse the order of the rows and rotate the columns of each row with its label.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<i64> = Vec::new();
        for (row, &label) in data.rows.iter().zip(&data.labels).rev() {
            let shift = (rng.next() as usize) % row.len();
            let mut r = row.to_vec();
            r.rotate_left(shift);
            labels.push(((label + row.len() - shift) % row.len()) as i64);
            rows.push(r);
        }
        let permuted = labelled_logits(Rows::from_slices(&rows), &labels, true).unwrap();
        if let (Ok(a), Ok(b)) = (fit_temperature(&data, DEFAULT_BOUNDS), fit_temperature(&permuted, DEFAULT_BOUNDS)) {
            prop_assert!(close(a.temperature, b.temperature, 1e-9));
        }
    }

    #[test]
    fn log_probabilities_recipe(seed in 0u64..u64::MAX, t0 in 0.1f64..10.0) {
        let data = generate(seed, 80, &[2, 4, 5], 2.0, 1.5);
        let logs: Vec<Vec<f64>> = data
            .rows
            .iter()
            .map(|r| calibrated_softmax(r, t0).unwrap().iter().map(|p| p.ln()).collect())
            .collect();
        let labels: Vec<i64> = data.labels.iter().map(|l| *l as i64).collect();
        let relative = labelled_logits(Rows::from_slices(&logs), &labels, true).unwrap();
        let bounds = (1e-3, 1e4);
        if let (Ok(direct), Ok(via)) = (fit_temperature(&data, bounds), fit_temperature(&relative, bounds)) {
            prop_assert!(close(via.temperature * t0, direct.temperature, 1e-6));
        }
    }
}

#[test]
fn repeatable_and_bounded_iterations() {
    let data = generate(11, 500, &[2, 4, 10], 4.0, 2.0);
    let first = fit(&data);
    let second = fit(&data);
    assert_eq!(first, second);
    assert!(first.iterations >= 1 && first.iterations <= MAX_ITERATIONS);
    let from_threads: Vec<TemperatureFit> = std::thread::scope(|scope| {
        (0..4)
            .map(|_| scope.spawn(|| fit(&data)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });
    assert!(from_threads.iter().all(|r| *r == first));
}

#[test]
fn recovers_2_0_and_0_5_and_mixed_k() {
    for (true_t, ks) in [(2.0, vec![4]), (0.5, vec![4]), (2.0, vec![2, 4, 10])] {
        let data = generate(2024, 20_000, &ks, 3.0, true_t);
        let result = fit(&data);
        assert!(
            close(result.temperature, true_t, 0.05),
            "true {true_t}, got {} for K {ks:?}",
            result.temperature
        );
    }
}

#[test]
fn uniform_rows_rejected() {
    let data = dataset(&[&[2.0, 2.0], &[0.0, 0.0, 0.0]], &[0, 2]);
    assert_eq!(
        refusal(fit_temperature(&data, DEFAULT_BOUNDS)),
        NoCalibration::Uniform
    );
}

#[test]
fn always_argmax_rejected_even_n1_and_wider_bounds() {
    let data = dataset(&[&[0.0, 3.0, 2.5]], &[1]);
    for bounds in [DEFAULT_BOUNDS, (1e-3, 1e4), (1e-6, 1e9)] {
        assert_eq!(
            refusal(fit_temperature(&data, bounds)),
            NoCalibration::BelowLower { lower: bounds.0 }
        );
    }
    let many = dataset(
        &[&[1.0, 0.0], &[0.0, 5.0, 1.0], &[2.0, 2.0, 1.0]],
        &[0, 1, 1],
    );
    assert!(matches!(
        refusal(fit_temperature(&many, (1e-3, 1e4))),
        NoCalibration::BelowLower { .. }
    ));
}

#[test]
fn lowest_logit_rejected() {
    let data = dataset(&[&[0.0, 3.0, 2.5], &[5.0, 1.0], &[0.0, 4.0]], &[0, 1, 0]);
    assert_eq!(
        refusal(fit_temperature(&data, DEFAULT_BOUNDS)),
        NoCalibration::AboveUpper { upper: 100.0 }
    );
}

#[test]
fn optimum_outside_bounds() {
    // Labels drawn at T = 500 from logits with a large spread: the optimum is near 500.
    let data = generate(5, 20_000, &[4], 30.0, 500.0);
    assert!(matches!(
        refusal(fit_temperature(&data, DEFAULT_BOUNDS)),
        NoCalibration::AboveUpper { upper } if upper == 100.0
    ));
    let wide = fit_temperature(&data, (0.01, 1000.0)).unwrap();
    assert!(close(wide.temperature, 500.0, 0.3), "{}", wide.temperature);
}

#[test]
fn invalid_bounds() {
    let data = analytic();
    for bounds in [
        (0.0, 1.0),
        (2.0, 1.0),
        (1.0, f64::INFINITY),
        (-1.0, 5.0),
        (f64::NAN, 5.0),
    ] {
        let e = fit_temperature(&data, bounds).unwrap_err();
        assert!(
            e.to_string()
                .starts_with("bounds must satisfy 0 < lower < upper < inf, got ("),
            "{e}"
        );
    }
}

#[test]
fn log_probabilities_recipe_by_hand() {
    // T0 = 2: the relative temperature times T0 is the temperature of the raw logits.
    let t0 = 2.0;
    let z = [[1.0, -1.0], [0.5, 0.2], [-2.0, 1.0], [0.1, 0.0]];
    let labels = [0, 1, 1, 0];
    let direct = dataset(&[&z[0], &z[1], &z[2], &z[3]], &labels);
    let logs: Vec<Vec<f64>> = z
        .iter()
        .map(|r| {
            calibrated_softmax(r, t0)
                .unwrap()
                .iter()
                .map(|p| p.ln())
                .collect()
        })
        .collect();
    let relative = labelled_logits(Rows::from_slices(&logs), &labels, true).unwrap();
    let a = fit(&direct).temperature;
    let b = fit(&relative).temperature * t0;
    assert!(close(a, b, 1e-6), "{a} vs {b}");
}

#[test]
fn extreme_logits_1e6() {
    let data = generate(3, 2_000, &[3, 6], 1e6, 2e5);
    let result = fit_temperature(&data, (1.0, 1e9)).unwrap();
    assert!(result.temperature.is_finite() && result.loss_after.is_finite());
    assert!(
        close(result.temperature, 2e5, 0.25),
        "{}",
        result.temperature
    );
}

#[test]
fn n_100000_no_nan() {
    let data = generate(99, 100_000, &[4], 3.0, 2.0);
    let result = fit(&data);
    assert!(result.temperature.is_finite() && !result.loss_after.is_nan());
    assert!(
        close(result.temperature, 2.0, 0.05),
        "{}",
        result.temperature
    );
    assert!(result.iterations <= MAX_ITERATIONS);
}

#[test]
fn a_large_common_offset_does_not_loosen_the_tolerance() {
    // The loss does not depend on a constant added to every logit of a row. With an offset of
    // 1e12 the stopping tolerance must still be that of the spread (about 1), not of the offset.
    let base = analytic();
    let expected = fit(&base).temperature;
    for offset in [1e3, 1e7, 1e9, 1e12] {
        let rows = [[1.0 + offset, -1.0 + offset]; 3];
        let shifted = dataset(&[&rows[0], &rows[1], &rows[2]], &[0, 0, 1]);
        let got = fit(&shifted);
        assert!(
            close(got.temperature, expected, 1e-6),
            "offset {offset}: {}",
            got.temperature
        );
    }
}

#[test]
fn n1_interior_and_two_rejections() {
    let interior = dataset(&[&[0.0, 3.0, 2.5]], &[2]);
    let result = fit(&interior);
    assert!(
        close(result.temperature, 1.864, 1e-3),
        "{}",
        result.temperature
    );
    assert!(matches!(
        refusal(fit_temperature(
            &dataset(&[&[0.0, 3.0, 2.5]], &[1]),
            DEFAULT_BOUNDS
        )),
        NoCalibration::BelowLower { .. }
    ));
    assert!(matches!(
        refusal(fit_temperature(
            &dataset(&[&[0.0, 3.0, 2.5]], &[0]),
            DEFAULT_BOUNDS
        )),
        NoCalibration::AboveUpper { .. }
    ));
}

#[test]
fn clamping_to_the_edge_is_not_what_the_function_does() {
    // Negative control: a "fit" that clamps would return the lower bound for argmax-only labels.
    let data = dataset(&[&[0.0, 3.0, 2.5]], &[1]);
    assert!(fit_temperature(&data, DEFAULT_BOUNDS).is_err());
}

#[test]
fn logits_near_the_float_limit_never_give_a_made_up_result() {
    // The difference of these logits overflows: a refusal, not NaN read as "converged".
    let data = dataset(&[&[1e308, -1e308], &[1e308, -1e308]], &[0, 1]);
    let e = fit_temperature(&data, DEFAULT_BOUNDS).unwrap_err();
    assert_eq!(
        e.to_string(),
        "logits row 0 spans more than the largest float number (the difference between its largest and smallest logit overflows); scale the logits down"
    );
    // Huge but representable differences either fit with a finite temperature or are refused.
    let big = dataset(&[&[1e308, 0.9e308], &[0.9e308, 1e308]], &[0, 0]);
    match fit_temperature(&big, (1.0, 1e308)) {
        Ok(fit) => assert!(fit.temperature.is_finite() && fit.loss_after.is_finite()),
        Err(Error::NoCalibration(_)) => {}
        Err(other) => panic!("unexpected {other}"),
    }
}

#[test]
fn a_tiny_lower_bound_cannot_skip_the_refusal() {
    let data = dataset(&[&[0.0, 3.0, 2.5]], &[1]);
    let e = fit_temperature(&data, (1e-310, 100.0)).unwrap_err();
    assert_eq!(
        e.to_string(),
        "the lower bound 1e-310 of bounds is too small: 1 / lower must be a finite number"
    );
}
