//! 95 % intervals by resampling whole groups with replacement (spec 017 §3.5).
//!
//! Records of one group are not independent (the same customer, the same document), so the
//! unit of resampling is the group. For each group the sums that make the metrics are computed
//! once; a resample adds up the sums of the groups it draws, which costs `O(groups · bins)`
//! instead of `O(rows)`.

use std::collections::HashMap;

use crate::calibration::formulas::first_argmax;
use crate::calibration::rows::Dataset as Rows;
use crate::metrics::bin_index;
use crate::rng::SplitMix64;

/// The metrics with an interval, in this order.
pub const METRICS: [&str; 4] = ["accuracy", "ece", "brier", "log_loss"];

/// A group's sums: enough to compute every metric of any union of groups.
#[derive(Debug, Clone)]
struct Sums {
    n: f64,
    correct: f64,
    brier: f64,
    log_loss: f64,
    /// Per bin: rows, correct rows, sum of confidences.
    bins: Vec<(f64, f64, f64)>,
}

impl Sums {
    fn new(n_bins: usize) -> Self {
        Sums {
            n: 0.0,
            correct: 0.0,
            brier: 0.0,
            log_loss: 0.0,
            bins: vec![(0.0, 0.0, 0.0); n_bins],
        }
    }

    fn add(&mut self, other: &Sums) {
        self.n += other.n;
        self.correct += other.correct;
        self.brier += other.brier;
        self.log_loss += other.log_loss;
        for (mine, theirs) in self.bins.iter_mut().zip(&other.bins) {
            mine.0 += theirs.0;
            mine.1 += theirs.1;
            mine.2 += theirs.2;
        }
    }

    /// `accuracy, ece, brier, log_loss`.
    fn metrics(&self) -> [f64; 4] {
        let ece: f64 = self
            .bins
            .iter()
            .filter(|b| b.0 > 0.0)
            .map(|b| b.0 / self.n * (b.1 / b.0 - b.2 / b.0).abs())
            .sum();
        [
            self.correct / self.n,
            ece,
            self.brier / self.n,
            self.log_loss / self.n,
        ]
    }
}

/// Per-group sums of a scope. `group_of_row[i]` is the group index (of the dataset) of row `i`.
fn group_sums(data: &Rows, group_of_row: &[usize], n_bins: usize) -> Vec<Sums> {
    let mut by_group: Vec<Sums> = Vec::new();
    let mut slot_of: HashMap<usize, usize> = HashMap::new();
    for ((row, &label), &group) in data.rows.iter().zip(&data.labels).zip(group_of_row) {
        let slot = *slot_of.entry(group).or_insert_with(|| {
            by_group.push(Sums::new(n_bins));
            by_group.len() - 1
        });
        let Some(sums) = by_group.get_mut(slot) else {
            continue;
        };
        let top = first_argmax(row);
        let confidence = row.get(top).copied().unwrap_or(0.0);
        sums.n += 1.0;
        if top == label {
            sums.correct += 1.0;
        }
        if let Some(b) = sums.bins.get_mut(bin_index(confidence, n_bins)) {
            b.0 += 1.0;
            b.1 += f64::from(u8::from(top == label));
            b.2 += confidence;
        }
        let mut squares = 0.0;
        for (k, p) in row.iter().enumerate() {
            let y = if k == label { 1.0 } else { 0.0 };
            squares += (p - y) * (p - y);
        }
        sums.brier += squares;
        sums.log_loss += 0.0 - row.get(label).copied().unwrap_or(0.0).ln();
    }
    by_group
}

/// The groups a resample draws: `groups` indices in `0..groups`, with replacement.
pub(crate) fn draw(rng: &mut SplitMix64, groups: usize) -> Vec<usize> {
    (0..groups)
        .map(|_| usize::try_from(rng.below(groups as u64)).unwrap_or(0))
        .collect()
}

/// The `q` percentile (0..100) of sorted values, by linear interpolation between order
/// statistics (`numpy.percentile`, method `linear`).
pub(crate) fn percentile(sorted: &[f64], q: f64) -> f64 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f64::NAN;
    };
    let position = q / 100.0 * last as f64;
    let low = position.floor() as usize;
    let fraction = position - low as f64;
    let a = sorted.get(low).copied().unwrap_or(f64::NAN);
    let b = sorted.get((low + 1).min(last)).copied().unwrap_or(a);
    if fraction == 0.0 || a == b {
        a
    } else {
        a * (1.0 - fraction) + b * fraction
    }
}

/// `(lower, upper)` 95 % intervals of `accuracy, ece, brier, log_loss`.
///
/// `group_of_row` gives the dataset group of every row. Only the groups that have rows in this
/// scope are resampled. With one group every resample is the whole data: a degenerate interval.
pub fn intervals(
    data: &Rows,
    group_of_row: &[usize],
    n_bins: usize,
    samples: usize,
    seed: u64,
) -> [(f64, f64); 4] {
    let sums = group_sums(data, group_of_row, n_bins);
    let mut rng = SplitMix64::new(seed);
    let mut collected: [Vec<f64>; 4] = Default::default();
    for _ in 0..samples {
        let mut total = Sums::new(n_bins);
        for g in draw(&mut rng, sums.len()) {
            if let Some(s) = sums.get(g) {
                total.add(s);
            }
        }
        for (slot, value) in collected.iter_mut().zip(total.metrics()) {
            slot.push(value);
        }
    }
    let mut out = [(0.0, 0.0); 4];
    for (target, values) in out.iter_mut().zip(collected.iter_mut()) {
        values.sort_by(f64::total_cmp);
        *target = (percentile(values, 2.5), percentile(values, 97.5));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::rows::{Rows as RowStore, SumTolerance, labelled_probabilities};
    use crate::metrics::calibration_report;

    fn dataset(rows: &[Vec<f64>], labels: &[i64]) -> Rows {
        labelled_probabilities(RowStore::from_slices(rows), labels, SumTolerance::default())
            .unwrap()
    }

    fn small() -> (Rows, Vec<usize>) {
        let rows = vec![
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.3, 0.7],
            vec![0.6, 0.4],
            vec![0.5, 0.5],
            vec![0.2, 0.8],
        ];
        (dataset(&rows, &[0, 1, 1, 0, 1, 1]), vec![0, 0, 1, 1, 2, 3])
    }

    #[test]
    fn percentile_linear_definition() {
        // numpy.percentile([1, 2, 3, 4, 5], q) with the default method
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 50.0), 3.0);
        assert_eq!(percentile(&v, 100.0), 5.0);
        assert!((percentile(&v, 2.5) - 1.1).abs() < 1e-12);
        assert!((percentile(&v, 97.5) - 4.9).abs() < 1e-12);
        assert_eq!(percentile(&[7.0], 97.5), 7.0);
        assert!(percentile(&[], 50.0).is_nan());
        assert_eq!(percentile(&[1.0, f64::INFINITY], 50.0), f64::INFINITY);
        assert_eq!(
            percentile(&[f64::INFINITY, f64::INFINITY], 50.0),
            f64::INFINITY
        );
    }

    #[test]
    fn same_seed_same_intervals() {
        let (data, groups) = small();
        let a = intervals(&data, &groups, 10, 200, 7);
        assert_eq!(a, intervals(&data, &groups, 10, 200, 7));
        assert_ne!(a, intervals(&data, &groups, 10, 200, 8));
    }

    #[test]
    fn literals_small_example() {
        let (data, groups) = small();
        let got = intervals(&data, &groups, 10, 100, 0);
        // Cross-checked with an independent Python implementation (SplitMix64, the same draws,
        // numpy.percentile with the linear method).
        let expected: [(f64, f64); 4] = PIN_INTERVALS;
        for (g, e) in got.iter().zip(expected) {
            assert!(
                (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                "{got:?}"
            );
        }
    }

    const PIN_INTERVALS: [(f64, f64); 4] = [
        (0.4135714285714286, 1.0),
        (0.2000000000000001, 0.44583333333333314),
        (0.165575, 0.5685714285714287),
        (0.32318389612015275, 0.7667912622271414),
    ];

    #[test]
    fn lower_le_upper_and_accuracy_in_unit_interval() {
        let (data, groups) = small();
        for seed in 0..5 {
            for (lo, hi) in intervals(&data, &groups, 10, 300, seed) {
                assert!(lo <= hi);
            }
            let (lo, hi) = intervals(&data, &groups, 10, 300, seed)[0];
            assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
        }
    }

    #[test]
    fn one_group_is_degenerate() {
        let (data, _) = small();
        let one = vec![0; 6];
        let report = calibration_report(&data, 10);
        let got = intervals(&data, &one, 10, 50, 3);
        let estimate = [report.accuracy, report.ece, report.brier, report.log_loss];
        for ((lo, hi), e) in got.iter().zip(estimate) {
            assert!(
                (lo - e).abs() < 1e-12 && (hi - e).abs() < 1e-12,
                "{got:?} vs {estimate:?}"
            );
        }
    }

    #[test]
    fn matches_direct_recomputation_on_resampled_groups() {
        let (data, groups) = small();
        let sums = group_sums(&data, &groups, 10);
        let mut rng = SplitMix64::new(11);
        for _ in 0..20 {
            let drawn = draw(&mut rng, sums.len());
            let mut total = Sums::new(10);
            for g in &drawn {
                total.add(&sums[*g]);
            }
            // the same rows, gathered by hand and run through the metrics of 004
            let distinct: Vec<usize> = {
                let mut seen = Vec::new();
                for g in &groups {
                    if !seen.contains(g) {
                        seen.push(*g);
                    }
                }
                seen
            };
            let mut rows = Vec::new();
            let mut labels = Vec::new();
            for d in drawn {
                let group = distinct[d];
                for (i, g) in groups.iter().enumerate() {
                    if *g == group {
                        rows.push(data.rows.row(i).to_vec());
                        labels.push(data.labels[i] as i64);
                    }
                }
            }
            let direct = calibration_report(&dataset(&rows, &labels), 10);
            let m = total.metrics();
            for (a, b) in m
                .iter()
                .zip([direct.accuracy, direct.ece, direct.brier, direct.log_loss])
            {
                assert!((a - b).abs() < 1e-12, "{m:?}");
            }
        }
    }
}
