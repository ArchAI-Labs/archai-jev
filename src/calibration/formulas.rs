//! The public formulas of the answers: confidence of a choice or a score, and the value of a score.
//!
//! These are the functions `ask` uses; the public functions of `archai_jev.calibration` validate
//! their input and then call exactly these, so both give the same bits.

/// Index of the first maximum (the tie-break of every answer).
pub fn first_argmax(p: &[f64]) -> usize {
    let mut best = 0;
    let mut best_value = f64::NEG_INFINITY;
    for (i, v) in p.iter().copied().enumerate() {
        if v > best_value {
            best = i;
            best_value = v;
        }
    }
    best
}

/// `p / sum(p)`, uniform when the sum is 0 (Kev's `_normalize`).
pub(crate) fn normalised(p: &[f64]) -> Vec<f64> {
    let sum: f64 = p.iter().sum();
    if sum > 0.0 {
        p.iter().map(|x| x / sum).collect()
    } else {
        vec![1.0 / p.len().max(1) as f64; p.len()]
    }
}

/// Choice confidence: 1 for a single option, otherwise `clamp01((max p - 1/K) / (1 - 1/K))`
/// on the normalised probabilities.
pub fn choice_confidence(p: &[f64]) -> f64 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let max = normalised(p).into_iter().fold(0.0_f64, f64::max);
    let inv = 1.0 / k as f64;
    ((max - inv) / (1.0 - inv)).clamp(0.0, 1.0)
}

/// Expected level `sum(i * p_i)`, 0-based.
pub fn score_value(p: &[f64]) -> f64 {
    p.iter().enumerate().map(|(i, p)| i as f64 * p).sum()
}

/// `D`, the mean absolute deviation of a uniform distribution over `l` levels (closed form).
pub fn score_deviation(l: usize) -> f64 {
    let lf = l as f64;
    if l.is_multiple_of(2) {
        lf / 4.0
    } else {
        (lf * lf - 1.0) / (4.0 * lf)
    }
}

/// Score confidence: 1 for one level, otherwise `max(0, 1 - sum(p_i |i - m|) / D)` on the
/// normalised probabilities, `m` the first maximum.
pub fn score_confidence(p: &[f64]) -> f64 {
    let l = p.len();
    if l <= 1 {
        return 1.0;
    }
    let mode = first_argmax(p) as f64;
    let spread: f64 = normalised(p)
        .iter()
        .enumerate()
        .map(|(i, p)| p * (i as f64 - mode).abs())
        .sum();
    (1.0 - spread / score_deviation(l)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moved_functions_keep_their_values() {
        // Choice: (0.7 - 1/3) / (1 - 1/3) = 0.55
        assert!((choice_confidence(&[0.7, 0.2, 0.1]) - 0.55).abs() < 1e-12);
        assert_eq!(choice_confidence(&[1.0]), 1.0);
        assert!(choice_confidence(&[0.25; 4]).abs() < 1e-12);
        // Score: value and the two clamped cases of Kev's `main`.
        assert!((score_value(&[0.1, 0.2, 0.7]) - 1.6).abs() < 1e-12);
        assert_eq!(score_confidence(&[1.0]), 1.0);
        assert_eq!(score_confidence(&[0.5, 0.0, 0.5]), 0.0);
        assert!(score_confidence(&[0.5, 0.5]).abs() < 1e-12);
        // Tie-break and normalisation.
        assert_eq!(first_argmax(&[0.4, 0.4, 0.2]), 0);
        assert_eq!(first_argmax(&[]), 0);
        assert!(choice_confidence(&[0.0, 0.0, 0.0]).abs() < 1e-12);
        assert_eq!(score_deviation(2), 0.5);
        assert!((score_deviation(3) - 2.0 / 3.0).abs() < 1e-15);
    }
}
