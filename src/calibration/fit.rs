//! `fit_temperature`: the temperature that minimises the log-loss of labelled logits.
//!
//! With `beta = 1 / T` the mean log-loss `L(beta)` is convex (a sum of log-sum-exp minus linear
//! terms), so it has a single minimum and its derivative
//! `D(beta) = mean_i [ E_{p_i(beta)}[z_i] - z_{i,y_i} ]` never decreases. The fit is therefore the
//! zero of a monotone function: Newton's method kept inside a bracket (so it cannot diverge), and
//! no optimiser or grid. If the zero is not inside the bounds the data do not determine a
//! calibration and the function refuses (D20): it never returns a temperature at the edge.

use crate::error::{Error, Result};
use crate::metrics::mean_logit_loss;

use super::problem::{NoCalibration, RowsProblem};
use super::rows::{Dataset, Neumaier};

/// The most iterations of the search. Bisection alone needs about 45 to close the default
/// bounds to `1e-12`, so reaching this limit is a bug, not a data problem.
pub const MAX_ITERATIONS: usize = 200;

/// The default bounds of the search for `T`.
pub const DEFAULT_BOUNDS: (f64, f64) = (0.01, 100.0);

/// The result of [`fit_temperature`].
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureFit {
    /// The `T` that minimises the log-loss.
    pub temperature: f64,
    /// The mean log-loss at `T = 1`.
    pub loss_before: f64,
    /// The mean log-loss at `temperature`.
    pub loss_after: f64,
    /// Number of rows.
    pub n: usize,
    /// Steps of the search for the zero.
    pub iterations: usize,
    /// The bounds the search ran in.
    pub bounds: (f64, f64),
}

/// The data of the fit, shifted so every row has its maximum at 0.
///
/// Subtracting the row maximum changes neither the probabilities nor `D` (the shift cancels
/// between `E[z]` and `z_y`) and makes every `exp` argument `<= 0`: no overflow even with
/// logits of `1e6`, and the loss does not depend on an additive constant per row.
///
/// The preparation is done once; a fit on some of the rows (the folds of `calibrate`) reads the
/// same buffers with a mask of the rows in use, instead of copying them.
pub(crate) struct Prepared {
    shifted: Vec<f64>,
    offsets: Vec<usize>,
    target: Vec<f64>,
    /// Mean of `|z - max z|` of each row, for the scale of the stopping rule.
    row_magnitude: Vec<f64>,
    /// Whether every logit of the row is equal.
    row_uniform: Vec<bool>,
    /// `max(1, mean |z - max z|)` over all rows: the scale of the stopping rule. It is the
    /// spread **inside** the rows, like `D` itself: a large offset common to all the logits must
    /// not loosen the tolerance.
    pub(crate) scale: f64,
}

impl Prepared {
    /// # Errors
    /// `RowsProblem::RangeTooLarge` if the logits of a row are so far apart that their
    /// difference overflows (such a row would turn the sums into NaN, and a NaN must never be
    /// read as "converged").
    pub(crate) fn new(data: &Dataset) -> Result<Self> {
        let mut shifted = Vec::with_capacity(data.rows.values().len());
        let mut offsets = vec![0];
        let mut target = Vec::with_capacity(data.rows.len());
        let mut row_magnitude = Vec::with_capacity(data.rows.len());
        let mut row_uniform = Vec::with_capacity(data.rows.len());
        // Running means: the sum of |z| could overflow where the mean does not.
        let (mut mean_magnitude, mut seen) = (0.0_f64, 0.0_f64);
        for (i, (row, &label)) in data.rows.iter().zip(&data.labels).enumerate() {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mut row_mean, mut in_row) = (0.0_f64, 0.0_f64);
            let mut uniform = true;
            for v in row {
                let u = v - max;
                if !u.is_finite() {
                    return Err(Error::Rows(RowsProblem::RangeTooLarge { row: i }));
                }
                uniform &= u == 0.0;
                shifted.push(u);
                seen += 1.0;
                in_row += 1.0;
                mean_magnitude += (u.abs() - mean_magnitude) / seen;
                row_mean += (u.abs() - row_mean) / in_row;
            }
            offsets.push(shifted.len());
            target.push(row.get(label).copied().unwrap_or(max) - max);
            row_magnitude.push(row_mean);
            row_uniform.push(uniform);
        }
        Ok(Prepared {
            shifted,
            offsets,
            target,
            row_magnitude,
            row_uniform,
            scale: mean_magnitude.max(1.0),
        })
    }

    /// The rows in use: `(index, shifted logits, shifted target)`.
    fn rows<'a>(
        &'a self,
        active: Option<&'a [bool]>,
    ) -> impl Iterator<Item = (usize, &'a [f64], f64)> + 'a {
        self.offsets
            .windows(2)
            .zip(&self.target)
            .enumerate()
            .filter(move |(i, _)| active.is_none_or(|a| a.get(*i).copied().unwrap_or(false)))
            .map(|(i, (w, &t))| match (w.first(), w.get(1)) {
                (Some(&a), Some(&b)) => (i, self.shifted.get(a..b).unwrap_or(&[]), t),
                _ => (i, &[][..], t),
            })
    }

    /// True when every logit of every row in use is equal: the loss does not depend on `T`.
    fn is_uniform(&self, active: Option<&[bool]>) -> bool {
        self.rows(active)
            .all(|(i, _, _)| self.row_uniform.get(i).copied().unwrap_or(true))
    }

    /// `max(1, mean |z - max z|)` over the rows in use.
    fn scale_where(&self, active: Option<&[bool]>) -> f64 {
        if active.is_none() {
            return self.scale;
        }
        let (mut mean, mut seen) = (0.0_f64, 0.0_f64);
        for (i, row, _) in self.rows(active) {
            let len = row.len() as f64;
            seen += len;
            let row_mean = self.row_magnitude.get(i).copied().unwrap_or(0.0);
            if seen > 0.0 {
                mean += (row_mean - mean) * len / seen;
            }
        }
        mean.max(1.0)
    }

    /// `D(beta)` and `D'(beta)` over all rows (what the tests check the result against).
    #[cfg(test)]
    pub(crate) fn derivative(&self, beta: f64) -> (f64, f64) {
        self.derivative_where(beta, None)
    }

    /// `D(beta)` and `D'(beta)` (the mean variance of the logits under the model) over the rows
    /// in use.
    fn derivative_where(&self, beta: f64, active: Option<&[bool]>) -> (f64, f64) {
        let mut d = Neumaier::new();
        let mut curvature = Neumaier::new();
        let mut n = 0usize;
        for (_, row, target) in self.rows(active) {
            let (mut z, mut s1, mut s2) = (0.0_f64, 0.0_f64, 0.0_f64);
            for &u in row {
                let e = (beta * u).exp();
                z += e;
                s1 += e * u;
                s2 += (e * u) * u;
            }
            let mean = s1 / z;
            d.add(mean - target);
            curvature.add((s2 / z - mean * mean).max(0.0));
            n += 1;
        }
        let n = n.max(1) as f64;
        (d.total() / n, curvature.total() / n)
    }
}

/// # Errors
/// `RowsProblem::InvalidBounds` unless `0 < lower < upper < inf`.
pub fn check_bounds(bounds: (f64, f64)) -> Result<()> {
    let (lower, upper) = bounds;
    if !(lower.is_finite() && upper.is_finite() && 0.0 < lower && lower < upper) {
        return Err(Error::Rows(RowsProblem::InvalidBounds { lower, upper }));
    }
    if !(1.0 / lower).is_finite() {
        return Err(Error::Rows(RowsProblem::LowerBoundTooSmall { lower }));
    }
    Ok(())
}

/// The temperature that minimises the log-loss of `data`, searched in `bounds`.
///
/// # Errors
/// - `RowsProblem::InvalidBounds` unless `0 < lower < upper < inf`.
/// - `NoCalibration::Uniform` if every row has all its logits equal.
/// - `NoCalibration::BelowLower` / `AboveUpper` if the best temperature is not inside `bounds`
///   (labels that always match the top option; logits that carry no information about the
///   labels or point the wrong way).
pub fn fit_temperature(data: &Dataset, bounds: (f64, f64)) -> Result<TemperatureFit> {
    check_bounds(bounds)?;
    let prepared = Prepared::new(data)?;
    fit_prepared(&prepared, data, bounds)
}

/// [`fit_temperature`] over data already prepared (`data` is what `prepared` was made from).
pub(crate) fn fit_prepared(
    prepared: &Prepared,
    data: &Dataset,
    bounds: (f64, f64),
) -> Result<TemperatureFit> {
    let (temperature, iterations) = solve(prepared, None, bounds)?;
    Ok(TemperatureFit {
        temperature,
        loss_before: mean_logit_loss(&data.rows, &data.labels, 1.0),
        loss_after: mean_logit_loss(&data.rows, &data.labels, temperature),
        n: data.rows.len(),
        iterations,
        bounds,
    })
}

/// The best temperature over the rows in use (`active`, or all) and the steps it took.
///
/// # Errors
/// As [`fit_temperature`].
pub(crate) fn solve(
    prepared: &Prepared,
    active: Option<&[bool]>,
    bounds: (f64, f64),
) -> Result<(f64, usize)> {
    let (lower, upper) = bounds;
    if prepared.is_uniform(active) {
        return Err(Error::NoCalibration(NoCalibration::Uniform));
    }
    let (beta_min, beta_max) = (1.0 / upper, 1.0 / lower);
    let (d_min, _) = prepared.derivative_where(beta_min, active);
    if d_min >= 0.0 {
        return Err(Error::NoCalibration(NoCalibration::AboveUpper { upper }));
    }
    let (d_max, _) = prepared.derivative_where(beta_max, active);
    if d_max <= 0.0 {
        return Err(Error::NoCalibration(NoCalibration::BelowLower { lower }));
    }
    let (beta, iterations) = find_zero(prepared, active, beta_min, beta_max)?;
    Ok((1.0 / beta, iterations))
}

/// The zero of `D` in `(low, high)`, where `D(low) < 0 < D(high)`: Newton's method with a
/// bracket (rtsafe), the bracket shrinking in log space when Newton is not making progress.
fn find_zero(
    prepared: &Prepared,
    active: Option<&[bool]>,
    mut low: f64,
    mut high: f64,
) -> Result<(f64, usize)> {
    let tolerance = 1e-12 * prepared.scale_where(active);
    let mut beta = 1.0_f64.clamp(low, high);
    if beta <= low || beta >= high {
        beta = (low * high).sqrt();
    }
    let mut step = high - low;
    let mut previous_step = step;
    let (mut f, mut df) = prepared.derivative_where(beta, active);
    for iteration in 1..=MAX_ITERATIONS {
        if f.abs() <= tolerance {
            return Ok((beta, iteration));
        }
        if f < 0.0 {
            low = beta;
        } else {
            high = beta;
        }
        if high - low <= 1e-12 * beta {
            return Ok((beta, iteration));
        }
        let newton_outside = ((beta - high) * df - f) * ((beta - low) * df - f) > 0.0;
        let newton_slow = (2.0 * f).abs() > (previous_step * df).abs();
        let next = if newton_outside || newton_slow || df <= 0.0 {
            (low * high).sqrt()
        } else {
            beta - f / df
        };
        previous_step = step;
        step = next - beta;
        beta = next;
        (f, df) = prepared.derivative_where(beta, active);
    }
    Err(Error::NoCalibration(NoCalibration::DidNotConverge))
}
