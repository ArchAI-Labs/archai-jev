//! The bins of the reliability diagram.

/// The bin of a confidence `c` in `[0, 1]` among `bins` uniform bins closed on the right.
///
/// Bin `i` is `(i / B, (i + 1) / B]` and `c` goes to the smallest `i` with `c <= (i + 1) / B`,
/// compared in `f64` exactly as written. So `0.1`, `0.3` and `0.5` fall in the bin they close,
/// `1.0` in the last one, and `0.0` in the first (which is therefore closed on both sides).
///
/// `ceil(c * B) - 1` is only the starting guess: `c * B` is not exact (`0.3 * 10` is
/// `3.0000000000000004`), so the two loops move to the bin the definition names.
pub(crate) fn bin_index(c: f64, bins: usize) -> usize {
    let top = bins.saturating_sub(1);
    let b = bins as f64;
    let guess = (c * b).ceil() - 1.0;
    let mut i = if guess.is_nan() || guess < 0.0 {
        0
    } else {
        (guess as usize).min(top)
    };
    while i > 0 && c <= i as f64 / b {
        i -= 1;
    }
    while i < top && c > (i + 1) as f64 / b {
        i += 1;
    }
    i
}
