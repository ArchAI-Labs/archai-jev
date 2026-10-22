//! The inference engine seen from the core (spec 006b section 7).
//!
//! A [`Forward`] turns token **ids** into numbers: the hidden state of the last layer (after the
//! final norm) at some positions, or the model's logits at a position restricted to some ids.
//! It never sees text (D24) and it never reads an output by "row of output" index, which in
//! llama.cpp silently returns the wrong numbers (spike S4): the only way to ask for a number is
//! [`OutputSpec`], by position in the branch you sent.
//!
//! The pattern is the one proven in S4: the state is decoded **once** per request, then every
//! question runs on a private copy of it, so no question can see another and the state is never
//! changed. One [`Session`] is one request.

pub mod cpu;
pub mod device;
#[cfg(any(test, feature = "testing"))]
pub mod fake;
pub mod llama;
mod waves;

use std::any::Any;
use std::sync::Arc;

use crate::control::Control;
use crate::error::{Error, Result};

/// What to read from a row of the model. More kinds can be added without touching the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputSpec {
    /// The hidden state (last layer, after the final norm, as f32) at `position` of the branch.
    Hidden {
        /// Index inside the branch.
        position: usize,
    },
    /// The logits of the model at `position` of the branch, only for `ids`.
    Logits {
        /// Index inside the branch.
        position: usize,
        /// Token ids whose logits are wanted, in this order.
        ids: Vec<u32>,
    },
    /// For each of `keys` (token sequences), the sum of the log-probabilities of its tokens as
    /// the continuation of the branch up to and including `position`: the first token is scored
    /// at `position`, each next one after the previous ones. The log-probabilities are
    /// normalised over the **whole vocabulary** (spec 010). `position` must be the last token of
    /// the row. The keys do not see each other.
    KeyLogProbs {
        /// Index inside the branch: its last token.
        position: usize,
        /// The token ids of each key, none empty.
        keys: Vec<Vec<u32>>,
    },
}

/// A number read from the model: the answer to one [`OutputSpec`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Output {
    /// `n_embd` values.
    Hidden(Vec<f32>),
    /// One value per requested id.
    Logits(Vec<f32>),
    /// One sum of log-probabilities per requested key.
    KeyLogProbs(Vec<f64>),
}

/// The shared part of a request, saved so a later request can start from it without decoding it
/// again (spec 007). Opaque: only the engine that made it can restore it. It owns its bytes and
/// is cheap to clone (a reference count).
#[derive(Clone)]
pub struct Snapshot {
    tokens: usize,
    bytes: usize,
    inner: Arc<dyn Any + Send + Sync>,
}

impl Snapshot {
    /// Wrap an engine's own saved state: `tokens` positions that take `bytes` bytes of memory.
    pub fn new<T: Any + Send + Sync>(tokens: usize, bytes: usize, state: T) -> Self {
        Snapshot {
            tokens,
            bytes,
            inner: Arc::new(state),
        }
    }

    /// How many positions (tokens) the snapshot holds.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Memory it takes, in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The engine's own state, if it is of type `T` (an engine asks for its own type; any other
    /// snapshot is refused).
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("tokens", &self.tokens)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

fn no_segments() -> Error {
    Error::Inference {
        detail: "this engine does not decode a row in segments".to_string(),
    }
}

/// One request being answered: the state is already decoded and kept.
pub trait Session {
    /// Run one question: decode `branch` after the state, on a copy that is discarded, and return
    /// one [`Output`] per entry of `outputs`, in order. Values that are not finite are an error
    /// ([`crate::Error::NonFiniteHidden`], with an empty question name: the caller names it).
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine fails; the session stays usable.
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>>;

    /// How many tokens the shared part of the session holds now.
    fn shared_len(&self) -> usize {
        0
    }

    /// Start from a saved shared part. Only on a session that holds nothing yet.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the session is not empty, if the snapshot was not made by
    /// this engine with these parameters, or if the engine cannot load it.
    fn restore(&mut self, _snapshot: &Snapshot) -> Result<()> {
        Err(no_segments())
    }

    /// Decode `tokens` onto the shared part, in one decode, at the positions that follow it.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine fails.
    fn extend(&mut self, _tokens: &[u32]) -> Result<()> {
        Err(no_segments())
    }

    /// A copy of the shared part as it is now.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine cannot save it.
    fn snapshot(&mut self) -> Result<Snapshot> {
        Err(no_segments())
    }

    /// Run one question given as consecutive segments after the shared part: one decode **per
    /// segment**, outputs only from the last. The segments are what the question's own cut points
    /// define, so the numbers do not depend on what the other questions or a cache did.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine fails; the session stays usable.
    fn row_segments(
        &mut self,
        _segments: &[&[u32]],
        _outputs: &[OutputSpec],
    ) -> Result<Vec<Output>> {
        Err(no_segments())
    }
}

/// A loaded model that can answer requests.
pub trait Forward: Send + Sync {
    /// Start a request: decode `state` once (positions `0..state.len()`) and keep it. The model
    /// is not thread-safe, so the returned session holds it: concurrent requests wait their turn.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the state cannot be decoded.
    fn begin<'a>(&'a self, state: &[u32]) -> Result<Box<dyn Session + 'a>>;

    /// Start a request with nothing decoded: the shared part is built with
    /// [`Session::restore`] and [`Session::extend`]. `begin(state)` is this plus `extend(state)`.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine cannot start (or does not decode in segments).
    fn begin_empty<'a>(&'a self) -> Result<Box<dyn Session + 'a>> {
        Err(no_segments())
    }

    /// Like [`begin`](Self::begin), but the wait for the engine looks at `control`, so a request
    /// that is queued can give up (spec 008). An engine with no queue keeps this default.
    ///
    /// # Errors
    /// [`crate::Error::Cancelled`] if `control` says stop while waiting; otherwise as `begin`.
    fn begin_with<'a>(
        &'a self,
        state: &[u32],
        _control: &dyn Control,
    ) -> Result<Box<dyn Session + 'a>> {
        self.begin(state)
    }

    /// Like [`begin_empty`](Self::begin_empty), with the wait looking at `control`.
    ///
    /// # Errors
    /// [`crate::Error::Cancelled`] if `control` says stop while waiting; otherwise as
    /// `begin_empty`.
    fn begin_empty_with<'a>(&'a self, _control: &dyn Control) -> Result<Box<dyn Session + 'a>> {
        self.begin_empty()
    }
}

/// The logits of `ids` out of a **whole** row of the model's logits (one value per token of the
/// vocabulary), read at `position`.
///
/// Every value of the row is checked, not only the wanted ones: a `nan` or an infinity anywhere
/// means a corrupt file or an overflow, and the wanted numbers cannot be trusted (spike S5, 8.1).
/// The wanted tokens usually have almost no mass in the full softmax (the letters head was trained
/// on the restricted one), so the mass on them is deliberately **not** checked.
///
/// # Errors
/// [`Error::NonFiniteHidden`] (with an empty question name: the caller names it) if the row has a
/// value that is not finite; [`Error::Inference`] if an id is outside the vocabulary.
pub fn pick_logits(row: &[f32], ids: &[u32], position: usize) -> Result<Vec<f32>> {
    if let Some(bad) = row.iter().copied().find(|v| !v.is_finite()) {
        return Err(Error::NonFiniteHidden {
            question: String::new(),
            position,
            value: f64::from(bad),
        });
    }
    ids.iter()
        .map(|&id| {
            row.get(id as usize)
                .copied()
                .ok_or_else(|| Error::Inference {
                    detail: format!("token id {id} is outside the vocabulary"),
                })
        })
        .collect()
}

/// The log-probabilities of `ids` out of a **whole** row of logits, normalised over the whole
/// vocabulary: `logit[id] - log(sum(exp(logit)))`, computed in `f64`.
///
/// Like [`pick_logits`], every value of the row is checked first.
///
/// # Errors
/// [`Error::NonFiniteHidden`] (with an empty question name) if the row has a value that is not
/// finite; [`Error::Inference`] if an id is outside the vocabulary.
pub fn log_probs_of(row: &[f32], ids: &[u32], position: usize) -> Result<Vec<f64>> {
    let picked = pick_logits(row, ids, position)?;
    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let top = f64::from(top);
    let sum: f64 = row.iter().map(|&v| (f64::from(v) - top).exp()).sum();
    let log_z = top + sum.ln();
    Ok(picked.iter().map(|&v| f64::from(v) - log_z).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wanted_logits_come_in_the_order_of_the_ids() {
        let row = [0.5f32, -1.0, 2.0, 3.5];
        assert_eq!(pick_logits(&row, &[3, 1], 7).unwrap(), vec![3.5, -1.0]);
        assert!(pick_logits(&row, &[], 7).unwrap().is_empty());
    }

    #[test]
    fn log_probs_are_normalised_over_the_whole_row() {
        // Four logits 0, ln 2, ln 3, ln 4 sum to 10 in exp space: the log-probs of ids 0..4 are
        // ln(1/10), ln(2/10), ln(3/10), ln(4/10), and they exponentiate to a total of 1.
        let row: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x.ln()).collect();
        let lp = log_probs_of(&row, &[0, 1, 2, 3], 0).unwrap();
        for (got, want) in lp.iter().zip([0.1f64, 0.2, 0.3, 0.4]) {
            assert!((got - want.ln()).abs() < 1e-6, "{got} vs {}", want.ln());
        }
        assert!((lp.iter().map(|v| v.exp()).sum::<f64>() - 1.0).abs() < 1e-6);
        // only the wanted ids are returned, but the mass of the others still counts
        let two = log_probs_of(&row, &[3, 0], 0).unwrap();
        assert!((two[0] - 0.4f64.ln()).abs() < 1e-6 && (two[1] - 0.1f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn log_probs_refuse_what_logits_refuse() {
        assert!(matches!(
            log_probs_of(&[1.0, f32::NAN], &[0], 3),
            Err(Error::NonFiniteHidden { position: 3, .. })
        ));
        assert!(matches!(
            log_probs_of(&[1.0, 2.0], &[2], 0),
            Err(Error::Inference { .. })
        ));
        // large logits do not overflow: the maximum is subtracted before exponentiating
        let lp = log_probs_of(&[1000.0, 1000.0], &[0], 0).unwrap();
        assert!((lp[0] - 0.5f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn a_value_that_is_not_finite_anywhere_in_the_row_is_refused() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // The wanted ids (0 and 1) are fine: the bad value is elsewhere in the row.
            let mut row = vec![0.0f32; 16];
            row[11] = bad;
            let err = pick_logits(&row, &[0, 1], 4).unwrap_err();
            match err {
                Error::NonFiniteHidden {
                    position, value, ..
                } => {
                    assert_eq!(position, 4);
                    assert!(value.is_nan() == bad.is_nan() && !value.is_finite());
                }
                other => panic!("{other}"),
            }
        }
    }

    #[test]
    fn tokens_with_almost_no_mass_in_the_full_vocabulary_are_not_a_fault() {
        // The wanted tokens are 25 nats below the best one: ~1e-11 of the full softmax. The model
        // was trained on the softmax restricted to them, so this is the normal case.
        let mut row = vec![-10.0f32; 32];
        row[5] = 15.0;
        row[0] = -10.0;
        row[1] = -11.0;
        assert_eq!(pick_logits(&row, &[0, 1], 0).unwrap(), vec![-10.0, -11.0]);
    }

    #[test]
    fn an_id_outside_the_vocabulary_is_an_error_not_a_panic() {
        let err = pick_logits(&[1.0, 2.0], &[0, 2], 0).unwrap_err();
        assert!(matches!(err, Error::Inference { .. }), "{err}");
    }
}
