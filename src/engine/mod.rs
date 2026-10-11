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
#[cfg(any(test, feature = "testing"))]
pub mod fake;
pub mod llama;

use std::any::Any;
use std::sync::Arc;

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
}

/// A number read from the model: the answer to one [`OutputSpec`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Output {
    /// `n_embd` values.
    Hidden(Vec<f32>),
    /// One value per requested id.
    Logits(Vec<f32>),
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
