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

use crate::error::Result;

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

/// One request being answered: the state is already decoded and kept.
pub trait Session {
    /// Run one question: decode `branch` after the state, on a copy that is discarded, and return
    /// one [`Output`] per entry of `outputs`, in order. Values that are not finite are an error
    /// ([`crate::Error::NonFiniteHidden`], with an empty question name: the caller names it).
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine fails; the session stays usable.
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>>;
}

/// A loaded model that can answer requests.
pub trait Forward: Send + Sync {
    /// Start a request: decode `state` once (positions `0..state.len()`) and keep it. The model
    /// is not thread-safe, so the returned session holds it: concurrent requests wait their turn.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the state cannot be decoded.
    fn begin<'a>(&'a self, state: &[u32]) -> Result<Box<dyn Session + 'a>>;
}
