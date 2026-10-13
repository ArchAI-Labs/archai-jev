//! Tests of the model-loading pipeline with fake checkpoints (feature `testing`).
//! Run them with `cargo test --features testing`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod harness;
mod hub_flow;
mod mini;
mod pipeline;
mod reject;
mod robustness;
mod source;
mod variant_choice;

pub(crate) use harness::Harness;
