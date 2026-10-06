//! Test support (feature `testing`, off in release wheels): a builder of tiny fake
//! checkpoints, a GGUF writer, a fake backend and fake model, used by the tests of spec 005.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

pub mod assets;
pub mod builder;
pub mod fake;
pub mod fake_converter;
pub mod gguf_writer;
pub mod golden;
pub mod jsonedit;
pub mod server;

pub use builder::{Builder, Fixture};
pub use fake::{FakeBackend, FakeModel};
pub use fake_converter::FakeConverter;
