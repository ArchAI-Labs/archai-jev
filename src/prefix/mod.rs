//! Reuse of the prefix of a prompt (spec 007): what is decoded once per request, and what is
//! kept between requests. Pure Rust: nothing here runs a model.

pub mod admission;
pub mod key;
pub mod plan;
pub mod store;

pub use admission::{MIN_CACHED_TOKENS, Seen, admit};
pub use key::{Fingerprint, FingerprintParts, KeyHasher, PrefixKey};
pub use plan::{RequestPlan, RowPlan};
pub use store::{Counters, Entry, PrefixStore};
