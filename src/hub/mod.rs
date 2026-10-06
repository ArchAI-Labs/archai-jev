//! Where model files live and how they get there: cache layout, download with resume,
//! integrity, offline mode (spec 005, section 9).

pub mod blobs;
pub mod cache;
pub mod config;
pub mod download;
pub mod events;

#[cfg(all(test, feature = "testing"))]
mod tests;
