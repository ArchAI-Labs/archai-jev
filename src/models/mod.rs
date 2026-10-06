//! Model loading: manifests, registry, validation, self-check, download and cache (spec 005).

pub mod backend;
pub mod calibration_gate;
pub mod convert;
pub mod failures;
pub mod families;
pub mod files;
pub mod gguf;
pub mod hash;
pub mod head;
pub mod incompat;
pub mod llama_backend;
pub mod load;
pub mod manifest;
pub mod record;
pub mod registry;
pub mod resolve;
pub mod tokenizer_check;
pub mod tolerance;
pub mod validate;
pub mod vectors;
pub mod verify;

#[cfg(feature = "testing")]
pub mod testing;

#[cfg(all(test, feature = "testing"))]
mod tests;
