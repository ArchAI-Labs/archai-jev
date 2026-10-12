//! A pool of slots for a model (spec 008): how many requests run together, and with how many
//! threads each. Pure Rust: nothing here runs a model.

pub mod policy;
pub mod slots;

#[cfg(test)]
mod tests;

pub use policy::{Admission, Mode, Policy, QUEUE_THRESHOLD};
pub use slots::{Demand, PoolStats, Slot, SlotPool};
