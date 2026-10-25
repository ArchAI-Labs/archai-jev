//! Cost of the check on the whole row of logits (spec 006c, section 5), measured by hand:
//! `cargo test --release --features testing --test logit_check_cost -- --ignored --nocapture`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::hint::black_box;
use std::time::Instant;

use _core::engine::pick_logits;

const VOCAB: usize = 151_936;

/// A row of plausible logits (a spread of values in [-20, 25], deterministic).
fn row() -> Vec<f32> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..VOCAB)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 45.0 - 20.0
        })
        .collect()
}

#[test]
#[ignore = "manual benchmark: run in release mode on a quiet machine"]
fn the_check_of_the_whole_row_costs_microseconds() {
    let row = row();
    let ids = [32u32, 33, 34, 35];
    let mut micros: Vec<f64> = Vec::new();
    for i in 0..2_050 {
        let started = Instant::now();
        let picked = pick_logits(black_box(&row), black_box(&ids), 0).unwrap();
        let elapsed = started.elapsed().as_secs_f64() * 1e6;
        black_box(picked);
        if i >= 50 {
            micros.push(elapsed);
        }
    }
    micros.sort_by(f64::total_cmp);
    let q = |p: f64| micros[((micros.len() - 1) as f64 * p) as usize];
    eprintln!(
        "pick_logits over {VOCAB} values: p50 {:.1} us, p95 {:.1} us, max {:.1} us (n = {})",
        q(0.5),
        q(0.95),
        micros[micros.len() - 1],
        micros.len()
    );
    // The same loop without the finiteness scan, to isolate what the check adds.
    let mut bare: Vec<f64> = Vec::new();
    for i in 0..2_050 {
        let started = Instant::now();
        let picked: Vec<f32> = ids.iter().map(|&id| black_box(&row)[id as usize]).collect();
        let elapsed = started.elapsed().as_secs_f64() * 1e6;
        black_box(picked);
        if i >= 50 {
            bare.push(elapsed);
        }
    }
    bare.sort_by(f64::total_cmp);
    eprintln!("only picking the 4 ids: p50 {:.3} us", bare[bare.len() / 2]);
}
