//! A real decode of the default model (Q8_0), for the build jobs of CI (spec 011, `gpu-build.yml`)
//! and for a person with a GPU. Set `ARCHAI_JEV_TEST_V2_GGUF` to the file (the same variable as
//! `v2_parity.rs`); without it the tests say so and pass. It needs no `testing` feature.
//!
//! * `cpu_plan_answers_with_finite_numbers`: in **any** build, also one with a GPU backend and no
//!   GPU or driver (a CI runner, a virtual machine), the CPU plan loads the model and answers
//!   within a minute. A Metal build that hung for minutes on such a machine is the fault it would
//!   catch.
//! * `the_named_device_agrees_with_the_cpu`: with `ARCHAI_JEV_TEST_DEVICE=cuda:0` (or `vulkan:0`,
//!   the name that `available_devices()` prints) the same decode runs on that device and agrees
//!   with the CPU. This is the check that a Vulkan build runs on Mesa's software GPU (lavapipe).
//!   Without the variable it does nothing.
//!
//! (The tiny checkpoint of `tests/data/qwen35-mini` cannot be used: it has the structure of a model
//! for the converter, and llama.cpp aborts on its rope dimensions.)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use _core::engine::device::{DeviceProbe, DeviceRequest, OffloadPlan};
use _core::engine::llama::{LlamaConfig, LlamaForward, LlamaProbe, cpu_plan};
use _core::engine::{Forward, Output, OutputSpec};

const MAX_CONTEXT: u32 = 512;
/// A minute by default; a software GPU (Mesa lavapipe) is slower and CI says so with
/// `ARCHAI_JEV_TEST_LIMIT_SECS`.
fn limit() -> Duration {
    let secs = std::env::var("ARCHAI_JEV_TEST_LIMIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    Duration::from_secs(secs)
}

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("ARCHAI_JEV_TEST_V2_GGUF").map(PathBuf::from);
    if path.is_none() {
        eprintln!("SKIPPED: set ARCHAI_JEV_TEST_V2_GGUF to the Q8_0 file of the default model");
    }
    path
}

fn load(path: &Path, plan: OffloadPlan) -> LlamaForward {
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get() as i32);
    LlamaForward::load(
        path,
        LlamaConfig {
            threads,
            max_context: MAX_CONTEXT,
            hidden_states: false,
            plan,
        },
    )
    .expect("the model loads")
}

/// The logits of four tokens at the last token of a short prompt.
fn logits(forward: &LlamaForward) -> Vec<f32> {
    let vocab = forward.n_vocab().max(2) as u32;
    let state: Vec<u32> = (0..10).map(|i| (3 * i + 1) % vocab).collect();
    let branch: Vec<u32> = (0..5).map(|i| (7 * i + 2) % vocab).collect();
    let mut session = forward.begin(&state).expect("the state decodes");
    let out = session
        .row(
            &branch,
            &[OutputSpec::Logits {
                position: branch.len() - 1,
                ids: vec![10, 11, 12, 13],
            }],
        )
        .expect("the row decodes");
    match out.into_iter().next() {
        Some(Output::Logits(l)) => l,
        other => panic!("unexpected output: {other:?}"),
    }
}

#[test]
fn cpu_plan_answers_with_finite_numbers() {
    let Some(path) = fixture() else { return };
    let started = Instant::now();
    let forward = load(&path, cpu_plan().unwrap());
    let l = logits(&forward);
    assert_eq!(l.len(), 4);
    assert!(l.iter().all(|x| x.is_finite()), "{l:?}");
    assert!(l.iter().any(|x| *x != 0.0), "all zeros");
    assert_eq!(l, logits(&forward), "repeatable");
    assert!(
        started.elapsed() < limit(),
        "the CPU plan took {:?}, more than {:?}",
        started.elapsed(),
        limit()
    );
}

#[test]
fn the_named_device_agrees_with_the_cpu() {
    let Some(path) = fixture() else { return };
    let Ok(name) = std::env::var("ARCHAI_JEV_TEST_DEVICE") else {
        eprintln!("SKIPPED: set ARCHAI_JEV_TEST_DEVICE (for example cuda:0 or vulkan:0)");
        return;
    };
    let request = DeviceRequest::parse(&name).expect("a valid device name");
    let DeviceRequest::Named { backend, index } = request else {
        panic!("name a GPU device, not {name:?}");
    };
    let devices = LlamaProbe.devices();
    let device = devices
        .iter()
        .find(|d| d.backend == backend && d.index == index)
        .unwrap_or_else(|| panic!("{name} is not among {devices:?}"));
    eprintln!("on {} ({})", device.name(), device.description);

    let started = Instant::now();
    let on_cpu = logits(&load(&path, cpu_plan().unwrap()));
    let forward = load(&path, OffloadPlan::for_device(device));
    let on_device = logits(&forward);
    assert!(on_device.iter().all(|x| x.is_finite()), "{on_device:?}");
    let scale = on_cpu.iter().fold(1.0_f32, |m, x| m.max(x.abs()));
    let gap = on_cpu
        .iter()
        .zip(&on_device)
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("largest gap {gap:e} on values up to {scale}");
    assert!(
        gap <= 5e-2 * scale,
        "{name} disagrees with the CPU by {gap} (values up to {scale})"
    );
    assert_eq!(on_device, logits(&forward), "repeatable");
    assert!(started.elapsed() < limit(), "took {:?}", started.elapsed());
}
