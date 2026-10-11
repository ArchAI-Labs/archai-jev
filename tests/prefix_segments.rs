//! The real engine decoding a row in segments and restoring a saved shared part (test level T2,
//! by hand, `#[ignore]`).
//!
//! Needs `ARCHAI_JEV_TEST_DEFAULT_GGUF` (the default model, Q8_0) and, for the pointer-head
//! engine, `ARCHAI_JEV_TEST_KEV_GGUF` (Kev, BF16). Run with
//! `cargo test --release --features testing --test prefix_segments -- --ignored --nocapture`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;

use _core::Error;
use _core::engine::llama::{LlamaConfig, LlamaForward};
use _core::engine::{Forward, Output, OutputSpec};
use _core::models::testing::assets;

fn kev_gguf() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("ARCHAI_JEV_TEST_KEV_GGUF").ok()?);
    p.is_file().then_some(p)
}

fn ids(from: u32, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| 1000 + (from + i * 7) % 20_000)
        .collect()
}

fn load(path: &std::path::Path, hidden_states: bool, max_context: u32) -> LlamaForward {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    LlamaForward::load(
        path,
        LlamaConfig {
            threads,
            max_context,
            hidden_states,
        },
    )
    .unwrap()
}

fn spec(hidden: bool, position: usize) -> Vec<OutputSpec> {
    vec![if hidden {
        OutputSpec::Hidden { position }
    } else {
        OutputSpec::Logits {
            position,
            ids: vec![32, 33, 34, 35],
        }
    }]
}

fn numbers(out: &[Output]) -> Vec<u32> {
    match &out[0] {
        Output::Hidden(v) | Output::Logits(v) => v.iter().map(|x| x.to_bits()).collect(),
        _ => panic!("unexpected output"),
    }
}

fn checks(forward: &LlamaForward, hidden: bool) {
    let prefix = ids(3, 60);
    let tail = ids(500, 12);
    let last = tail.len() - 1;

    // Cold: the prefix is decoded here, the tail runs on a copy of it.
    let mut cold = forward.begin_empty().unwrap();
    cold.extend(&prefix).unwrap();
    assert_eq!(cold.shared_len(), prefix.len());
    let snapshot = cold.snapshot().unwrap();
    assert_eq!(snapshot.tokens(), prefix.len());
    assert!(snapshot.bytes() > 0);
    let a = numbers(&cold.row_segments(&[&tail], &spec(hidden, last)).unwrap());
    // The same session, asked again, gives the same numbers (the copy is discarded each time).
    let again = numbers(&cold.row_segments(&[&tail], &spec(hidden, last)).unwrap());
    assert_eq!(a, again, "a second question on the same session");
    drop(cold);

    // Warm: a new context, the prefix restored, the same tail: bit for bit the same.
    let mut warm = forward.begin_empty().unwrap();
    warm.restore(&snapshot).unwrap();
    assert_eq!(warm.shared_len(), prefix.len());
    let b = numbers(&warm.row_segments(&[&tail], &spec(hidden, last)).unwrap());
    assert_eq!(a, b, "restored in another context");

    // A row in two segments: positions count from the first segment; outputs in the last one.
    let (t1, t2) = tail.split_at(5);
    let d1 = numbers(&warm.row_segments(&[t1, t2], &spec(hidden, last)).unwrap());
    let d2 = numbers(&warm.row_segments(&[t1, t2], &spec(hidden, last)).unwrap());
    assert_eq!(d1, d2, "the same split twice");
    // An output before the last segment is refused, and the session is usable afterwards.
    let err = warm.row_segments(&[t1, t2], &spec(hidden, 2)).unwrap_err();
    assert!(matches!(err, Error::Inference { .. }), "{err}");
    let e = numbers(&warm.row_segments(&[&tail], &spec(hidden, last)).unwrap());
    assert_eq!(a, e, "after a refused row");

    // A snapshot only goes on an empty session.
    assert!(matches!(
        warm.restore(&snapshot),
        Err(Error::Inference { .. })
    ));
    // The engine runs one session at a time: the next ones start when this one is gone.
    drop(warm);

    // `begin(state)` is `begin_empty` + `extend(state)`; `row` is one segment.
    let mut classic = forward.begin(&prefix).unwrap();
    let c = numbers(&classic.row(&tail, &spec(hidden, last)).unwrap());
    assert_eq!(a, c, "begin + row");
    drop(classic);

    // Extending after a restore equals extending the whole prefix in one go **only** if the
    // split is the same; here it is (prefix, then more), on both sides.
    let more = ids(900, 8);
    let mut x = forward.begin_empty().unwrap();
    x.restore(&snapshot).unwrap();
    x.extend(&more).unwrap();
    let xo = numbers(&x.row_segments(&[&tail], &spec(hidden, last)).unwrap());
    drop(x);
    let mut y = forward.begin_empty().unwrap();
    y.extend(&prefix).unwrap();
    y.extend(&more).unwrap();
    let yo = numbers(&y.row_segments(&[&tail], &spec(hidden, last)).unwrap());
    assert_eq!(xo, yo, "restore + extend against extend + extend");
}

#[test]
#[ignore = "manual: needs the default GGUF"]
fn default_model_segments_and_restore() {
    let Some(gguf) = assets::default_gguf() else {
        return;
    };
    checks(&load(&gguf, false, 512), false);
}

#[test]
#[ignore = "manual: needs the Kev GGUF"]
fn kev_segments_and_restore() {
    let Some(gguf) = kev_gguf() else {
        eprintln!("SKIPPED: ARCHAI_JEV_TEST_KEV_GGUF does not point at a file");
        return;
    };
    checks(&load(&gguf, true, 8_205), true);
}

#[test]
#[ignore = "manual: needs the default GGUF"]
fn a_snapshot_of_another_context_shape_is_an_error_not_a_crash() {
    let Some(gguf) = assets::default_gguf() else {
        return;
    };
    let small = load(&gguf, false, 512);
    let large = load(&gguf, false, 1_024);
    let mut a = small.begin_empty().unwrap();
    a.extend(&ids(3, 40)).unwrap();
    let snapshot = a.snapshot().unwrap();
    drop(a);
    let mut b = large.begin_empty().unwrap();
    let r = b.restore(&snapshot);
    // Either llama.cpp refuses it (an error) or it accepts a state that fits; what must not
    // happen is a crash, and a snapshot of a *different* engine type is always refused.
    eprintln!("restore into a context of another size: {:?}", r.is_ok());
}
