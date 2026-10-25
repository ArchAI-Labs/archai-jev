//! The engine alone (no prompt building, no tokenizer, no head) under 16 callers, with the golden ids of
//! the default model (spec 008). It guards against the silent failure of task T6: contexts that share the
//! cores stall ggml's spinning thread pool, and a pool of 4 took 2.5 times as long as a pool of 1 with no
//! error anywhere. The pool may be a little faster or a little slower than K = 1, never much slower.
//! Level T2 (needs the GGUF):
//! `cargo test --release --features testing --test pool_engine_direct -- --ignored --nocapture --test-threads=1`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use _core::engine::llama::{LlamaConfig, LlamaForward, cpu_plan};
use _core::engine::{Forward, OutputSpec};
use _core::models::testing::{assets, golden};

fn load(gguf: &std::path::Path, k: usize) -> Arc<LlamaForward> {
    // POOL_DIRECT_THREADS=16 sets the thread budget (default: all logical cores)
    let threads = std::env::var("POOL_DIRECT_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() as i32));
    Arc::new(
        LlamaForward::load(
            gguf,
            LlamaConfig {
                threads,
                max_context: 512,
                hidden_states: false,
                plan: cpu_plan().unwrap(),
            },
        )
        .unwrap()
        .with_max_concurrency(k),
    )
}

#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF"]
fn the_engine_alone_with_sixteen_callers() {
    let Some(gguf) = assets::default_gguf() else {
        return;
    };
    // POOL_DIRECT_TASK=safety keeps only the short requests of that task (default: the whole mix)
    let only = std::env::var("POOL_DIRECT_TASK").ok();
    let rows: Arc<Vec<Vec<u32>>> = Arc::new(
        golden::load_default()
            .into_iter()
            .filter(|c| only.as_deref().is_none_or(|t| c.task == t))
            .map(|c| c.ids)
            .collect(),
    );
    let ks: Vec<usize> = std::env::var("POOL_DIRECT_KS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 4]);
    let mut k1_seconds: Option<f64> = None;
    for k in ks {
        let forward = load(&gguf, k);
        // warm-up, like the probe: touch the weights once
        {
            let mut s = forward.begin_empty().unwrap();
            let ids = &rows[0];
            s.row(
                ids,
                &[OutputSpec::Logits {
                    position: ids.len() - 1,
                    ids: vec![32, 33, 34, 35],
                }],
            )
            .unwrap();
        }
        let started = Instant::now();
        let callers: usize = std::env::var("POOL_DIRECT_CALLERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let handles: Vec<_> = (0..callers)
            .map(|w| {
                let (forward, rows) = (Arc::clone(&forward), Arc::clone(&rows));
                thread::spawn(move || {
                    for i in 0..(48 / callers) {
                        let ids = &rows[(w + i) % rows.len()];
                        let mut s = forward.begin_empty().unwrap();
                        s.row(
                            ids,
                            &[OutputSpec::Logits {
                                position: ids.len() - 1,
                                ids: vec![32, 33, 34, 35],
                            }],
                        )
                        .unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let stats = forward.pool_stats();
        let seconds = started.elapsed().as_secs_f64();
        if k == 1 {
            k1_seconds = Some(seconds);
        } else if let Some(base) = k1_seconds {
            assert!(
                seconds <= base * 1.4,
                "a pool of {k} took {seconds:.1}s, a pool of 1 took {base:.1}s: the contexts are stalling each other"
            );
        }
        assert!(stats.peak_threads <= stats.thread_budget, "{stats:?}");
        eprintln!(
            "ENGINE ONLY task={} threads={} K={k}: 48 requests in {:.1}s; peak_busy {}; slots held {} ms in all",
            only.as_deref().unwrap_or("mix"),
            forward.pool_stats().thread_budget,
            started.elapsed().as_secs_f64(),
            stats.peak_busy,
            stats.held_ms
        );
    }
}
