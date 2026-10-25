//! A pool of contexts must not change the numbers (spec 008, D46). Level T2: needs the 1.65 GB GGUF
//! of the default model (`ARCHAI_JEV_TEST_DEFAULT_GGUF`) and its `tokenizer.json`
//! (`ARCHAI_JEV_TEST_DEFAULT_TOKENIZER`):
//! `cargo test --release --features testing --test pool_parity -- --ignored --nocapture`.
//!
//! The 48 golden requests of the default model are answered one at a time by a pool of one (the
//! reference, the behaviour of 0.2.0) and then by 16 threads at once over pools of 2, 4 and 8 slots.
//! The raw logits must be **identical bit for bit**: the slot that ran a request, the other
//! requests and the number of threads of its context change nothing (spike S9: 84 of 84 identical
//! from 1 to 20 threads).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::Arc;
use std::thread;

use _core::Calibration;
use _core::engine::llama::{LlamaConfig, LlamaForward, cpu_plan};
use _core::heads::LettersHead;
use _core::model_scorer::{ModelScorer, Run};
use _core::models::manifest::Target;
use _core::models::testing::assets;
use _core::models::testing::golden;
use _core::prompt::chatml::ChatMlLetters;
use _core::prompt::tasks::{self, default_tasks};
use _core::prompt::{PromptTokenizer, Roles};
use _core::schema::{Questions, State};

fn target(label: &str, id: u32) -> Target {
    Target {
        label: label.to_string(),
        id,
    }
}

fn scorer(forward: Arc<LlamaForward>, tokenizer: &std::path::Path, cache: u64) -> Arc<ModelScorer> {
    Arc::new(
        ModelScorer::new(
            "default",
            Box::new(ChatMlLetters::new(Some(
                tasks::parse(&default_tasks(), 4).unwrap(),
            ))),
            PromptTokenizer::from_file(
                tokenizer,
                Roles::new([("im_start", 151_644u32), ("im_end", 151_645)]),
            )
            .unwrap(),
            forward,
            Box::new(LettersHead::new(
                &[
                    target("A", 32),
                    target("B", 33),
                    target("C", 34),
                    target("D", 35),
                ],
                Some(&(target("FALSE", 30_351), target("TRUE", 20_611))),
            )),
            Calibration::new(2.09, true).unwrap(),
            512,
            ["choice", "noul"]
                .iter()
                .map(|t| (*t).to_string())
                .collect(),
        )
        .with_prefix_cache(cache),
    )
}

fn load(gguf: &std::path::Path, k: usize) -> Arc<LlamaForward> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
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

fn requests() -> Vec<(String, State, Questions)> {
    golden::load_default()
        .into_iter()
        .map(|case| {
            let (st, qs) = golden::request_to_domain(&case.request).unwrap();
            (case.id, st, qs)
        })
        .collect()
}

#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF and the default tokenizer"]
fn the_golden_is_bit_identical_for_every_pool_size_and_every_caller() {
    let (Some(tokenizer), Some(gguf)) = (assets::default_tokenizer(), assets::default_gguf())
    else {
        return;
    };
    let cases = Arc::new(requests());
    assert_eq!(cases.len(), 48);

    for cache in [0u64, 128 << 20] {
        // The reference for this setting of the memory of prefixes: a pool of one, one request at a
        // time (0.2.0). With the memory on the numbers differ from the ones with it off within the
        // tolerances of spec 007, so each setting is compared with itself.
        let reference_forward = load(&gguf, 1);
        let reference_scorer = scorer(reference_forward.clone(), &tokenizer, cache);
        let reference: Vec<Run> = cases
            .iter()
            .map(|(_, st, qs)| reference_scorer.run(st, qs).unwrap())
            .collect();
        let stats = reference_forward.pool_stats();
        assert_eq!((stats.peak_busy, stats.parallel_openings), (1, 0));
        drop(reference_scorer);
        drop(reference_forward);

        // POOL_PARITY_KS=1,4 limits the pool sizes (to look at a case without waiting for all)
        let ks: Vec<usize> = std::env::var("POOL_PARITY_KS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 2, 4, 8]);
        for k in ks {
            let forward = load(&gguf, k);
            let s = scorer(forward.clone(), &tokenizer, cache);
            let started = std::time::Instant::now();
            let handles: Vec<_> = (0..16)
                .map(|w| {
                    let (s, cases) = (Arc::clone(&s), Arc::clone(&cases));
                    thread::spawn(move || {
                        (w..cases.len())
                            .step_by(16)
                            .map(|i| {
                                let (_, st, qs) = &cases[i];
                                (i, s.run(st, qs).unwrap())
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut answered = 0;
            for h in handles {
                for (i, run) in h.join().unwrap() {
                    assert_eq!(
                        run.questions, reference[i].questions,
                        "K={k} cache={cache}: {} is not bit-identical",
                        cases[i].0
                    );
                    assert_eq!(run.input_ids, reference[i].input_ids);
                    answered += 1;
                }
            }
            assert_eq!(answered, 48);
            let stats = forward.pool_stats();
            eprintln!(
                "K={k} cache={}MiB: 48 requests in {:.1}s by 16 threads; peak_busy {}, parallel opened {} times; waited {} ms in all, slots held {} ms in all",
                cache >> 20,
                started.elapsed().as_secs_f64(),
                stats.peak_busy,
                stats.parallel_openings,
                stats.waited_ms,
                stats.held_ms
            );
            assert!(stats.peak_busy <= k, "K={k}: {stats:?}");
            // never more threads together than the budget: ggml's spinning thread pool stalls everybody
            // when contexts share cores (35 threads on 20 cores made 60-token decodes last 8 to 30 s)
            assert!(
                stats.peak_threads <= stats.thread_budget,
                "K={k}: {stats:?}"
            );
            assert_eq!((stats.busy, stats.waiting), (0, 0));
            if k == 1 {
                assert_eq!(stats.parallel_openings, 0);
            } else {
                assert!(
                    stats.peak_busy >= 2 && stats.parallel_openings >= 1,
                    "K={k}: 16 callers must open the parallel mode: {stats:?}"
                );
            }
            // one context per request, never more than K alive together
            assert_eq!(stats.contexts_created, 48);
        }
    }
}

#[test]
#[ignore = "needs the 1.65 GB GGUF: set ARCHAI_JEV_TEST_DEFAULT_GGUF and the default tokenizer"]
fn a_lone_request_keeps_every_thread_and_never_opens_the_pool() {
    let (Some(tokenizer), Some(gguf)) = (assets::default_tokenizer(), assets::default_gguf())
    else {
        return;
    };
    let forward = load(&gguf, 4);
    let s = scorer(forward.clone(), &tokenizer, 0);
    let cases = requests();
    for (_, st, qs) in cases.iter().take(6) {
        s.run(st, qs).unwrap();
    }
    let stats = forward.pool_stats();
    assert_eq!(
        (stats.peak_busy, stats.parallel_openings, stats.completed),
        (1, 0, 6)
    );
    assert!(!stats.parallel);
}
