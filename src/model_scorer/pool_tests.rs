//! The scorer over a pool of slots (spec 008): control between the rows, results that do not
//! depend on the slot, and a pool that comes back clean after a failure or a cancellation.
//!
//! The fake can be made **thread sensitive** (its numbers depend on the threads of the slot), so
//! "the result does not depend on the slot" is a claim these tests can fail, and one of them shows
//! that it does.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use super::prefix_tests::{SOURCE, hypotheses, letters, request, state, tokenizer};
use super::*;
use crate::engine::fake::{FakeForward, Faults};
use crate::prompt::chatml::ChatMlLetters;
use crate::prompt::tasks::{self, default_tasks};

const BIG: u64 = 64 << 20;

fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .expect("timed out: the scorer or the pool may be blocked")
}

/// A scorer over a fake with a pool of `k` slots and `budget` threads. `limit` > 0 takes the
/// segmented path with the memory of prefixes.
fn pooled(
    k: usize,
    budget: usize,
    thread_sensitive: bool,
    limit: u64,
) -> (Arc<ModelScorer>, Arc<FakeForward>) {
    let mut fake = FakeForward::new(4)
        .partition_sensitive()
        .with_pool(k, budget);
    if thread_sensitive {
        fake = fake.thread_sensitive();
    }
    let fake = Arc::new(fake);
    let s = ModelScorer::new(
        "default",
        Box::new(ChatMlLetters::new(Some(
            tasks::parse(&default_tasks(), 4).unwrap(),
        ))),
        tokenizer(),
        fake.clone(),
        letters(),
        Calibration::new(1.0, true).unwrap(),
        100_000,
        ["choice", "noul"]
            .iter()
            .map(|t| (*t).to_string())
            .collect(),
    )
    .with_prefix_cache(limit);
    (Arc::new(s), fake)
}

fn latency(fake: &FakeForward, ms: u64) {
    fake.set_faults(Faults {
        latency: Duration::from_millis(ms),
        ..Faults::default()
    });
}

/// Says stop the `n`-th time it is asked.
struct StopAtCall {
    asked: AtomicUsize,
    n: usize,
}

impl Control for StopAtCall {
    fn checkpoint(&self) -> Result<()> {
        if self.asked.fetch_add(1, Ordering::SeqCst) + 1 >= self.n {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

struct Flag(AtomicBool);

impl Control for Flag {
    fn checkpoint(&self) -> Result<()> {
        if self.0.load(Ordering::SeqCst) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn no_control_means_the_same_run() {
    let (s, _) = pooled(1, 8, false, 0);
    let st = state(SOURCE);
    let qs = request(&hypotheses(4));
    assert_eq!(
        s.run(&st, &qs).unwrap(),
        s.run_with(&st, &qs, &NoControl).unwrap()
    );
}

#[test]
fn a_control_that_stops_at_row_k_starts_no_later_row_and_returns_nothing() {
    for limit in [0, BIG] {
        let st = state(SOURCE);
        let qs = request(&hypotheses(6));
        let (reference, _) = pooled(1, 8, false, limit);
        let expected = reference.run(&st, &qs).unwrap();
        // How many looks does a whole cold request make? One before the wait for a slot, one per
        // segment of the shared part (segmented path only), one per row.
        let counting = StopAtCall {
            asked: AtomicUsize::new(0),
            n: usize::MAX,
        };
        let (fresh, _) = pooled(1, 8, false, limit);
        fresh.run_with(&st, &qs, &counting).unwrap();
        let shared_looks = counting.asked.load(Ordering::SeqCst) - 1 - 6;
        assert_eq!(
            shared_looks > 0,
            limit > 0,
            "only the segmented path has them"
        );
        for k in [0usize, 1, 3, 5] {
            let (s, fake) = pooled(1, 8, false, limit);
            let control = StopAtCall {
                asked: AtomicUsize::new(0),
                n: 1 + shared_looks + k + 1,
            };
            let verdict = s.run_with(&st, &qs, &control);
            assert!(
                matches!(verdict, Err(Error::Cancelled)),
                "limit {limit} k {k}: {verdict:?}"
            );
            assert_eq!(fake.stats().rows, k, "limit {limit}: rows started");
            // the slot is free and the next request is what it always was
            let pool = fake.pool_stats();
            assert_eq!((pool.busy, pool.waiting), (0, 0));
            assert_eq!(s.run(&st, &qs).unwrap(), expected, "limit {limit} k {k}");
        }
    }
}

#[test]
fn a_row_that_has_started_is_not_interrupted() {
    for limit in [0, BIG] {
        let (s, fake) = pooled(1, 8, false, limit);
        latency(&fake, 150);
        let flag = Arc::new(Flag(AtomicBool::new(false)));
        let started = Instant::now();
        let stopper = {
            let flag = Arc::clone(&flag);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(60)); // inside the first row
                flag.0.store(true, Ordering::SeqCst);
            })
        };
        let verdict = s.run_with(&state(SOURCE), &request(&hypotheses(5)), flag.as_ref());
        stopper.join().unwrap();
        assert!(matches!(verdict, Err(Error::Cancelled)));
        // the first row ran to its end (150 ms), the second never started
        assert_eq!(fake.stats().rows, 1, "limit {limit}");
        assert!(started.elapsed() >= Duration::from_millis(150));
    }
}

#[test]
fn a_request_cancelled_while_queued_never_runs() {
    within(30, || {
        let (s, fake) = pooled(1, 8, false, 0);
        latency(&fake, 120);
        let st = state(SOURCE);
        let qs = request(&hypotheses(3));
        let busy = {
            let (s, st, qs) = (Arc::clone(&s), st.clone(), qs.clone());
            thread::spawn(move || s.run(&st, &qs).unwrap())
        };
        thread::sleep(Duration::from_millis(40)); // it holds the only slot
        let flag = Arc::new(Flag(AtomicBool::new(true)));
        let verdict = s.run_with(&st, &qs, flag.as_ref());
        assert!(matches!(verdict, Err(Error::Cancelled)));
        busy.join().unwrap();
        // only the three rows of the first request ran
        assert_eq!(fake.stats().rows, 3);
        assert_eq!(fake.stats().sessions, 1);
    });
}

#[test]
fn a_failure_in_one_request_leaves_the_pool_clean() {
    within(30, || {
        let (s, fake) = pooled(2, 8, false, BIG);
        let st = state(SOURCE);
        let qs = request(&hypotheses(4));
        let expected = s.run(&st, &qs).unwrap();
        s.clear_cache();
        fake.set_faults(Faults {
            fail_row: Some(fake.stats().rows + 1),
            ..Faults::default()
        });
        assert!(matches!(s.run(&st, &qs), Err(Error::Inference { .. })));
        let pool = fake.pool_stats();
        assert_eq!((pool.busy, pool.waiting), (0, 0), "the slot came back");
        fake.set_faults(Faults::default());
        s.clear_cache();
        assert_eq!(s.run(&st, &qs).unwrap(), expected);
    });
}

#[test]
fn concurrent_requests_give_the_sequential_results_bit_for_bit() {
    within(120, || {
        // The numbers do not depend on the slot or on the threads (the real engine, S9: 84/84)...
        let (s, fake) = pooled(4, 20, false, BIG);
        let cases: Vec<(State, Questions)> = (0..6)
            .map(|i| {
                (
                    state(&format!("{SOURCE} Variant {i}.")),
                    request(&hypotheses(2 + i % 3)),
                )
            })
            .collect();
        let expected: Vec<_> = cases
            .iter()
            .map(|(st, qs)| s.run(st, qs).unwrap())
            .collect();
        latency(&fake, 3);
        let handles: Vec<_> = (0..16)
            .map(|w| {
                let (s, cases, expected) = (Arc::clone(&s), cases.clone(), expected.clone());
                thread::spawn(move || {
                    for i in 0..20 {
                        let (st, qs) = cases.get((w + i) % cases.len()).unwrap();
                        let want = expected.get((w + i) % cases.len()).unwrap();
                        assert_eq!(&s.run(st, qs).unwrap(), want, "worker {w} call {i}");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let stats = fake.stats();
        assert!(stats.max_concurrent <= 4);
        let pool = fake.pool_stats();
        assert!(
            pool.peak_busy <= 4 && pool.parallel_openings >= 1,
            "{pool:?}"
        );
        assert_eq!((pool.busy, pool.waiting), (0, 0));
        // no oversubscription, ever: 35 threads on 20 cores stall ggml's spinning thread pool
        assert!(pool.peak_threads <= 20, "{pool:?}");
        assert_eq!(pool.threads_in_use, 0);
    });
}

#[test]
fn the_thread_sensitive_fake_shows_the_test_can_fail() {
    within(120, || {
        // ...but a fake whose numbers do depend on the threads of the slot gives other numbers when
        // the pool is open (fewer threads per request), so the comparison above is not vacuous.
        let (s, fake) = pooled(4, 20, true, BIG);
        let st = state(SOURCE);
        let qs = request(&hypotheses(3));
        let alone = s.run(&st, &qs).unwrap(); // serial mode: all 20 threads
        latency(&fake, 30);
        let results: Vec<_> = (0..8)
            .map(|_| {
                let (s, st, qs) = (Arc::clone(&s), st.clone(), qs.clone());
                thread::spawn(move || s.run(&st, &qs).unwrap())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        assert!(
            results.iter().any(|r| r != &alone),
            "no request ran with fewer threads"
        );
    });
}
