//! Tests of the pool with real threads. Every test runs under a time limit: a pool that blocks
//! must fail the test, not hang it.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::*;
use crate::control::{Control, NoControl};
use crate::error::{Error, Result};

fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .expect("timed out: the pool may be blocked")
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// Stops after it has been asked `limit` times.
struct StopAfter {
    asked: AtomicUsize,
    limit: usize,
}

impl Control for StopAfter {
    fn checkpoint(&self) -> Result<()> {
        if self.asked.fetch_add(1, Ordering::SeqCst) >= self.limit {
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

/// Run `n` threads that take a slot, hold it for `hold`, and report the most that held one together.
fn contend(pool: &Arc<SlotPool>, n: usize, hold: Duration) -> (usize, Vec<usize>) {
    let now = Arc::new(AtomicUsize::new(0));
    let max = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let barrier = Arc::new(Barrier::new(n));
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let (pool, now, max, seen, barrier) = (
                Arc::clone(pool),
                Arc::clone(&now),
                Arc::clone(&max),
                Arc::clone(&seen),
                Arc::clone(&barrier),
            );
            thread::spawn(move || {
                barrier.wait();
                let slot = pool.acquire(&NoControl).unwrap();
                let inside = now.fetch_add(1, Ordering::SeqCst) + 1;
                max.fetch_max(inside, Ordering::SeqCst);
                seen.lock().unwrap().push(slot.threads());
                thread::sleep(hold);
                now.fetch_sub(1, Ordering::SeqCst);
                drop(slot);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let threads = seen.lock().unwrap().clone();
    (max.load(Ordering::SeqCst), threads)
}

#[test]
fn below_the_threshold_one_request_runs_at_a_time() {
    within(30, || {
        let pool = Arc::new(SlotPool::new(4, 20));
        // three callers: the queue never reaches four, so the pool stays serial
        let (max, threads) = contend(&pool, 3, ms(60));
        assert_eq!(max, 1);
        assert!(threads.iter().all(|&t| t == 20), "{threads:?}");
        let s = pool.stats();
        assert_eq!((s.peak_busy, s.completed, s.parallel_openings), (1, 3, 0));
        assert!(!s.parallel);
    });
}

#[test]
fn k_is_reached_with_a_queue_of_four_and_never_k_plus_one() {
    within(30, || {
        let pool = Arc::new(SlotPool::new(4, 20));
        let (max, threads) = contend(&pool, 8, ms(100));
        assert_eq!(max, 4, "threads seen: {threads:?}");
        // the first request to open the parallel mode sees at least five present: 20 / 4 threads
        assert!(threads.contains(&5), "{threads:?}");
        let s = pool.stats();
        assert_eq!((s.peak_busy, s.completed), (4, 8));
        assert!(s.peak_threads <= 20, "{s:?}");
        assert_eq!(s.threads_in_use, 0);
        assert!(s.parallel_openings >= 1);
        assert!(!s.parallel, "the pool is serial again once it drained");
    });
}

#[test]
fn a_pool_of_one_never_runs_two_requests() {
    within(30, || {
        let pool = Arc::new(SlotPool::new(1, 20));
        let (max, threads) = contend(&pool, 8, ms(20));
        assert_eq!(max, 1);
        assert!(threads.iter().all(|&t| t == 20));
        assert_eq!(pool.stats().parallel_openings, 0);
    });
}

#[test]
fn the_states_of_an_ask_many_count_as_queue() {
    within(30, || {
        let pool = SlotPool::new(4, 20);
        let demand = pool.demand(8);
        let slot = pool.acquire_claiming(&NoControl, &demand).unwrap();
        // one running, seven still to come: the parallel mode is open from the first one
        assert_eq!(slot.threads(), 5);
        assert!(pool.stats().parallel);
        drop(slot);
        drop(demand);
        let s = pool.stats();
        assert!(!s.parallel, "no demand left, nothing running: serial again");
        assert_eq!(s.waiting, 0);
        // a demand of two does not reach the threshold
        let demand = pool.demand(2);
        let slot = pool.acquire_claiming(&NoControl, &demand).unwrap();
        assert_eq!(slot.threads(), 20);
    });
}

#[test]
fn requests_are_served_in_arrival_order() {
    within(30, || {
        let pool = Arc::new(SlotPool::new(1, 20));
        let first = pool.acquire(&NoControl).unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for id in 1..=5usize {
            let (pool, order) = (Arc::clone(&pool), Arc::clone(&order));
            handles.push(thread::spawn(move || {
                let slot = pool.acquire(&NoControl).unwrap();
                order.lock().unwrap().push(id);
                drop(slot);
            }));
            thread::sleep(ms(25)); // they arrive one after the other
        }
        drop(first);
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3, 4, 5]);
    });
}

#[test]
fn a_request_that_gives_up_does_not_hold_back_the_ones_behind() {
    within(30, || {
        let pool = Arc::new(SlotPool::new(1, 20));
        let running = pool.acquire(&NoControl).unwrap();
        // order of arrival: head (waits), quitter (gives up while somebody is still ahead of it), follower
        let head = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || pool.acquire(&NoControl).map(|s| s.threads()))
        };
        thread::sleep(ms(60));
        let flag = Arc::new(Flag(AtomicBool::new(false)));
        let quitter = {
            let (pool, flag) = (Arc::clone(&pool), Arc::clone(&flag));
            thread::spawn(move || pool.acquire(flag.as_ref()).map(|_| ()))
        };
        thread::sleep(ms(60));
        let follower = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || pool.acquire(&NoControl).map(|s| s.threads()))
        };
        thread::sleep(ms(60));
        flag.0.store(true, Ordering::SeqCst);
        assert!(matches!(quitter.join().unwrap(), Err(Error::Cancelled)));
        assert_eq!(
            pool.stats().waiting,
            2,
            "the quitter is not counted any more"
        );
        drop(running);
        assert_eq!(head.join().unwrap().unwrap(), 20);
        assert_eq!(follower.join().unwrap().unwrap(), 20);
        let s = pool.stats();
        assert_eq!((s.cancelled, s.waiting, s.busy), (1, 0, 0));
    });
}

#[test]
fn waiting_looks_at_the_control() {
    within(30, || {
        let pool = Arc::new(SlotPool::new(1, 20));
        let _running = pool.acquire(&NoControl).unwrap();
        let control = StopAfter {
            asked: AtomicUsize::new(0),
            limit: 3,
        };
        let started = Instant::now();
        let verdict = pool.acquire(&control);
        assert!(matches!(verdict, Err(Error::Cancelled)));
        // the first look is before taking a ticket, the next ones are one every 20 ms
        assert!(control.asked.load(Ordering::SeqCst) >= 4);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(pool.stats().cancelled, 1);
    });
}

#[test]
fn a_demand_dropped_early_gives_the_queue_back() {
    within(30, || {
        let pool = SlotPool::new(4, 20);
        let demand = pool.demand(10);
        drop(demand);
        let slot = pool.acquire(&NoControl).unwrap();
        assert_eq!(slot.threads(), 20);
        assert_eq!(pool.stats().waiting, 0);
    });
}

#[test]
fn many_threads_many_requests_no_block_and_k_respected() {
    within(120, || {
        let pool = Arc::new(SlotPool::new(4, 20));
        let now = Arc::new(AtomicUsize::new(0));
        let max = Arc::new(AtomicUsize::new(0));
        let total = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..24)
            .map(|w| {
                let (pool, now, max, total) = (
                    Arc::clone(&pool),
                    Arc::clone(&now),
                    Arc::clone(&max),
                    Arc::clone(&total),
                );
                thread::spawn(move || {
                    for i in 0..40usize {
                        let slot = pool.acquire(&NoControl).unwrap();
                        let inside = now.fetch_add(1, Ordering::SeqCst) + 1;
                        max.fetch_max(inside, Ordering::SeqCst);
                        assert!(slot.threads() >= 1 && slot.threads() <= 20);
                        thread::sleep(Duration::from_micros(200 + ((w * 7 + i) % 5) as u64 * 100));
                        now.fetch_sub(1, Ordering::SeqCst);
                        total.fetch_add(1, Ordering::SeqCst);
                        drop(slot);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(max.load(Ordering::SeqCst) <= 4);
        assert_eq!(total.load(Ordering::SeqCst), 24 * 40);
        let s = pool.stats();
        assert_eq!((s.completed, s.busy, s.waiting), (24 * 40, 0, 0));
        assert!(s.peak_busy <= 4 && s.parallel_openings >= 1);
        assert!(s.peak_threads <= 20, "{s:?}");
        assert!(!s.parallel);
    });
}

#[test]
fn contexts_are_counted() {
    let pool = SlotPool::new(2, 8);
    let slot = pool.acquire(&NoControl).unwrap();
    slot.note_context_created();
    slot.note_context_created();
    drop(slot);
    let s = pool.stats();
    assert_eq!(
        (s.contexts_created, s.max_concurrency, s.thread_budget),
        (2, 2, 8)
    );
}
