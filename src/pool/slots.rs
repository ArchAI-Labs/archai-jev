//! The slots of the pool: who runs now, who waits, in arrival order.
//!
//! A slot is a **permit** to run one request; the llama.cpp context is built inside the request
//! (a context borrows the model, so keeping contexts next to it is a self-referential type that
//! safe Rust does not allow: plan 008, 3.1). Waiting is in arrival order (tickets), and a request
//! that gives up is skipped.
//!
//! **No lock of the pool is ever held while the [`Control`] runs**: the Python control takes the
//! GIL, and a thread that holds the GIL and asks the pool for its numbers must not meet a thread
//! that holds the pool and waits for the GIL.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use super::policy::{Mode, Policy};
use crate::control::Control;
use crate::error::Result;

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// How often a waiting request looks at its [`Control`].
const CHECK_EVERY: Duration = Duration::from_millis(20);

#[derive(Default)]
struct Inner {
    next_ticket: u64,
    serving: u64,
    /// Tickets of requests that gave up while somebody else was ahead of them.
    abandoned: HashSet<u64>,
    busy: usize,
    mode: Mode,
    /// States of `ask_many` calls that have not asked for a slot yet.
    extra: usize,
    peak_busy: usize,
    completed: u64,
    cancelled: u64,
    parallel_openings: u64,
    /// Total time requests waited for a slot, and total time slots were held, in microseconds.
    waited_us: u64,
    held_us: u64,
    /// Threads of the requests that run now, and the most there ever were together.
    threads_in_use: usize,
    peak_threads: usize,
}

impl Inner {
    /// Requests waiting for a slot (the head included).
    fn waiting(&self) -> usize {
        usize::try_from(self.next_ticket.saturating_sub(self.serving))
            .unwrap_or(usize::MAX)
            .saturating_sub(self.abandoned.len())
    }

    /// What the policy calls the queue: waiting requests plus states not yet started.
    fn queued(&self) -> usize {
        self.waiting().saturating_add(self.extra)
    }

    fn skip_abandoned(&mut self) {
        while self.abandoned.remove(&self.serving) {
            self.serving += 1;
        }
    }

    fn abandon(&mut self, ticket: u64) {
        if ticket == self.serving {
            self.serving += 1;
            self.skip_abandoned();
        } else {
            self.abandoned.insert(ticket);
        }
    }
}

/// What the pool has done and is doing now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// The most requests that run together.
    pub max_concurrency: usize,
    /// Threads of the whole model.
    pub thread_budget: usize,
    /// Whether the parallel mode is open now.
    pub parallel: bool,
    /// How many times the parallel mode has opened.
    pub parallel_openings: u64,
    /// Contexts built so far (the engine reports them through [`Slot::note_context_created`]).
    pub contexts_created: u64,
    /// Requests running now.
    pub busy: usize,
    /// Requests waiting for a slot now.
    pub waiting: usize,
    /// The most requests that ran together.
    pub peak_busy: usize,
    /// Requests that finished (well or badly).
    pub completed: u64,
    /// Requests that gave up while waiting.
    pub cancelled: u64,
    /// Total time requests waited for a slot, in milliseconds.
    pub waited_ms: u64,
    /// Total time slots were held (the context, the decode, the reading), in milliseconds.
    pub held_ms: u64,
    /// Threads of the requests that run now.
    pub threads_in_use: usize,
    /// The most threads that ran together: never above `thread_budget` unless a single request
    /// has more than the budget.
    pub peak_threads: usize,
}

/// The permits of one model, handed out by [`Policy`].
pub struct SlotPool {
    policy: Policy,
    inner: Mutex<Inner>,
    cv: Condvar,
    contexts_created: AtomicU64,
}

/// A permit to run one request. Dropping it gives the slot back.
pub struct Slot<'a> {
    pool: &'a SlotPool,
    threads: usize,
    since: Instant,
}

impl Slot<'_> {
    /// Threads for the context of this request.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The engine tells the pool it built a context for this request.
    pub fn note_context_created(&self) {
        self.pool.contexts_created.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.pool.release(self.since.elapsed(), self.threads);
    }
}

/// The states of an `ask_many` that have not started: they count as queue for the policy, or one
/// call alone could never open the parallel mode (its workers are at most `K`).
pub struct Demand<'a> {
    pool: &'a SlotPool,
    remaining: AtomicUsize,
}

impl Demand<'_> {
    /// Take one state off the demand; the request that asks for the slot takes it, under the
    /// same lock that creates its ticket, so the queue never dips.
    fn take_one(&self) -> bool {
        self.remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |r| r.checked_sub(1))
            .is_ok()
    }
}

impl Drop for Demand<'_> {
    fn drop(&mut self) {
        let left = self.remaining.swap(0, Ordering::SeqCst);
        let mut g = self.pool.lock();
        g.extra = g.extra.saturating_sub(left);
        let queued = g.queued();
        g.mode = self.pool.policy.after_release(g.mode, g.busy, queued);
        drop(g);
        self.pool.cv.notify_all();
    }
}

impl SlotPool {
    /// A pool of `max_concurrency` slots for a model that has `thread_budget` threads.
    pub fn new(max_concurrency: usize, thread_budget: usize) -> Self {
        SlotPool {
            policy: Policy::new(max_concurrency, thread_budget),
            inner: Mutex::new(Inner::default()),
            cv: Condvar::new(),
            contexts_created: AtomicU64::new(0),
        }
    }

    /// The rule this pool follows.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Declare `n` states that will ask for a slot, one by one, with
    /// [`acquire_claiming`](Self::acquire_claiming).
    pub fn demand(&self, n: usize) -> Demand<'_> {
        let mut g = self.lock();
        g.extra = g.extra.saturating_add(n);
        drop(g);
        self.cv.notify_all();
        Demand {
            pool: self,
            remaining: AtomicUsize::new(n),
        }
    }

    /// Wait for a slot, in arrival order.
    ///
    /// # Errors
    /// Whatever the [`Control`] returns (`Error::Cancelled`): the request gives up its place and
    /// nobody behind it is held back.
    pub fn acquire(&self, control: &dyn Control) -> Result<Slot<'_>> {
        self.acquire_with(control, None)
    }

    /// Like [`acquire`](Self::acquire), for one state of a [`Demand`].
    ///
    /// # Errors
    /// Whatever the [`Control`] returns (`Error::Cancelled`).
    pub fn acquire_claiming(&self, control: &dyn Control, demand: &Demand<'_>) -> Result<Slot<'_>> {
        self.acquire_with(control, Some(demand))
    }

    fn acquire_with(&self, control: &dyn Control, demand: Option<&Demand<'_>>) -> Result<Slot<'_>> {
        control.checkpoint()?;
        let asked_at = Instant::now();
        let mut g = self.lock();
        let ticket = g.next_ticket;
        g.next_ticket += 1;
        if let Some(d) = demand
            && d.take_one()
        {
            g.extra = g.extra.saturating_sub(1);
        }
        // a new arrival can make the head of the queue eligible (the queue got longer)
        self.cv.notify_all();
        let mut last_check = Instant::now();
        loop {
            if ticket == g.serving {
                let queued = g.queued();
                if let Some(adm) = self.policy.admit(g.mode, g.busy, g.threads_in_use, queued) {
                    if adm.mode == Mode::Parallel && g.mode == Mode::Serial {
                        g.parallel_openings += 1;
                    }
                    g.mode = adm.mode;
                    g.serving += 1;
                    g.skip_abandoned();
                    g.busy += 1;
                    g.peak_busy = g.peak_busy.max(g.busy);
                    g.threads_in_use += adm.threads;
                    g.peak_threads = g.peak_threads.max(g.threads_in_use);
                    g.waited_us += micros(asked_at.elapsed());
                    drop(g);
                    self.cv.notify_all();
                    return Ok(Slot {
                        pool: self,
                        threads: adm.threads,
                        since: Instant::now(),
                    });
                }
            }
            let (guard, _) = self
                .cv
                .wait_timeout(g, CHECK_EVERY)
                .unwrap_or_else(PoisonError::into_inner);
            g = guard;
            if last_check.elapsed() >= CHECK_EVERY {
                drop(g);
                let verdict = control.checkpoint();
                g = self.lock();
                last_check = Instant::now();
                if let Err(e) = verdict {
                    g.abandon(ticket);
                    g.cancelled += 1;
                    drop(g);
                    self.cv.notify_all();
                    return Err(e);
                }
            }
        }
    }

    fn release(&self, held: Duration, threads: usize) {
        let mut g = self.lock();
        g.busy = g.busy.saturating_sub(1);
        g.threads_in_use = g.threads_in_use.saturating_sub(threads);
        g.completed += 1;
        g.held_us += micros(held);
        let queued = g.queued();
        g.mode = self.policy.after_release(g.mode, g.busy, queued);
        drop(g);
        self.cv.notify_all();
    }

    /// The counters and the state of the pool now.
    pub fn stats(&self) -> PoolStats {
        let g = self.lock();
        PoolStats {
            max_concurrency: self.policy.max_concurrency(),
            thread_budget: self.policy.thread_budget(),
            parallel: g.mode == Mode::Parallel,
            parallel_openings: g.parallel_openings,
            contexts_created: self.contexts_created.load(Ordering::Relaxed),
            busy: g.busy,
            waiting: g.waiting(),
            peak_busy: g.peak_busy,
            completed: g.completed,
            cancelled: g.cancelled,
            waited_ms: g.waited_us / 1000,
            held_ms: g.held_us / 1000,
            threads_in_use: g.threads_in_use,
            peak_threads: g.peak_threads,
        }
    }
}
