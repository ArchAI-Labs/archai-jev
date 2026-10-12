//! When may a request start, and with how many threads? (spec 008, D46.)
//!
//! A pool of contexts is a device for **capacity**, not for latency: splitting the threads of one
//! model between several requests makes each of them slower (measured: two contexts of ten
//! threads give 1.23 times the capacity, so each request takes about 1.65 times longer). So the
//! pool works in two modes:
//!
//! * [`Mode::Serial`]: one request at a time with every thread, exactly like a pool of one;
//! * [`Mode::Parallel`]: up to `K` requests together, each with a share of the threads. It opens
//!   only when [`QUEUE_THRESHOLD`] requests are waiting and stays open until the pool drains.
//!
//! The value 4 and the sizing come from spike S9 (`results/s9d`): neutral at light and medium
//! load, 0.85 of the p95 at 3 requests per second and 0.48 at overload. Everything here is a pure
//! function of a few numbers, so it is tested without threads.
//!
//! **Invariant: the threads of the requests that run together never add up to more than the
//! budget.** ggml's own thread pool spins at its synchronisation points, so a wide context (all
//! the threads) next to narrow ones (35 threads on 20 cores) does not just share the cores: it
//! stalls everybody (decodes of 60 tokens that took 8 to 30 seconds instead of under one, measured
//! on the library in task T6). A request is therefore admitted only if it fits in what is left.

/// Requests waiting (the head of the queue included) that open the parallel mode.
pub const QUEUE_THRESHOLD: usize = 4;

/// How the pool is working now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// One request at a time, with the whole thread budget.
    #[default]
    Serial,
    /// Up to `K` requests together, each with a share of the thread budget.
    Parallel,
}

/// A request that may start now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admission {
    /// The mode the pool is in after admitting it.
    pub mode: Mode,
    /// Threads for the context of this request.
    pub threads: usize,
}

/// The rule: `K` slots and a thread budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    k: usize,
    budget: usize,
}

impl Policy {
    /// A policy for at most `max_concurrency` requests together and `thread_budget` threads in
    /// all (both at least 1).
    pub fn new(max_concurrency: usize, thread_budget: usize) -> Self {
        Policy {
            k: max_concurrency.max(1),
            budget: thread_budget.max(1),
        }
    }

    /// The most requests that run together.
    pub fn max_concurrency(&self) -> usize {
        self.k
    }

    /// The threads of the whole model.
    pub fn thread_budget(&self) -> usize {
        self.budget
    }

    /// Threads of one request in the parallel mode, given how many requests are present (running,
    /// plus waiting): the budget divided between them, never more than `K` ways.
    pub fn threads_parallel(&self, present: usize) -> usize {
        (self.budget / present.clamp(1, self.k)).max(1)
    }

    fn admission(&self, mode: Mode, busy: usize, queued: usize) -> Admission {
        let threads = match mode {
            Mode::Serial => self.budget,
            Mode::Parallel => self.threads_parallel(busy.saturating_add(queued)),
        };
        Admission { mode, threads }
    }

    /// May the request at the head of the queue start?
    ///
    /// `busy` is how many requests run now and `threads_in_use` how many threads they have in
    /// all; `queued` is how many wait for a slot, **counting the head itself** and the states of
    /// an `ask_many` that have not started yet. A request that would take the threads in use over
    /// the budget is not admitted: it waits for one of the running ones to finish.
    pub fn admit(
        &self,
        mode: Mode,
        busy: usize,
        threads_in_use: usize,
        queued: usize,
    ) -> Option<Admission> {
        if busy >= self.k {
            return None;
        }
        if self.k == 1 {
            return Some(self.admission(Mode::Serial, busy, queued));
        }
        let wanted = if busy == 0 {
            // Nothing is running: a queue that is already there opens the parallel mode at once.
            let mode = if queued >= QUEUE_THRESHOLD {
                Mode::Parallel
            } else {
                Mode::Serial
            };
            self.admission(mode, busy, queued)
        } else {
            match mode {
                Mode::Parallel => self.admission(Mode::Parallel, busy, queued),
                Mode::Serial if queued >= QUEUE_THRESHOLD => {
                    self.admission(Mode::Parallel, busy, queued)
                }
                Mode::Serial => return None,
            }
        };
        (threads_in_use.saturating_add(wanted.threads) <= self.budget).then_some(wanted)
    }

    /// The mode after a request finished: back to serial when nothing runs and nothing waits.
    pub fn after_release(&self, mode: Mode, busy: usize, queued: usize) -> Mode {
        if self.k == 1 || (busy == 0 && queued == 0) {
            Mode::Serial
        } else {
            mode
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pool_of_one_is_always_serial_with_every_thread() {
        let p = Policy::new(1, 20);
        for queued in 0..10 {
            let a = p.admit(Mode::Serial, 0, 0, queued).unwrap();
            assert_eq!(
                a,
                Admission {
                    mode: Mode::Serial,
                    threads: 20
                }
            );
            assert_eq!(p.admit(Mode::Serial, 1, 20, queued), None);
        }
        assert_eq!(p.after_release(Mode::Parallel, 3, 3), Mode::Serial);
    }

    #[test]
    fn below_the_threshold_one_request_runs_at_a_time_with_every_thread() {
        let p = Policy::new(4, 20);
        for queued in 1..QUEUE_THRESHOLD {
            assert_eq!(
                p.admit(Mode::Serial, 0, 0, queued),
                Some(Admission {
                    mode: Mode::Serial,
                    threads: 20
                })
            );
            assert_eq!(p.admit(Mode::Serial, 1, 20, queued), None);
        }
    }

    #[test]
    fn a_queue_at_the_threshold_opens_the_parallel_mode_once_the_wide_request_is_gone() {
        let p = Policy::new(4, 20);
        // a wide request runs (20 threads in use) and four wait: nobody else fits, whatever the queue
        assert_eq!(p.admit(Mode::Serial, 1, 20, QUEUE_THRESHOLD), None);
        assert_eq!(p.admit(Mode::Serial, 1, 20, 40), None);
        // it finishes and the queue is still there: parallel from the start, five requests present
        // at least, four ways to split the threads
        assert_eq!(
            p.admit(Mode::Serial, 0, 0, QUEUE_THRESHOLD),
            Some(Admission {
                mode: Mode::Parallel,
                threads: 5
            })
        );
        // fewer present than K: a larger share (two present, ten threads each)
        assert_eq!(p.threads_parallel(2), 10);
    }

    #[test]
    fn the_parallel_mode_admits_up_to_k_and_no_more() {
        let p = Policy::new(4, 20);
        for busy in 1..4 {
            // `busy` requests of five threads each
            assert!(
                p.admit(Mode::Parallel, busy, 5 * busy, 8).is_some(),
                "busy {busy}"
            );
        }
        assert_eq!(p.admit(Mode::Parallel, 4, 20, 10), None);
        assert_eq!(p.admit(Mode::Serial, 4, 20, 10), None);
    }

    #[test]
    fn a_request_that_does_not_fit_in_what_is_left_waits() {
        let p = Policy::new(4, 20);
        // two requests of ten threads run; the head would get five: 25 > 20
        assert_eq!(p.admit(Mode::Parallel, 2, 20, 8), None);
        // one of them is gone: 10 + 5 fits
        assert!(p.admit(Mode::Parallel, 1, 10, 8).is_some());
    }

    #[test]
    fn threads_for_every_pair() {
        for budget in 1..=64 {
            for k in 1..=16 {
                let p = Policy::new(k, budget);
                for present in 1..=40 {
                    let t = p.threads_parallel(present);
                    assert!(
                        t >= 1 && t <= budget,
                        "budget {budget} k {k} present {present}"
                    );
                    assert_eq!(t, (budget / present.min(k)).max(1));
                }
            }
        }
    }

    #[test]
    fn mode_transitions() {
        let p = Policy::new(4, 20);
        assert_eq!(p.after_release(Mode::Parallel, 0, 0), Mode::Serial);
        assert_eq!(p.after_release(Mode::Parallel, 2, 0), Mode::Parallel);
        assert_eq!(p.after_release(Mode::Parallel, 0, 2), Mode::Parallel);
        assert_eq!(p.after_release(Mode::Serial, 1, 3), Mode::Serial);
        // with nothing running the next admission decides again from the queue
        assert_eq!(
            p.admit(Mode::Parallel, 0, 0, 2).map(|a| a.mode),
            Some(Mode::Serial)
        );
    }

    /// Random arrivals and releases: whatever the policy admits, the threads in use never add up
    /// to more than the budget, no more than K requests run, and nothing that fits is refused for
    /// ever (the queue drains).
    #[test]
    fn the_threads_in_use_never_exceed_the_budget() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for budget in [1usize, 2, 5, 8, 12, 16, 20, 32] {
            for k in [1usize, 2, 3, 4, 8] {
                let p = Policy::new(k, budget);
                let mut mode = Mode::Serial;
                let mut running: Vec<usize> = Vec::new(); // threads of each running request
                let mut queued = 0usize;
                let mut finished = 0usize;
                for _ in 0..4000 {
                    match next(3) {
                        0 => queued += 1,
                        1 if !running.is_empty() => {
                            let i = next(running.len() as u64) as usize;
                            running.swap_remove(i);
                            finished += 1;
                            mode = p.after_release(mode, running.len(), queued);
                        }
                        _ => {}
                    }
                    while queued > 0 {
                        let in_use: usize = running.iter().sum();
                        match p.admit(mode, running.len(), in_use, queued) {
                            Some(a) => {
                                mode = a.mode;
                                running.push(a.threads);
                                queued -= 1;
                            }
                            None => break,
                        }
                        let in_use: usize = running.iter().sum();
                        assert!(running.len() <= k, "k {k}");
                        assert!(
                            in_use <= budget || running.len() == 1,
                            "budget {budget} k {k}: {running:?}"
                        );
                    }
                }
                // drain: every request that waits is eventually admitted as the others finish
                while !running.is_empty() || queued > 0 {
                    let in_use: usize = running.iter().sum();
                    if let Some(a) = (queued > 0)
                        .then(|| p.admit(mode, running.len(), in_use, queued))
                        .flatten()
                    {
                        mode = a.mode;
                        running.push(a.threads);
                        queued -= 1;
                    } else {
                        assert!(
                            !running.is_empty(),
                            "budget {budget} k {k}: stuck with {queued} waiting"
                        );
                        running.pop();
                        finished += 1;
                        mode = p.after_release(mode, running.len(), queued);
                    }
                }
                assert!(finished > 0);
            }
        }
    }

    #[test]
    fn a_trace_of_arrivals_and_releases() {
        // a burst of 6 against K = 4, budget 20: the first four start together (five threads each)
        let p = Policy::new(4, 20);
        let mut mode = Mode::Serial;
        let mut running: Vec<usize> = Vec::new();
        let mut queued = 6usize;
        let mut started = Vec::new();
        while queued > 0 || !running.is_empty() {
            let in_use: usize = running.iter().sum();
            match p.admit(mode, running.len(), in_use, queued) {
                Some(a) if queued > 0 => {
                    mode = a.mode;
                    running.push(a.threads);
                    queued -= 1;
                    started.push(a.threads);
                }
                _ => {
                    running.pop();
                    mode = p.after_release(mode, running.len(), queued);
                }
            }
        }
        assert_eq!(started.len(), 6);
        assert_eq!(started.first(), Some(&5));
        assert_eq!(mode, Mode::Serial);
    }
}
