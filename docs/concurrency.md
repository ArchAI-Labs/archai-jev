# Interrupting, cancelling and sharing a `Jev`

A `Jev` answers **one request at a time**. You can share it between threads and `asyncio` tasks: the
calls wait their turn, in the order they arrived. What this page covers is how to stop a call that is
running, what happens when you cancel a task, and why there is no setting to run requests in parallel.

## `Ctrl-C` stops the call

```python
try:
    answers = jev.ask_many(thousand_messages, {"safety": SAFETY})
except KeyboardInterrupt:
    ...   # the call stopped; the Jev is fine, ask again whenever you like
```

The model is asked "may I go on?" between two rows of its work (a row is one question, a fraction of a
second), so `Ctrl-C` is seen at the next row instead of at the end of the call. Nothing partial is
returned and nothing is left in a strange state: the next call gives the same answers it would have
given before. On the reference machine, with the default model, the interrupt arrived 32 ms after the
signal (the slowest was 98 ms), where before this release it arrived when the whole call finished.
Only the main thread receives `Ctrl-C` (that is how Python works); a call in another thread is not
stopped by it.

A signal handler, or a `__del__`, that runs while a `Jev` call is in progress and calls `ask` itself
would wait for the call it interrupted for ever. It gets a `RuntimeError` instead.

## Cancelling an `aask` stops the work

```python
task = asyncio.create_task(jev.aask(state, questions))
...
task.cancel()          # also what asyncio.wait_for() does when its timeout expires
```

A request that has not started never starts. A request that is running stops at its next row and gives
the model back. Before this release the result was only thrown away and the model kept computing it, so
a timeout did not free the model. With twelve requests waiting and all of them cancelled, the next
`aask` completed in 0.21 s instead of 1.2 s on the reference machine.

`aask` and `aask_many` run on a small pool of four threads (named `archai-jev-*`) that belongs to the
`Jev`, not on the event loop's default executor: a program that keeps that one busy no longer slows
down `aask`, and a flood of `aask` no longer slows down the program's own `to_thread` calls. The threads
go away when the `Jev` does.

## One failure should not lose the rest

```python
results = jev.ask_many(states, questions, return_exceptions=True)   # list[Answers | JevError]
good = [r for r in results if isinstance(r, Answers)]
```

With `return_exceptions=True` a state whose calculation fails (a `NumericalError`, for instance) gives
its error in its place and the others are still answered. A state or a question that is invalid still
raises before anything is computed, and `Ctrl-C` always stops the call.

## Why not run requests in parallel?

llama.cpp already uses every core inside one request, so a second request at the same time has no idle
CPU to use. We measured a pool of contexts that share the weights on a 20-thread CPU: at best a gain of
about 1.1 to 1.2 times in throughput, with the same latency per request at best and worse at light load.
So `Jev` has no `max_concurrency` setting, and `ask_many` goes through its states one after the other.
The pool exists in the Rust core for a server that serves many clients (it is tested to give the same
numbers); it is not exposed in Python.

One rule came out of that work and is worth knowing if you tune threads yourself:
`ARCHAI_JEV_NUM_THREADS` is the number of threads of the *whole* model. Contexts that share the cores
make llama.cpp's own thread pool (it spins while it waits at synchronisation points) stall each other:
we saw 60-token decodes last 8 to 30 seconds with 35 threads on 20 cores. The pool never allows more
threads in total than that number.

## Two processes on one machine

`ARCHAI_JEV_NUM_THREADS` defaults to every logical core. That is right for one process and **wrong for
two**: llama.cpp's own thread pool spins while it waits at its synchronisation points, so when two
processes each want every core, each one keeps waiting for a thread the other has taken. On the
reference machine (20 logical cores, the default model, one short message again and again for 25 s):

| | calls in 25 s | median per call |
|---|---|---|
| one process | 239 and 253 | 92 ms |
| two processes at once, every core each | **neither finished in 120 s** (stopped) | |
| two processes at once, `ARCHAI_JEV_NUM_THREADS=10` each | 147 and 146 | 152 ms |

If you run several processes that load a model (worker processes of a web server, a notebook next to a
script), give each a share: `ARCHAI_JEV_NUM_THREADS` = cores divided by the number of processes, or a
bit less. One process serving many clients (with the requests waiting their turn) is the better shape.

## Summary

| You want to | Do |
|---|---|
| stop a long `ask`/`ask_many` | `Ctrl-C` (main thread); the `Jev` stays usable |
| give up on an `aask` | `task.cancel()` or `asyncio.wait_for(...)`: the model is freed at the next row |
| not lose the good answers when one state fails | `ask_many(..., return_exceptions=True)` |
| limit the CPU used | `ARCHAI_JEV_NUM_THREADS` (the whole model), not a number of requests |
| run several processes | divide the cores between them with `ARCHAI_JEV_NUM_THREADS` |
