"""Async calls, cancellation and `return_exceptions` (spec 008).

The mock splits its ``latency`` between the questions and looks at the control before each slice
(see ``test_signals.py``), so cancelling a call part-way can be told from letting it finish.
"""

import asyncio
import gc
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from archai_jev import (
    Answers,
    InvalidStateError,
    Jev,
    JevError,
    NumericalError,
    YesNo,
)
from archai_jev.testing import Fault, MockScorer

QUESTIONS = {f"q{i}": YesNo(f"Is it number {i}?") for i in range(4)}
STATE = "A short text."


def make(latency: float = 0.0) -> tuple[Jev, MockScorer]:
    scorer = MockScorer(1, latency=latency)
    return Jev.from_scorer(scorer), scorer


def archai_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("archai-jev")]


def run(coro: Callable[[], Awaitable[Any]]) -> Any:
    return asyncio.run(coro())  # type: ignore[arg-type]


def test_aask_and_aask_many_equal_the_sync_calls() -> None:
    jev, _ = make()

    async def main() -> None:
        assert await jev.aask(STATE, QUESTIONS) == jev.ask(STATE, QUESTIONS)
        assert await jev.aask_many([STATE, STATE], QUESTIONS) == jev.ask_many(
            [STATE, STATE], QUESTIONS
        )

    asyncio.run(main())


def test_a_busy_default_executor_does_not_slow_aask() -> None:
    jev, _ = make(latency=0.1)

    async def main() -> None:
        loop = asyncio.get_running_loop()
        # the program keeps the loop's default executor busy for a while with its own jobs
        blockers = [loop.run_in_executor(None, time.sleep, 0.8) for _ in range(40)]
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        await jev.aask(STATE, QUESTIONS)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5, f"aask waited for the user's jobs: {elapsed:.2f}s"
        await asyncio.gather(*blockers)

    asyncio.run(main())


def test_many_aask_do_not_slow_the_default_executor() -> None:
    jev, _ = make(latency=0.1)

    async def main() -> None:
        flood = [asyncio.create_task(jev.aask(STATE, QUESTIONS)) for _ in range(64)]
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        assert await asyncio.to_thread(lambda: 42) == 42
        assert time.perf_counter() - started < 0.2
        for task in flood:
            task.cancel()
        await asyncio.gather(*flood, return_exceptions=True)

    asyncio.run(main())


def test_cancelling_aask_raises_at_once_and_the_jev_stays_usable() -> None:
    jev, _ = make(latency=0.4)
    expected = jev.ask(STATE, QUESTIONS)

    async def main() -> None:
        task = asyncio.create_task(jev.aask(STATE, QUESTIONS))
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.perf_counter() - started < 0.1
        assert await jev.aask(STATE, QUESTIONS) == expected

    asyncio.run(main())


def test_a_request_cancelled_in_the_queue_never_starts() -> None:
    jev, scorer = make(latency=0.4)

    async def main() -> None:
        # four requests take the four threads; the fifth waits in the queue
        running = [asyncio.create_task(jev.aask(STATE, QUESTIONS)) for _ in range(4)]
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(jev.aask(STATE, QUESTIONS))
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await asyncio.gather(*running)

    asyncio.run(main())
    assert scorer.calls == 4, "the cancelled request must never have reached the scorer"


def test_a_running_request_stops_at_its_next_row_and_frees_its_thread() -> None:
    jev, _ = make(latency=0.4)  # four rows of 0.1 s

    async def main() -> None:
        busy = [asyncio.create_task(jev.aask(STATE, QUESTIONS)) for _ in range(4)]
        await asyncio.sleep(0.05)
        for task in busy:
            task.cancel()
        await asyncio.gather(*busy, return_exceptions=True)
        # the threads are free again within a row (0.1 s), not after the whole call (0.4 s)
        started = time.perf_counter()
        await jev.aask(STATE, QUESTIONS)
        assert (
            time.perf_counter() - started < 0.4 + 0.2
        ), "the cancelled calls ran to the end"

    asyncio.run(main())


def test_cancelling_aask_many_stops_between_states() -> None:
    jev, scorer = make(latency=0.2)

    async def main() -> None:
        task = asyncio.create_task(jev.aask_many([STATE] * 10, QUESTIONS))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.4)  # let the worker reach its next look

    asyncio.run(main())
    assert scorer.calls <= 2, scorer.calls  # not ten


def test_executor_threads_are_at_most_four_and_end_with_the_jev() -> None:
    def use_a_jev() -> None:
        jev, _ = make(latency=0.05)

        async def main() -> None:
            await asyncio.gather(*(jev.aask(STATE, QUESTIONS) for _ in range(40)))
            assert 1 <= len(archai_threads()) <= 4

        asyncio.run(main())

    use_a_jev()  # the Jev is gone when it returns
    gc.collect()
    deadline = time.perf_counter() + 2.0
    while archai_threads() and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert archai_threads() == []


def test_exiting_with_requests_pending_does_not_hang() -> None:
    code = textwrap.dedent("""
        import asyncio
        from archai_jev import Jev, YesNo
        from archai_jev.testing import MockScorer

        jev = Jev.from_scorer(MockScorer(1, latency=0.4))
        questions = {f"q{i}": YesNo(f"Is it {i}?") for i in range(4)}

        async def main():
            for _ in range(30):
                asyncio.create_task(jev.aask("text", questions))
            await asyncio.sleep(0.1)
            raise SystemExit(0)

        asyncio.run(main())
        """)
    started = time.perf_counter()
    done = subprocess.run([sys.executable, "-c", code], timeout=30, capture_output=True)
    assert done.returncode == 0, done.stderr.decode()
    # thirty requests of 0.4 s on four threads would take 3 s if they all ran
    assert time.perf_counter() - started < 3.0


def test_return_exceptions_in_order_and_types() -> None:
    faulty = Jev.from_scorer(MockScorer(1).with_fault(Fault.nan("q1", 0)))
    good, _ = make()
    # default: the first failure ends the call
    with pytest.raises(NumericalError):
        faulty.ask_many([STATE, STATE], QUESTIONS)
    # with the flag: an error object in the place of each state that failed
    items = faulty.ask_many([STATE, STATE], QUESTIONS, return_exceptions=True)
    assert len(items) == 2 and all(isinstance(i, NumericalError) for i in items)
    # and the answers of the states that worked are what a plain call gives
    answers = good.ask_many([STATE, "another"], QUESTIONS, return_exceptions=True)
    assert answers == good.ask_many([STATE, "another"], QUESTIONS)
    assert all(isinstance(a, Answers) for a in answers)
    # the async method has the same flag
    assert asyncio.run(faulty.aask_many([STATE], QUESTIONS, return_exceptions=True))


def test_return_exceptions_does_not_hide_invalid_input() -> None:
    jev, scorer = make()
    with pytest.raises(InvalidStateError, match=r"states\[1\]"):
        jev.ask_many([STATE, {1, 2}], QUESTIONS, return_exceptions=True)
    assert scorer.calls == 0, "invalid states are refused before any calculation"


def test_the_types_of_ask_many_follow_the_flag() -> None:
    jev, _ = make()
    plain: list[Answers] = jev.ask_many([STATE], QUESTIONS)
    explicit_false: list[Answers] = jev.ask_many(
        [STATE], QUESTIONS, return_exceptions=False
    )
    mixed: list[Answers | JevError] = jev.ask_many(
        [STATE], QUESTIONS, return_exceptions=True
    )
    # with the flag the items may be errors: it is not a list of Answers for the type checker
    wrong: list[Answers] = jev.ask_many(  # type: ignore[assignment]
        [STATE], QUESTIONS, return_exceptions=True
    )
    assert plain == explicit_false == wrong and len(mixed) == 1
