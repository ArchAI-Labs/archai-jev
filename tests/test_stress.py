"""Stress and deadlock tests (spec 008): many threads, many coroutines, a busy GIL.

Every test arms ``faulthandler`` so that a hang prints the stack of every thread instead of
sitting there until the CI gives up.
"""

import _thread
import asyncio
import faulthandler
import signal
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from archai_jev import Answers, Jev, YesNo
from archai_jev.testing import MockScorer

QUESTIONS = {f"q{i}": YesNo(f"Is it number {i}?") for i in range(3)}
STATES = [f"text number {i}" for i in range(8)]


@pytest.fixture(autouse=True)
def dump_stacks_if_stuck() -> Iterator[None]:
    faulthandler.dump_traceback_later(90, exit=False)
    yield
    faulthandler.cancel_dump_traceback_later()


def expected_answers(jev: Jev) -> list[Answers]:
    return [jev.ask(s, QUESTIONS) for s in STATES]


def test_threads_asks_and_aasks_together_give_the_sequential_answers() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.003))
    expected = expected_answers(jev)
    failures: list[str] = []

    def sync_work(worker: int) -> None:
        for i in range(20):
            k = (worker + i) % len(STATES)
            if jev.ask(STATES[k], QUESTIONS) != expected[k]:
                failures.append(f"ask {worker}/{i}")
            if jev.ask_many([STATES[k]], QUESTIONS) != [expected[k]]:
                failures.append(f"ask_many {worker}/{i}")

    def async_side() -> None:
        async def main() -> None:
            results = await asyncio.gather(
                *(jev.aask(STATES[i % len(STATES)], QUESTIONS) for i in range(64))
            )
            for i, r in enumerate(results):
                if r != expected[i % len(STATES)]:
                    failures.append(f"aask {i}")

        asyncio.run(main())

    threads = [threading.Thread(target=sync_work, args=(w,)) for w in range(16)]
    threads.append(threading.Thread(target=async_side))
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a thread is stuck"
    assert failures == []
    assert time.perf_counter() - started < 60


def test_no_deadlock_while_other_threads_keep_the_gil_busy() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.004))
    expected = expected_answers(jev)
    stop = threading.Event()
    spun = [0] * 8

    def spin(index: int) -> None:
        # pure Python: it asks for the GIL all the time and gives it back every few ms
        while not stop.is_set():
            spun[index] += sum(i * i for i in range(2000))

    spinners = [threading.Thread(target=spin, args=(i,)) for i in range(8)]
    for t in spinners:
        t.start()
    done: list[int] = []
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:

            def work(i: int) -> bool:
                k = i % len(STATES)
                ok = jev.ask(STATES[k], QUESTIONS) == expected[k]
                ok &= jev.ask_many([STATES[k]], QUESTIONS) == [expected[k]]
                done.append(i)
                return ok

            results = list(pool.map(work, range(40), timeout=60))

        async def main() -> list[Answers]:
            return await asyncio.gather(*(jev.aask(s, QUESTIONS) for s in STATES))

        assert asyncio.run(asyncio.wait_for(main(), timeout=60)) == expected
        # and the main thread, the one that looks at signals at every row, also gets through
        assert [jev.ask(s, QUESTIONS) for s in STATES] == expected
    finally:
        stop.set()
        for t in spinners:
            t.join()
    assert all(results) and len(done) == 40
    assert all(n > 0 for n in spun), "the busy threads must have run"


def test_ctrl_c_is_still_prompt_when_the_gil_is_contended() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.4))  # four rows of 0.1 s
    questions = {f"q{i}": YesNo(f"Is it {i}?") for i in range(4)}
    stop = threading.Event()
    go = threading.Event()
    fired = threading.Event()
    in_call = threading.Event()
    marks: dict[str, float] = {}

    def spin() -> None:
        while not stop.is_set():
            sum(i * i for i in range(2000))

    def firer() -> None:
        # Started before the busy threads (starting a thread under a contended GIL can take longer
        # than the whole call). What is measured is OUR part: from the signal to the moment the
        # handler runs inside the look between two rows.
        while not stop.is_set():
            if go.wait(0.05):
                go.clear()
                time.sleep(0.1)
                marks["signal"] = time.perf_counter()
                _thread.interrupt_main()
                fired.set()

    def handler(signum: int, frame: object) -> None:
        marks["handler"] = time.perf_counter()
        marks["during_call"] = 1.0 if in_call.is_set() else 0.0
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, handler)
    firing = threading.Thread(target=firer)
    firing.start()
    spinners = [threading.Thread(target=spin) for _ in range(8)]
    for t in spinners:
        t.start()
    measured = False
    try:
        for _ in range(
            5
        ):  # a starved firer can signal after the call is over: try again
            marks.clear()
            fired.clear()
            in_call.set()
            go.set()
            try:
                jev.ask("a text", questions)
            except KeyboardInterrupt:
                pass
            finally:
                in_call.clear()
            try:
                fired.wait(5)
                time.sleep(0.05)  # a late signal is handled here, outside the call
            except KeyboardInterrupt:
                pass
            if marks.get("during_call") == 1.0:
                measured = True
                break
    finally:
        stop.set()
        for t in spinners:
            t.join()
        firing.join()
        signal.signal(signal.SIGINT, previous)
    assert measured, "the signal never arrived inside the call"
    # one row (0.1 s) to the next look, and the time to get the GIL back from busy threads
    assert marks["handler"] - marks["signal"] < 0.1 + 0.3, marks


def test_every_thread_gets_its_turn_when_many_wait() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.002))
    counts = [0] * 16

    def work(index: int) -> None:
        for i in range(50):
            jev.ask(STATES[(index + i) % len(STATES)], QUESTIONS)
            counts[index] += 1

    threads = [threading.Thread(target=work, args=(w,)) for w in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert counts == [50] * 16


def test_a_cancelled_storm_leaves_the_jev_clean() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.05))
    expected = expected_answers(jev)

    async def main() -> None:
        for _ in range(5):
            tasks: list[Any] = [
                asyncio.create_task(jev.aask(STATES[i % len(STATES)], QUESTIONS))
                for i in range(40)
            ]
            await asyncio.sleep(0.03)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # after five rounds of start-and-cancel everything still answers correctly
        again = await asyncio.gather(*(jev.aask(s, QUESTIONS) for s in STATES))
        assert again == expected

    asyncio.run(main())
