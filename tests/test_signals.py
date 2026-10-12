"""Ctrl-C and cancellation of a running call (spec 008).

The mock splits its ``latency`` between the questions (its "rows") and looks at the control before
each slice, as the model scorer does between rows: a call can be stopped part-way, never in the
middle of a slice.
"""

import _thread
import signal
import threading
import time
from typing import Any

import pytest

from archai_jev import Jev, NumericalError, YesNo, _core
from archai_jev._exceptions import _StoppedError
from archai_jev._jev import _question_pairs
from archai_jev.testing import Fault, MockScorer

QUESTIONS = {f"q{i}": YesNo(f"Is it number {i}?") for i in range(4)}
STATE = "A short text."
LATENCY = 0.4  # four questions: four slices of 0.1 s
SLICE = LATENCY / len(QUESTIONS)


def make(latency: float = LATENCY, scorer: MockScorer | None = None) -> Jev:
    return Jev.from_scorer(scorer or MockScorer(1, latency=latency))


def fire_at(delay: float) -> threading.Timer:
    """Simulate a Ctrl-C ``delay`` seconds from now."""
    timer = threading.Timer(delay, _thread.interrupt_main)
    timer.start()
    return timer


def test_ctrl_c_stops_ask_within_one_row_and_the_jev_stays_usable() -> None:
    jev = make()
    expected = jev.ask(STATE, QUESTIONS)
    started = time.perf_counter()
    timer = fire_at(0.15)  # inside the second slice
    with pytest.raises(KeyboardInterrupt):
        jev.ask(STATE, QUESTIONS)
    elapsed = time.perf_counter() - started
    timer.join()
    # seen at the next look (0.2 s), not at the end of the call (0.4 s)
    assert SLICE <= elapsed < LATENCY - 0.05, elapsed
    assert jev.ask(STATE, QUESTIONS) == expected  # nothing left behind


def test_ctrl_c_stops_ask_many_between_states_and_inside_one() -> None:
    jev = make()
    started = time.perf_counter()
    timer = fire_at(0.15)
    with pytest.raises(KeyboardInterrupt):
        jev.ask_many([STATE] * 5, QUESTIONS)  # 2 s if nobody stops it
    elapsed = time.perf_counter() - started
    timer.join()
    assert elapsed < LATENCY, elapsed
    assert jev.ask_many([STATE], QUESTIONS) == [jev.ask(STATE, QUESTIONS)]


def test_ctrl_c_in_a_secondary_thread_does_not_stop_its_call() -> None:
    # Python only delivers signals to the main thread: a call in another thread is not stopped.
    jev = make()
    expected = jev.ask(STATE, QUESTIONS)
    result: list[Any] = []
    worker = threading.Thread(target=lambda: result.append(jev.ask(STATE, QUESTIONS)))
    worker.start()
    time.sleep(0.1)
    with pytest.raises(KeyboardInterrupt):
        _thread.interrupt_main()
        time.sleep(1)
    worker.join()
    assert result == [expected]


def test_a_signal_handler_cannot_call_ask_inside_a_jev_call() -> None:
    jev = make()
    seen: list[str] = []

    def handler(signum: int, frame: object) -> None:
        try:
            jev.ask(STATE, QUESTIONS)
        except RuntimeError as error:
            seen.append(str(error))
            raise

    previous = signal.signal(signal.SIGINT, handler)
    try:
        timer = fire_at(0.15)
        with pytest.raises(RuntimeError, match="signal handler"):
            jev.ask(STATE, QUESTIONS)
        timer.join()
    finally:
        signal.signal(signal.SIGINT, previous)
    assert seen and "destructor" in seen[0]
    assert jev.ask(STATE, QUESTIONS) == jev.ask(STATE, QUESTIONS)  # and it is usable


def test_a_token_set_before_the_call_stops_it_at_once() -> None:
    jev = make()
    token = _core._CancelToken()
    assert not token.is_set()
    token.set()
    assert token.is_set()
    with pytest.raises(_StoppedError):
        jev._model.ask(STATE, _question_pairs(QUESTIONS), token)


def test_a_token_set_during_the_call_stops_it_at_the_next_row() -> None:
    jev = make()
    token = _core._CancelToken()
    timer = threading.Timer(0.15, token.set)
    started = time.perf_counter()
    timer.start()
    with pytest.raises(_StoppedError):
        jev._model.ask(STATE, _question_pairs(QUESTIONS), token)
    elapsed = time.perf_counter() - started
    timer.join()
    assert SLICE <= elapsed < LATENCY - 0.05, elapsed
    # the same token does not stop a call that does not carry it
    assert jev.ask(STATE, QUESTIONS) == jev.ask(STATE, QUESTIONS)


def test_a_token_that_is_never_set_changes_nothing() -> None:
    jev = make(latency=0.0)
    token = _core._CancelToken()
    pairs = _question_pairs(QUESTIONS)
    assert len(jev._model.ask(STATE, pairs, token)) == len(QUESTIONS)
    assert len(jev._model.ask_many([STATE, STATE], pairs, token)) == 2


def test_return_exceptions_gives_the_error_in_its_place() -> None:
    faulty = MockScorer(1).with_fault(Fault.nan("q1", 0))
    jev = make(scorer=faulty)
    pairs = _question_pairs(QUESTIONS)
    with pytest.raises(NumericalError):
        jev._model.ask_many([STATE, STATE], pairs)
    items = jev._model.ask_many([STATE, STATE], pairs, None, True)
    assert len(items) == 2
    assert all(isinstance(item, NumericalError) for item in items)


def test_return_exceptions_never_swallows_a_stop() -> None:
    jev = make()
    token = _core._CancelToken()
    token.set()
    with pytest.raises(_StoppedError):
        jev._model.ask_many([STATE, STATE], _question_pairs(QUESTIONS), token, True)
