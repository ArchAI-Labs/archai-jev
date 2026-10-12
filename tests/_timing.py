"""Helpers for the few tests whose verdict depends on how late the machine is.

A test that asserts "this stopped within a row, not at the end of the call" needs a wall-clock
bound. A busy CI runner (two cores shared with other jobs) is late by a random amount; a real
fault (nobody looks at the signal between rows) is late every time. So these tests are tried up
to ``attempts`` times and fail only when every try is slow: the lateness of a loaded machine does
not repeat, the fault does.
"""

import _thread
import contextlib
import functools
import threading
import time
from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar

import pytest

P = ParamSpec("P")
R = TypeVar("R")

# What a failed check looks like: an ``assert``, or ``pytest.raises`` that saw nothing raised
# because the event it waited for came too late.
_TIMING_FAILURES = (AssertionError, pytest.fail.Exception)


def timing_sensitive(attempts: int = 3) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Run the test up to ``attempts`` times; fail with the last failure if every try failed."""

    def decorate(test: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(test)
        def run(*args: P.args, **kwargs: P.kwargs) -> R:
            failure: BaseException | None = None
            for _ in range(attempts):
                try:
                    return test(*args, **kwargs)
                except _TIMING_FAILURES as error:
                    failure = error
            assert failure is not None
            raise failure

        return run

    return decorate


@contextlib.contextmanager
def interrupt_main_at(delay: float) -> Iterator[None]:
    """Simulate a Ctrl-C ``delay`` seconds from now, and never later than the end of the block.

    If the block ends first (the machine was slow and the call finished), the timer is cancelled:
    a signal that arrived later would interrupt whatever pytest is running by then.
    """
    timer = threading.Timer(delay, _thread.interrupt_main)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        try:
            timer.join()
            # a signal that fired just now is handled here, not further on
            time.sleep(0)
        except KeyboardInterrupt:
            pass
