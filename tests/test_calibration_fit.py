"""``fit_temperature``: result, degenerate data, arguments, threads and the GIL."""

from __future__ import annotations

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from _timing import timing_sensitive
from archai_jev import CalibrationError, JevError, TemperatureFit, fit_temperature
from archai_jev.calibration import calibrated_softmax


def make_dataset(
    seed: int, n: int, k: int, sigma: float, true_t: float
) -> tuple[list[list[float]], list[int]]:
    """Gaussian logits; labels drawn from ``softmax(z / true_t)``."""
    rng = random.Random(seed)
    rows = [[rng.gauss(0.0, sigma) for _ in range(k)] for _ in range(n)]
    labels = [
        rng.choices(range(k), weights=calibrated_softmax(row, true_t))[0]
        for row in rows
    ]
    return rows, labels


def test_analytic_vector() -> None:
    fit = fit_temperature([[1.0, -1.0]] * 3, labels=[0, 0, 1])
    assert isinstance(fit, TemperatureFit)
    assert fit.temperature == pytest.approx(2 / math.log(2), rel=1e-9)
    assert fit.n == 3 and 1 <= fit.iterations <= 200
    assert fit.bounds == (0.01, 100.0)

    def loss(t: float) -> float:
        right, wrong = math.log1p(math.exp(-2 / t)), math.log1p(math.exp(2 / t))
        return (2 * right + wrong) / 3

    assert fit.loss_before == pytest.approx(loss(1.0), rel=1e-12)
    assert fit.loss_after == pytest.approx(loss(fit.temperature), rel=1e-12)
    assert fit.loss_after < fit.loss_before


def test_example_of_the_spec() -> None:
    fit = fit_temperature([[1.0, -1.0]] * 3, labels=[0, 0, 1])
    assert abs(fit.temperature - 2 / math.log(2)) < 1e-9
    with pytest.raises(CalibrationError, match=r"above the upper bound 100\.0"):
        fit_temperature([[1.0, -1.0]] * 3, labels=[1, 1, 1])


def test_determinism_two_runs_and_threads() -> None:
    rows, labels = make_dataset(5, 2000, 4, 3.0, 2.0)
    first = fit_temperature(rows, labels)
    assert fit_temperature(rows, labels) == first
    assert hash(first) == hash(fit_temperature(rows, labels))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: fit_temperature(rows, labels), range(8)))
    assert all(r == first for r in results)


UNIFORM = (
    "cannot fit a temperature: the logits do not discriminate between options "
    "(every row is uniform), so every temperature gives the same loss"
)
ARGMAX = (
    "the best temperature is below the lower bound {lo}: the labels always match the "
    "top option, so the loss keeps decreasing as the temperature goes to 0; "
    "provide more examples that include mistakes"
)
NO_INFORMATION = (
    "the best temperature is above the upper bound {hi}: the logits carry no usable "
    "information about the labels (or point the wrong way), so calibrating them would "
    "be meaningless; check the labels and the model"
)


def test_degenerate_cases_exact_messages() -> None:
    cases: list[tuple[Any, Any, str]] = [
        ([[2.0, 2.0], [0.0, 0.0, 0.0]], [0, 2], UNIFORM),
        ([[0.0, 3.0, 2.5]], [1], ARGMAX.format(lo="0.01")),
        ([[0.0, 3.0, 2.5], [5.0, 1.0]], [1, 0], ARGMAX.format(lo="0.01")),
        ([[0.0, 3.0, 2.5]], [0], NO_INFORMATION.format(hi="100.0")),
        ([[0.0, 3.0, 2.5], [5.0, 1.0]], [0, 1], NO_INFORMATION.format(hi="100.0")),
    ]
    for logits, labels, message in cases:
        with pytest.raises(CalibrationError) as info:
            fit_temperature(logits, labels)
        assert str(info.value) == message
        assert isinstance(info.value, JevError) and isinstance(info.value, ValueError)
    rows, labels = make_dataset(5, 20_000, 4, 30.0, 500.0)
    with pytest.raises(CalibrationError) as info:
        fit_temperature(rows, labels)
    assert str(info.value) == NO_INFORMATION.format(hi="100.0")
    wide = fit_temperature(rows, labels, bounds=(0.01, 1000.0))
    assert wide.temperature == pytest.approx(500.0, rel=0.3)


def test_a_tiny_lower_bound_cannot_skip_the_refusal() -> None:
    with pytest.raises(ValueError) as info:
        fit_temperature([[0.0, 3.0, 2.5]], [1], bounds=(1e-310, 100.0))
    assert str(info.value) == (
        "the lower bound 1e-310 of bounds is too small: 1 / lower must be a finite number"
    )


def test_logits_whose_difference_overflows_are_refused() -> None:
    with pytest.raises(ValueError, match="spans more than the largest float number"):
        fit_temperature([[1e308, -1e308]] * 2, [0, 1])
    # representable but enormous: a finite answer or a refusal, never NaN read as converged
    try:
        fit = fit_temperature(
            [[1e308, 0.9e308], [0.9e308, 1e308]], [0, 0], bounds=(1.0, 1e308)
        )
    except CalibrationError:
        return
    assert math.isfinite(fit.temperature) and math.isfinite(fit.loss_after)


def test_a_large_common_offset_does_not_change_the_answer() -> None:
    expected = fit_temperature([[1.0, -1.0]] * 3, [0, 0, 1]).temperature
    for offset in (1e3, 1e7, 1e9, 1e12):
        shifted = fit_temperature([[1.0 + offset, -1.0 + offset]] * 3, [0, 0, 1])
        assert shifted.temperature == pytest.approx(expected, rel=1e-6), offset


def test_wider_bounds_still_reject_always_argmax() -> None:
    for bounds in [(1e-3, 1e4), (1e-6, 1e9)]:
        with pytest.raises(CalibrationError, match="below the lower bound"):
            fit_temperature([[0.0, 3.0, 2.5]], [1], bounds=bounds)


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        (([], []), ValueError, "logits must not be empty"),
        (
            ([[0.0, 1.0]] * 3, [0, 1]),
            ValueError,
            "logits has 3 rows but labels has 2; they must have the same length",
        ),
        (
            ([[0.0, 1.0], [2.0]], [0, 0]),
            ValueError,
            "logits row 1 has a single option; it carries no calibration information, "
            "remove it",
        ),
        (
            ([[0.0, 1.0], []], [0, 0]),
            ValueError,
            "logits row 1 is empty; every row needs at least one option",
        ),
        (
            ([[0.0, float("nan")]], [0]),
            ValueError,
            "logits row 0 has the non-finite value NaN at option 1",
        ),
        (
            ([[0.0, float("inf")]], [0]),
            ValueError,
            "logits row 0 has the non-finite value inf at option 1",
        ),
        (
            ([[0.0, float("-inf")]], [0]),
            ValueError,
            "logits row 0 has the non-finite value -inf at option 1",
        ),
        (
            ([[0.0, 1.0], [0.0, 1.0]], [0, 2]),
            ValueError,
            "label of row 1 is 2 but the row has 2 options (valid labels are 0..1)",
        ),
        (
            ([[0.0, 1.0]], [-1]),
            ValueError,
            "label of row 0 is -1 but the row has 2 options (valid labels are 0..1)",
        ),
        (
            ([[0.0, 1.0]], [True]),
            TypeError,
            "labels must be integers, got bool at position 0",
        ),
        (
            ([[0.0, 1.0]], [1.0]),
            TypeError,
            "labels must be integers, got float at position 0",
        ),
        (
            ([[0.0, 1.0]], ["1"]),
            TypeError,
            "labels must be integers, got str at position 0",
        ),
    ],
)
def test_invalid_arguments_table(
    args: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error) as info:
        fit_temperature(*args)
    assert str(info.value) == message


@pytest.mark.parametrize(
    ("bounds", "text"),
    [
        ((0, 1), "(0.0, 1.0)"),
        ((2, 1), "(2.0, 1.0)"),
        ((1, math.inf), "(1.0, inf)"),
        ((-1.0, 5.0), "(-1.0, 5.0)"),
        ((1.0, 1.0), "(1.0, 1.0)"),
    ],
)
def test_invalid_bounds(bounds: tuple[float, float], text: str) -> None:
    with pytest.raises(ValueError) as info:
        fit_temperature([[1.0, -1.0]] * 3, [0, 0, 1], bounds=bounds)
    assert str(info.value) == f"bounds must satisfy 0 < lower < upper < inf, got {text}"


def test_extremes_and_n1() -> None:
    rows, labels = make_dataset(1, 100_000, 4, 3.0, 2.0)
    fit = fit_temperature(rows, labels)
    assert math.isfinite(fit.temperature) and not math.isnan(fit.loss_after)
    assert fit.temperature == pytest.approx(2.0, rel=0.05)
    big, big_labels = make_dataset(2, 500, 5, 1e6, 2e5)
    huge = fit_temperature(big, big_labels, bounds=(1.0, 1e9))
    assert math.isfinite(huge.temperature) and math.isfinite(huge.loss_after)
    one = fit_temperature([[0.0, 3.0, 2.5]], [2])
    assert one.temperature == pytest.approx(1.864, abs=1e-3)
    with pytest.raises(CalibrationError):
        fit_temperature([[0.0, 3.0, 2.5]], [1])
    with pytest.raises(CalibrationError):
        fit_temperature([[0.0, 3.0, 2.5]], [0])


def test_ragged_rows_are_one_temperature() -> None:
    rows, labels = make_dataset(3, 3000, 2, 3.0, 2.0)
    rows3, labels3 = make_dataset(4, 3000, 10, 3.0, 2.0)
    mixed = fit_temperature(rows + rows3, labels + labels3)
    assert mixed.temperature == pytest.approx(2.0, rel=0.05)


@timing_sensitive()
def test_gil_released_during_fit() -> None:
    rows, labels = make_dataset(7, 100_000, 10, 3.0, 2.0)
    ticks = 0
    stop = threading.Event()

    def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            time.sleep(0.01)
            ticks += 1

    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        start = time.perf_counter()
        fit_temperature(rows, labels)
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        thread.join()
    # Measured on the development machine: a fit that releases the GIL gets ~80% of the ticks
    # an idle process would (19 of 23 in 0.24 s); a call that holds it, like sorting 3 million
    # floats, gets ~3% (3 in 0.85 s). The reading of the input holds the GIL, hence not 100%.
    assert elapsed >= 0.05
    assert ticks >= max(
        3, 0.3 * elapsed / 0.01
    ), f"only {ticks} ticks in {elapsed:.3f} s: the GIL was held"


def test_eight_threads_equal_sequential() -> None:
    datasets = [make_dataset(s, 3000, 4, 3.0, 2.0) for s in range(8)]
    sequential = [fit_temperature(r, ls) for r, ls in datasets]
    with ThreadPoolExecutor(8) as pool:
        parallel = list(pool.map(lambda d: fit_temperature(*d), datasets))
    assert parallel == sequential
