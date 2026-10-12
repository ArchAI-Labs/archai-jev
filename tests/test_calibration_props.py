"""Properties against references, and inputs that must never crash the interpreter."""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from archai_jev import CalibrationError, fit_temperature
from archai_jev import calibration as cal


def reference_metrics(
    rows: list[list[float]], labels: list[int], n_bins: int
) -> dict[str, float]:
    """Plain Python with ``math.fsum`` (exactly rounded sums) — the numpy reference is below."""
    n = len(rows)
    conf = [max(r) for r in rows]
    hit = [r.index(c) == y for r, c, y in zip(rows, conf, labels, strict=True)]
    bins: dict[int, list[int]] = {}
    for i, c in enumerate(conf):
        b = next(b for b in range(n_bins) if c <= (b + 1) / n_bins)
        bins.setdefault(b, []).append(i)
    ece = math.fsum(
        len(ix)
        / n
        * abs(
            math.fsum(hit[i] for i in ix) / len(ix)
            - math.fsum(conf[i] for i in ix) / len(ix)
        )
        for ix in bins.values()
    )
    brier = math.fsum(
        math.fsum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(r))
        for r, y in zip(rows, labels, strict=True)
    )
    loss = math.fsum(-math.log(r[y]) for r, y in zip(rows, labels, strict=True))
    return {
        "ece": ece,
        "brier": brier / n,
        "log_loss": loss / n,
        "mean_confidence": math.fsum(conf) / n,
    }


def random_probabilities(
    seed: int, n: int, k: int
) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    rows = [
        cal.calibrated_softmax([rng.gauss(0, 2) for _ in range(k)]) for _ in range(n)
    ]
    return rows, [rng.randrange(k) for _ in range(n)]


@pytest.mark.parametrize(("n", "k"), [(2000, 2), (2000, 7), (500, 255), (10_000, 4)])
def test_reference_parity_ece_brier_logloss(n: int, k: int) -> None:
    rows, labels = random_probabilities(n + k, n, k)
    report = cal.calibration_report(rows, labels)
    want = reference_metrics(rows, labels, 10)
    assert report.ece == pytest.approx(want["ece"], abs=1e-12)
    assert report.brier == pytest.approx(want["brier"], abs=1e-12)
    assert report.log_loss == pytest.approx(want["log_loss"], abs=1e-12)
    assert report.mean_confidence == pytest.approx(want["mean_confidence"], abs=1e-12)


def test_reference_parity_with_numpy() -> None:
    np = pytest.importorskip("numpy")
    for n, k, seed in [(10_000, 4, 1), (3000, 255, 2), (10_000, 2, 3)]:
        rows, labels = random_probabilities(seed, n, k)
        p = np.array(rows)
        y = np.array(labels)
        conf = p.max(axis=1)
        hit = p.argmax(axis=1) == y
        edges = np.arange(1, 11) / 10
        bin_of = np.minimum(np.searchsorted(edges, conf, side="left"), 9)
        ece = 0.0
        for b in range(10):
            mask = bin_of == b
            if mask.any():
                ece += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
        onehot = np.zeros_like(p)
        onehot[np.arange(n), y] = 1.0
        report = cal.calibration_report(p, y)
        assert report.ece == pytest.approx(ece, abs=1e-12)
        assert report.brier == pytest.approx(
            ((p - onehot) ** 2).sum(axis=1).mean(), abs=1e-12
        )
        assert report.log_loss == pytest.approx(
            -np.log(p[np.arange(n), y]).mean(), abs=1e-12
        )
        assert report.mean_confidence == pytest.approx(conf.mean(), abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**31),
    t0=st.floats(0.1, 10.0),
    true_t=st.floats(0.5, 5.0),
)
def test_recipe_with_log_probabilities(seed: int, t0: float, true_t: float) -> None:
    rng = random.Random(seed)
    k = 4
    logits = [[rng.gauss(0, 2) for _ in range(k)] for _ in range(120)]
    labels = [
        rng.choices(range(k), weights=cal.calibrated_softmax(r, true_t))[0]
        for r in logits
    ]
    bounds = (1e-3, 1e4)
    try:
        direct = fit_temperature(logits, labels, bounds=bounds)
    except CalibrationError:
        return
    probabilities = [cal.calibrated_softmax(r, t0) for r in logits]
    relative = fit_temperature(
        [[math.log(p) for p in row] for row in probabilities], labels, bounds=bounds
    )
    assert relative.temperature * t0 == pytest.approx(direct.temperature, rel=1e-6)


class Hostile:
    """An object whose ``__float__`` and ``__index__`` raise."""

    def __float__(self) -> float:
        raise KeyError("float")

    def __index__(self) -> int:
        raise KeyError("index")


class Weird(float):
    def __float__(self) -> float:
        return math.nan


numbers = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(-(10**400), 10**400),
    st.booleans(),
    st.none(),
    st.text(max_size=3),
    st.just(Hostile()),
    st.just(Weird(1.0)),
    st.just(b"x"),
)
rows_strategy = st.one_of(
    st.lists(st.lists(numbers, max_size=4), max_size=4),
    st.lists(numbers, max_size=4),
    numbers,
    st.lists(st.lists(st.floats(-5, 5), min_size=1, max_size=4), max_size=4),
)
labels_strategy = st.one_of(
    st.lists(numbers, max_size=4), numbers, st.lists(st.integers(-3, 6), max_size=4)
)
ALLOWED = (ValueError, TypeError, OverflowError, KeyError, CalibrationError)


@settings(max_examples=400, deadline=None)
@given(rows=rows_strategy, labels=labels_strategy)
def test_fuzz_no_crash_only_standard_errors(rows: Any, labels: Any) -> None:
    calls: list[Any] = [
        lambda: fit_temperature(rows, labels),
        lambda: cal.calibration_report(rows, labels),
        lambda: cal.log_loss_from_logits(rows, labels),
        lambda: cal.calibrated_softmax_batch(rows),
        lambda: cal.yes_no_probability_batch(rows),
        lambda: cal.choice_confidence_batch(rows),
        lambda: cal.score_value(rows),
        lambda: cal.reliability_curve(rows, labels, n_bins=3),
    ]
    for call in calls:
        with contextlib.suppress(*ALLOWED):
            call()


def test_user_exceptions_propagate() -> None:
    with pytest.raises(KeyError, match="float"):
        cal.calibrated_softmax_batch([[1.0, Hostile()]])  # type: ignore[list-item]
    with pytest.raises(KeyError, match="index"):
        cal.expected_calibration_error([[1.0]], [Hostile()])  # type: ignore[list-item]
    for scalar_call in (
        lambda: cal.yes_no_probability(Hostile(), 1.0),  # type: ignore[arg-type]
        lambda: cal.calibrated_softmax_batch([[1.0]], Hostile()),  # type: ignore[arg-type]
    ):
        with pytest.raises(KeyError):
            scalar_call()
