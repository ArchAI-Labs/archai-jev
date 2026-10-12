"""Metrics and reliability diagram: definitions checked by hand, invariants, errors, classes."""

from __future__ import annotations

import dataclasses
import math
import pickle
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from archai_jev import calibration as cal
from archai_jev.calibration import CalibrationReport, ReliabilityBin

HAND_ROWS = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
HAND_LABELS = [0, 1, 1, 0]


def test_hand_vectors() -> None:
    report = cal.calibration_report(HAND_ROWS, HAND_LABELS)
    assert report.n == 4
    assert report.ece == pytest.approx(0.4, abs=1e-12)
    assert report.brier == pytest.approx(0.45, abs=1e-12)
    assert report.log_loss == pytest.approx(0.6455747489491624, abs=1e-12)
    assert report.accuracy == pytest.approx(0.75, abs=1e-12)
    assert report.mean_confidence == pytest.approx(0.75, abs=1e-12)
    assert report.mce == pytest.approx(0.8, abs=1e-12)
    assert cal.expected_calibration_error(HAND_ROWS, HAND_LABELS) == report.ece


def test_reliability_curve_shape_and_none() -> None:
    bins = cal.reliability_curve(HAND_ROWS, HAND_LABELS)
    assert len(bins) == 10
    empty = [b for b in bins if b.count == 0]
    assert len(empty) == 6
    assert all(
        b.accuracy is None and b.mean_confidence is None and b.gap is None
        for b in empty
    )
    assert bins[7].gap == pytest.approx(0.8)
    assert (bins[0].lower, bins[9].upper) == (0.0, 1.0)
    assert sum(b.count for b in bins) == 4


def counts(rows: list[list[float]], n_bins: int = 10) -> list[int]:
    labels = [0] * len(rows)
    return [b.count for b in cal.reliability_curve(rows, labels, n_bins=n_bins)]


def test_bin_edges_table() -> None:
    def at(bin_index: int) -> list[int]:
        return [1 if i == bin_index else 0 for i in range(10)]

    assert counts([[0.1] * 10]) == at(0)
    assert counts([[0.3, 0.3, 0.2, 0.2]]) == at(2)
    assert counts([[0.5, 0.5]]) == at(4)
    assert counts([[1.0, 0.0]]) == at(9)
    # a hair above an edge goes to the next bin
    assert counts([[0.5000000001, 0.4999999999]]) == at(5)
    assert counts([[0.9, 0.1]]) == at(8)
    assert counts([[0.9000000001, 0.0999999999]]) == at(9)
    assert counts([[1.0 / 255] * 255]) == at(0)
    assert counts([[1.0, 0.0]], n_bins=1) == [1]


@st.composite
def datasets(draw: Any) -> tuple[list[list[float]], list[int], int]:
    n = draw(st.integers(1, 40))
    k = draw(st.integers(2, 9))
    rows, labels = [], []
    for _ in range(n):
        logits = draw(st.lists(st.floats(-8, 8), min_size=k, max_size=k))
        rows.append(cal.calibrated_softmax(logits))
        labels.append(draw(st.integers(0, k - 1)))
    return rows, labels, draw(st.integers(1, 30))


@settings(max_examples=100, deadline=None)
@given(datasets())
def test_invariants_hypothesis(data: tuple[list[list[float]], list[int], int]) -> None:
    rows, labels, n_bins = data
    r = cal.calibration_report(rows, labels, n_bins=n_bins)
    assert 0 <= r.ece <= 1 and 0 <= r.mce <= 1 and 0 <= r.brier <= 2
    assert r.log_loss >= 0 and sum(b.count for b in r.bins) == r.n == len(rows)
    by_bins = sum(b.count / r.n * b.gap for b in r.bins if b.gap is not None)
    assert r.ece == pytest.approx(by_bins, abs=1e-12)
    one = cal.calibration_report(rows, labels, n_bins=1)
    assert one.ece == pytest.approx(abs(one.accuracy - one.mean_confidence), abs=1e-12)
    # the bins follow the definition: smallest i with c <= (i + 1) / B, in floats
    expected = [0] * n_bins
    for row in rows:
        c = max(row)
        expected[next(i for i in range(n_bins) if c <= (i + 1) / n_bins)] += 1
    assert [b.count for b in r.bins] == expected
    reordered = cal.calibration_report(rows[::-1], labels[::-1], n_bins=n_bins)
    assert reordered.ece == pytest.approx(r.ece, abs=1e-12)
    assert reordered.log_loss == pytest.approx(r.log_loss, abs=1e-12)


def test_report_equals_single_functions() -> None:
    report = cal.calibration_report(HAND_ROWS, HAND_LABELS, n_bins=7)
    assert report.ece == cal.expected_calibration_error(
        HAND_ROWS, HAND_LABELS, n_bins=7
    )
    assert report.brier == cal.brier_score(HAND_ROWS, HAND_LABELS)
    assert report.log_loss == cal.log_loss(HAND_ROWS, HAND_LABELS)
    assert list(report.bins) == cal.reliability_curve(HAND_ROWS, HAND_LABELS, n_bins=7)


def test_log_loss_inf_and_logits() -> None:
    assert cal.log_loss([[1.0, 0.0], [0.5, 0.5]], [1, 0]) == math.inf
    assert cal.log_loss([[1.0, 0.0]], [0]) == 0.0
    assert cal.log_loss_from_logits([[1e6, -1e6]], [0]) == pytest.approx(
        0.0, abs=1e-300
    )
    assert cal.log_loss_from_logits([[1e6, -1e6]], [1]) == pytest.approx(2e6)
    logits = [[2.0, 1.0, 0.1], [0.3, -1.2, 2.2]]
    labels = [0, 2]
    probabilities = [cal.calibrated_softmax(r, 2.3) for r in logits]
    assert cal.log_loss_from_logits(logits, labels, 2.3) == pytest.approx(
        cal.log_loss(probabilities, labels), abs=1e-9
    )
    # the function that fit_temperature minimises: its value at the fitted T is loss_after
    from archai_jev import fit_temperature

    rows, y = [[1.0, -1.0]] * 3, [0, 0, 1]
    fit = fit_temperature(rows, y)
    assert cal.log_loss_from_logits(rows, y, 1.0) == pytest.approx(fit.loss_before)
    assert cal.log_loss_from_logits(rows, y, fit.temperature) == pytest.approx(
        fit.loss_after
    )
    with pytest.raises(ValueError, match="temperature must be a finite number > 0"):
        cal.log_loss_from_logits(rows, y, 0.0)


def test_brier_two_classes() -> None:
    # (0.3 - 1)^2 is the binary Brier score; the multi-class one is twice that.
    assert cal.brier_score([[0.7, 0.3]], [1]) == pytest.approx(0.98, abs=1e-12)
    assert cal.brier_score([[1.0, 0.0]], [1]) == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize("bad", [0, -1, 1001, 2.5, "10", None, True])
def test_n_bins_errors(bad: Any) -> None:
    with pytest.raises(
        ValueError, match="n_bins must be an integer between 1 and 1000"
    ):
        cal.expected_calibration_error(HAND_ROWS, HAND_LABELS, n_bins=bad)
    with pytest.raises(ValueError, match="n_bins must be"):
        cal.reliability_curve(HAND_ROWS, HAND_LABELS, n_bins=bad)


def test_n_bins_accepts_numpy_integers() -> None:
    np = pytest.importorskip("numpy")
    assert len(cal.reliability_curve(HAND_ROWS, HAND_LABELS, n_bins=np.int64(4))) == 4
    with pytest.raises(ValueError, match="n_bins must be"):
        cal.reliability_curve(HAND_ROWS, HAND_LABELS, n_bins=np.float64(4.0))


def test_n_bins_message_shows_the_value() -> None:
    with pytest.raises(ValueError) as info:
        cal.calibration_report(HAND_ROWS, HAND_LABELS, n_bins=2.5)  # type: ignore[arg-type]
    assert str(info.value) == "n_bins must be an integer between 1 and 1000, got 2.5"
    assert len(cal.reliability_curve(HAND_ROWS, HAND_LABELS, n_bins=1000)) == 1000


def test_errors_with_index() -> None:
    with pytest.raises(ValueError, match="probabilities must not be empty"):
        cal.brier_score([], [])
    with pytest.raises(ValueError) as info:
        cal.log_loss(HAND_ROWS, HAND_LABELS[:3])
    assert (
        str(info.value)
        == "probabilities has 4 rows but labels has 3; they must have the same length"
    )
    with pytest.raises(ValueError) as info:
        cal.log_loss([[0.5, 0.5], [0.9, 0.9]], [0, 0])
    assert str(info.value) == "probabilities row 1: sums to 1.8, expected 1 within 1e-6"
    with pytest.raises(ValueError) as info:
        cal.log_loss([[0.5, 0.5]], [2])
    assert (
        str(info.value)
        == "label of row 0 is 2 but the row has 2 options (valid labels are 0..1)"
    )
    with pytest.raises(ValueError, match="sum_tolerance must be a number in"):
        cal.brier_score(HAND_ROWS, HAND_LABELS, sum_tolerance=0.1)


def test_results_immutable_hashable_picklable_repr() -> None:
    report = cal.calibration_report(HAND_ROWS, HAND_LABELS)
    assert isinstance(report, CalibrationReport)
    assert isinstance(report.bins, tuple) and isinstance(report.bins[0], ReliabilityBin)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.n = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.bins[0].count = 5  # type: ignore[misc]
    assert hash(report) == hash(cal.calibration_report(HAND_ROWS, HAND_LABELS))
    assert pickle.loads(pickle.dumps(report)) == report
    assert repr(report).startswith("CalibrationReport(n=4, accuracy=0.75")
    assert "ReliabilityBin(lower=0.0, upper=0.1, count=0" in repr(report)
    assert {report: 1}[cal.calibration_report(HAND_ROWS, HAND_LABELS)] == 1
