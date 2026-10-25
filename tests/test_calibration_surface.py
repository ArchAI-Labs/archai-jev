"""The public surface of the calibration feature: exact names, classes, exceptions."""

from __future__ import annotations

import dataclasses
import pickle

import pytest

import archai_jev
from archai_jev import CalibrationError, JevError, TemperatureFit, fit_temperature
from archai_jev import calibration as cal

CALIBRATION_NAMES = [
    "CalibrationReport",
    "ReliabilityBin",
    "brier_score",
    "calibrated_softmax",
    "calibrated_softmax_batch",
    "calibration_report",
    "choice_confidence",
    "choice_confidence_batch",
    "expected_calibration_error",
    "log_loss",
    "log_loss_from_logits",
    "reliability_curve",
    "score_confidence",
    "score_confidence_batch",
    "score_value",
    "score_value_batch",
    "yes_no_probability",
    "yes_no_probability_batch",
]


def test_all_snapshots() -> None:
    assert sorted(cal.__all__) == CALIBRATION_NAMES
    assert len(cal.__all__) == len(set(cal.__all__)) == 18
    for name in cal.__all__:
        assert hasattr(cal, name), name
    # The top level gained exactly three names.
    for name in ("fit_temperature", "TemperatureFit", "CalibrationError"):
        assert name in archai_jev.__all__
        assert hasattr(archai_jev, name)
    assert "calibration" not in archai_jev.__all__


def test_calibration_error_is_jev_and_value_error() -> None:
    assert issubclass(CalibrationError, JevError)
    assert issubclass(CalibrationError, ValueError)
    assert CalibrationError.__module__ == "archai_jev"
    assert (CalibrationError.__doc__ or "").strip()
    with pytest.raises(CalibrationError):
        fit_temperature([[1.0, -1.0]] * 3, [1, 1, 1])
    # `except ValueError` keeps working
    with pytest.raises(ValueError):
        fit_temperature([[1.0, -1.0]] * 3, [1, 1, 1])


def test_dataclasses_contract() -> None:
    fit = fit_temperature([[1.0, -1.0]] * 3, [0, 0, 1])
    assert isinstance(fit, TemperatureFit)
    assert dataclasses.is_dataclass(fit)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.temperature = 1.0  # type: ignore[misc]
    again = fit_temperature([[1.0, -1.0]] * 3, [0, 0, 1])
    assert fit == again and hash(fit) == hash(again)
    assert pickle.loads(pickle.dumps(fit)) == fit
    text = repr(fit)
    assert (
        text.startswith("TemperatureFit(temperature=2.88539") and "bounds=(0.01" in text
    )
    assert TemperatureFit.__module__ == "archai_jev"
    assert cal.ReliabilityBin.__module__ == cal.CalibrationReport.__module__
    assert cal.ReliabilityBin.__module__ == "archai_jev.calibration"
    names = [f.name for f in dataclasses.fields(TemperatureFit)]
    assert names == [
        "temperature",
        "loss_before",
        "loss_after",
        "n",
        "iterations",
        "bounds",
    ]
