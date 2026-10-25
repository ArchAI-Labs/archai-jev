"""Calibration: public formulas, ``fit_temperature``, metrics and the reliability diagram data.

Import this module on its own (like :mod:`archai_jev.testing`)::

    from archai_jev.calibration import calibration_report

``fit_temperature``, ``TemperatureFit`` and ``CalibrationError`` live in ``archai_jev`` itself.
"""

from ._calibration import (
    CalibrationReport,
    ReliabilityBin,
    brier_score,
    calibrated_softmax,
    calibrated_softmax_batch,
    calibration_report,
    choice_confidence,
    choice_confidence_batch,
    expected_calibration_error,
    log_loss,
    log_loss_from_logits,
    reliability_curve,
    score_confidence,
    score_confidence_batch,
    score_value,
    score_value_batch,
    yes_no_probability,
    yes_no_probability_batch,
)

__all__ = [
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
