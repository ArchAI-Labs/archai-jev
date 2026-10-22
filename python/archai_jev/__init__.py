"""archai_jev: typed decisions with calibrated probabilities, computed by a Rust core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._answers import Answer, Answers, ChoiceAnswer, ScoreAnswer, YesNoAnswer
from ._calibration import TemperatureFit, fit_temperature
from ._core import calibrated_softmax, render_state
from ._devices import DeviceInfo, available_devices, compiled_backends
from ._exceptions import (
    CalibrationError,
    DeviceFallbackWarning,
    DeviceUnavailableError,
    IncompatibleModelError,
    InferenceError,
    InvalidDatasetError,
    InvalidQuestionError,
    InvalidStateError,
    JevError,
    ModelDownloadError,
    ModelVerificationError,
    NumericalError,
    UnsupportedRequestError,
)
from ._jev import CacheInfo, Jev, ModelInfo, VariantInfo, list_models
from ._questions import Choice, Noul, Score, YesNo

JSONValue = str | int | float | bool | None | Mapping[str, Any] | Sequence[Any]
State = str | Mapping[str, Any] | Sequence[Any]
Question = Choice | Score | YesNo

__all__ = [
    "Answer",
    "Answers",
    "CacheInfo",
    "CalibrationError",
    "Choice",
    "ChoiceAnswer",
    "DeviceFallbackWarning",
    "DeviceInfo",
    "DeviceUnavailableError",
    "IncompatibleModelError",
    "InferenceError",
    "InvalidDatasetError",
    "InvalidQuestionError",
    "InvalidStateError",
    "JSONValue",
    "Jev",
    "JevError",
    "ModelDownloadError",
    "ModelInfo",
    "ModelVerificationError",
    "Noul",
    "NumericalError",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "TemperatureFit",
    "UnsupportedRequestError",
    "VariantInfo",
    "YesNo",
    "YesNoAnswer",
    "available_devices",
    "calibrated_softmax",
    "compiled_backends",
    "fit_temperature",
    "list_models",
    "render_state",
]
