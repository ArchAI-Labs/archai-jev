"""archai_jev: typed decisions with calibrated probabilities, computed by a Rust core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._answers import Answer, Answers, ChoiceAnswer, ScoreAnswer, YesNoAnswer
from ._core import calibrated_softmax, render_state
from ._exceptions import (
    IncompatibleModelError,
    InferenceError,
    InvalidQuestionError,
    InvalidStateError,
    JevError,
    ModelDownloadError,
    ModelVerificationError,
    NumericalError,
    UnsupportedRequestError,
)
from ._jev import Jev, ModelInfo, list_models
from ._questions import Choice, Noul, Score, YesNo

JSONValue = str | int | float | bool | None | Mapping[str, Any] | Sequence[Any]
State = str | Mapping[str, Any] | Sequence[Any]
Question = Choice | Score | YesNo

__all__ = [
    "Answer",
    "Answers",
    "Choice",
    "ChoiceAnswer",
    "IncompatibleModelError",
    "InferenceError",
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
    "UnsupportedRequestError",
    "YesNo",
    "YesNoAnswer",
    "calibrated_softmax",
    "list_models",
    "render_state",
]
