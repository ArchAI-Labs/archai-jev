"""Exception hierarchy of ``archai_jev``.

Every error about *content* (state, questions, models, numbers) is a :class:`JevError`.
Errors in how the API is *used* (wrong argument types) are plain ``TypeError``/``ValueError``.
"""

from __future__ import annotations

__all__ = [
    "IncompatibleModelError",
    "InferenceError",
    "InvalidQuestionError",
    "InvalidStateError",
    "JevError",
    "ModelDownloadError",
    "ModelVerificationError",
    "NumericalError",
    "UnsupportedRequestError",
]


class JevError(Exception):
    """Base class of every error raised by archai_jev about its inputs, models or results."""


class InvalidStateError(JevError, ValueError):
    """The state cannot be used: unsupported type, non-string key, NaN/infinity, a cycle,
    nesting deeper than 32 levels, or more than 1 MiB of text. Raised before any computation.
    """


class InvalidQuestionError(JevError, ValueError):
    """A question or the set of questions is invalid: empty instructions, a wrong number of
    options, duplicate keys or names, wrong types. Raised before any computation."""


class UnsupportedRequestError(JevError, ValueError):
    """The request is valid in itself but the loaded model cannot answer it faithfully
    (for example the state is longer than the model's declared context). Nothing is truncated.
    """


class IncompatibleModelError(JevError, RuntimeError):
    """A checkpoint or its manifest cannot be loaded: missing or invalid manifest, unknown
    family, wrong tensors, wrong tokenizer, failed integrity check, or no calibration was
    declared and none was given. Raised when loading, never later."""


class ModelVerificationError(JevError, RuntimeError):
    """The model loaded but its self-check vectors gave different numbers than expected, so its
    probabilities cannot be trusted. The model is not returned."""


class NumericalError(JevError, RuntimeError):
    """The model produced non-finite logits, a wrong number of logits, or probabilities that are
    not a valid distribution. No result is returned."""


class InferenceError(JevError, RuntimeError):
    """The inference engine could not run: a decoding error, not enough memory for the context,
    or a CPU without the instructions the engine needs (AVX2, FMA, F16C, BMI2). No result is
    returned and the model stays usable for the next request."""


class ModelDownloadError(JevError, OSError):
    """A model file could not be downloaded or found: no network, HTTP error, not enough disk
    space, or offline mode with the file missing from the cache."""


for _cls in (
    JevError,
    InvalidStateError,
    InvalidQuestionError,
    UnsupportedRequestError,
    IncompatibleModelError,
    InferenceError,
    ModelVerificationError,
    NumericalError,
    ModelDownloadError,
):
    _cls.__module__ = "archai_jev"
del _cls
