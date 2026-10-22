"""Exception hierarchy of ``archai_jev``.

Every error about *content* (state, questions, models, numbers) is a :class:`JevError`.
Errors in how the API is *used* (wrong argument types) are plain ``TypeError``/``ValueError``.
"""

from __future__ import annotations

__all__ = [
    "CalibrationError",
    "DeviceUnavailableError",
    "IncompatibleModelError",
    "InferenceError",
    "InvalidDatasetError",
    "InvalidQuestionError",
    "InvalidStateError",
    "JevError",
    "ModelDownloadError",
    "ModelVerificationError",
    "NumericalError",
    "UnsupportedRequestError",
]


class _StoppedError(Exception):
    """Private: a call stopped by its token (the cancellation of ``aask``). The async methods
    turn it into ``asyncio.CancelledError``; it never reaches a user."""


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


class CalibrationError(JevError, ValueError):
    """The data are valid but do not determine a calibration, so none is returned.

    Raised by :func:`archai_jev.fit_temperature` when the best temperature is not inside
    ``bounds``: every row is uniform, the labels always match the top option, or the logits
    carry no usable information about the labels. Using a temperature at the edge would make
    the probabilities look calibrated without any basis, so the function refuses.
    """


class InvalidDatasetError(JevError, ValueError):
    """A labelled dataset cannot be used: an empty file, a line that is not valid JSON, fields the
    format does not have, a label that does not fit its question, or one question name with two
    types. The message names the file and line (``data.jsonl:41``) or the record (``record 3``).
    Nothing is skipped and no partial dataset is returned.
    """


class InferenceError(JevError, RuntimeError):
    """The inference engine could not run: a decoding error, not enough memory for the context,
    or a CPU without the instructions the engine needs (AVX2, FMA, F16C, BMI2). No result is
    returned and the model stays usable for the next request."""


class DeviceUnavailableError(JevError, RuntimeError):
    """A device named in ``device=`` cannot be used: this build of archai_jev was compiled without
    that backend (the wheels on PyPI are CPU-only), or the machine has no such device (a missing
    driver, a virtual machine, a wrong index). ``device="cpu"`` and ``device="auto"`` never raise
    it. The message lists what the build contains."""


class ModelDownloadError(JevError, OSError):
    """A model file could not be downloaded or found: no network, HTTP error, not enough disk
    space, or offline mode with the file missing from the cache."""


for _cls in (
    JevError,
    CalibrationError,
    InvalidDatasetError,
    InvalidStateError,
    InvalidQuestionError,
    UnsupportedRequestError,
    IncompatibleModelError,
    InferenceError,
    DeviceUnavailableError,
    ModelVerificationError,
    NumericalError,
    ModelDownloadError,
):
    _cls.__module__ = "archai_jev"
del _cls
