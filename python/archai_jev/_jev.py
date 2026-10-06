"""The :class:`Jev` model object and :class:`ModelInfo`."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import _core
from ._answers import Answers, build_answers
from ._exceptions import InvalidQuestionError
from ._questions import Choice, Score, YesNo

__all__ = ["Jev", "ModelInfo", "list_models"]


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelInfo:
    """What is known about a loaded model. Immutable and hashable.

    Attributes:
        name: Model name (``"mock"`` for the fake scorer).
        revision: Pinned revision of the checkpoint, if any.
        family: Supported family (architecture + head + template).
        head: Decision head: ``"pointer"``, ``"letters"`` or ``"mock"``.
        template: Versioned prompt template.
        temperature: Temperature applied to the raw logits.
        calibrated: ``False`` if no calibration is declared (every answer says so).
        license: License of the model.
        max_context: Declared maximum context, in tokens.
        dtype: Weights variant in use (``"q8_0"``, ``"bf16"``, ``"f32"``).
        source: ``"registry"`` or ``"local"``.
        calibration_source: ``"manifest"``, ``"user"`` or ``"none"``.
        tasks: Ids of the tasks the model answers, or ``None`` if it has no closed set.
        notice: A warning from the model's author that is shown on loading (the default model
            says it is a demo).
        is_default: Only for :func:`list_models`: whether this is the default model.
    """

    name: str
    revision: str | None
    family: str
    head: str
    template: str
    temperature: float
    calibrated: bool
    license: str | None
    max_context: int | None
    dtype: str | None = None
    source: str | None = None
    calibration_source: str | None = None
    tasks: tuple[str, ...] | None = None
    notice: str | None = None
    is_default: bool | None = None


def _info(raw: dict[str, Any]) -> ModelInfo:
    tasks = raw.get("tasks")
    return ModelInfo(**{**raw, "tasks": None if tasks is None else tuple(tasks)})


def list_models() -> tuple[ModelInfo, ...]:
    """The models this version of archai-jev knows by name, with the default marked.

    Static data: no network and no disk access.
    """
    return tuple(_info(d) for d in _core.list_models())


def _question_pairs(questions: Any) -> list[tuple[str, Any]]:
    if not isinstance(questions, Mapping):
        raise TypeError(
            f"questions must be a Mapping of name to question, got {type(questions).__name__}"
        )
    pairs: list[tuple[str, Any]] = []
    for name, q in questions.items():
        if not isinstance(name, str):
            raise InvalidQuestionError(
                f"question name must be a str, got {type(name).__name__}"
            )
        if not isinstance(q, (Choice, Score, YesNo)):
            raise InvalidQuestionError(
                f'question "{name}" must be a Choice, Score or YesNo, got {type(q).__name__}'
            )
        pairs.append((name, q._q))
    return pairs


class Jev:
    """A loaded decision model. Get one with :meth:`from_pretrained` (or :meth:`from_scorer`
    for the test scorer); ``Jev()`` is not allowed.

    A ``Jev`` is immutable and safe to share between threads and coroutines.
    """

    __slots__ = ("_info", "_model")

    _model: Any
    _info: ModelInfo

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "Jev cannot be instantiated directly; use Jev.from_pretrained(...) "
            "(or Jev.from_scorer(archai_jev.testing.MockScorer()) to try the API)"
        )

    @classmethod
    def _create(cls, model: Any, info: ModelInfo) -> Jev:
        obj = object.__new__(cls)
        object.__setattr__(obj, "_model", model)
        object.__setattr__(obj, "_info", info)
        return obj

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike[str] | None = None,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        temperature: float | None = None,
        allow_uncalibrated: bool = False,
        cache_dir: str | os.PathLike[str] | None = None,
        offline: bool = False,
        manifest: str | os.PathLike[str] | None = None,
    ) -> Jev:
        """Load a model: the default one, a model of the registry, or a local folder.

        Everything is checked before a probability is returned: the manifest, the files (SHA-256),
        the weights file, the tokenizer, the head, and a self-check of the model's numbers. The
        first call may download the model (about 1.6 GB for the default).

        Args:
            name_or_path: ``None`` for the default model, a registry name, or a folder with an
                ``archai-jev-manifest.json`` (pass a :class:`pathlib.Path` to force a folder).
            revision: A pinned revision of a registry model (full hash or at least 8 characters).
            device: Only ``"cpu"``.
            dtype: Weights variant (``"q8_0"``, ``"bf16"``, ``"f32"``); default: the model's.
            temperature: Your own calibration: the temperature that divides the model's raw logits
                (greater than 0). Every answer then says ``calibrated=True``.
            allow_uncalibrated: Accept a model without a declared calibration; every answer then
                says ``calibrated=False``.
            cache_dir: Where models are stored (default: the OS cache folder, or
                ``ARCHAI_JEV_CACHE``).
            offline: Never open a network connection.
            manifest: The manifest file of a local folder, if it is not in the folder.

        Raises:
            IncompatibleModelError: The model cannot be trusted: bad manifest, wrong file, no
                calibration, unsupported family...
            ModelVerificationError: The self-check gave different numbers than expected.
            ModelDownloadError: A file could not be downloaded or found.
            InferenceError: The engine cannot run (for example a CPU without AVX2).
        """
        for label, value in (
            ("revision", revision),
            ("dtype", dtype),
            ("device", device),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{label} must be a str, got {type(value).__name__}")
        if isinstance(temperature, bool) or not (
            temperature is None or isinstance(temperature, (int, float))
        ):
            raise TypeError(
                f"temperature must be a number or None, got {type(temperature).__name__}"
            )
        for label, flag in (
            ("allow_uncalibrated", allow_uncalibrated),
            ("offline", offline),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{label} must be a bool, got {type(flag).__name__}")
        if name_or_path is None:
            kind, value_ = "default", None
        elif isinstance(name_or_path, str):
            kind, value_ = "str", name_or_path
        elif isinstance(name_or_path, os.PathLike):
            kind, value_ = "path", os.fspath(name_or_path)
        else:
            raise TypeError(
                "name_or_path must be a str, a path or None, "
                f"got {type(name_or_path).__name__}"
            )

        def path_or_none(label: str, p: Any) -> str | None:
            if p is None:
                return None
            if isinstance(p, (str, os.PathLike)):
                return os.fspath(p)
            raise TypeError(f"{label} must be a str or a path, got {type(p).__name__}")

        model, raw = _core.load_model(
            kind,
            value_,
            revision,
            device,
            dtype,
            None if temperature is None else float(temperature),
            allow_uncalibrated,
            path_or_none("cache_dir", cache_dir),
            offline,
            path_or_none("manifest", manifest),
        )
        return cls._create(model, _info(raw))

    @classmethod
    def from_scorer(cls, scorer: _core.MockScorer) -> Jev:
        """Wrap a test scorer from :mod:`archai_jev.testing`.

        Raises:
            TypeError: If ``scorer`` is anything else (models of your own go through a manifest).
        """
        if not isinstance(scorer, _core.MockScorer):
            raise TypeError(
                "from_scorer accepts only scorers from archai_jev.testing, "
                f"got {type(scorer).__name__}"
            )
        info = ModelInfo(
            name="mock",
            revision=None,
            family="mock",
            head="mock",
            template="mock",
            temperature=scorer.temperature,
            calibrated=scorer.calibrated,
            license=None,
            max_context=None,
        )
        return cls._create(_core._Model.from_mock(scorer), info)

    @property
    def model_info(self) -> ModelInfo:
        """Facts about the loaded model."""
        return self._info

    def ask(self, state: Any, questions: Mapping[str, Any]) -> Answers:
        """Ask every question about ``state``; all or nothing.

        Args:
            state: A string, a dict or a list (nested JSON-like values).
            questions: Mapping of question name to :class:`Choice`, :class:`Score` or
                :class:`YesNo`.

        Raises:
            InvalidStateError: Unsupported state. Raised before any computation.
            InvalidQuestionError: Invalid questions or names. Raised before any computation.
            NumericalError: The model gave unusable numbers; no result is returned.
        """
        pairs = _question_pairs(questions)
        return build_answers(self._model.ask(state, pairs))

    def ask_many(
        self, states: Iterable[Any], questions: Mapping[str, Any]
    ) -> list[Answers]:
        """Same questions on several states, in order; every state is validated first."""
        pairs = _question_pairs(questions)
        if isinstance(states, (str, bytes, Mapping)):
            raise TypeError(
                f"states must be an iterable of states, got {type(states).__name__}"
            )
        listed = list(states)
        return [build_answers(raw) for raw in self._model.ask_many(listed, pairs)]

    async def aask(self, state: Any, questions: Mapping[str, Any]) -> Answers:
        """Like :meth:`ask`, in a worker thread so the event loop is not blocked."""
        return await asyncio.to_thread(self.ask, state, questions)

    async def aask_many(
        self, states: Iterable[Any], questions: Mapping[str, Any]
    ) -> list[Answers]:
        """Like :meth:`ask_many`, in a worker thread."""
        listed = (
            list(states) if not isinstance(states, (str, bytes, Mapping)) else states
        )
        return await asyncio.to_thread(self.ask_many, listed, questions)

    def __repr__(self) -> str:
        return f"Jev(model={self._info.name!r})"
