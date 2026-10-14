"""The :class:`Jev` model object and :class:`ModelInfo`."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, overload

from . import _core
from ._answers import Answers, build_answers
from ._calibration import TemperatureFit
from ._evaluation import LabelledRecord, _prepare
from ._exceptions import InvalidQuestionError, JevError, _StoppedError
from ._questions import Choice, Score, YesNo
from ._reports import (
    CalibrationStudy,
    EvaluationReport,
    evaluation_from,
    study_from,
)

__all__ = ["CacheInfo", "Jev", "ModelInfo", "VariantInfo", "list_models"]

# Threads of the executor that runs ``aask`` / ``aask_many``. The model answers one request at a
# time whatever this number is; the others wait here, where they can still be cancelled.
_ASYNC_WORKERS = 4


@dataclass(frozen=True, kw_only=True, slots=True)
class CacheInfo:
    """What a model's memory of reused prefixes holds and has done. Immutable and hashable.

    The memory keeps the part of a prompt that questions share (the fixed part of a task, and the
    state when it is asked about again) so it is not computed twice. Everything is zero when
    ``prefix_cache_mib=0`` and for the test scorer. The counters add up over the life of the model
    until the model is loaded again.

    Attributes:
        limit_bytes: The most memory the saved prefixes may take.
        bytes: Memory they take now.
        entries: How many saved prefixes there are now.
        hits: Requests that restored a saved prefix.
        misses: Requests that looked for one and found none.
        evictions: Saved prefixes removed to make room.
        tokens_reused: Tokens that were not computed because they were shared or restored.
        tokens_decoded: Tokens the model computed.
    """

    limit_bytes: int
    bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int
    tokens_reused: int
    tokens_decoded: int


@dataclass(frozen=True, kw_only=True, slots=True)
class VariantInfo:
    """One variant of a model's weights (one ``dtype``), as the model's manifest declares it.

    A quantized variant is a model of its own (spec 009): it has its own calibration, and the
    temperature declared for one variant is never applied to another.

    Attributes:
        dtype: Name to pass as ``dtype=`` to :meth:`Jev.from_pretrained` (``"q8_0"``, ...).
        size_bytes: Size of the weights file, or ``None`` when the variant is produced by
            converting the original files on first use (its size is not known before that).
        calibrated: Whether the manifest declares a calibration for **this** variant. If not,
            loading it needs ``temperature=`` or ``allow_uncalibrated=True``.
        temperature: The declared temperature of this variant, or ``None``.
        is_default: Whether this variant is the one used when no ``dtype`` is given.
    """

    dtype: str
    size_bytes: int | None
    calibrated: bool
    temperature: float | None
    is_default: bool


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
        variants: Every variant of the weights the model offers (not only the one loaded), in
            the manifest's order; ``None`` for the test scorer.
        max_options: The most options of a ``Choice`` the model answers; a question with more is
            refused with :class:`UnsupportedRequestError`. ``None`` if the model has no limit of
            its own.
        max_key_tokens: The most tokens of an option key (only for models that score the keys).
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
    variants: tuple[VariantInfo, ...] | None = None
    max_options: int | None = None
    max_key_tokens: int | None = None


def _info(raw: dict[str, Any]) -> ModelInfo:
    tasks = raw.get("tasks")
    variants = raw.get("variants")
    return ModelInfo(
        **{
            **raw,
            "tasks": None if tasks is None else tuple(tasks),
            "variants": (
                None if variants is None else tuple(VariantInfo(**v) for v in variants)
            ),
        }
    )


def list_models() -> tuple[ModelInfo, ...]:
    """The models this version of archai-jev knows by name, with the default marked.

    One entry per name, at its current revision; an older or newer revision of the same name is
    loaded by asking for it (``from_pretrained(name, revision=...)``). Static data: no network and
    no disk access.
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

    A ``Jev`` is immutable and safe to share between threads and coroutines. It answers one
    request at a time: concurrent calls wait their turn.

    ``Ctrl-C`` interrupts a running call (it is looked at between the model's rows, so within a
    fraction of a second), and cancelling an ``aask`` stops the work: a request that has not
    started never starts, one that has started stops at its next row. The ``Jev`` stays usable.
    """

    __slots__ = ("_executor", "_executor_lock", "_info", "_model")

    _model: Any
    _info: ModelInfo
    _executor: ThreadPoolExecutor | None
    _executor_lock: threading.Lock

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
        object.__setattr__(obj, "_executor", None)
        object.__setattr__(obj, "_executor_lock", threading.Lock())
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
        prefix_cache_mib: int | None = None,
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
            prefix_cache_mib: Memory, in MiB, for reusing what questions share (the fixed part of
                a task, and the state when several questions are asked about it). ``None`` uses
                ``ARCHAI_JEV_PREFIX_CACHE_MIB`` or 128; ``0`` turns the reuse off and decodes every
                question whole, as 0.1.0 did. Answers are the same within the tolerances of the
                model; asking many questions about one state is much faster.

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
        if prefix_cache_mib is not None:
            if isinstance(prefix_cache_mib, bool) or not isinstance(
                prefix_cache_mib, int
            ):
                raise TypeError(
                    "prefix_cache_mib must be an int or None, "
                    f"got {type(prefix_cache_mib).__name__}"
                )
            if prefix_cache_mib < 0:
                raise ValueError(
                    f"prefix_cache_mib must be 0 or more, got {prefix_cache_mib}"
                )
            if prefix_cache_mib >= 2**64:
                raise ValueError(f"prefix_cache_mib is too large: {prefix_cache_mib}")
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
            prefix_cache_mib,
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

    def cache_info(self) -> CacheInfo:
        """The memory of reused prefixes: its limit, what it holds and what it saved.

        Reading it is cheap and does not change anything. Everything is zero when the memory is
        off (``prefix_cache_mib=0``).
        """
        return CacheInfo(
            **dict(zip(CacheInfo.__slots__, self._model.cache_info(), strict=True))
        )

    def clear_cache(self) -> None:
        """Forget every saved prefix (the counters of :meth:`cache_info` stay).

        Answers never change because of it: the next requests only start from scratch. Use it to
        free memory, or if the texts you asked about are private and must not stay in memory.
        """
        self._model.clear_cache()

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
            KeyboardInterrupt: ``Ctrl-C`` while the model works (main thread only). The call
                stops between two rows of the model, with no result, and the ``Jev`` is still
                usable.
            RuntimeError: Called from a signal handler or a destructor that runs inside another
                call of this ``Jev`` (it would wait for itself).
        """
        return self._ask(state, questions, None)

    def _ask(
        self, state: Any, questions: Mapping[str, Any], token: _core._CancelToken | None
    ) -> Answers:
        pairs = _question_pairs(questions)
        return build_answers(self._model.ask(state, pairs, token))

    @overload
    def ask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[Answers]: ...

    @overload
    def ask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        *,
        return_exceptions: bool,
    ) -> list[Answers | JevError]: ...

    def ask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        *,
        return_exceptions: bool = False,
    ) -> list[Answers] | list[Answers | JevError]:
        """Same questions on several states, in order; every state is validated first.

        Args:
            states: The states, one list of answers each.
            questions: As in :meth:`ask`.
            return_exceptions: If ``True`` a state whose calculation fails (for example with a
                :class:`NumericalError`) gives its error object in its place and the others are
                still answered. If ``False`` (the default) the first failure ends the call. A
                state or question that is invalid always raises, before any calculation, and
                ``Ctrl-C`` always stops the call.

        Raises:
            InvalidStateError: A state is unsupported (the message starts with ``states[i]:``).
            KeyboardInterrupt: As in :meth:`ask`.
        """
        return self._ask_many(states, questions, None, return_exceptions)

    def _ask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        token: _core._CancelToken | None,
        return_exceptions: bool,
    ) -> list[Answers] | list[Answers | JevError]:
        pairs = _question_pairs(questions)
        if isinstance(states, (str, bytes, Mapping)):
            raise TypeError(
                f"states must be an iterable of states, got {type(states).__name__}"
            )
        listed = list(states)
        raw = self._model.ask_many(listed, pairs, token, return_exceptions)
        return [
            item if isinstance(item, JevError) else build_answers(item) for item in raw
        ]

    def _executor_for_async(self) -> ThreadPoolExecutor:
        """The threads that run the async calls of this ``Jev`` (made at the first use).

        Not the loop's default executor: a program that already keeps that one busy would make
        every ``aask`` wait for its own jobs, and cancelling a queued request could not be
        told apart from theirs.
        """
        with self._executor_lock:
            executor = self._executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=_ASYNC_WORKERS, thread_name_prefix="archai-jev"
                )
                # No explicit close: the executor ends its threads when it is collected, which is
                # when the Jev is (a request still waiting keeps the Jev alive).
                object.__setattr__(self, "_executor", executor)
            return executor

    async def _run_cancellable(self, work: Callable[[_core._CancelToken], Any]) -> Any:
        """Run ``work(token)`` in the executor; stop it if this coroutine is cancelled."""
        token = _core._CancelToken()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor_for_async(), work, token)
        try:
            return await future
        except asyncio.CancelledError:
            # A request still in the queue never starts; one that runs stops at its next row.
            token.set()
            raise
        except _StoppedError:
            raise asyncio.CancelledError from None

    async def aask(self, state: Any, questions: Mapping[str, Any]) -> Answers:
        """Like :meth:`ask`, in a worker thread so the event loop is not blocked.

        Cancelling the task stops the work: a request that has not started never starts, and one
        that is running stops at the next row of the model. The ``Jev`` stays usable.
        """
        answers: Answers = await self._run_cancellable(
            lambda token: self._ask(state, questions, token)
        )
        return answers

    @overload
    async def aask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[Answers]: ...

    @overload
    async def aask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        *,
        return_exceptions: bool,
    ) -> list[Answers | JevError]: ...

    async def aask_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any],
        *,
        return_exceptions: bool = False,
    ) -> list[Answers] | list[Answers | JevError]:
        """Like :meth:`ask_many`, in a worker thread; cancelling the task stops the work."""
        listed = (
            list(states) if not isinstance(states, (str, bytes, Mapping)) else states
        )
        result: list[Answers] | list[Answers | JevError] = await self._run_cancellable(
            lambda token: self._ask_many(listed, questions, token, return_exceptions)
        )
        return result

    def evaluate(
        self,
        dataset: Iterable[LabelledRecord | Mapping[str, Any]],
        *,
        n_bins: int = 10,
        bootstrap_samples: int = 0,
        seed: int = 0,
    ) -> EvaluationReport:
        """How accurate **and** calibrated the model is on your labelled data.

        The whole dataset is validated before the model is asked anything. Each record goes
        through the same path and the same checks as :meth:`ask`; the probabilities measured are
        the ones the model returns, with its current temperature. All or nothing: a failure
        (``NumericalError``, ``UnsupportedRequestError`` naming the record, ``Ctrl-C``) gives no
        partial metrics. The model is not changed.

        The confidence of the ECE is the largest probability of each row, not the ``confidence``
        field of an answer (they are normalised differently). For a ``YesNo`` the row is
        ``[1 - p, p]``.

        Args:
            dataset: An iterable of :class:`~archai_jev.evaluation.LabelledRecord` or of ``dict``
                in the format of :func:`~archai_jev.evaluation.load_dataset`.
            n_bins: Bins of the reliability diagram, an integer in ``1..=1000``.
            bootstrap_samples: If greater than 0, 95 % intervals (``ci95``) from resampling whole
                groups; 0 for none.
            seed: Seed of the bootstrap, an integer in ``0 <= seed < 2**64``.

        Returns:
            An :class:`~archai_jev.evaluation.EvaluationReport`.

        Raises:
            InvalidDatasetError: The dataset is empty or a record or label is not valid.
            InvalidStateError: A state is not usable (the message names the record).
            InvalidQuestionError: A question is not valid (the message names the record).
            UnsupportedRequestError: The model cannot answer a record (the message names it).
            NumericalError: The model gave unusable numbers.
            ValueError: ``n_bins``, ``bootstrap_samples`` or ``seed`` is out of range.
            KeyboardInterrupt: ``Ctrl-C`` between two records; no result.

        Warns:
            UserWarning: Some of the records were also used to calibrate this model: the
                metrics are optimistic.

        Examples:
            >>> from archai_jev import Jev, YesNo
            >>> from archai_jev.evaluation import LabelledRecord
            >>> from archai_jev.testing import MockScorer
            >>> rows = [LabelledRecord(f"s{i}", {"ok": YesNo("Fine?")}, {"ok": True}) for i in range(4)]
            >>> report = Jev.from_scorer(MockScorer(1)).evaluate(rows)
            >>> report.n_records, sorted(report.by_type)
            (4, ['yes_no'])
        """
        return self._evaluate_prepared(
            _prepare(dataset), n_bins, bootstrap_samples, seed, None
        )

    def _evaluate_prepared(
        self,
        prepared: Any,
        n_bins: int,
        bootstrap_samples: int,
        seed: int,
        token: _core._CancelToken | None,
    ) -> EvaluationReport:
        raw = self._model.evaluate(prepared, n_bins, bootstrap_samples, seed, token)
        report = evaluation_from(raw, self._info)
        if report.calibration_overlap:
            prefix = (
                f"{report.calibration_overlap} of {report.n_records} records were also"
            )
            for note in report.notes:
                if note.startswith(prefix):
                    warnings.warn(note, UserWarning, stacklevel=3)
        return report

    def calibrate(
        self,
        dataset: Iterable[LabelledRecord | Mapping[str, Any]],
        *,
        folds: int = 5,
        seed: int = 0,
        question_types: Collection[str] | None = None,
        bounds: tuple[float, float] = (0.01, 100.0),
    ) -> CalibrationStudy:
        """Find one global temperature for this model on your data, and measure what it gains.

        The raw logits of every selected question are computed once. The temperature ``T*`` is
        fitted on all of them (:func:`~archai_jev.fit_temperature`; it is **absolute**, whatever
        the model's current temperature is). To say honestly what it gains, groups of records are
        spread over ``folds`` folds and each fold is measured with a temperature fitted on the
        *other* folds (``out_of_fold``); ``in_sample`` is optimistic and only shows how much.
        The study does not change the model: apply it with :meth:`with_temperature`.

        Args:
            dataset: As for :meth:`evaluate`.
            folds: Number of folds, an integer in ``2..=50``; at least as many groups are needed.
            seed: Seed of the assignment of groups to folds.
            question_types: A collection of ``"choice"``, ``"score"``, ``"yes_no"`` to calibrate
                on only those kinds; ``None`` for all.
            bounds: Limits for the temperature, as in :func:`~archai_jev.fit_temperature`.

        Returns:
            A :class:`~archai_jev.evaluation.CalibrationStudy`.

        Raises:
            CalibrationError: The data do not determine a temperature inside ``bounds`` (the
                message of :func:`~archai_jev.fit_temperature`, with ``fold k: `` when it is a
                fold that cannot be fitted). No study is returned.
            InvalidDatasetError: The dataset is empty or a record or label is not valid.
            UnsupportedRequestError: The model cannot answer a record (the message names it).
            NumericalError: The model gave unusable numbers.
            TypeError: ``question_types`` is a ``str``.
            ValueError: ``folds``, ``seed``, ``question_types`` or ``bounds`` is out of range, or
                there are fewer groups than folds.
            KeyboardInterrupt: ``Ctrl-C`` between two records; no result.

        Examples:
            >>> from archai_jev import Jev, YesNo
            >>> from archai_jev.evaluation import LabelledRecord
            >>> from archai_jev.testing import MockScorer
            >>> rows = [LabelledRecord(f"s{i}", {"ok": YesNo("Fine?")}, {"ok": i % 4 != 0}) for i in range(60)]
            >>> study = Jev.from_scorer(MockScorer.scripted([[0.0, 2.0]])).calibrate(rows, folds=3)
            >>> study.folds, len(study.fold_temperatures), round(study.temperature, 2)
            (3, 3, 1.82)
        """
        if isinstance(question_types, str):
            raise TypeError(
                "question_types must be a collection of names such as ['yes_no'], not a str"
            )
        types = None if question_types is None else [str(t) for t in question_types]
        raw = self._model.calibrate(_prepare(dataset), folds, seed, types, bounds, None)
        return study_from(raw, folds, seed, bounds)

    def with_temperature(
        self, calibration: float | TemperatureFit | CalibrationStudy
    ) -> Jev:
        """A **new** ``Jev`` that applies this temperature; the original does not change.

        The new one shares the model (nothing is copied), has ``model_info.temperature`` equal to
        the new value and ``calibrated=True`` on every answer: it is a calibration you provided.
        The temperature is absolute, so ``jev.with_temperature(a).with_temperature(b)`` is just
        ``b``. Given a :class:`~archai_jev.evaluation.CalibrationStudy` it also remembers which
        records fitted it, so :meth:`evaluate` can warn if you measure on them; a number or a
        :class:`~archai_jev.TemperatureFit` has no such origin.

        Args:
            calibration: A temperature, a ``TemperatureFit`` or a ``CalibrationStudy``.

        Returns:
            A ``Jev``.

        Raises:
            TypeError: ``calibration`` is none of the three (a ``bool`` is not a temperature).
            ValueError: The temperature is not a finite number greater than 0.

        Examples:
            >>> from archai_jev import Jev
            >>> from archai_jev.testing import MockScorer
            >>> jev = Jev.from_scorer(MockScorer(1))
            >>> tuned = jev.with_temperature(2.5)
            >>> (tuned.model_info.temperature, tuned.model_info.calibrated, jev.model_info.temperature)
            (2.5, True, 1.0)
        """
        provenance: list[int] | None = None
        if isinstance(calibration, CalibrationStudy):
            temperature = calibration.temperature
            provenance = list(calibration.record_fingerprints)
        elif isinstance(calibration, TemperatureFit):
            temperature = calibration.temperature
        elif isinstance(calibration, (int, float)) and not isinstance(
            calibration, bool
        ):
            temperature = float(calibration)
        else:
            raise TypeError(
                "calibration must be a temperature (float), a TemperatureFit or a "
                f"CalibrationStudy, got {type(calibration).__name__}"
            )
        model = self._model.with_temperature(temperature, provenance)
        info = dataclasses.replace(
            self._info,
            temperature=temperature,
            calibrated=True,
            calibration_source="user",
        )
        return self._create(model, info)

    def __repr__(self) -> str:
        return f"Jev(model={self._info.name!r})"
