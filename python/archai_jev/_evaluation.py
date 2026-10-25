"""Labelled data: :class:`LabelledRecord`, :func:`load_dataset`, :func:`split_dataset` and
:func:`compare` (the methods ``evaluate``, ``calibrate`` and ``with_temperature`` are on ``Jev``).

Records are validated in Rust, all of them, before any model is asked anything. Nothing is
skipped and no label is interpreted: a label is exactly right or an error naming the file and
line (``data.jsonl:41``) or the record (``record 3``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from . import _core
from ._exceptions import InvalidQuestionError, JevError
from ._questions import Choice, Score, YesNo
from ._reports import Comparison, EvaluationReport
from ._values import canon, freeze, thaw

if TYPE_CHECKING:
    from ._jev import Jev

__all__ = ["LabelledRecord", "compare", "load_dataset", "split_dataset"]

Question = Choice | Score | YesNo
Label = str | bool | int
T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class LabelledRecord:
    """A state, the questions asked about it and the right answer to each.

    Labels are exact and never interpreted: a ``Choice`` label is the key of the right option
    (case included), a ``YesNo`` label is ``True`` or ``False`` (not ``1``), a ``Score`` label is
    the index of the right level as an ``int`` counted from 0 (not ``2.0``, not ``True``).

    Attributes:
        state: A string, a dict or a list (nested JSON-like values), copied when the record is
            built.
        questions: Name to :class:`~archai_jev.Choice`, :class:`~archai_jev.Score` or
            :class:`~archai_jev.YesNo`, in order.
        labels: Name to label; the same names as ``questions``.
        id: Optional, for messages (at most 200 characters).
        group: Optional. Records with the same group are never separated by
            :func:`split_dataset` or by the folds of ``Jev.calibrate`` (a customer's several
            tickets, the variants of one document).

    Examples:
        >>> from archai_jev import YesNo
        >>> r = LabelledRecord("Late again", {"angry": YesNo("Is the customer angry?")}, {"angry": True})
        >>> r.labels["angry"]
        True
    """

    state: Any
    questions: Mapping[str, Any]
    labels: Mapping[str, Any]
    id: str | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", freeze(self.state))
        object.__setattr__(self, "questions", MappingProxyType(dict(self.questions)))
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    def _key(self) -> tuple[Any, ...]:
        return (
            canon(self.state),
            tuple(self.questions.items()),
            tuple((k, canon(v)) for k, v in self.labels.items()),
            self.id,
            self.group,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is LabelledRecord and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            LabelledRecord,
            (
                thaw(self.state),
                dict(self.questions),
                dict(self.labels),
                self.id,
                self.group,
            ),
        )


def _pairs(record: LabelledRecord, n: int) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for name, q in record.questions.items():
        if not isinstance(name, str):
            raise InvalidQuestionError(
                f"record {n}: question name must be a str, got {type(name).__name__}"
            )
        if not isinstance(q, (Choice, Score, YesNo)):
            raise InvalidQuestionError(
                f'record {n}: question "{name}" must be a Choice, Score or YesNo, '
                f"got {type(q).__name__}"
            )
        pairs.append((name, q._q))
    return pairs


def _item(item: Any, n: int) -> Any:
    if isinstance(item, LabelledRecord):
        return (
            thaw(item.state),
            _pairs(item, n),
            list(item.labels.items()),
            item.id,
            item.group,
        )
    return item


def _prepare(dataset: Any) -> _core._Dataset:
    """Validate a dataset (an iterable of records or dicts) once."""
    if isinstance(dataset, _core._Dataset):
        return dataset
    if isinstance(dataset, (str, bytes, Mapping)):
        raise TypeError(
            "dataset must be an iterable of LabelledRecord or dict, got "
            f"{type(dataset).__name__} (use load_dataset(path) for a file)"
        )
    items = [_item(item, n) for n, item in enumerate(dataset)]
    return _core._dataset_from_items(items)


def _make_question(spec: Mapping[str, Any]) -> Question:
    kind = spec["type"]
    if kind == "choice":
        return Choice(spec["instructions"], spec["criteria"])
    if kind == "score":
        return Score(spec["instructions"], spec["criteria"])
    return YesNo(spec["instructions"], spec.get("criteria"))


def load_dataset(path: str | os.PathLike[str]) -> list[LabelledRecord]:
    """Read a JSONL dataset (the format of ``kev.data.load_records``): all of it, or an error.

    One JSON object per line with the request of the TypeSafe shape (``state`` and ``questions``
    with ``type``, ``instructions`` and ``criteria``) plus a ``label`` in every question and,
    optionally, ``id`` and ``group``. Blank lines are skipped. Any field not named here
    (``_meta``, ``src``, ``target``, ``model``…) is an error that lists the names.

    Args:
        path: The file, UTF-8 without byte-order mark.

    Returns:
        The records, in file order.

    Raises:
        FileNotFoundError: The file does not exist.
        InvalidDatasetError: The file is empty, a line is not valid JSON, a field is not
            supported, a label does not fit its question, or a name has two types. The message
            names the file and line.
        InvalidStateError: A state is not usable (the message names the line).
        InvalidQuestionError: A question is not valid (the message names the line).

    Examples:
        >>> import os, tempfile
        >>> line = '{"state": "late", "questions": {"angry": {"type": "noul", "instructions": "Angry?", "label": true}}}'
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "d.jsonl")
        ...     with open(path, "w", encoding="utf-8") as f:
        ...         _ = f.write(line)
        ...     records = load_dataset(path)
        >>> records[0].labels["angry"]
        True
    """
    name = os.fspath(path)
    if isinstance(name, bytes):
        name = os.fsdecode(name)
    with open(name, "rb") as handle:
        data = handle.read()
    _core._dataset_from_jsonl(data, name)  # validates every line, or raises
    records: list[LabelledRecord] = []
    # The very bytes that were validated, split as the Rust reader splits them.
    for raw in data.split(b"\n"):
        text = raw.removesuffix(b"\r").decode("utf-8")
        if not text.strip(" \t"):
            continue
        obj = json.loads(text)
        specs: Mapping[str, Mapping[str, Any]] = obj["questions"]
        records.append(
            LabelledRecord(
                obj["state"],
                {n: _make_question(q) for n, q in specs.items()},
                {n: q["label"] for n, q in specs.items()},
                obj.get("id"),
                obj.get("group"),
            )
        )
    return records


def split_dataset(
    dataset: Iterable[T],
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list[T], list[T]]:
    """Split a dataset into ``(train, test)`` by group: calibrate on ``train``, evaluate on ``test``.

    Every group goes whole to one side (a record without ``group`` is a group of its own). The
    split is deterministic for a ``seed`` (the same on every platform), keeps the original order of
    the records on each side and adds whole groups to ``test`` until it holds at least
    ``test_fraction`` of the records.

    Args:
        dataset: An iterable of :class:`LabelledRecord` or ``dict``; the same objects come back.
        test_fraction: Share of the records for ``test``, strictly between 0 and 1.
        seed: An integer in ``0 <= seed < 2**64``.

    Returns:
        Two lists, ``train`` and ``test``, with the objects that were given.

    Raises:
        InvalidDatasetError: The dataset is empty or a record is not valid.
        TypeError: ``test_fraction`` is not a number, or ``dataset`` is not an iterable of
            records.
        ValueError: ``test_fraction`` is outside ``(0, 1)``, ``seed`` is out of range, or the
            fraction leaves one side empty.

    Examples:
        >>> from archai_jev import YesNo
        >>> q = {"ok": YesNo("Fine?")}
        >>> rows = [LabelledRecord(f"s{i}", q, {"ok": True}, group=f"g{i // 2}") for i in range(10)]
        >>> train, test = split_dataset(rows, test_fraction=0.4, seed=1)
        >>> len(train) + len(test)
        10
        >>> {r.group for r in train}.isdisjoint({r.group for r in test})
        True
    """
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise TypeError(
            f"test_fraction must be a number, got {type(test_fraction).__name__}"
        )
    if isinstance(dataset, (str, bytes, Mapping)):
        _prepare(dataset)  # raises the TypeError that explains what is expected
    items = list(dataset)
    prepared = _prepare(items)
    train, test = _core._split(prepared, float(test_fraction), seed)
    return [items[i] for i in train], [items[i] for i in test]


def compare(
    models: Mapping[str, Jev],
    dataset: Iterable[LabelledRecord | Mapping[str, Any]],
    *,
    n_bins: int = 10,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> Comparison:
    """Evaluate several models on the **same** dataset and put the numbers side by side.

    The dataset is read and validated once. If a model refuses a record (for example the model of
    4 tasks and a question that is not one of them) or fails, the whole comparison fails with
    ``model {label}: …``: comparing models on different subsets would falsify the comparison.
    Restrict the dataset to what every model can do. There is no ranking.

    Args:
        models: Label to :class:`~archai_jev.Jev`, in the order to be shown.
        dataset: An iterable of :class:`LabelledRecord` or ``dict``.
        n_bins: Bins of the reliability diagram, an integer in ``1..=1000``.
        bootstrap_samples: Resamples for the 95 % intervals (0 for none).
        seed: Seed of the bootstrap.

    Returns:
        A :class:`Comparison`; its ``reports[label]`` equals ``models[label].evaluate(dataset)``.

    Raises:
        TypeError: A value of ``models`` is not a ``Jev``.
        ValueError: ``models`` is empty or an argument is out of range.
        InvalidDatasetError: The dataset is not valid.
        JevError: A model failed; the message starts with ``model {label}: ``.

    Examples:
        >>> from archai_jev import Jev, YesNo
        >>> from archai_jev.testing import MockScorer
        >>> rows = [LabelledRecord(f"s{i}", {"ok": YesNo("Fine?")}, {"ok": i % 2 == 0}) for i in range(4)]
        >>> a, b = Jev.from_scorer(MockScorer(1)), Jev.from_scorer(MockScorer(2))
        >>> list(compare({"a": a, "b": b}, rows).reports)
        ['a', 'b']
    """
    from ._jev import Jev  # imported late: _jev imports this module

    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty mapping of label to Jev")
    for label, jev in models.items():
        if not isinstance(jev, Jev):
            raise TypeError(
                f'models["{label}"] must be a Jev, got {type(jev).__name__}'
            )
    _core._check_evaluate_options(n_bins, bootstrap_samples, seed)
    prepared = _prepare(dataset)
    reports: dict[str, EvaluationReport] = {}
    for label, jev in models.items():
        try:
            reports[label] = jev._evaluate_prepared(
                prepared, n_bins, bootstrap_samples, seed, None
            )
        except (JevError, ValueError) as error:
            message = f"model {label}: {error}"
            try:
                raised = type(error)(message)
            except Exception:  # an exception class with another constructor
                raised = JevError(message)
            raise raised from error
    return Comparison(reports=reports)
