"""Result classes of ``evaluate``, ``calibrate`` and ``compare``, with ``table()`` and ``to_dict()``.

All the numbers come from Rust; this module only gives them names, immutable classes and a text
layout. The layout of ``table()`` is part of the contract: fixed columns, three decimals, ``-``
where a value does not exist.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ._calibration import CalibrationReport, ReliabilityBin, TemperatureFit

if TYPE_CHECKING:
    from ._jev import ModelInfo

__all__ = [
    "CalibrationStudy",
    "Comparison",
    "EvaluationReport",
    "QuestionReport",
    "StudyByType",
    "TypeReport",
]

SCHEMA = "archai-jev.evaluation/1"
COMPARISON_SCHEMA = "archai-jev.comparison/1"
_KINDS = ("choice", "score", "yes_no")


# --------------------------------------------------------------------------- text layout


def _num(x: float | None) -> str:
    if x is None:
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.3f}"


def _header(first: str, width: int) -> str:
    return (
        f"{first:<{width}} {'n':>6} {'acc':>7} {'conf':>7} {'ece':>7} "
        f"{'brier':>7} {'logloss':>9}"
    )


def _line(label: str, width: int, r: CalibrationReport, suffix: str = "") -> str:
    return (
        f"{label:<{width}} {r.n:>6} {_num(r.accuracy):>7} {_num(r.mean_confidence):>7} "
        f"{_num(r.ece):>7} {_num(r.brier):>7} {_num(r.log_loss):>9}{suffix}"
    )


def _notes(notes: tuple[str, ...]) -> list[str]:
    return ["notes:", *[f"  - {n}" for n in notes]] if notes else []


# --------------------------------------------------------------------------- dict forms


def _bin_dict(b: ReliabilityBin) -> dict[str, Any]:
    return {
        "lower": b.lower,
        "upper": b.upper,
        "count": b.count,
        "mean_confidence": b.mean_confidence,
        "accuracy": b.accuracy,
    }


def _report_dict(r: CalibrationReport) -> dict[str, Any]:
    return {
        "n": r.n,
        "accuracy": r.accuracy,
        "mean_confidence": r.mean_confidence,
        "ece": r.ece,
        "mce": r.mce,
        "brier": r.brier,
        "log_loss": r.log_loss,
        "bins": [_bin_dict(b) for b in r.bins],
    }


def _ci_dict(
    ci: Mapping[str, tuple[float, float]] | None,
) -> dict[str, list[float]] | None:
    return None if ci is None else {k: [lo, hi] for k, (lo, hi) in ci.items()}


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _restore(cls: type, fields: dict[str, Any]) -> Any:
    return cls(**fields)


def _reduce(obj: Any) -> tuple[Any, ...]:
    """Pickle a result class: read-only mappings travel as plain dicts and are frozen again."""
    fields = {
        f.name: (
            dict(getattr(obj, f.name))
            if isinstance(getattr(obj, f.name), MappingProxyType)
            else getattr(obj, f.name)
        )
        for f in dataclasses.fields(obj)
    }
    return (_restore, (type(obj), fields))


# --------------------------------------------------------------------------- scopes


@dataclass(frozen=True, kw_only=True)
class TypeReport:
    """Metrics for one kind of question (or, as :class:`QuestionReport`, for one question name).

    The fields of :class:`~archai_jev.calibration.CalibrationReport` plus the extras of the
    kind: ``expected_level_mae`` (Score), ``positive_rate`` and ``mean_probability`` (YesNo),
    ``None`` where they do not apply. ``ci95`` maps ``accuracy``, ``ece``, ``brier`` and
    ``log_loss`` to a ``(lower, upper)`` interval and is ``None`` unless ``bootstrap_samples``
    was given. The confidence of the ECE is the largest probability, not the ``confidence`` field
    of an answer.
    """

    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    mce: float
    brier: float
    log_loss: float
    bins: tuple[ReliabilityBin, ...]
    expected_level_mae: float | None = None
    positive_rate: float | None = None
    mean_probability: float | None = None
    ci95: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.ci95 is not None:
            object.__setattr__(self, "ci95", _frozen(self.ci95))

    def _hash_key(self) -> tuple[Any, ...]:
        ci = None if self.ci95 is None else tuple(self.ci95.items())
        return (
            self.n,
            self.accuracy,
            self.mean_confidence,
            self.ece,
            self.mce,
            self.brier,
            self.log_loss,
            self.bins,
            self.expected_level_mae,
            self.positive_rate,
            self.mean_probability,
            ci,
        )

    def __hash__(self) -> int:
        return hash(self._hash_key())

    def __reduce__(self) -> tuple[Any, ...]:
        return _reduce(self)

    def _as_report(self) -> CalibrationReport:
        return CalibrationReport(
            n=self.n,
            accuracy=self.accuracy,
            mean_confidence=self.mean_confidence,
            ece=self.ece,
            mce=self.mce,
            brier=self.brier,
            log_loss=self.log_loss,
            bins=self.bins,
        )

    def _extras_text(self) -> str:
        if self.expected_level_mae is not None:
            return f"  (expected-level MAE {_num(self.expected_level_mae)})"
        if self.positive_rate is not None:
            return f"  (positive rate {_num(self.positive_rate)})"
        return ""

    def to_dict(self) -> dict[str, Any]:
        """A nested dict of plain Python values with the same keys as the fields."""
        out = _report_dict(self._as_report())
        out["expected_level_mae"] = self.expected_level_mae
        out["positive_rate"] = self.positive_rate
        out["mean_probability"] = self.mean_probability
        out["ci95"] = _ci_dict(self.ci95)
        return out


@dataclass(frozen=True, kw_only=True)
class QuestionReport(TypeReport):
    """Metrics for one question name; ``type`` is ``"choice"``, ``"score"`` or ``"yes_no"``."""

    type: str = ""

    def _hash_key(self) -> tuple[Any, ...]:
        return (*super()._hash_key(), self.type)

    def __hash__(self) -> int:
        return hash(self._hash_key())

    def to_dict(self) -> dict[str, Any]:
        """A nested dict of plain Python values with the same keys as the fields."""
        out = super().to_dict()
        out["type"] = self.type
        return out


def _bins(raw: list[tuple[Any, ...]]) -> tuple[ReliabilityBin, ...]:
    return tuple(ReliabilityBin(*b) for b in raw)


def report_from(raw: tuple[Any, ...]) -> CalibrationReport:
    n, accuracy, mean_confidence, ece, mce, brier, log_loss, bins = raw
    return CalibrationReport(
        n=n,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        ece=ece,
        mce=mce,
        brier=brier,
        log_loss=log_loss,
        bins=_bins(bins),
    )


_CI_KEYS = ("accuracy", "ece", "brier", "log_loss")


def _scope_fields(raw: tuple[Any, ...]) -> dict[str, Any]:
    report, extras, ci = raw
    r = report_from(report)
    fields: dict[str, Any] = {
        "n": r.n,
        "accuracy": r.accuracy,
        "mean_confidence": r.mean_confidence,
        "ece": r.ece,
        "mce": r.mce,
        "brier": r.brier,
        "log_loss": r.log_loss,
        "bins": r.bins,
        "ci95": None if ci is None else dict(zip(_CI_KEYS, ci, strict=True)),
    }
    fields.update(dict(extras))
    return fields


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True, kw_only=True)
class EvaluationReport:
    """How a model does on a labelled dataset (:meth:`archai_jev.Jev.evaluate`).

    Attributes:
        model: The :class:`~archai_jev.ModelInfo` of the model that was evaluated.
        n_records: Number of records.
        n_questions: Number of questions over all records.
        overall: Metrics over all rows of all kinds.
        by_type: Metrics per kind, only the kinds present, in the order choice, score, yes_no.
        by_question: Metrics per question name, in order of first appearance.
        dataset_fingerprint: 16 hexadecimal digits that identify the dataset (a heuristic to
            warn about reuse, not a security hash).
        calibration_overlap: How many records were also used to calibrate this model, or
            ``None`` if its calibration has no known origin.
        notes: Factual warnings, deterministic; they never change a number.
    """

    model: ModelInfo
    n_records: int
    n_questions: int
    overall: CalibrationReport
    by_type: Mapping[str, TypeReport]
    by_question: Mapping[str, QuestionReport]
    dataset_fingerprint: str
    calibration_overlap: int | None
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_type", _frozen(self.by_type))
        object.__setattr__(self, "by_question", _frozen(self.by_question))

    def __reduce__(self) -> tuple[Any, ...]:
        return _reduce(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.model,
                self.n_records,
                self.n_questions,
                self.overall,
                tuple(self.by_type.items()),
                tuple(self.by_question.items()),
                self.dataset_fingerprint,
                self.calibration_overlap,
                self.notes,
            )
        )

    def table(self) -> str:
        """The report as text with fixed columns, three decimals and ``-`` for a missing value."""
        names = [f"question {q}" for q in self.by_question]
        width = max([12, *[len(n) for n in names]])
        m = self.model
        lines = [
            f"model: {m.name} (T={m.temperature:.3f}, calibrated={m.calibrated})   "
            f"records: {self.n_records}   questions: {self.n_questions}   "
            f"dataset: {self.dataset_fingerprint}",
            _header("scope", width),
            _line("overall", width, self.overall),
        ]
        for kind, rep in self.by_type.items():
            lines.append(_line(kind, width, rep._as_report(), rep._extras_text()))
        for name, rep in self.by_question.items():
            lines.append(
                _line(f"question {name}", width, rep._as_report(), rep._extras_text())
            )
        lines.extend(_notes(self.notes))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A nested dict of plain Python values (``log_loss`` may be ``inf``; no ``NaN``)."""
        return {
            "schema": SCHEMA,
            "model": _plain(dataclasses.asdict(self.model)),
            "n_records": self.n_records,
            "n_questions": self.n_questions,
            "overall": _report_dict(self.overall),
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "by_question": {k: v.to_dict() for k, v in self.by_question.items()},
            "dataset_fingerprint": self.dataset_fingerprint,
            "calibration_overlap": self.calibration_overlap,
            "notes": list(self.notes),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def evaluation_from(raw: tuple[Any, ...], model: ModelInfo) -> EvaluationReport:
    (
        counts,
        overall,
        by_type,
        by_question,
        (fingerprint, overlap, notes),
        _calibration,
    ) = raw
    return EvaluationReport(
        model=model,
        n_records=counts[0],
        n_questions=counts[1],
        overall=report_from(overall[0]),
        by_type={k: TypeReport(**_scope_fields(s)) for k, s in by_type},
        by_question={
            name: QuestionReport(type=kind, **_scope_fields(s))
            for name, kind, s in by_question
        },
        dataset_fingerprint=fingerprint,
        calibration_overlap=overlap,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- calibration


@dataclass(frozen=True, kw_only=True)
class StudyByType:
    """The two arms of a :class:`CalibrationStudy` for one kind of question."""

    current: CalibrationReport
    out_of_fold: CalibrationReport


@dataclass(frozen=True, kw_only=True)
class CalibrationStudy:
    """The result of :meth:`archai_jev.Jev.calibrate`.

    ``temperature`` is the new global temperature (absolute, fitted on every row). Three arms say
    what it gains: ``current`` is the model as it is, ``in_sample`` the new temperature on the very
    rows that fitted it (optimistic), ``out_of_fold`` each row measured with a temperature fitted
    on the *other* folds, the honest estimate. Folds are made of whole groups.

    Attributes:
        temperature: The temperature to apply (see :meth:`archai_jev.Jev.with_temperature`).
        fit: The :class:`~archai_jev.TemperatureFit` behind it.
        fold_temperatures: The temperature fitted without each fold, in fold order.
        current: Metrics of the rows as the model serves them now.
        in_sample: The new temperature on the rows that fitted it.
        out_of_fold: Each row with the temperature of the other folds.
        by_type: ``current`` and ``out_of_fold`` per kind present.
        n: Number of rows used (one per selected question).
        n_groups: Number of groups that have rows.
        folds: Number of folds.
        seed: Seed of the assignment of groups to folds.
        dataset_fingerprint: Fingerprint of the dataset.
        record_fingerprints: Fingerprints of the records that contributed rows.
        notes: Factual warnings.
    """

    temperature: float
    fit: TemperatureFit
    fold_temperatures: tuple[float, ...]
    current: CalibrationReport
    in_sample: CalibrationReport
    out_of_fold: CalibrationReport
    by_type: Mapping[str, StudyByType]
    n: int
    n_groups: int
    folds: int
    seed: int
    dataset_fingerprint: str
    record_fingerprints: tuple[int, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_type", _frozen(self.by_type))

    def __reduce__(self) -> tuple[Any, ...]:
        return _reduce(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.temperature,
                self.fit,
                self.fold_temperatures,
                self.current,
                self.in_sample,
                self.out_of_fold,
                tuple(self.by_type.items()),
                self.n,
                self.n_groups,
                self.folds,
                self.seed,
                self.dataset_fingerprint,
                self.record_fingerprints,
                self.notes,
            )
        )

    def table(self) -> str:
        """The arms as text with the same columns as :meth:`EvaluationReport.table`."""
        arms: list[tuple[str, CalibrationReport]] = [
            ("current", self.current),
            ("in_sample", self.in_sample),
            ("out_of_fold", self.out_of_fold),
        ]
        for kind, pair in self.by_type.items():
            arms.append((f"{kind} current", pair.current))
            arms.append((f"{kind} out_of_fold", pair.out_of_fold))
        width = max([12, *[len(label) for label, _ in arms]])
        lines = [
            f"temperature: {self.temperature:.3f}   folds: {self.folds}   "
            f"groups: {self.n_groups}   rows: {self.n}   dataset: {self.dataset_fingerprint}",
            _header("arm", width),
            *[_line(label, width, rep) for label, rep in arms],
            *_notes(self.notes),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A nested dict of plain Python values."""
        return {
            "schema": "archai-jev.calibration-study/1",
            "temperature": self.temperature,
            "fit": dataclasses.asdict(self.fit) | {"bounds": list(self.fit.bounds)},
            "fold_temperatures": list(self.fold_temperatures),
            "current": _report_dict(self.current),
            "in_sample": _report_dict(self.in_sample),
            "out_of_fold": _report_dict(self.out_of_fold),
            "by_type": {
                k: {
                    "current": _report_dict(v.current),
                    "out_of_fold": _report_dict(v.out_of_fold),
                }
                for k, v in self.by_type.items()
            },
            "n": self.n,
            "n_groups": self.n_groups,
            "folds": self.folds,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "record_fingerprints": list(self.record_fingerprints),
            "notes": list(self.notes),
        }


def study_from(
    raw: tuple[Any, ...], folds: int, seed: int, bounds: tuple[float, float]
) -> CalibrationStudy:
    (
        (temperature, fit, fold_ts),
        (cur, ins, oof, by_type),
        (n, groups, fp, fps),
        (
            notes,
            _calibration,
        ),
    ) = raw
    t, before, after, fit_n, iterations = fit
    return CalibrationStudy(
        temperature=temperature,
        fit=TemperatureFit(
            temperature=t,
            loss_before=before,
            loss_after=after,
            n=fit_n,
            iterations=iterations,
            bounds=(float(bounds[0]), float(bounds[1])),
        ),
        fold_temperatures=tuple(fold_ts),
        current=report_from(cur),
        in_sample=report_from(ins),
        out_of_fold=report_from(oof),
        by_type={
            kind: StudyByType(current=report_from(c), out_of_fold=report_from(o))
            for kind, c, o in by_type
        },
        n=n,
        n_groups=groups,
        folds=folds,
        seed=seed,
        dataset_fingerprint=fp,
        record_fingerprints=tuple(fps),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- comparison


@dataclass(frozen=True, kw_only=True)
class Comparison:
    """Several models on the same dataset (:func:`archai_jev.evaluation.compare`).

    ``reports`` maps the label of each model to its :class:`EvaluationReport`, in the order the
    models were given. There is no ranking: the table puts the numbers side by side.
    """

    reports: Mapping[str, EvaluationReport]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reports", _frozen(self.reports))

    def __hash__(self) -> int:
        return hash(tuple(self.reports.items()))

    def __reduce__(self) -> tuple[Any, ...]:
        return _reduce(self)

    def table(self) -> str:
        """One row per model for ``overall``, then one block per kind of question."""
        reports = list(self.reports.items())
        first = reports[0][1]
        kinds = [k for k in _KINDS if any(k in r.by_type for _, r in reports)]
        labels = [label for label, _ in reports] + [
            f"{label} [{k}]" for k in kinds for label, r in reports if k in r.by_type
        ]
        width = max([12, *[len(x) for x in labels]])
        lines = [
            f"dataset: {first.dataset_fingerprint}   records: {first.n_records}   "
            f"questions: {first.n_questions}",
            _header("model", width),
        ]
        for label, r in reports:
            lines.append(_line(label, width, r.overall))
        for k in kinds:
            for label, r in reports:
                if k in r.by_type:
                    lines.append(
                        _line(f"{label} [{k}]", width, r.by_type[k]._as_report())
                    )
        notes = [f"  - [{label}] {n}" for label, r in reports for n in r.notes]
        if notes:
            lines.extend(["notes:", *notes])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A nested dict of plain Python values."""
        first = next(iter(self.reports.values()))
        return {
            "schema": COMPARISON_SCHEMA,
            "dataset_fingerprint": first.dataset_fingerprint,
            "reports": {k: v.to_dict() for k, v in self.reports.items()},
        }
