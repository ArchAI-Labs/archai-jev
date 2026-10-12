"""The public surface of the evaluation feature: names, classes, docs, the example of the spec."""

from __future__ import annotations

import doctest
import importlib.metadata
import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import pytest

import archai_jev
from _evaluation_data import as_dict, drawn_records, scripted
from archai_jev import InvalidDatasetError, Jev, JevError
from archai_jev import evaluation as ev

EVALUATION_NAMES = [
    "CalibrationStudy",
    "Comparison",
    "EvaluationReport",
    "LabelledRecord",
    "QuestionReport",
    "TypeReport",
    "compare",
    "load_dataset",
    "split_dataset",
]


def test_all_snapshots() -> None:
    assert sorted(ev.__all__) == EVALUATION_NAMES
    assert len(ev.__all__) == len(set(ev.__all__)) == 9
    for name in ev.__all__:
        assert hasattr(ev, name), name
    assert "InvalidDatasetError" in archai_jev.__all__
    assert "evaluation" not in archai_jev.__all__
    for method in ("evaluate", "calibrate", "with_temperature"):
        assert callable(getattr(Jev, method))


def test_invalid_dataset_error_is_jev_and_value_error() -> None:
    assert issubclass(InvalidDatasetError, JevError)
    assert issubclass(InvalidDatasetError, ValueError)
    assert InvalidDatasetError.__module__ == "archai_jev"
    assert (InvalidDatasetError.__doc__ or "").strip()


def test_result_classes_contract() -> None:
    rows = drawn_records(60, 2.0, 1, group_size=2)
    jev = scripted()
    report = jev.evaluate(rows, bootstrap_samples=10)
    study = jev.calibrate(rows, folds=3)
    comparison = ev.compare({"a": jev}, rows)
    for obj in (
        report,
        report.by_type["choice"],
        report.by_question["team"],
        study,
        comparison,
        rows[0],
    ):
        assert pickle.loads(pickle.dumps(obj)) == obj
        assert hash(obj) == hash(pickle.loads(pickle.dumps(obj)))
        assert repr(obj)
    for obj in (report, study, comparison):
        assert isinstance(obj.table(), str) and obj.to_dict()
        json.dumps(obj.to_dict(), allow_nan=True)


def _public_functions() -> list[tuple[str, Any]]:
    return [
        ("load_dataset", ev.load_dataset),
        ("split_dataset", ev.split_dataset),
        ("compare", ev.compare),
        ("Jev.evaluate", Jev.evaluate),
        ("Jev.calibrate", Jev.calibrate),
        ("Jev.with_temperature", Jev.with_temperature),
    ]


@pytest.mark.parametrize(
    ("name", "obj"), _public_functions(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_docs_and_doctests(name: str, obj: Any) -> None:
    doc = obj.__doc__ or ""
    assert "Raises:" in doc and ">>>" in doc, name
    examples = doctest.DocTestParser().get_examples(doc)
    test = doctest.DocTest(examples, {**vars(ev), "Jev": Jev}, name, None, 0, doc)
    runner = doctest.DocTestRunner(verbose=False)
    runner.run(test)
    assert runner.failures == 0, name


@pytest.mark.parametrize(
    "cls",
    [
        ev.LabelledRecord,
        ev.EvaluationReport,
        ev.TypeReport,
        ev.QuestionReport,
        ev.CalibrationStudy,
        ev.Comparison,
    ],
)
def test_classes_are_documented(cls: type) -> None:
    assert (cls.__doc__ or "").strip()


def test_spec_example(tmp_path: Path) -> None:
    # the example of §6 with the scripted model: labels drawn so that a temperature exists
    path = tmp_path / "tickets.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(as_dict(r)) for r in drawn_records(200, 2.0, 5, group_size=2)
        ),
        encoding="utf-8",
    )
    dataset = ev.load_dataset(path)
    calib, held_out = ev.split_dataset(dataset, test_fraction=0.5, seed=0)
    jev = scripted()
    assert "overall" in jev.evaluate(held_out).table()
    study = jev.calibrate(calib, folds=5, seed=0)
    assert "out_of_fold" in study.table()
    tuned = jev.with_temperature(study)
    assert (
        tuned.model_info.calibrated
        and tuned.model_info.temperature == study.temperature
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # held_out was not used to calibrate: no warning
        comparison = ev.compare({"base": jev, "base + T": tuned}, held_out)
    assert all(r.calibration_overlap in (None, 0) for r in comparison.reports.values())
    assert (
        comparison.reports["base + T"].overall.ece
        < comparison.reports["base"].overall.ece
    )
    assert "base + T" in comparison.table()


def test_no_runtime_dependencies() -> None:
    requires = importlib.metadata.requires("archai-jev")
    assert not requires, requires


def test_core_does_not_name_pyo3() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "evaluation"
    for source in root.glob("*.rs"):
        assert "pyo3" not in source.read_text(encoding="utf-8"), source.name
    assert math.isfinite(1.0)
