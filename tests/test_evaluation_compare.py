"""``compare`` and ``Comparison`` (spec 017 §4)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from _evaluation_data import QUESTIONS, drawn_records, record, scripted
from archai_jev import Jev, YesNo
from archai_jev.evaluation import Comparison, LabelledRecord, compare
from archai_jev.testing import Fault, MockScorer

DATA = Path(__file__).parent / "data" / "evaluation"


def models() -> dict[str, Jev]:
    base = scripted()
    return {"base": base, "tuned": base.with_temperature(2.0)}


def rows() -> list[LabelledRecord]:
    return drawn_records(30, 2.0, 4, group_size=2)


def test_reports_equal_individual_evaluations() -> None:
    comparison = compare(models(), rows())
    assert isinstance(comparison, Comparison)
    for label, jev in models().items():
        assert comparison.reports[label] == jev.evaluate(rows())


def test_order_and_fingerprint() -> None:
    swapped = dict(reversed(list(models().items())))
    assert list(compare(models(), rows()).reports) == ["base", "tuned"]
    assert list(compare(swapped, rows()).reports) == ["tuned", "base"]
    fingerprints = {
        r.dataset_fingerprint for r in compare(models(), rows()).reports.values()
    }
    assert len(fingerprints) == 1


def test_options_reach_every_model() -> None:
    comparison = compare(models(), rows(), n_bins=4, bootstrap_samples=20, seed=3)
    for report in comparison.reports.values():
        assert len(report.overall.bins) == 4
        assert report.by_type["choice"].ci95 is not None


def test_table_reference_string() -> None:
    text = compare(models(), drawn_records(30, 2.0, 4, group_size=2)).table()
    assert text == (DATA / "table_comparison.txt").read_text(encoding="utf-8").rstrip(
        "\n"
    )


def test_comparison_contract() -> None:
    a = compare(models(), rows())
    assert a == compare(models(), rows())
    assert hash(a) == hash(compare(models(), rows()))
    assert pickle.loads(pickle.dumps(a)) == a
    d = a.to_dict()
    assert d["schema"] == "archai-jev.comparison/1" and list(d["reports"]) == [
        "base",
        "tuned",
    ]


def test_a_failure_names_the_model_and_gives_no_comparison() -> None:
    faulty = Jev.from_scorer(MockScorer(1).with_fault(Fault.nan("team", 0)))
    with pytest.raises(Exception) as info:
        compare({"good": scripted(), "bad": faulty}, rows())
    assert type(info.value).__name__ == "NumericalError"
    assert str(info.value).startswith("model bad: record 0: ")


def test_empty_models_and_non_jev() -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        compare({}, rows())
    with pytest.raises(TypeError, match=r'models\["x"\] must be a Jev, got str'):
        compare({"x": "not a model"}, rows())  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty mapping"):
        compare([scripted()], rows())  # type: ignore[arg-type]


def test_the_dataset_is_validated_once_before_any_model_runs() -> None:
    scorer = MockScorer.scripted([[0.0, 1.0]])
    bad = [
        LabelledRecord("s", {"q": YesNo("Ok?")}, {"q": True}),
        LabelledRecord("s", {"q": YesNo("Ok?")}, {"q": 1}),
    ]
    with pytest.raises(Exception, match="must be true or false"):
        compare({"a": Jev.from_scorer(scorer)}, bad)
    assert scorer.calls == 0


def test_a_record_the_models_share_is_enough() -> None:
    only = [record(0), record(1)]
    assert set(compare(models(), only).reports) == {"base", "tuned"}
    assert QUESTIONS  # keep the shared fixture imported


def test_bad_options_are_not_blamed_on_a_model() -> None:
    scorer = MockScorer.scripted([[0.0, 1.0]])
    rows = [LabelledRecord("s", {"q": YesNo("Ok?")}, {"q": True})]
    with pytest.raises(ValueError) as info:
        compare({"a": Jev.from_scorer(scorer)}, rows, n_bins=0)
    assert not str(info.value).startswith("model ")
    assert "n_bins must be an integer" in str(info.value)
    assert scorer.calls == 0


def test_the_overlap_warning_points_at_the_caller() -> None:
    import warnings

    jev = scripted(temperature=1.7, calibrated=True)
    data = drawn_records(60, 2.0, 3, group_size=2)
    tuned = jev.with_temperature(jev.calibrate(data, folds=3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compare({"tuned": tuned}, data[:10])
        tuned.evaluate(data[:10])
    files = [
        Path(w.filename).name for w in caught if issubclass(w.category, UserWarning)
    ]
    assert files == ["test_evaluation_compare.py", "test_evaluation_compare.py"]
