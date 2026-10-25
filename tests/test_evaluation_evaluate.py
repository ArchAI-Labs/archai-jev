"""``Jev.evaluate``: numbers by hand, coherence with ``ask``, notes, tables, dict form."""

from __future__ import annotations

import dataclasses
import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import pytest

from _evaluation_data import KEYS, LOGITS, QUESTIONS, drawn_records, record, scripted
from archai_jev import Jev, calibrated_softmax
from archai_jev.calibration import calibration_report
from archai_jev.evaluation import EvaluationReport, LabelledRecord
from archai_jev.testing import Fault, MockScorer

DATA = Path(__file__).parent / "data" / "evaluation"
T = 1.7


def reference_text(name: str) -> str:
    return (DATA / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")


def three_records() -> list[LabelledRecord]:
    return [
        record(0, "returns", True, 2),
        record(1, "returns", True, 1),
        record(2, "other", False, 1),
    ]


def reference(rows: list[list[float]], labels: list[int]) -> dict[str, float]:
    """The definitions of spec 004 §3, in plain Python."""
    n = len(rows)
    bins: dict[int, list[float]] = {}
    hits = conf = brier = loss = 0.0
    for row, y in zip(rows, labels, strict=True):
        top = row.index(max(row))
        c = row[top]
        hit = float(top == y)
        b = next(i for i in range(10) if c <= (i + 1) / 10)
        cell = bins.setdefault(b, [0.0, 0.0, 0.0])
        cell[0] += 1
        cell[1] += hit
        cell[2] += c
        hits += hit
        conf += c
        brier += sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(row))
        loss += -math.log(row[y]) if row[y] > 0 else math.inf
    gaps = [(c[0] / n, abs(c[1] / c[0] - c[2] / c[0])) for c in bins.values()]
    return {
        "accuracy": hits / n,
        "mean_confidence": conf / n,
        "ece": sum(w * g for w, g in gaps),
        "mce": max(g for _, g in gaps),
        "brier": brier / n,
        "log_loss": loss / n,
    }


def check(rep: Any, rows: list[list[float]], labels: list[int]) -> None:
    want = reference(rows, labels)
    assert rep.n == len(rows)
    for key, value in want.items():
        assert getattr(rep, key) == pytest.approx(value, abs=1e-12), key


def test_exact_values_by_hand() -> None:
    jev = scripted(temperature=T, calibrated=True)
    report = jev.evaluate(three_records())
    team = calibrated_softmax(LOGITS[0], T)
    urgent_p = calibrated_softmax(LOGITS[1], T)[1]
    prio = calibrated_softmax(LOGITS[2], T)
    team_rows, team_y = [team] * 3, [0, 0, 2]
    urgent_rows, urgent_y = [[1 - urgent_p, urgent_p]] * 3, [1, 1, 0]
    prio_rows, prio_y = [prio] * 3, [2, 1, 1]
    check(
        report.overall, team_rows + urgent_rows + prio_rows, team_y + urgent_y + prio_y
    )
    assert list(report.by_type) == ["choice", "score", "yes_no"]
    check(report.by_type["choice"], team_rows, team_y)
    check(report.by_type["score"], prio_rows, prio_y)
    check(report.by_type["yes_no"], urgent_rows, urgent_y)
    assert list(report.by_question) == ["team", "urgent", "prio"]
    check(report.by_question["prio"], prio_rows, prio_y)
    assert report.by_question["prio"].type == "score"
    level = sum(i * p for i, p in enumerate(prio))
    mae = sum(abs(level - y) for y in prio_y) / 3
    assert report.by_type["score"].expected_level_mae == pytest.approx(mae, abs=1e-12)
    assert report.by_type["yes_no"].positive_rate == pytest.approx(2 / 3)
    assert report.by_type["yes_no"].mean_probability == pytest.approx(urgent_p)
    assert report.by_type["choice"].positive_rate is None
    assert report.n_records == 3 and report.n_questions == 9
    assert report.overall.n == 9 == sum(t.n for t in report.by_type.values())
    assert report.by_type["yes_no"].ci95 is None


def test_equals_calibration_report_of_ask() -> None:
    jev = Jev.from_scorer(MockScorer(5, temperature=1.4, calibrated=True))
    rows = [
        record(i, KEYS[i % 3], i % 2 == 0, (i * 7) % 3, group=f"g{i // 3}")
        for i in range(40)
    ]
    rows = [
        LabelledRecord(f"state number {i}", r.questions, r.labels, group=r.group)
        for i, r in enumerate(rows)
    ]
    report = jev.evaluate(rows)
    scopes: dict[str, tuple[list[list[float]], list[int]]] = {
        "choice": ([], []),
        "score": ([], []),
        "yes_no": ([], []),
    }
    by_name: dict[str, tuple[list[list[float]], list[int]]] = {
        n: ([], []) for n in QUESTIONS
    }
    for r in rows:
        answers = jev.ask(r.state, r.questions)
        team = answers.choices["team"]
        prio = answers.scores["prio"]
        urgent = answers.yes_nos["urgent"]
        items = [
            (
                "choice",
                "team",
                list(team.probabilities.values()),
                KEYS.index(r.labels["team"]),
            ),
            (
                "yes_no",
                "urgent",
                [1 - urgent.probability, urgent.probability],
                int(r.labels["urgent"]),
            ),
            ("score", "prio", list(prio.probabilities), r.labels["prio"]),
        ]
        for kind, name, row, y in items:
            scopes[kind][0].append(row)
            scopes[kind][1].append(y)
            by_name[name][0].append(row)
            by_name[name][1].append(y)
    all_rows = [x for k in ("choice", "yes_no", "score") for x in scopes[k][0]]
    # `overall` is in record order, so build it the same way
    ordered_rows: list[list[float]] = []
    ordered_labels: list[int] = []
    for i in range(len(rows)):
        for name in QUESTIONS:
            ordered_rows.append(by_name[name][0][i])
            ordered_labels.append(by_name[name][1][i])
    assert len(all_rows) == len(ordered_rows)
    assert report.overall == calibration_report(ordered_rows, ordered_labels)
    for kind, (r_, y_) in scopes.items():
        want = calibration_report(r_, y_)
        got = report.by_type[kind]
        for field in (
            "n",
            "accuracy",
            "mean_confidence",
            "ece",
            "mce",
            "brier",
            "log_loss",
            "bins",
        ):
            assert getattr(got, field) == getattr(want, field), (kind, field)
    for name, (r_, y_) in by_name.items():
        want = calibration_report(r_, y_)
        got = report.by_question[name]
        assert (got.ece, got.brier, got.log_loss) == (
            want.ece,
            want.brier,
            want.log_loss,
        )


def test_confidence_is_the_largest_probability() -> None:
    # p = [0.5, 0.3, 0.2]: the `confidence` field of the answer is 0.25, the ECE uses 0.5.
    z = [math.log(p) * T for p in (0.5, 0.3, 0.2)]
    jev = scripted(temperature=T, calibrated=True, logits=[z])
    team_only = {"team": QUESTIONS["team"]}
    rows = [LabelledRecord("s", team_only, {"team": "returns"})]
    report = jev.evaluate(rows)
    answer = jev.ask("s", team_only).choices["team"]
    assert answer.confidence == pytest.approx(0.25)
    assert report.overall.mean_confidence == pytest.approx(0.5)


def test_bootstrap_intervals() -> None:
    jev = scripted(temperature=T, calibrated=True)
    rows = [
        record(i, KEYS[i % 3], i % 2 == 0, i % 3, group=f"g{i // 2}") for i in range(30)
    ]
    plain = jev.evaluate(rows)
    assert plain.by_type["choice"].ci95 is None
    ci = jev.evaluate(rows, bootstrap_samples=100, seed=4)
    again = jev.evaluate(rows, bootstrap_samples=100, seed=4)
    assert ci == again
    other = jev.evaluate(rows, bootstrap_samples=100, seed=5)
    assert ci != other
    for rep in [*ci.by_type.values(), *ci.by_question.values()]:
        assert rep.ci95 is not None and list(rep.ci95) == [
            "accuracy",
            "ece",
            "brier",
            "log_loss",
        ]
        for lo, hi in rep.ci95.values():
            assert lo <= hi
        assert 0.0 <= rep.ci95["accuracy"][0] <= rep.ci95["accuracy"][1] <= 1.0
    # one group: the interval is the estimate itself
    single = jev.evaluate(
        [record(i, KEYS[i % 3], True, i % 3, group="same") for i in range(8)],
        bootstrap_samples=20,
    )
    rep = single.by_type["choice"]
    assert rep.ci95 is not None
    assert rep.ci95["accuracy"] == (rep.accuracy, rep.accuracy)
    assert rep.ci95["log_loss"] == pytest.approx((rep.log_loss, rep.log_loss))


@pytest.mark.parametrize("bad", [-1, 2.5, "10", None, True, 100_001])
def test_bootstrap_arguments(bad: Any) -> None:
    with pytest.raises(
        ValueError, match="bootstrap_samples must be an integer between 0 and"
    ):
        scripted().evaluate(three_records(), bootstrap_samples=bad)


@pytest.mark.parametrize("bad", [-1, 2**64, 1.5, "0", True])
def test_seed_and_bins_arguments(bad: Any) -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        scripted().evaluate(three_records(), seed=bad)
    with pytest.raises(ValueError, match="n_bins must be an integer"):
        scripted().evaluate(three_records(), n_bins=bad if bad != -1 else 0)


def test_nothing_partial_and_the_jev_stays_usable() -> None:
    faulty = Jev.from_scorer(MockScorer(1).with_fault(Fault.nan("team", 0)))
    rows = [LabelledRecord("s", QUESTIONS, record(0).labels)]
    with pytest.raises(Exception) as info:
        faulty.evaluate(rows)
    assert type(info.value).__name__ == "NumericalError"
    assert str(info.value).startswith("record 0: ")
    ok = scripted()
    before = ok.ask("x", QUESTIONS)
    ok.evaluate(three_records())
    assert ok.ask("x", QUESTIONS) == before


def test_notes_and_warning_once() -> None:
    jev = scripted(temperature=T, calibrated=True)
    rows = drawn_records(60, 2.0, 1, group_size=2)
    study = jev.calibrate(rows, folds=3)
    tuned = jev.with_temperature(study)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = tuned.evaluate(rows[:12])
    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    expected = (
        "12 of 12 records were also used to calibrate this model; these metrics are optimistic, "
        "evaluate on records not used for calibration"
    )
    assert report.calibration_overlap == 12
    assert texts == [expected]
    assert expected in report.notes
    other = [record(100 + i, "returns", True, 2) for i in range(5)]
    other = [
        LabelledRecord(f"unrelated {i}", r.questions, r.labels)
        for i, r in enumerate(other)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean = tuned.evaluate(other)
    assert clean.calibration_overlap == 0
    assert not any("also used to calibrate" in n for n in clean.notes)


def test_overlap_is_none_for_float_and_fit() -> None:
    from archai_jev import fit_temperature

    jev = scripted()
    rows = three_records()
    assert jev.evaluate(rows).calibration_overlap is None
    assert jev.with_temperature(2.0).evaluate(rows).calibration_overlap is None
    fit = fit_temperature([[2.0, 0.0]] * 4, [0, 0, 0, 1])
    assert jev.with_temperature(fit).evaluate(rows).calibration_overlap is None


def test_not_calibrated_note_disappears() -> None:
    jev = scripted()
    note = "the model has no declared calibration (calibrated=False); its probabilities are not guaranteed to be calibrated"
    assert note in jev.evaluate(three_records()).notes
    assert note not in jev.with_temperature(1.5).evaluate(three_records()).notes
    assert note not in scripted(calibrated=True).evaluate(three_records()).notes


def test_model_in_report_reflects_temperature() -> None:
    jev = scripted()
    assert jev.evaluate(three_records()).model.temperature == 1.0
    tuned = jev.with_temperature(2.5)
    model = tuned.evaluate(three_records()).model
    assert (model.temperature, model.calibrated, model.calibration_source) == (
        2.5,
        True,
        "user",
    )
    assert jev.evaluate(three_records()).model.calibrated is False


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def field(data: bytes) -> bytes:
    return len(data).to_bytes(8, "little") + data


def test_dataset_fingerprint_stable() -> None:
    """The fingerprint follows its definition (field lengths, FNV-1a), checked independently."""
    from archai_jev import YesNo

    rows = [
        LabelledRecord("hello", {"ok": YesNo("Fine?")}, {"ok": True}),
        LabelledRecord(
            "world", {"ok": YesNo("Fine?")}, {"ok": False}, id="x", group="g"
        ),
    ]
    stream = b""
    for r in rows:
        body = field(r.state.encode()) + (1).to_bytes(8, "little")
        body += field(b"ok") + field(b"yes_no") + field(b"Fine?")
        body += (2).to_bytes(8, "little") + field(b"no") + field(b"yes")
        stream += fnv1a(body).to_bytes(8, "little")
    expected = f"{fnv1a(stream):016x}"
    report = (
        scripted().evaluate(rows)
        if False
        else Jev.from_scorer(MockScorer(1)).evaluate(rows)
    )
    assert report.dataset_fingerprint == expected
    again = Jev.from_scorer(MockScorer(2)).evaluate(rows)
    assert again.dataset_fingerprint == expected
    # labels, id and group are not part of it; the state is
    relabelled = [
        LabelledRecord("hello", {"ok": YesNo("Fine?")}, {"ok": False}),
        LabelledRecord("world", {"ok": YesNo("Fine?")}, {"ok": True}),
    ]
    assert (
        Jev.from_scorer(MockScorer(1)).evaluate(relabelled).dataset_fingerprint
        == expected
    )
    changed = [LabelledRecord("hellp", {"ok": YesNo("Fine?")}, {"ok": True}), rows[1]]
    assert (
        Jev.from_scorer(MockScorer(1)).evaluate(changed).dataset_fingerprint != expected
    )
    swapped = [
        LabelledRecord(
            "s", {"a": YesNo("A?"), "b": YesNo("B?")}, {"a": True, "b": True}
        ),
        LabelledRecord(
            "s", {"b": YesNo("B?"), "a": YesNo("A?")}, {"a": True, "b": True}
        ),
    ]
    jev = Jev.from_scorer(MockScorer(1))
    first = jev.evaluate(swapped[:1]).dataset_fingerprint
    second = jev.evaluate(swapped[1:]).dataset_fingerprint
    assert first != second


def test_to_dict_schema_and_inf() -> None:
    # logits [0, 800] with label 0 give probability exactly 0 on the label: log-loss is inf
    jev = scripted(logits=[[0.0, 800.0, 0.0], [0.0, 1.0], [0.0, 0.0, 2.0]])
    report = jev.evaluate(three_records())
    d = report.to_dict()
    assert d["schema"] == "archai-jev.evaluation/1"
    assert set(d) == {
        "schema", "model", "n_records", "n_questions", "overall", "by_type",
        "by_question", "dataset_fingerprint", "calibration_overlap", "notes",
    }  # fmt: skip
    assert d["overall"]["log_loss"] == math.inf
    assert d["by_type"]["choice"]["log_loss"] == math.inf
    assert (
        d["by_type"]["score"]["expected_level_mae"]
        == report.by_type["score"].expected_level_mae
    )
    assert d["by_question"]["team"]["type"] == "choice"
    assert d["model"]["name"] == "mock" and isinstance(d["notes"], list)

    def has_nan(x: Any) -> bool:
        if isinstance(x, float):
            return math.isnan(x)
        if isinstance(x, dict):
            return any(has_nan(v) for v in x.values())
        if isinstance(x, list):
            return any(has_nan(v) for v in x)
        return False

    assert not has_nan(d)
    # the numbers convert back to the report
    assert d["overall"]["accuracy"] == report.overall.accuracy
    assert d["overall"]["bins"][0]["lower"] == report.overall.bins[0].lower


def test_report_classes_contract() -> None:
    report = scripted().evaluate(three_records(), bootstrap_samples=10)
    assert isinstance(report, EvaluationReport)
    assert report == scripted().evaluate(three_records(), bootstrap_samples=10)
    assert hash(report) == hash(
        scripted().evaluate(three_records(), bootstrap_samples=10)
    )
    assert pickle.loads(pickle.dumps(report)) == report
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.n_records = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.by_type["x"] = report.by_type["choice"]  # type: ignore[index]
    assert "EvaluationReport(" in repr(report)


def test_table_matches_the_reference_string() -> None:
    jev = scripted(temperature=T, calibrated=True)
    assert jev.evaluate(three_records()).table() == reference_text("table_evaluation")


def test_table_with_intervals_missing_values_and_inf() -> None:
    jev = scripted(logits=[[0.0, 800.0, 0.0], [0.0, 1.0], [0.0, 0.0, 2.0]])
    text = jev.evaluate(three_records()).table()
    assert text == reference_text("table_evaluation_inf")
