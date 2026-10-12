"""``calibrate``, ``with_temperature`` and ``split_dataset`` (spec 017 §3)."""

from __future__ import annotations

import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from _evaluation_data import (
    KEYS,
    LOGITS,
    QUESTIONS,
    as_dict,
    drawn_records,
    record,
    scripted,
)
from archai_jev import (
    CalibrationError,
    Jev,
    YesNo,
    calibrated_softmax,
    fit_temperature,
)
from archai_jev.evaluation import CalibrationStudy, LabelledRecord, split_dataset
from archai_jev.testing import MockScorer

DATA = Path(__file__).parent / "data" / "evaluation"
M = (1 << 64) - 1


# ------------------------------------------------------------------ an independent SplitMix64


def splitmix(seed: int) -> Any:
    state = seed
    while True:
        state = (state + 0x9E3779B97F4A7C15) & M
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        yield z ^ (z >> 31)


def below(it: Any, n: int) -> int:
    block = M - M % n
    while True:
        x = next(it)
        if x < block:
            return int(x % n)


def shuffled(seed: int, n: int) -> list[int]:
    it, items = splitmix(seed), list(range(n))
    for i in range(n - 1, 0, -1):
        j = below(it, i + 1)
        items[i], items[j] = items[j], items[i]
    return items


def reference_split(groups: list[str | None], fraction: float, seed: int) -> list[bool]:
    """Per record: is it in the test side. The definition of spec 017 §3.3, in plain Python."""
    keys: list[Any] = [g if g is not None else ("own", i) for i, g in enumerate(groups)]
    names: list[Any] = []
    for key in keys:
        if key not in names:
            names.append(key)
    sizes = [keys.count(k) for k in names]
    target = fraction * len(groups)
    test_groups: set[int] = set()
    taken = 0
    for g in shuffled(seed, len(names)):
        if taken >= target and taken > 0:
            break
        test_groups.add(g)
        taken += sizes[g]
    return [names.index(key) in test_groups for key in keys]


# ------------------------------------------------------------------ split_dataset


def test_split_dataset_contract() -> None:
    rows = [record(i, group=f"g{i // 3}") for i in range(30)]
    train, test = split_dataset(rows, test_fraction=0.3, seed=0)
    assert split_dataset(rows, test_fraction=0.3, seed=0) == (train, test)
    assert split_dataset(rows, test_fraction=0.3, seed=1) != (train, test)
    assert {r.group for r in train}.isdisjoint({r.group for r in test})
    assert sorted(train + test, key=lambda r: r.state) == sorted(
        rows, key=lambda r: r.state
    )
    assert train == [r for r in rows if r in train] and test == [
        r for r in rows if r in test
    ]
    assert len(test) >= 9 and len(test) < 9 + 3
    # the assignment is the one of the definition, checked against an independent implementation
    for seed in (0, 1, 7, 12345):
        flags = reference_split([r.group for r in rows], 0.3, seed)
        got_train, got_test = split_dataset(rows, test_fraction=0.3, seed=seed)
        assert got_test == [r for r, f in zip(rows, flags, strict=True) if f]
        assert got_train == [r for r, f in zip(rows, flags, strict=True) if not f]


def test_split_dataset_without_groups_and_with_dicts() -> None:
    rows = [record(i) for i in range(10)]
    train, test = split_dataset(rows, test_fraction=0.5, seed=2)
    flags = reference_split([None] * 10, 0.5, 2)
    assert len(test) == sum(flags) and len(train) + len(test) == 10
    dicts = [as_dict(r) for r in rows]
    _, d_test = split_dataset(dicts, test_fraction=0.5, seed=2)
    assert d_test == [d for d, f in zip(dicts, flags, strict=True) if f]


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5, math.nan, math.inf])
def test_split_dataset_fraction_errors(fraction: float) -> None:
    with pytest.raises(
        ValueError, match="test_fraction must be a number strictly between 0 and 1"
    ):
        split_dataset([record(i) for i in range(5)], test_fraction=fraction)


def test_split_dataset_other_errors() -> None:
    rows = [record(i) for i in range(4)]
    with pytest.raises(TypeError, match="test_fraction must be a number"):
        split_dataset(rows, test_fraction="0.3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leaves the train set empty"):
        split_dataset([record(0, group="a"), record(1, group="a")], test_fraction=0.5)
    with pytest.raises(ValueError, match="seed must be an integer"):
        split_dataset(rows, seed=-1)
    with pytest.raises(Exception, match="dataset is empty"):
        split_dataset([])


# ------------------------------------------------------------------ with_temperature


def raw_probabilities(jev: Jev, temperature: float) -> list[list[float]]:
    answers = jev.ask("s", QUESTIONS)
    assert answers.choices["team"].calibrated is (
        temperature != 1.0 or jev.model_info.calibrated
    )
    return [
        list(answers.choices["team"].probabilities.values()),
        [
            1 - answers.yes_nos["urgent"].probability,
            answers.yes_nos["urgent"].probability,
        ],
        list(answers.scores["prio"].probabilities),
    ]


def test_with_temperature_new_jev_old_unchanged() -> None:
    jev = scripted()
    before = jev.ask("s", QUESTIONS)
    tuned = jev.with_temperature(2.5)
    assert tuned is not jev
    assert jev.ask("s", QUESTIONS) == before
    assert jev.model_info.temperature == 1.0 and jev.model_info.calibrated is False
    info = tuned.model_info
    assert (info.temperature, info.calibrated, info.calibration_source) == (
        2.5,
        True,
        "user",
    )
    got = raw_probabilities(tuned, 2.5)
    assert got[0] == calibrated_softmax(LOGITS[0], 2.5)
    assert got[2] == calibrated_softmax(LOGITS[2], 2.5)
    assert all(a.calibrated for a in tuned.ask("s", QUESTIONS).values())
    # the last temperature wins, and it is absolute
    chained = jev.with_temperature(2.5).with_temperature(4.0)
    assert chained.ask("s", QUESTIONS) == jev.with_temperature(4.0).ask("s", QUESTIONS)
    assert chained.model_info.temperature == 4.0


def test_with_temperature_inputs_and_errors() -> None:
    jev = scripted()
    for bad in [0, -1.0, math.nan, math.inf, -math.inf]:
        with pytest.raises(ValueError, match="temperature must be a finite number > 0"):
            jev.with_temperature(bad)
    for wrong in [True, "2.0", None, [1.0]]:
        with pytest.raises(TypeError, match="calibration must be a temperature"):
            jev.with_temperature(wrong)  # type: ignore[arg-type]
    fit = fit_temperature([[2.0, 0.0]] * 4, [0, 0, 0, 1])
    assert jev.with_temperature(fit).model_info.temperature == fit.temperature
    study = jev.calibrate(drawn_records(60, 2.0, 1, group_size=2), folds=3)
    assert jev.with_temperature(study).model_info.temperature == study.temperature
    assert jev.with_temperature(3).model_info.temperature == 3.0


def test_with_temperature_is_thread_safe() -> None:
    jev = scripted()
    tuned = jev.with_temperature(2.0)
    expected = tuned.ask("s", QUESTIONS)
    original = jev.ask("s", QUESTIONS)

    def work(i: int) -> bool:
        pick = tuned if i % 2 else jev
        return pick.ask("s", QUESTIONS) == (expected if i % 2 else original)

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(work, range(64)))


# ------------------------------------------------------------------ calibrate


def test_calibrate_recovers_the_temperature() -> None:
    jev = scripted()
    study = jev.calibrate(drawn_records(3000, 2.0, 3), folds=5)
    assert study.temperature == pytest.approx(2.0, rel=0.05)
    assert study.n == 9000 and study.folds == 5 and study.n_groups == 3000
    assert len(study.fold_temperatures) == 5 and len(study.record_fingerprints) == 3000
    assert isinstance(study, CalibrationStudy)
    assert study.fit.temperature == study.temperature and study.fit.n == 9000
    assert study.out_of_fold.ece < study.current.ece
    assert set(study.by_type) == {"choice", "score", "yes_no"}


def test_calibrate_matches_fit_temperature_on_the_same_logits() -> None:
    rows = drawn_records(400, 2.0, 5, group_size=2)
    study = scripted().calibrate(rows, folds=4)
    logits: list[list[float]] = []
    labels: list[int] = []
    for r in rows:
        logits += LOGITS
        labels += [
            KEYS.index(r.labels["team"]),
            int(r.labels["urgent"]),
            r.labels["prio"],
        ]
    assert study.temperature == pytest.approx(
        fit_temperature(logits, labels).temperature, abs=1e-9
    )


def test_the_temperature_is_absolute() -> None:
    rows = drawn_records(300, 2.0, 6)
    a = scripted().calibrate(rows, folds=3, seed=1)
    b = scripted(temperature=3.7, calibrated=True).calibrate(rows, folds=3, seed=1)
    assert (a.temperature, a.fold_temperatures, a.in_sample) == (
        b.temperature,
        b.fold_temperatures,
        b.in_sample,
    )
    assert a.current != b.current


def test_current_equals_evaluate_overall() -> None:
    rows = drawn_records(200, 2.0, 7, group_size=2)
    jev = scripted(temperature=1.3, calibrated=True)
    study = jev.calibrate(rows, folds=4)
    assert study.current == jev.evaluate(rows).overall
    assert study.by_type["score"].current.n == 200


def test_question_types_filter_and_errors() -> None:
    rows = drawn_records(200, 2.0, 8, group_size=2)
    jev = scripted()
    only = jev.calibrate(rows, folds=4, question_types=["yes_no"])
    assert only.n == 200 and list(only.by_type) == ["yes_no"]
    labels = [int(r.labels["urgent"]) for r in rows]
    assert only.temperature == pytest.approx(
        fit_temperature([LOGITS[1]] * 200, labels).temperature, abs=1e-9
    )
    with pytest.raises(TypeError, match="not a str"):
        jev.calibrate(rows, question_types="yes_no")
    with pytest.raises(ValueError, match="question_types must contain only"):
        jev.calibrate(rows, question_types=["text"])
    with pytest.raises(ValueError, match="question_types is empty"):
        jev.calibrate(rows, question_types=[])
    with pytest.raises(ValueError, match="the dataset has no score questions"):
        jev.calibrate(
            [LabelledRecord("s", {"q": YesNo("Ok?")}, {"q": True}) for _ in range(5)],
            question_types=["score"],
        )


@pytest.mark.parametrize("bad", [1, 51, 2.5, "5", None, True])
def test_folds_errors(bad: Any) -> None:
    with pytest.raises(ValueError, match="folds must be an integer between 2 and 50"):
        scripted().calibrate(drawn_records(60, 2.0, 1), folds=bad)


def test_fewer_groups_than_folds_is_reported_before_the_model_runs() -> None:
    scorer = MockScorer.scripted(LOGITS)
    jev = Jev.from_scorer(scorer)
    rows = drawn_records(20, 2.0, 1, group_size=10)  # 2 groups
    with pytest.raises(ValueError) as info:
        jev.calibrate(rows, folds=3)
    assert (
        str(info.value)
        == "dataset has 2 groups but folds=3; use more records or fewer folds"
    )
    assert scorer.calls == 0


def test_degenerate_data_gives_no_study() -> None:
    # the label is always the option with the highest logit
    rows = [record(i, "returns", True, 2) for i in range(30)]
    with pytest.raises(CalibrationError, match=r"below the lower bound 0\.01"):
        scripted().calibrate(rows, folds=3)
    with pytest.raises(CalibrationError, match="below the lower bound"):
        scripted().calibrate(rows, folds=3, bounds=(1e-6, 1e9))
    uniform = scripted(logits=[[1.0, 1.0, 1.0], [0.0, 0.0], [2.0, 2.0, 2.0]])
    with pytest.raises(CalibrationError, match="every row is uniform"):
        uniform.calibrate(drawn_records(30, 2.0, 1), folds=3)
    # a fold that cannot be fitted is named
    mixed = [record(i, "returns", True, 2, group="A") for i in range(6)]
    mixed += [
        record(6 + i, KEYS[(i + 1) % 3], i % 2 == 0, i % 3, group="B") for i in range(6)
    ]
    with pytest.raises(CalibrationError, match=r"^fold [01]: ") as info:
        scripted().calibrate(mixed, folds=2)
    assert "the best temperature is" in str(info.value)


def test_a_failing_record_names_its_place() -> None:
    from archai_jev.testing import Fault

    jev = Jev.from_scorer(MockScorer(1).with_fault(Fault.nan("team", 0)))
    rows = [
        LabelledRecord(f"s{i}", QUESTIONS, record(0).labels, group=f"g{i}")
        for i in range(6)
    ]
    with pytest.raises(Exception, match=r"^record 0: ") as info:
        jev.calibrate(rows, folds=3)
    assert type(info.value).__name__ == "NumericalError"


def test_study_contract_and_table() -> None:
    rows = drawn_records(60, 2.0, 11, group_size=2)
    study = scripted().calibrate(rows, folds=3, seed=2)
    assert study == scripted().calibrate(rows, folds=3, seed=2)
    assert hash(study) == hash(scripted().calibrate(rows, folds=3, seed=2))
    assert pickle.loads(pickle.dumps(study)) == study
    d = study.to_dict()
    assert d["temperature"] == study.temperature and d["folds"] == 3
    assert d["fit"]["bounds"] == [0.01, 100.0]
    assert study.table() == (DATA / "table_study.txt").read_text(
        encoding="utf-8"
    ).rstrip("\n")
