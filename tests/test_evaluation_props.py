"""Repeatability, threads, the GIL, Ctrl-C, fuzzing and references (spec 017 AC-10, 14, 17, 31)."""

from __future__ import annotations

import contextlib
import math
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from _evaluation_data import KEYS, LOGITS, QUESTIONS, drawn_records, record, scripted
from _timing import interrupt_main_at, timing_sensitive
from archai_jev import (
    InvalidDatasetError,
    InvalidQuestionError,
    InvalidStateError,
    Jev,
    JevError,
    YesNo,
    calibrated_softmax,
)
from archai_jev.calibration import calibration_report
from archai_jev.evaluation import LabelledRecord, load_dataset
from archai_jev.testing import MockScorer

M = (1 << 64) - 1


def test_repeatable_and_eight_threads() -> None:
    rows = drawn_records(80, 2.0, 2, group_size=2)
    jev = scripted(temperature=1.5, calibrated=True)
    first = jev.evaluate(rows, bootstrap_samples=20, seed=3)
    assert jev.evaluate(rows, bootstrap_samples=20, seed=3) == first
    with ThreadPoolExecutor(8) as pool:
        reports = list(
            pool.map(
                lambda _: jev.evaluate(rows, bootstrap_samples=20, seed=3), range(8)
            )
        )
    assert all(r == first for r in reports)
    studies = {jev.calibrate(rows, folds=4, seed=1).temperature for _ in range(3)}
    assert len(studies) == 1


@timing_sensitive()
def test_gil_released_during_evaluate() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.05))
    rows = [
        LabelledRecord(f"s{i}", {"q": YesNo("Ok?")}, {"q": True}) for i in range(10)
    ]
    ticks = 0
    stop = threading.Event()

    def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            time.sleep(0.01)
            ticks += 1

    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        jev.evaluate(rows)
    finally:
        stop.set()
        thread.join()
    assert ticks >= 10, f"only {ticks} ticks during a 0.5 s evaluation"


@timing_sensitive()
def test_ctrl_c_stops_between_records() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.2))
    rows = [
        LabelledRecord(f"s{i}", {"q": YesNo("Ok?")}, {"q": True}) for i in range(10)
    ]
    start = time.perf_counter()
    with pytest.raises(KeyboardInterrupt), interrupt_main_at(0.5):
        jev.evaluate(rows)
    assert time.perf_counter() - start < 1.4  # the whole evaluation takes 2 s
    # the Jev is still usable
    assert jev.evaluate(rows[:1]).n_records == 1


# ------------------------------------------------------------------ fuzzing

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(10**30), 10**30)
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(max_size=8),
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(max_size=6), children, max_size=3),
    max_leaves=12,
)
record_dicts = st.fixed_dictionaries(
    {},
    optional={
        "state": json_values,
        "questions": json_values,
        "id": json_values,
        "group": json_values,
        "extra": json_values,
    },
)
ALLOWED = (
    InvalidDatasetError,
    InvalidStateError,
    InvalidQuestionError,
    TypeError,
    ValueError,
)


@settings(max_examples=300, deadline=None)
@given(items=st.lists(st.one_of(record_dicts, json_values), max_size=3))
def test_fuzz_dicts_never_crash(items: list[Any]) -> None:
    with contextlib.suppress(*ALLOWED):
        scripted().evaluate(items)


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=120))
def test_fuzz_lines_never_crash(text: str) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fuzz.jsonl"
        path.write_text(text, encoding="utf-8", newline="")
        with contextlib.suppress(JevError, ValueError, TypeError):
            load_dataset(path)


@settings(max_examples=100, deadline=None)
@given(data=st.binary(max_size=200))
def test_fuzz_bytes_never_crash(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fuzz.jsonl"
        path.write_bytes(data)
        with contextlib.suppress(JevError, ValueError, TypeError, UnicodeDecodeError):
            load_dataset(path)


# ------------------------------------------------------------------ references


def test_reference_parity_with_numpy() -> None:
    np = pytest.importorskip("numpy")
    jev = Jev.from_scorer(MockScorer(9, temperature=1.6, calibrated=True))
    rows = [
        LabelledRecord(
            f"state {i}",
            QUESTIONS,
            record(i, KEYS[i % 3], i % 3 == 0, (i * 5) % 3).labels,
        )
        for i in range(300)
    ]
    report = jev.evaluate(rows)
    probabilities: list[list[float]] = []
    labels: list[int] = []
    for r in rows:
        answers = jev.ask(r.state, r.questions)
        p = answers.yes_nos["urgent"].probability
        probabilities += [
            list(answers.choices["team"].probabilities.values()),
            [1 - p, p],
            list(answers.scores["prio"].probabilities),
        ]
        labels += [
            KEYS.index(r.labels["team"]),
            int(r.labels["urgent"]),
            r.labels["prio"],
        ]
    # the rows have different lengths, so build the numpy reference per bucket of equal length
    n = len(probabilities)
    conf = np.array([max(p) for p in probabilities])
    hit = np.array(
        [p.index(max(p)) == y for p, y in zip(probabilities, labels, strict=True)]
    )
    edges = np.arange(1, 11) / 10
    bin_of = np.minimum(np.searchsorted(edges, conf, side="left"), 9)
    ece = 0.0
    for b in range(10):
        mask = bin_of == b
        if mask.any():
            ece += mask.sum() / n * abs(hit[mask].mean() - conf[mask].mean())
    assert report.overall.ece == pytest.approx(ece, abs=1e-12)
    assert report.overall.mean_confidence == pytest.approx(conf.mean(), abs=1e-12)
    assert report.overall.accuracy == pytest.approx(hit.mean(), abs=1e-12)
    log_loss = -np.mean(
        [math.log(p[y]) for p, y in zip(probabilities, labels, strict=True)]
    )
    assert report.overall.log_loss == pytest.approx(float(log_loss), abs=1e-12)
    assert report.overall == calibration_report(probabilities, labels)


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


def test_bootstrap_percentiles_match_numpy() -> None:
    np = pytest.importorskip("numpy")
    sizes = [3, 1, 2, 4, 1, 2, 3]
    labels_team = [
        "returns",
        "billing",
        "other",
        "returns",
        "returns",
        "billing",
        "other",
    ]
    rows: list[LabelledRecord] = []
    for g, size in enumerate(sizes):
        for k in range(size):
            rows.append(
                record(len(rows), labels_team[(g + k) % 7], True, 1, group=f"g{g}")
            )
    jev = scripted()
    report = jev.evaluate(rows, bootstrap_samples=200, seed=5)
    got = report.by_type["choice"].ci95
    assert got is not None
    # reference: the definition of spec 017 §3.5, in plain Python and numpy
    p = calibrated_softmax(LOGITS[0], 1.0)
    group_rows: dict[str, list[int]] = {}
    for r in rows:
        group_rows.setdefault(r.group or "", []).append(KEYS.index(r.labels["team"]))
    groups = list(group_rows.values())
    top = p.index(max(p))
    conf = p[top]
    b = next(i for i in range(10) if conf <= (i + 1) / 10)
    it = splitmix(5)
    samples: list[list[float]] = [[], [], [], []]
    for _ in range(200):
        chosen = [groups[below(it, len(groups))] for _ in groups]
        ys = [y for g in chosen for y in g]
        n = len(ys)
        correct = sum(y == top for y in ys)
        accuracy = correct / n
        ece = abs(accuracy - conf) if n else 0.0
        brier = (
            sum(
                sum((q - (1.0 if k == y else 0.0)) ** 2 for k, q in enumerate(p))
                for y in ys
            )
            / n
        )
        loss = sum(-math.log(p[y]) for y in ys) / n
        for slot, value in zip(samples, [accuracy, ece, brier, loss], strict=True):
            slot.append(value)
    del b
    for name, values in zip(
        ["accuracy", "ece", "brier", "log_loss"], samples, strict=True
    ):
        lo, hi = np.percentile(sorted(values), [2.5, 97.5])
        assert got[name][0] == pytest.approx(float(lo), abs=1e-12), name
        assert got[name][1] == pytest.approx(float(hi), abs=1e-12), name
