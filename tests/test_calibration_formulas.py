"""Public formulas of ``archai_jev.calibration``: values, validation, equality with ``ask``."""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import archai_jev
from archai_jev import Choice, Jev, Score, YesNo, calibrated_softmax
from archai_jev import calibration as cal
from archai_jev.testing import MockScorer
from test_calibration_fixture import load_formula_vectors

LOGITS = [[2.0, 1.0, 0.1], [math.log(0.1), math.log(0.2), math.log(0.7)], [1.0, -1.0]]
STATE = {"ticket": "Shoes arrived late and in the wrong size. Charged twice."}
QUESTIONS = {
    "team": Choice(
        "Which team should handle this?",
        {"returns": "Exchanges, refunds", "shipping": "Delays", "billing": "Charges"},
    ),
    "urgency": Score("How urgent is it?", ["low", "medium", "high"]),
    "angry": YesNo("Is the customer angry?"),
}


def test_calibrated_softmax_unchanged_and_same_object() -> None:
    assert cal.calibrated_softmax is calibrated_softmax is archai_jev.calibrated_softmax
    assert calibrated_softmax([2.0, 1.0, 0.1]) == pytest.approx(
        [0.659, 0.242, 0.099], abs=1e-3
    )
    for logits, temperature in [([], 1.0), ([1.0, float("nan")], 1.0), ([1.0], 0.0)]:
        with pytest.raises(ValueError):
            calibrated_softmax(logits, temperature)


def test_batch_equals_loop_exactly() -> None:
    rows = [[2.0, 1.0, 0.1], [-3.0, 4.0], [1e300, 1e300, 5.0], [7.0]]
    for temperature in (1.0, 2.3, 0.05):
        batch = cal.calibrated_softmax_batch(rows, temperature)
        assert batch == [calibrated_softmax(r, temperature) for r in rows]


def test_38_vectors() -> None:
    seen = {"noul": 0, "choice": 0, "score": 0}
    for v in load_formula_vectors():
        if v["kind"] == "noul":
            no, yes = v["logits_raw"]
            p = cal.yes_no_probability(no, yes, v["temperature"])
            assert p == pytest.approx(v["p_true"], abs=1e-12)
            seen["noul"] += 1
            continue
        if abs(sum(v["p"]) - 1.0) > 1e-6:
            continue
        seen[v["kind"]] += 1
        if v["kind"] == "choice":
            got = cal.choice_confidence(v["p"])
        else:
            got = cal.score_confidence(v["p"])
            assert cal.score_value(v["p"]) == pytest.approx(v["value"], abs=1e-12)
        assert got == pytest.approx(v["confidence"], abs=1e-12), v["p"]
    # 8 yes/no vectors plus the 26 distributions that sum to 1 (choice and score).
    assert seen["noul"] == 8
    assert seen["choice"] + seen["score"] == 26


def test_four_rejected_with_message() -> None:
    rejected = 0
    for v in load_formula_vectors():
        if v["kind"] == "noul" or abs(sum(v["p"]) - 1.0) <= 1e-6:
            continue
        rejected += 1
        expected = (
            "probabilities row 0: value 2.0 at option 0 is outside [0, 1]"
            if max(v["p"]) > 1
            else "probabilities row 0: sums to 0.0, expected 1 within 1e-6"
        )
        for fn in (cal.choice_confidence, cal.score_confidence, cal.score_value):
            with pytest.raises(ValueError) as info:
                fn(v["p"])
            assert str(info.value) == expected
    assert rejected == 4


def test_equals_ask_bit_for_bit_with_mock() -> None:
    jev = Jev.from_scorer(MockScorer.scripted(LOGITS, calibrated=True))
    answers = jev.ask(STATE, QUESTIONS)
    team = answers.choices["team"]
    probabilities = list(team.probabilities.values())
    assert cal.choice_confidence(probabilities) == team.confidence
    assert cal.choice_confidence_batch([probabilities]) == [team.confidence]
    urgency = answers.scores["urgency"]
    assert cal.score_confidence(urgency.probabilities) == urgency.confidence
    assert cal.score_value(urgency.probabilities) == urgency.value
    assert cal.score_value_batch([urgency.probabilities]) == [urgency.value]
    assert cal.score_confidence_batch([urgency.probabilities]) == [urgency.confidence]
    angry = answers.yes_nos["angry"]
    assert cal.yes_no_probability(1.0, -1.0) == pytest.approx(
        angry.probability, abs=1e-12
    )


@settings(max_examples=200, deadline=None)
@given(
    no=st.floats(-1e300, 1e300),
    yes=st.floats(-1e300, 1e300),
    temperature=st.floats(1e-3, 1e3),
)
def test_yes_no_extremes_hypothesis(no: float, yes: float, temperature: float) -> None:
    p = cal.yes_no_probability(no, yes, temperature)
    assert 0.0 <= p <= 1.0 and not math.isnan(p)
    assert p + cal.yes_no_probability(yes, no, temperature) == pytest.approx(
        1.0, abs=1e-12
    )
    d = (yes - no) / temperature
    if abs(d) < 700:
        assert p == pytest.approx(1.0 / (1.0 + math.exp(-d)), abs=1e-12)


def test_yes_no_batch_rows_are_pairs() -> None:
    assert cal.yes_no_probability_batch([[0.0, 0.0], [0.0, 1e300]]) == [0.5, 1.0]
    with pytest.raises(ValueError) as info:
        cal.yes_no_probability_batch([[0.0, 1.0], [1.0]])
    assert (
        str(info.value) == "logits row 1 must have exactly 2 options ([no, yes]), got 1"
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: cal.choice_confidence([]), "probabilities row 0 is empty"),
        (
            lambda: cal.choice_confidence_batch([]),
            "probabilities must not be empty",
        ),
        (
            lambda: cal.choice_confidence([0.5, float("nan")]),
            "probabilities row 0: contains NaN or infinity",
        ),
        (
            lambda: cal.choice_confidence([0.5, float("inf")]),
            "probabilities row 0: contains NaN or infinity",
        ),
        (
            lambda: cal.score_value_batch([[1.0], [-0.5, 1.5]]),
            "probabilities row 1: value -0.5 at option 0 is outside [0, 1]",
        ),
        (
            lambda: cal.score_value([0.0, 0.0]),
            "probabilities row 0: sums to 0.0, expected 1 within 1e-6",
        ),
        (
            lambda: cal.score_confidence([0.5, 0.4]),
            "probabilities row 0: sums to 0.9, expected 1 within 1e-6",
        ),
        (
            lambda: cal.calibrated_softmax_batch([[1.0], [2.0, float("inf")]]),
            "logits row 1 has the non-finite value inf at option 1",
        ),
        (
            lambda: cal.calibrated_softmax_batch([[1.0], []]),
            "logits row 1 is empty; every row needs at least one option",
        ),
        (
            lambda: cal.calibrated_softmax_batch([[1.0]], 0.0),
            "temperature must be a finite number > 0, got 0",
        ),
        (
            lambda: cal.choice_confidence([1.0], sum_tolerance=0.06),
            "sum_tolerance must be a number in [0, 0.05], got 0.06",
        ),
        (
            lambda: cal.choice_confidence([1.0], sum_tolerance=-1.0),
            "sum_tolerance must be a number in [0, 0.05], got -1.0",
        ),
    ],
)
def test_validation_messages(call: Any, message: str) -> None:
    with pytest.raises(ValueError) as info:
        call()
    assert str(info.value) == message or str(info.value).startswith(message)


def test_sum_tolerance_255_options() -> None:
    row = [round(1 / 255, 4)] * 255
    with pytest.raises(ValueError, match="sums to"):
        cal.choice_confidence(row)
    assert cal.choice_confidence(row, sum_tolerance=0.02) == pytest.approx(
        0.0, abs=1e-3
    )
    assert cal.choice_confidence_batch([row], sum_tolerance=0.02)[0] == pytest.approx(
        0.0, abs=1e-3
    )


def test_ragged_rows_all_batches() -> None:
    probs = [[1.0], [0.5, 0.5], [0.25, 0.25, 0.5]]
    assert len(cal.choice_confidence_batch(probs)) == 3
    assert len(cal.score_confidence_batch(probs)) == 3
    assert len(cal.score_value_batch(probs)) == 3
    assert [len(r) for r in cal.calibrated_softmax_batch(probs)] == [1, 2, 3]
    assert cal.choice_confidence_batch(probs)[0] == 1.0
