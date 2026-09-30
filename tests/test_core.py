import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from archai_jev import calibrated_softmax


def test_sums_to_one() -> None:
    probs = calibrated_softmax([2.0, 1.0, 0.1])
    assert math.isclose(sum(probs), 1.0)


def test_known_values() -> None:
    probs = calibrated_softmax([2.0, 1.0, 0.1])
    assert probs == pytest.approx([0.659, 0.242, 0.099], abs=1e-3)


def test_higher_temperature_flattens() -> None:
    sharp = calibrated_softmax([2.0, 1.0, 0.1], temperature=0.5)
    flat = calibrated_softmax([2.0, 1.0, 0.1], temperature=2.0)
    assert max(flat) < max(sharp)


def test_large_logits_are_stable() -> None:
    probs = calibrated_softmax([1000.0, 1000.0])
    assert probs == pytest.approx([0.5, 0.5])


def test_accepts_any_sequence() -> None:
    assert calibrated_softmax((1.0, 1.0)) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("logits", "temperature"),
    [
        ([], 1.0),
        ([1.0, float("nan")], 1.0),
        ([1.0, float("inf")], 1.0),
        ([1.0, 2.0], 0.0),
        ([1.0, 2.0], -1.0),
        ([1.0, 2.0], float("nan")),
        ([1.0, 2.0], float("inf")),
    ],
)
def test_invalid_input_raises(logits: list[float], temperature: float) -> None:
    with pytest.raises(ValueError):
        calibrated_softmax(logits, temperature)


@given(
    logits=st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=300),
    temperature=st.floats(min_value=1e-3, max_value=1e3),
)
def test_is_a_probability_distribution(logits: list[float], temperature: float) -> None:
    probs = calibrated_softmax(logits, temperature)
    assert len(probs) == len(logits)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
