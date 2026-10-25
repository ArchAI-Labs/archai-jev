"""How Python values become rows of numbers: types, containers, numpy, user objects."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from archai_jev import calibration as cal


class Exotic:
    """Not a float, but it has ``__float__``."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


class BoomError(Exception):
    pass


class Exploding:
    def __float__(self) -> float:
        raise BoomError("the user's own error")


class FloatSub(float):
    pass


class IntSub(int):
    pass


@pytest.mark.parametrize(
    ("bad", "type_name"),
    [(True, "bool"), ("1", "str"), (None, "NoneType"), (b"1", "bytes"), ([1], "list")],
)
def test_bool_str_none_are_type_errors_with_message(bad: Any, type_name: str) -> None:
    with pytest.raises(TypeError) as info:
        cal.calibrated_softmax_batch([[0.0, 1.0], [0.5, bad]])
    assert str(info.value) == f"logits row 1 option 1 must be a number, got {type_name}"


def test_label_types() -> None:
    rows = [[0.9, 0.1], [0.2, 0.8]]
    for bad, name in [(True, "bool"), (1.0, "float"), ("1", "str"), (None, "NoneType")]:
        with pytest.raises(TypeError) as info:
            labels: list[Any] = [0, bad]
            cal.expected_calibration_error(rows, labels)
        assert str(info.value) == f"labels must be integers, got {name} at position 1"
    with pytest.raises(TypeError, match="labels must be a sequence of integers"):
        cal.expected_calibration_error(rows, 3)  # type: ignore[arg-type]
    # int subclasses and objects with __index__ are integers
    assert cal.expected_calibration_error(rows, [IntSub(0), 1]) == pytest.approx(0.15)


def test_ragged_rows() -> None:
    assert [len(r) for r in cal.calibrated_softmax_batch([[1.0], [1.0, 2.0]])] == [1, 2]


def test_subclasses_and_tuples() -> None:
    expected = cal.calibrated_softmax_batch([[1.0, 2.0, 3.0]])
    assert cal.calibrated_softmax_batch(((1.0, 2.0, 3.0),)) == expected
    assert cal.calibrated_softmax_batch([[FloatSub(1.0), IntSub(2), 3]]) == expected
    assert cal.calibrated_softmax_batch(iter([iter([1.0, 2.0, 3.0])])) == expected  # type: ignore[arg-type]
    mixed: list[list[Any]] = [[Fraction(1), Exotic(2.0), 3]]
    assert cal.calibrated_softmax_batch(mixed) == expected


def test_rows_must_be_sequences() -> None:
    with pytest.raises(TypeError, match="logits must be a sequence of rows, got int"):
        cal.calibrated_softmax_batch(3)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="logits row 1 must be a sequence of numbers, got float"
    ):
        cal.calibrated_softmax_batch([[1.0], 2.0])  # type: ignore[list-item]
    with pytest.raises(
        TypeError, match="probabilities must be a sequence of numbers, got float"
    ):
        cal.choice_confidence(0.5)  # type: ignore[arg-type]


def test_numpy_scalars_and_arrays() -> None:
    np = pytest.importorskip("numpy")
    rows = [[0.9, 0.1], [0.3, 0.7]]
    labels = [0, 1]
    expected = cal.calibration_report(rows, labels)
    assert cal.calibration_report(np.array(rows), np.array(labels)) == expected
    scalars = [[np.float64(0.9), np.float32(0.5)], [np.float64(0.3), np.float64(0.7)]]
    assert cal.calibrated_softmax_batch(scalars) == cal.calibrated_softmax_batch(
        [[0.9, float(np.float32(0.5))], [0.3, 0.7]]
    )
    ints = [[np.int64(1), np.int32(2)]]
    assert cal.calibrated_softmax_batch(ints) == cal.calibrated_softmax_batch([[1, 2]])
    assert cal.choice_confidence(np.array([0.5, 0.5])) == 0.0
    with pytest.raises(TypeError, match="must be a number, got bool"):
        cal.calibrated_softmax_batch(np.array([[True, False]]))


def test_user_float_exception_propagates() -> None:
    with pytest.raises(BoomError, match="the user's own error"):
        cal.calibrated_softmax_batch([[1.0, Exploding()]])  # type: ignore[list-item]


def test_huge_int_overflow() -> None:
    with pytest.raises(OverflowError):
        cal.calibrated_softmax_batch([[10**400, 1]])
    # A label too big for 64 bits is outside every row: a ValueError like any other label.
    with pytest.raises(
        ValueError, match="label at position 0 is 1000000000000000000000000000000"
    ):
        cal.expected_calibration_error([[1.0]], [10**30])


def test_a_long_first_row_does_not_reserve_memory_for_every_row() -> None:
    # The size of the buffer is a hint taken from the first row: 20,001 rows of 1,000,000 numbers
    # would be 160 GB. It must stay a bounded reservation.
    rows = [[0.0] * 1_000_000] + [[0.0, 1.0]] * 20_000
    out = cal.calibrated_softmax_batch(rows)
    assert len(out) == 20_001 and len(out[0]) == 1_000_000
