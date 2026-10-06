import collections
import datetime
import decimal
import enum
import re
import types
from collections import OrderedDict
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from archai_jev import InvalidStateError, render_state


def test_spec_vectors() -> None:
    state = {
        "order": {"id": 1042, "charges": [12.5, 12.5], "note": None, "paid": True},
        "tags": [],
        "history": [{"at": "mon", "ev": "shipped"}, "late"],
    }
    assert render_state(state) == (
        "order:\n  id: 1042\n  charges:\n    - 12.5\n    - 12.5\n  note: \n  paid: True\n"
        "tags:\n\nhistory:\n  - at: mon\n    ev: shipped\n  - late"
    )
    assert render_state(["  a", "\x1c\x1db", "\n\nc", "", None, [1, [2]]]) == (
        "- a\n- b\n- c\n- \n- \n- - 1\n  - - 2"
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (1e16, "- 1e+16"),
        (-0.0, "- -0.0"),
        (1e-05, "- 1e-05"),
        (12345678901234567890123, "- 12345678901234567890123"),
        (True, "- True"),
        (None, "- "),
        (0.1, "- 0.1"),
    ],
)
def test_scalar_rendering(value: Any, text: str) -> None:
    assert render_state([value]) == text


def test_tuple_equals_list_and_order_is_kept() -> None:
    assert render_state((1, 2)) == render_state([1, 2])
    assert render_state(OrderedDict([("b", 1), ("a", 2)])) == "b: 1\na: 2"
    assert render_state("") == ""
    assert render_state({}) == ""
    assert render_state([]) == ""


def test_by_value_never_runs_user_code() -> None:
    class S(str):
        def __str__(self) -> str:
            raise RuntimeError("no")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("no")

    class I(enum.IntEnum):  # noqa: E742
        A = 7

    class F(float):
        def __repr__(self) -> str:
            raise RuntimeError("no")

    class D(dict):  # type: ignore[type-arg]
        def keys(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("no")

        def items(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("no")

    assert render_state([S("x"), I.A, F(1.5), True]) == "- x\n- 7\n- 1.5\n- True"
    assert render_state(D(a=1)) == "a: 1"


@pytest.mark.parametrize(
    "value",
    [
        {1, 2},
        frozenset([1]),
        b"bytes",
        bytearray(b"x"),
        datetime.datetime(2020, 1, 1),
        decimal.Decimal("1.5"),
        range(3),
        types.MappingProxyType({"a": 1}),
        collections.deque([1]),
        object(),
        (x for x in [1]),
    ],
)
def test_unsupported_types_are_rejected(value: Any) -> None:
    with pytest.raises(InvalidStateError, match="unsupported type"):
        render_state({"when": value})


def test_non_numeric_enum_and_dataclass_rejected() -> None:
    import dataclasses

    class Color(enum.Enum):
        RED = 1

    @dataclasses.dataclass
    class P:
        x: int = 1

    for v in (Color.RED, P()):
        with pytest.raises(InvalidStateError):
            render_state([v])


def test_numpy_integers_are_rejected_if_numpy_is_installed() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(InvalidStateError):
        render_state([np.int64(1)])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({1: "a"}, "key 1 at <root> is int, not str"),
        ({(1, 2): "a"}, "is tuple, not str"),
        ({None: "a"}, "key None at <root> is NoneType"),
        ({"k": "bad\ud800"}, "lone surrogate U+D800"),
        ({"bad\udc00": 1}, "lone surrogate U+DC00"),
        ([float("nan")], "non-finite number NaN at [0]"),
        ({"a": float("inf")}, "non-finite number Infinity at a"),
        ({"a": float("-inf")}, "-Infinity"),
        ({"a": 10**4400}, "more than 4300 digits"),
    ],
)
def test_conversion_errors(value: Any, message: str) -> None:
    with pytest.raises(InvalidStateError, match=re.escape(message)):
        render_state(value)


def test_limits() -> None:
    def nested(n: int) -> list[Any]:
        v: list[Any] = []
        for _ in range(n - 1):
            v = [v]
        return v

    render_state(nested(32))
    with pytest.raises(InvalidStateError, match="more than 32 levels"):
        render_state(nested(33))
    render_state("x" * 1_048_576)
    with pytest.raises(InvalidStateError, match="exceeds the maximum of 1048576"):
        render_state("x" * 1_048_577)


def test_cycles_do_not_crash() -> None:
    a: list[Any] = []
    a.append(a)
    with pytest.raises(InvalidStateError, match="more than 32 levels"):
        render_state(a)
    d: dict[str, Any] = {}
    d["self"] = d
    with pytest.raises(InvalidStateError, match="more than 32 levels"):
        render_state(d)


def test_scalar_roots_are_rejected_with_a_hint() -> None:
    for root in (None, 42, 1.5, True):
        with pytest.raises(InvalidStateError, match="wrap it in an object"):
            render_state(root)


json_like = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=5), children, max_size=4),
    max_leaves=12,
)


@given(
    st.one_of(
        st.text(),
        st.lists(json_like, max_size=4),
        st.dictionaries(st.text(max_size=5), json_like, max_size=4),
    )
)
def test_hypothesis_render_is_total_and_deterministic(value: Any) -> None:
    try:
        first = render_state(value)
    except InvalidStateError:
        return  # surrogates, too deep... still a typed error
    assert first == render_state(value)


@given(st.text())
def test_string_root_is_verbatim(s: str) -> None:
    try:
        assert render_state(s) == s
    except InvalidStateError:
        assert any("\ud800" <= c <= "\udfff" for c in s)
