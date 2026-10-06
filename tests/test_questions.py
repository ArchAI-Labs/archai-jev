import copy
import pickle
from collections import OrderedDict

import pytest
from hypothesis import given
from hypothesis import strategies as st

from archai_jev import Choice, InvalidQuestionError, Noul, Score, YesNo


def test_noul_is_yesno() -> None:
    assert Noul is YesNo
    assert repr(Noul("Is it?")).startswith("YesNo(")


def test_choice_accepts_dict_list_and_tuple_by_position_and_name() -> None:
    a = Choice("Which?", {"a": "first", "b": None})
    b = Choice(instructions="Which?", criteria={"a": "first", "b": None})
    c = Choice("Which?", ["a", "b"])
    d = Choice("Which?", ("a", "b"))
    assert a == b
    assert c == d
    assert list(a.criteria) == ["a", "b"]
    assert c.criteria == {"a": None, "b": None}


def test_score_and_yesno_shapes() -> None:
    assert Score("How?", ["low", "high"]).criteria == ("low", "high")
    assert Score("How?", ("low", "high")).criteria == ("low", "high")
    assert YesNo("Is it?").criteria == {}
    assert YesNo("Is it?", {"true": "yes!", "false": None}).criteria == {"true": "yes!"}


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Choice("q", []), "choice has 0 options; it needs between 1 and 255"),
        (
            lambda: Choice("q", [f"k{i}" for i in range(256)]),
            "choice has 256 options; it needs between 1 and 255",
        ),
        (lambda: Score("q", []), "score has 0 levels; it needs between 1 and 255"),
        (
            lambda: Score("q", [f"l{i}" for i in range(256)]),
            "score has 256 levels; it needs between 1 and 255",
        ),
        (lambda: Choice("", ["a"]), "instructions are empty or blank"),
        (lambda: Choice(chr(0xA0) + " ", ["a"]), "instructions are empty or blank"),
        (lambda: Choice("q", ["a", " "]), "the key of option 1 is empty or blank"),
        (
            lambda: Choice("q", ["a", "b", "a"]),
            'the key "a" appears at options 0 and 2',
        ),
        (lambda: Score("q", ["a", ""]), "level 1 is empty or blank"),
        (lambda: Score("q", ["a", "a"]), "options 0 and 1 would read exactly the same"),
        (
            lambda: Choice("q", {"a: b": None, "a": "b"}),
            "would read exactly the same",
        ),
        (lambda: Choice("x" * 262_144, ["k"]), "is 262145 bytes long"),
    ],
)
def test_construction_errors(make, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidQuestionError, match=message):
        make()


def test_boundaries_are_accepted() -> None:
    Choice("q", ["a"])
    Choice("q", [f"k{i}" for i in range(255)])
    Score("q", ["a"])
    Score("q", [f"l{i}" for i in range(255)])
    Choice("x" * 262_143, ["k"])


@pytest.mark.parametrize(
    "make",
    [
        lambda: Choice(42, ["a"]),
        lambda: Choice(None, ["a"]),
        lambda: Choice({"a"}, ["a"]),
        lambda: Choice("q", "abc"),
        lambda: Choice("q", 3),  # type: ignore[arg-type]
        lambda: Choice("q", [1, 2]),  # type: ignore[list-item]
        lambda: Choice("q", {1: "a"}),  # type: ignore[dict-item]
        lambda: Score("q", "abc"),
        lambda: Score("q", [1, 2]),
        lambda: Score("q", None),  # type: ignore[arg-type]
        lambda: YesNo("q", {"TRUE": "x"}),
        lambda: YesNo("q", {"maybe": "x"}),
        lambda: YesNo("q", ["true"]),  # type: ignore[arg-type]
    ],
)
def test_wrong_types_are_validation_errors(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidQuestionError):
        make()


def test_yesno_unknown_key_message() -> None:
    with pytest.raises(InvalidQuestionError) as e:
        YesNo("q", {"TRUE": "x"})
    assert 'only accepts the keys "true" and "false", got "TRUE"' in str(e.value)


def test_structured_entries() -> None:
    q = Choice({"potential": {"id": 1}, "question": "dup?"}, {"a": {"what": ["x", 1]}})
    assert q.instructions["potential"]["id"] == 1
    assert q.criteria["a"]["what"] == ("x", 1)
    assert q == pickle.loads(pickle.dumps(q))
    with pytest.raises(InvalidQuestionError, match="got number"):
        Choice("q", {"a": 5})
    with pytest.raises(InvalidQuestionError):
        Choice({}, ["a"])  # renders to nothing


def test_immutability_and_defensive_copy() -> None:
    crit = {"a": None}
    q = Choice("q", crit)
    crit["b"] = None
    assert list(q.criteria) == ["a"]
    with pytest.raises(AttributeError):
        q.instructions = "x"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        q.extra = 1  # type: ignore[attr-defined]
    assert not hasattr(q, "__dict__")
    with pytest.raises(TypeError):
        q.criteria["b"] = None  # type: ignore[index]
    levels = ["lo", "hi"]
    s = Score("q", levels)
    levels.append("x")
    assert s.criteria == ("lo", "hi")


def test_final_classes() -> None:
    with pytest.raises(TypeError):

        class X(Choice):
            pass


def test_equality_hash_repr_roundtrip() -> None:
    a = Choice("Which team?", {"a": "x", "b": None})
    assert a == Choice("Which team?", {"a": "x", "b": None})
    assert hash(a) == hash(Choice("Which team?", {"a": "x", "b": None}))
    assert a != Choice("Which team?", {"b": None, "a": "x"})  # order changes the prompt
    assert a != Choice("Which team!", {"a": "x", "b": None})
    assert a != Score("Which team?", ["a", "b"])
    assert len({a, Choice("Which team?", {"a": "x", "b": None})}) == 1
    for q in (a, Score("s", ["l", "h"]), YesNo("y", {"true": "t"}), YesNo("y")):
        ns = {"Choice": Choice, "Score": Score, "YesNo": YesNo}
        assert eval(repr(q), ns) == q


def test_equality_distinguishes_types_of_scalars() -> None:
    assert Choice({"a": 1}, ["k"]) != Choice({"a": True}, ["k"])
    assert Choice({"a": 1}, ["k"]) != Choice({"a": 1.0}, ["k"])


def test_pickle_copy_and_match() -> None:
    qs = [Choice("q", {"a": None}), Score("q", ["l", "h"]), YesNo("q", {"false": "no"})]
    for q in qs:
        assert pickle.loads(pickle.dumps(q)) == q
        assert copy.copy(q) == q
        assert copy.deepcopy(q) == q
    match qs[0]:
        case Choice(instructions=i):
            assert i == "q"
        case _:
            raise AssertionError("did not match")


def test_subclass_by_value_not_by_method() -> None:
    class Evil(str):
        def __str__(self) -> str:
            raise RuntimeError("must not run")

    q = Choice(Evil("pick"), OrderedDict([("a", None)]))
    assert q.instructions == "pick"


@given(
    st.text(min_size=1).filter(lambda s: s.strip() != "" and not s.isspace()),
    st.lists(
        st.text(min_size=1, max_size=6).filter(lambda s: not s.isspace()),
        min_size=1,
        max_size=6,
        unique=True,
    ),
)
def test_hypothesis_equality_and_roundtrip(instr: str, keys: list[str]) -> None:
    try:
        q = Choice(instr, keys)
    except InvalidQuestionError:
        return
    assert q == Choice(instr, list(keys))
    assert hash(q) == hash(Choice(instr, tuple(keys)))
    assert pickle.loads(pickle.dumps(q)) == q
