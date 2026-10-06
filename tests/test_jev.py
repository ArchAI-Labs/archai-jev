import asyncio
import copy
import inspect
import math
import pickle
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import archai_jev
from archai_jev import (
    Answers,
    Choice,
    ChoiceAnswer,
    IncompatibleModelError,
    InvalidQuestionError,
    InvalidStateError,
    Jev,
    JevError,
    ModelInfo,
    ModelVerificationError,
    Noul,
    NumericalError,
    Score,
    ScoreAnswer,
    UnsupportedRequestError,
    YesNo,
    YesNoAnswer,
)
from archai_jev.testing import Fault, MockScorer

STATE = {"ticket": "Shoes arrived late and in the wrong size. Charged twice."}
QUESTIONS = {
    "team": Choice(
        "Which team should handle this?",
        {"returns": "Exchanges, refunds", "shipping": "Delays", "billing": "Charges"},
    ),
    "urgency": Score("How urgent is it?", ["low", "medium", "high"]),
    "angry": YesNo("Is the customer angry?"),
}
LOGITS = [[2.0, 1.0, 0.1], [math.log(0.1), math.log(0.2), math.log(0.7)], [1.0, -1.0]]


def scripted(calibrated: bool = True) -> Jev:
    return Jev.from_scorer(MockScorer.scripted(LOGITS, calibrated=calibrated))


def test_public_surface_is_exactly_the_documented_one() -> None:
    assert sorted(archai_jev.__all__) == sorted(
        [
            "Answer", "Answers", "Choice", "ChoiceAnswer", "IncompatibleModelError",
            "InferenceError", "InvalidQuestionError", "InvalidStateError", "JSONValue", "Jev", "JevError",
            "ModelDownloadError", "ModelInfo", "ModelVerificationError", "Noul",
            "NumericalError", "Question", "Score", "ScoreAnswer", "State",
            "UnsupportedRequestError", "YesNo", "YesNoAnswer", "calibrated_softmax",
            "list_models", "render_state",
        ]
    )  # fmt: skip
    assert "_core" not in archai_jev.__all__
    import archai_jev.testing as t

    assert sorted(t.__all__) == ["Fault", "MockScorer"]
    assert Noul is YesNo
    for name in archai_jev.__all__:
        obj = getattr(archai_jev, name)
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
            assert (obj.__doc__ or "").strip(), name


def test_jev_cannot_be_built_directly_and_accepts_only_known_scorers() -> None:
    with pytest.raises(TypeError, match="from_pretrained"):
        Jev()

    class Fake:
        def score(self, *a: Any) -> Any: ...

    with pytest.raises(TypeError):
        Jev.from_scorer(Fake())  # type: ignore[arg-type]


def test_the_worked_example() -> None:
    result = scripted().ask(STATE, QUESTIONS)
    team = result.choices["team"]
    assert team.value == "returns" and team.index == 0 and team.calibrated
    assert round(team.confidence, 4) == 0.4885
    assert list(team.probabilities) == ["returns", "shipping", "billing"]
    assert sum(team.probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert round(result.scores["urgency"].value, 4) == 1.6
    assert result.scores["urgency"].confidence == pytest.approx(0.4, abs=1e-12)
    assert round(result.yes_nos["angry"].probability, 4) == 0.1192
    assert result.nouls is result.yes_nos
    # no rounding anywhere
    assert result.yes_nos["angry"].probability != round(
        result.yes_nos["angry"].probability, 4
    )


def test_calibrated_mirrors_the_scorer_and_model_info() -> None:
    for flag in (True, False):
        jev = scripted(flag)
        assert jev.model_info.calibrated is flag
        for answer in jev.ask(STATE, QUESTIONS).values():
            assert answer.calibrated is flag
    jev = Jev.from_scorer(MockScorer(temperature=2.5))
    assert (jev.model_info.temperature, jev.model_info.calibrated) == (2.5, False)


def test_answers_is_an_immutable_mapping_with_typed_views() -> None:
    r = scripted().ask(STATE, QUESTIONS)
    assert isinstance(r, Mapping) and len(r) == 3
    assert list(r) == ["team", "urgency", "angry"]
    assert "team" in r and "nope" not in r
    assert list(r.keys()) == list(r) and len(list(r.values())) == 3
    assert [type(v) for v in r.values()] == [ChoiceAnswer, ScoreAnswer, YesNoAnswer]
    assert list(r.choices) == ["team"] and list(r.scores) == ["urgency"]
    with pytest.raises(KeyError, match="available"):
        r["nope"]
    with pytest.raises(AttributeError):
        r.extra = 1  # type: ignore[attr-defined]
    assert r == scripted().ask(STATE, QUESTIONS)
    assert pickle.loads(pickle.dumps(r)) == r
    with pytest.raises(TypeError):
        r.choices["x"] = None  # type: ignore[index]


def test_answer_types_value_semantics() -> None:
    r = scripted().ask(STATE, QUESTIONS)
    for a in r.values():
        assert pickle.loads(pickle.dumps(a)) == a
        assert copy.deepcopy(a) == a
        assert hash(a) == hash(copy.copy(a))
        with pytest.raises(AttributeError):
            a.calibrated = False
        assert "calibrated=True" in repr(a)
    assert not hasattr(r.yes_nos["angry"], "confidence")
    match r["team"]:
        case ChoiceAnswer(value=v, confidence=c) if c < 0.5:
            assert v == "returns"
        case _:
            raise AssertionError("no match")


def test_input_errors_are_typed_and_happen_before_the_scorer() -> None:
    scorer = MockScorer(1)
    jev = Jev.from_scorer(scorer)
    for bad_state in ({"x": {1}}, None, 42, {1: 2}, float("nan")):
        with pytest.raises(InvalidStateError):
            jev.ask(bad_state, QUESTIONS)
    for bad_q in ({}, {"": QUESTIONS["angry"]}, {"x" * 129: QUESTIONS["angry"]},
                  {" a": QUESTIONS["angry"]}, {"a\x00": QUESTIONS["angry"]},
                  {"a": {"instructions": "x"}}, {1: QUESTIONS["angry"]}):  # fmt: skip
        with pytest.raises(InvalidQuestionError):
            jev.ask(STATE, bad_q)  # type: ignore[arg-type]
    many = {f"q{i}": YesNo("?") for i in range(129)}
    with pytest.raises(InvalidQuestionError, match="129 questions"):
        jev.ask(STATE, many)
    assert scorer.calls == 0
    not_mappings: list[Any] = [[], "abc", None]
    for not_a_mapping in not_mappings:
        with pytest.raises(TypeError):
            jev.ask(STATE, not_a_mapping)
    with pytest.raises(TypeError):
        jev.ask_many(5, QUESTIONS)  # type: ignore[arg-type]
    assert scorer.calls == 0
    with pytest.raises(
        InvalidQuestionError,
        match='question "x" must be a Choice, Score or YesNo, got dict',
    ):
        jev.ask(STATE, {"x": {}})


def test_errors_are_jeverrors_and_value_or_runtime_errors() -> None:
    assert issubclass(InvalidStateError, (JevError, ValueError))
    assert issubclass(InvalidQuestionError, (JevError, ValueError))
    assert issubclass(UnsupportedRequestError, (JevError, ValueError))
    runtime_errors: tuple[type[JevError], ...] = (
        IncompatibleModelError,
        ModelVerificationError,
        NumericalError,
    )
    for runtime_cls in runtime_errors:
        assert issubclass(runtime_cls, (JevError, RuntimeError))
    assert issubclass(archai_jev.ModelDownloadError, (JevError, OSError))
    for cls in (
        UnsupportedRequestError,
        IncompatibleModelError,
        ModelVerificationError,
    ):
        e = pickle.loads(pickle.dumps(cls("boom")))
        assert str(e) == "boom" and cls.__doc__


def test_ask_many_order_empty_generators_and_prefixes() -> None:
    jev = Jev.from_scorer(MockScorer(3))
    states = [{"a": str(i)} for i in range(5)]
    assert jev.ask_many(states, QUESTIONS) == [jev.ask(s, QUESTIONS) for s in states]
    assert jev.ask_many([], QUESTIONS) == []
    assert jev.ask_many((s for s in states), QUESTIONS) == jev.ask_many(
        states, QUESTIONS
    )
    scorer = MockScorer(3)
    jev2 = Jev.from_scorer(scorer)
    with pytest.raises(InvalidStateError, match=r"^states\[2\]: "):
        jev2.ask_many([{"a": "1"}, {"a": "2"}, {"a": {1}}], QUESTIONS)
    assert scorer.calls == 0


@pytest.mark.parametrize(
    "fault",
    [
        Fault.nan("urgency", 1),
        Fault.pos_inf("urgency", 1),
        Fault.neg_inf("urgency", 1),
        Fault.wrong_option_count("team", 2),
        Fault.wrong_question_count(2),
    ],
)
def test_faults_never_return_answers(fault: Fault) -> None:
    jev = Jev.from_scorer(MockScorer(7).with_fault(fault))
    with pytest.raises(NumericalError):
        jev.ask(STATE, QUESTIONS)
    with pytest.raises(NumericalError):
        jev.ask_many([STATE, STATE], QUESTIONS)
    with pytest.raises(NumericalError):
        asyncio.run(jev.aask(STATE, QUESTIONS))


def test_fault_message_names_question_and_option() -> None:
    jev = Jev.from_scorer(MockScorer(7).with_fault(Fault.nan("urgency", 1)))
    with pytest.raises(NumericalError, match='option 1 of question "urgency"'):
        jev.ask(STATE, QUESTIONS)


def test_mock_scorer_argument_validation_and_immutability() -> None:
    with pytest.raises(ValueError):
        MockScorer(-1)
    with pytest.raises(ValueError):
        MockScorer(2**64)
    with pytest.raises(TypeError):
        MockScorer("1")  # type: ignore[arg-type]
    for t in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MockScorer(temperature=t)
    with pytest.raises(ValueError):
        MockScorer(latency=-0.1)
    base = MockScorer(5)
    faulty = base.with_fault(Fault.nan("urgency", 0))
    jev = Jev.from_scorer(base)
    jev.ask(STATE, QUESTIONS)
    assert base.calls == 1 and faulty.calls == 0
    assert MockScorer(5).calls == 0


def test_mock_is_deterministic_and_isolated() -> None:
    a = Jev.from_scorer(MockScorer(11)).ask(STATE, QUESTIONS)
    b = Jev.from_scorer(MockScorer(11)).ask(STATE, QUESTIONS)
    c = Jev.from_scorer(MockScorer(12)).ask(STATE, QUESTIONS)
    assert a == b and a != c
    alone = Jev.from_scorer(MockScorer(11)).ask(STATE, {"x": QUESTIONS["angry"]})
    assert alone["x"] == a["angry"]
    reordered = Jev.from_scorer(MockScorer(11)).ask(
        STATE, {k: QUESTIONS[k] for k in ("angry", "team", "urgency")}
    )
    assert reordered["angry"] == a["angry"] and reordered["team"] == a["team"]


def test_repr_and_model_info() -> None:
    jev = scripted()
    assert repr(jev) == "Jev(model='mock')"
    info = jev.model_info
    assert info == jev.model_info and isinstance(info, ModelInfo) and hash(info)
    assert (info.family, info.head, info.template, info.revision) == ("mock",) * 3 + (
        None,
    )
    with pytest.raises(AttributeError):
        info.name = "x"  # type: ignore[misc]


def test_from_pretrained_has_no_switch_to_disable_a_check() -> None:
    sig = inspect.signature(Jev.from_pretrained)
    for forbidden in ("strict", "check", "verify", "validate"):
        assert forbidden not in sig.parameters


def test_threads_share_one_jev() -> None:
    jev = Jev.from_scorer(MockScorer(9))
    states = [{"n": str(i)} for i in range(8)]
    expected = [jev.ask(s, QUESTIONS) for s in states]

    def work(i: int) -> bool:
        s = states[i % 8]
        return jev.ask(s, QUESTIONS) == expected[i % 8] and jev.ask_many(
            [s], QUESTIONS
        ) == [expected[i % 8]]

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert all(pool.map(work, range(16 * 40), timeout=120))


def test_gil_is_released_during_ask() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.2))
    ticks = 0
    stop = threading.Event()

    def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            time.sleep(0.01)
            ticks += 1

    t = threading.Thread(target=ticker)
    t.start()
    try:
        # Reference: how many ticks fit in 0.2 s on this machine while the main thread merely
        # sleeps (a CI runner's timer can be several times coarser than 10 ms).
        before = ticks
        time.sleep(0.2)
        baseline = ticks - before
        before = ticks
        jev.ask(STATE, QUESTIONS)
        first = ticks - before
        before = ticks
        jev.ask_many([STATE], QUESTIONS)
        second = ticks - before
    finally:
        stop.set()
        t.join()
    # With the GIL held for the whole call the ticker would get no tick at all.
    floor = max(2, baseline // 2)
    assert first >= floor and second >= floor, (baseline, first, second)


def test_asyncio() -> None:
    jev = Jev.from_scorer(MockScorer(1, latency=0.2))

    async def main() -> None:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        # Reference for this machine: ticks that fit in 0.2 s of plain awaiting.
        await asyncio.sleep(0)
        ticks = 0
        await asyncio.sleep(0.2)
        baseline = ticks
        sync = jev.ask(STATE, QUESTIONS)
        ticks = 0
        assert await jev.aask(STATE, QUESTIONS) == sync
        # A blocking call would leave the event loop without a single tick.
        assert ticks >= max(2, baseline // 2), (baseline, ticks)
        assert await jev.aask_many([STATE], QUESTIONS) == [sync]
        task = asyncio.create_task(jev.aask(STATE, QUESTIONS))
        await asyncio.sleep(0.02)
        start = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.perf_counter() - start < 0.1
        assert await jev.aask(STATE, QUESTIONS) == sync  # still usable
        t.cancel()
        gathered = await asyncio.gather(*(jev.aask(STATE, QUESTIONS) for _ in range(3)))
        assert all(g == sync for g in gathered)

    asyncio.run(main())


def test_answers_constructor_roundtrip() -> None:
    r = scripted().ask(STATE, QUESTIONS)
    assert Answers(dict(r)) == r
