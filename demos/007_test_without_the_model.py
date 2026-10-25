"""007 - Test your own code without the model.

What you will learn
-------------------
* how to build a ``Jev`` backed by a FAKE scorer (``MockScorer``), with no download,
* how to force exact probabilities with ``MockScorer.scripted``,
* how to make the fake misbehave on purpose (``Fault``) to test your error handling,
* how to simulate a slow model (``latency``) and count the calls (``calls``).

Why
---
Your application will contain code AROUND the model: "if the model is unsure, ask a
human", "if it fails, use a default", "retry later". That code must be tested, and you do
not want every test to load 1.6 GB of weights or to depend on what the model decides today.

``archai_jev.testing.MockScorer`` is not a model. Its numbers are a deterministic function
of the input and mean nothing. It exists to run YOUR code deterministically.

Run it
------
    python demos/007_test_without_the_model.py     # runs the checks below, no download
    pytest demos/007_test_without_the_model.py     # the same, as ordinary pytest tests
"""

from __future__ import annotations

import time

from archai_jev import Choice, Jev, JevError
from archai_jev.testing import Fault, MockScorer

# ======================================================================================
# The code under test: the part that is YOURS.
# ======================================================================================

TEAMS = Choice("Which team should handle this message?", ["billing", "support"])


def route(jev: Jev, message: str, minimum_confidence: float = 0.6) -> str:
    """Send a message to a team, or to a human when the model is not sure or fails.

    This is the typical glue code around a decision model: the model proposes, the rules
    around it decide what to do with the proposal.
    """
    try:
        answer = jev.ask(message, {"team": TEAMS}).choices["team"]
    except JevError:
        return "human"  # the library refused to answer: never guess
    if answer.confidence < minimum_confidence:
        return "human"  # the model is not sure enough
    return answer.value


# ======================================================================================
# The tests. They are plain functions named test_*: pytest finds them, and the little
# runner at the bottom of this file also runs them without pytest.
# ======================================================================================


def test_route_follows_a_confident_answer() -> None:
    # ``scripted`` returns EXACTLY these raw scores (logits), one list per question, in
    # the order of the questions. Here the first option ("billing") is far ahead.
    jev = Jev.from_scorer(MockScorer.scripted([[4.0, 0.0]]))
    assert route(jev, "Why was I charged twice?") == "billing"


def test_route_picks_the_second_option() -> None:
    jev = Jev.from_scorer(MockScorer.scripted([[0.0, 4.0]]))
    assert route(jev, "The app does not open") == "support"


def test_route_asks_a_human_when_unsure() -> None:
    # Almost equal scores: the probabilities are close to 50/50, so the confidence is low.
    jev = Jev.from_scorer(MockScorer.scripted([[0.1, 0.0]]))
    assert route(jev, "Hello?") == "human"


def test_route_falls_back_when_the_model_misbehaves() -> None:
    # ``with_fault`` makes the fake return a NaN score. The library refuses to turn it into
    # a probability and raises NumericalError (a JevError): our code must fall back.
    broken = MockScorer(seed=1).with_fault(Fault.nan("team", 0))
    assert route(Jev.from_scorer(broken), "anything") == "human"


def test_the_same_seed_gives_the_same_answers() -> None:
    # Without ``scripted``, the fake invents numbers from the input and a seed. They are
    # arbitrary but reproducible: handy to check that a function is deterministic.
    first = Jev.from_scorer(MockScorer(seed=7)).ask("hello", {"team": TEAMS})
    second = Jev.from_scorer(MockScorer(seed=7)).ask("hello", {"team": TEAMS})
    assert first == second  # answers are immutable values, so == compares them


def test_the_scorer_counts_its_calls() -> None:
    scorer = MockScorer(seed=1)
    jev = Jev.from_scorer(scorer)
    route(jev, "one")
    route(jev, "two")
    assert scorer.calls == 2  # useful to test caching or batching in your code


def test_a_slow_model_can_be_simulated() -> None:
    # ``latency`` makes every call take at least that many seconds: test timeouts,
    # progress bars and async code without waiting for a real model.
    jev = Jev.from_scorer(MockScorer(seed=1, latency=0.2))
    started = time.perf_counter()
    route(jev, "slow")
    assert time.perf_counter() - started >= 0.2


def test_calibration_flags_are_reported() -> None:
    # By default the fake declares NO calibration, and every answer says so. You can
    # declare one to test code that checks ``calibrated``.
    plain = Jev.from_scorer(MockScorer(seed=1)).ask("x", {"team": TEAMS})
    declared = Jev.from_scorer(MockScorer(seed=1, temperature=2.0, calibrated=True))
    assert plain.choices["team"].calibrated is False
    assert declared.ask("x", {"team": TEAMS}).choices["team"].calibrated is True


# ======================================================================================
# A tiny runner, so the file also works without pytest: ``python demos/007_...py``.
# ======================================================================================


def main() -> None:
    print("007 - Testing your code without the model (no download, no weights)")
    print()
    tests = {name: fn for name, fn in globals().items() if name.startswith("test_")}
    failed = 0
    for name, test in tests.items():
        try:
            test()
        except AssertionError:
            failed += 1
            print(f"  FAIL  {name}")
        else:
            print(f"  ok    {name}")
    print()
    print(f"{len(tests) - failed} of {len(tests)} checks passed.")
    print("Remember: the MockScorer is not a model. It tests YOUR code, not the model.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
