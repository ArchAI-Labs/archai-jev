"""The logic of the pellet maze demo, played with a fake model (level T0: no download).

``Match`` is the game without the window. Three promises of the demo are checked:

* every move that is played is legal, over whole games;
* the model is not asked when the game has a single legal move, and is asked exactly once
  at every other crossing;
* the move that is played is the model's most probable one, even when the oracle wants another
  (there is no safety net), and the counter says how often that happened.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from archai_jev import Jev
from archai_jev.testing import MockScorer

DEMOS = Path(__file__).resolve().parent.parent / "demos"


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    sys.path.insert(0, str(DEMOS))  # the demo imports _games from its own folder
    try:
        spec = importlib.util.spec_from_file_location(
            "pellet_maze_demo", DEMOS / "010_pellet_maze.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["pellet_maze_demo"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DEMOS))
    return module


def fake_model_decider(demo: ModuleType, scorer: MockScorer, match: Any) -> Any:
    jev = Jev.from_scorer(scorer)
    return lambda state, legal: demo.decide_with_model(jev, state, legal, match.rng)


def test_a_whole_game_plays_only_legal_moves_and_skips_the_model_on_single_moves(
    demo: ModuleType,
) -> None:
    scorer = MockScorer()
    # a fake model plays badly and dies early: many lives, so that the game lasts 120 moves
    match = demo.Match(10**9, lives=50, max_moves=120)
    inner = fake_model_decider(demo, scorer, match)
    played: list[tuple[list[str], Any]] = []

    def decider(state: dict[str, Any], legal: list[str]) -> Any:
        decision = inner(state, legal)
        played.append((legal, decision))
        return decision

    # drive the game by hand so that the legality of EVERY move is checked before it is played
    while match.outcome is None:
        state, legal = match.observe()
        decision = decider(state, legal)
        assert decision.move in legal
        match.apply(decision)
    assert len(played) >= 50
    singles = sum(1 for legal, _d in played if len(legal) == 1)
    assert match.forced == singles
    assert match.asked == len(played) - singles
    # the fake model was asked once at every crossing with several moves, and never at the others
    assert scorer.calls == match.asked
    assert all(d.forced for legal, d in played if len(legal) == 1)
    assert all(not d.forced for legal, d in played if len(legal) > 1)


def test_the_move_played_is_the_models_even_when_the_oracle_disagrees(
    demo: ModuleType,
) -> None:
    scorer = MockScorer()
    match = demo.Match(10**9 + 17, lives=1, max_moves=80)
    inner = fake_model_decider(demo, scorer, match)
    seen: list[Any] = []

    def decider(state: dict[str, Any], legal: list[str]) -> Any:
        decision = inner(state, legal)
        seen.append((state, decision))
        return decision

    match.play(decider)
    asked = [(s, d) for s, d in seen if not d.forced]
    assert asked
    oracle = demo.games.maze_optimal
    expected = sum(1 for s, d in asked if d.move not in oracle(s))
    # the counter is exactly the number of times the played move was not one the oracle accepts;
    # a fake model has no reason to agree with the oracle, so this is not zero
    assert match.disagreements == expected
    assert expected > 0
    # the probabilities the demo shows are the model's and add up to one
    for _s, d in asked:
        assert sum(d.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert d.move == max(d.probabilities, key=d.probabilities.__getitem__)


def test_the_oracle_plays_the_same_game_as_games_play(demo: ModuleType) -> None:
    """``Match`` with one life is the game the model was measured on (same seed, same score)."""
    for seed in (10**9, 10**9 + 17, 10**9 + 34):
        match = demo.Match(seed, lives=1, max_moves=200)
        match.play(
            lambda state, legal, m=match: demo.decide_with_oracle(state, legal, m.rng)
        )
        reference = demo.games.play(
            "maze", seed, demo.games.oracle_controller, max_moves=200
        )
        assert match.score == reference["score"]
        assert match.game.moves == reference["moves"]
        assert (match.outcome == "won") == reference["won"]


def test_lives_start_again_from_the_first_cell_and_keep_the_pellets(
    demo: ModuleType,
) -> None:
    match = demo.Match(10**9, lives=3, max_moves=None)
    rng = random.Random(0)
    deaths_seen = 0
    for _ in range(3000):
        if match.outcome is not None:
            break
        before_lives = match.lives
        eaten_before = len(match.game.pellets) + len(match.game.powers)
        _state, legal = match.observe()
        # a random player dies often: enough to see a respawn
        match.apply(demo.Decision(rng.choice(legal), forced=len(legal) == 1))
        if match.lives < before_lives and match.outcome is None:
            deaths_seen += 1
            assert match.game.pos == match.start
            assert match.game.alive and match.game.power_left == 0
            assert all(
                g["pos"] == g["home"] and not g["scared"] for g in match.game.ghosts
            )
            assert len(match.game.pellets) + len(match.game.powers) <= eaten_before
    assert deaths_seen >= 1
    assert match.outcome in ("died", "won")


def test_the_number_of_ghosts_and_the_chase_are_options_and_the_default_is_the_measured_game(
    demo: ModuleType,
) -> None:
    for ghosts in (1, 2, 3):
        match = demo.Match(10**9, ghosts=ghosts, chase=0.65)
        assert len(match.game.ghosts) == ghosts and match.game.chase == 0.65
    default = demo.Match(10**9)
    reference = demo.games.new_game("maze", 10**9)
    assert len(default.game.ghosts) == 2 and default.game.chase == 0.5
    assert default.game.wall == reference.wall
    assert default.game.pos == reference.pos
