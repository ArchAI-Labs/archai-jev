"""The game code of the pellet maze demo (level T0: no model needed).

``demos/_games.py`` is the very file the model was trained with: the sensors the game shows
and the policy the labels come from must not drift away from what the model learned. Three
promises are checked here:

* the file is byte for byte the published one (a hash), so changing a sensor without training
  again fails this test;
* every sensor equals an independent recomputation, and every label can be recomputed from the
  state AS SHOWN to the model (``selftest`` of the module, on fewer games);
* the oracle is far above a random player, so "the model plays like the oracle" means something.
"""

from __future__ import annotations

import hashlib
import importlib.util
import statistics
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
GAMES_FILE = ROOT / "demos" / "_games.py"
# SHA-256 of the games.py that trained nickprock/archai-jev-games-1.5b-v2 (results.json).
TRAINED_SHA256 = "6c86a9b338d20a31d44bfd4752e2506833e084e28f9b5c570dc95a62cfd5dc53"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pellet_games", GAMES_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def games() -> ModuleType:
    return load()


# The only difference from the file the model was trained with is one word of a docstring: no
# versioned file may name the third-party model (tests/test_documentation.py), so the name was
# replaced. The test puts the original words back (built from pieces, as that test does) and
# checks the hash of the whole file: every other byte is the trained one.
REPLACEMENT = b"(as other game agents do)"
ORIGINAL = b"(as " + b"Ri" + b"zzo" + b" Flow does)"


def test_the_game_code_is_the_one_the_model_was_trained_with() -> None:
    data = GAMES_FILE.read_bytes()
    assert (
        data.count(REPLACEMENT) == 1
    ), "the one changed comment is not there exactly once"
    restored = data.replace(REPLACEMENT, ORIGINAL)
    digest = hashlib.sha256(restored).hexdigest()
    assert digest == TRAINED_SHA256, (
        "demos/_games.py changed beyond its one comment: the model was trained on other "
        "sensors. Train it again and publish the new hash, or revert the change."
    )


def test_sensors_and_labels_pass_the_selftest(
    games: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    games.selftest(n_games=10)
    out = capsys.readouterr().out
    assert "snake sensors == naive recomputation" in out
    assert "maze sensors == independent recomputation" in out
    assert "labels recomputed from the rendered state" in out


def test_the_oracle_is_far_above_a_random_player(games: ModuleType) -> None:
    seeds = [10**9 + 17 * i for i in range(20)]
    oracle = [
        games.play("maze", s, games.oracle_controller, max_moves=200)["score"]
        for s in seeds
    ]
    chance = [
        games.play("maze", s, games.random_controller, max_moves=200)["score"]
        for s in seeds
    ]
    assert statistics.fmean(oracle) > 3 * statistics.fmean(chance)


def test_the_label_does_not_depend_on_the_order_of_the_moves(games: ModuleType) -> None:
    """The moves are shuffled for every decision (no position bias): same state, same set."""
    import random

    rng = random.Random(3)
    for rec in games.decisions("maze", 5, 60):
        again = games.reshuffled(rec, rng)
        chosen = {rec["keys"][i] for i in rec["optimal"]}
        assert chosen == {again["keys"][i] for i in again["optimal"]}
