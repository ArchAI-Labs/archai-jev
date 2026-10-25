"""010 - Pellet maze: a maze game played by a small specialist model.

What you are looking at
-----------------------
A maze with pellets and ghosts. The player eats every pellet; a power pellet makes the ghosts
scared for a while, and a scared ghost can be eaten. Touching a ghost that is not scared costs
a life (you have three). At every crossing the model decides where to go, and the game plays
that move. There is no safety net: if the model walks into a ghost, the ghost wins.

What the model does, and what it does not (please read before recording a video)
--------------------------------------------------------------------------------
The game computes the SENSORS, the model only chooses the move. For every legal move the game
works out what is in the next cell, how far the nearest dangerous ghost is, how far the nearest
pellet is, and so on (``game-sensors-v1``, see ``_games.py``). The model reads those numbers as
a JSON state and answers ONE ``Choice`` question over the legal moves:

    state     = {"game": "maze", "power_left": 0, "pellets_left": 63, "moves": {"up": {...}, ...}}
    question  = Choice("Pellet maze: which move should the player make? ...", ["up", "left"])

and the answer is a typed value with the probability of every move. The model does NOT read a
board and does not reason about space: it imitates a fixed policy (the "oracle") that is a function
of the numbers it is shown. So it plays like that policy, not better, and the policy is not a
champion (it loses most of its games). Nothing here is a claim about spatial reasoning.

If a crossing has a single legal move, the game plays it and the model is not asked. The screen
shows the probability of every move on the cells next to the player, the decisions per second
measured LIVE on your machine, and how many times the oracle would have chosen another move.

Run it
------
    python demos/010_pellet_maze.py                     # play (first run downloads ~1.6 GB)
    python demos/010_pellet_maze.py --device cuda:0     # on a GPU (see demos/009 and docs/gpu.md)
    python demos/010_pellet_maze.py --oracle            # the oracle plays: the ceiling of the model
    python demos/010_pellet_maze.py --headless          # no window: whole games, numbers on the console

On a CPU the model needs a little more than half a second per decision, so the game is slow:
it waits for the model. Keys: SPACE = start / play again, ESC = quit.

Options: ``--seed N`` (first game), ``--tick-ms N`` (a minimum pause between moves, to slow the
game down), ``--device`` (default ``cpu``), ``--model PATH`` (a local folder with an
``archai-jev-manifest.json`` instead of the model of the registry), ``--max-seconds N``, ``--ghosts 1|2|3`` and ``--chase 0.35|0.5|0.65`` (the difficulty: the
model saw all of them in training, but it was measured only with 2 ghosts and a chase of 0.5,
the defaults).
For ``--headless``: ``--games N``, ``--max-moves N``, ``--regimes``, ``--pause S``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import queue
import random
import re
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A build of the library with CUDA needs the folder of the CUDA runtime added by hand on
# Windows (Python does not search PATH for DLLs), BEFORE the import: see docs/gpu.md.
if hasattr(os, "add_dll_directory") and os.environ.get("CUDA_PATH"):
    _cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
    if _cuda_bin.is_dir():
        os.add_dll_directory(str(_cuda_bin))

import _games as games  # noqa: E402  (the very file the model was trained with)

from archai_jev import Choice, Jev  # noqa: E402

try:  # tkinter ships with Python, but some Linux distros package it separately.
    import tkinter as tk
except ImportError:  # pragma: no cover - only matters for --headless on such systems
    tk = None  # type: ignore[assignment]

MODEL_NAME = "nickprock/archai-jev-games-1.5b-v2"
FIRST_SEED = 10**9  # the seeds of the evaluation of the model: never seen in training
SEED_STEP = 17


# ======================================================================================
# 1. ONE DECISION: the game shows the sensors, the model picks the move
# ======================================================================================
@dataclass(frozen=True)
class Decision:
    """What happened at one crossing."""

    move: str
    forced: bool = False  # a single legal move: the game played it, nobody was asked
    probabilities: dict[str, float] = field(default_factory=dict)  # move -> P, empty if forced
    ms: float = 0.0  # time the model took
    matches_oracle: bool = True  # is the move one the oracle would have accepted?


def decide_with_model(
    jev: Jev, state: dict[str, Any], legal: list[str], rng: random.Random
) -> Decision:
    """Ask the model which of the legal moves to play and return its most probable one.

    The order of the moves is shuffled for every decision (the model was trained like that, to
    have no bias for a position). The move that is played is ALWAYS the most probable one of
    the model: there is nothing that replaces it.
    """
    if len(legal) == 1:
        return Decision(legal[0], forced=True)
    shown = games.record_of(state, [], rng)  # the same state with the moves in another order
    question = Choice(shown["question"]["instructions"], {m: None for m in shown["keys"]})
    started = time.perf_counter()
    answer = jev.ask(shown["state"], {"move": question}).choices["move"]
    ms = (time.perf_counter() - started) * 1000.0
    move = str(answer.value)
    return Decision(
        move,
        probabilities={m: float(p) for m, p in answer.probabilities.items()},
        ms=ms,
        matches_oracle=move in games.maze_optimal(state),
    )


def decide_with_oracle(
    state: dict[str, Any], legal: list[str], rng: random.Random
) -> Decision:
    """The oracle plays (``--oracle``): a random move among the best ones, the ceiling."""
    if len(legal) == 1:
        return Decision(legal[0], forced=True)
    return Decision(rng.choice(games.maze_optimal(state)))


# ======================================================================================
# 2. A MATCH: the game, the lives and the counters (no window, no model inside)
# ======================================================================================
class Match:
    """One game of the maze. The window, ``--headless`` and the tests all play through it.

    ``observe()`` gives the state the model would see; ``apply(decision)`` plays the move. With
    ``lives=1`` and ``max_moves`` it is the same game as ``games.play`` (the one the model was
    measured on); the window uses three lives: after a death the player starts again from
    the first cell, the ghosts go home, the pellets already eaten stay eaten.
    """

    def __init__(
        self,
        seed: int,
        lives: int = 1,
        max_moves: int | None = None,
        ghosts: int = 2,
        chase: float = 0.5,
    ) -> None:
        self.seed = seed
        # The same maze as ``games.new_game("maze", seed)`` when ghosts=2 and chase=0.5 (the
        # setting the model was measured on). The model was trained with 1 to 3 ghosts and
        # a chase of 0.35, 0.5 or 0.65, so the others are inside what it saw, but not measured.
        self.game = games.Maze(random.Random(seed), 13, 11, ghosts=ghosts, chase=chase)
        # the generator of ``games.play``: it only shuffles the moves of the model's questions
        self.rng = random.Random(seed * 7919 + 1)
        self.lives = lives
        self.max_moves = max_moves
        self.start = self.game.pos
        self.outcome: str | None = None  # "won", "died" or "timeout"
        self.deaths = 0
        self.asked = 0  # decisions the model took
        self.forced = 0  # moves the game played because there was only one
        self.disagreements = 0  # the oracle would have chosen another move
        self.latencies: list[float] = []

    @property
    def score(self) -> int:
        return int(self.game.score())

    def observe(self) -> tuple[dict[str, Any], list[str]]:
        return self.game.sensors(), self.game.legal()

    def apply(self, decision: Decision) -> None:
        """Play ``decision.move`` and update lives, outcome and counters."""
        g = self.game
        if decision.forced:
            self.forced += 1
        else:
            self.asked += 1
            self.latencies.append(decision.ms)
            if not decision.matches_oracle:
                self.disagreements += 1
        g.step(decision.move)
        if g.won():
            self.outcome = "won"
        elif not g.alive:
            self.deaths += 1
            self.lives -= 1
            if self.lives <= 0:
                self.outcome = "died"
            else:
                self._respawn()
        elif self.max_moves is not None and g.moves >= self.max_moves:
            self.outcome = "timeout"

    def _respawn(self) -> None:
        g = self.game
        g.pos = self.start
        for ghost in g.ghosts:
            ghost["pos"], ghost["scared"] = ghost["home"], False
        g.power_left = 0
        g.alive = True

    def play(self, decider: Callable[[dict[str, Any], list[str]], Decision]) -> None:
        """Play to the end (death of the last life, win or ``max_moves``)."""
        while self.outcome is None:
            state, legal = self.observe()
            self.apply(decider(state, legal))


# ======================================================================================
# 3. HEADLESS MODE: whole games with fixed seeds, no window
# ======================================================================================
def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def run_regime(
    label: str,
    decider_for: Callable[[Match], Callable[[dict[str, Any], list[str]], Decision]],
    games_n: int,
    max_moves: int,
    pause: float,
    ghosts: int = 2,
    chase: float = 0.5,
) -> None:
    """Play ``games_n`` games and print what was measured. ``pause`` sleeps after every
    decision of the model (the "after a pause" regime: the processor has cooled down)."""
    seeds = [FIRST_SEED + SEED_STEP * i for i in range(games_n)]
    matches: list[Match] = []
    started = time.perf_counter()
    for seed in seeds:
        match = Match(seed, lives=1, max_moves=max_moves, ghosts=ghosts, chase=chase)
        decide = decider_for(match)

        def paused(
            state: dict[str, Any],
            legal: list[str],
            decide: Callable[[dict[str, Any], list[str]], Decision] = decide,
        ) -> Decision:
            decision = decide(state, legal)
            if pause and not decision.forced:
                time.sleep(pause)
            return decision

        match.play(paused)
        matches.append(match)
    wall = time.perf_counter() - started

    scores = [m.score for m in matches]
    asked = sum(m.asked for m in matches)
    latencies = [ms for m in matches for ms in m.latencies]
    disagreements = sum(m.disagreements for m in matches)
    print(
        f"\n[{label}] {games_n} games, at most {max_moves} moves, seeds {seeds[0]} + {SEED_STEP}*i, "
        f"{ghosts} ghosts, chase {chase}"
    )
    if (ghosts, chase) != (2, 0.5):
        print("  (the model was measured with 2 ghosts and chase 0.5, not with this setting)")
    print(f"  score      : mean {statistics.fmean(scores):.1f}, median {statistics.median(scores):.1f}")
    print(
        f"  outcomes   : died {sum(m.outcome == 'died' for m in matches)}, "
        f"won {sum(m.outcome == 'won' for m in matches)}, "
        f"time out {sum(m.outcome == 'timeout' for m in matches)}"
    )
    print(
        f"  decisions  : {asked} crossings with a choice, "
        f"{sum(m.forced for m in matches)} single moves played by the game"
    )
    if latencies and max(latencies) > 0:  # the oracle takes no time and has no model to measure
        print(
            f"  agreement  : {100 * (1 - disagreements / asked):.1f}% of the moves are ones the oracle accepts"
        )
        print(
            f"  latency    : mean {statistics.fmean(latencies):.0f} ms, p50 {statistics.median(latencies):.0f} ms, "
            f"p95 {percentile(latencies, 0.95):.0f} ms"
        )
        print(
            f"  throughput : {1000.0 / statistics.fmean(latencies):.2f} decisions/s of model time "
            f"({asked / wall:.2f}/s of wall time{', pauses included' if pause else ''})"
        )


def run_headless(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if args.oracle:
        print("The oracle plays (no model): this is the ceiling of the model.")
        run_regime(
            "oracle",
            lambda match: lambda state, legal: decide_with_oracle(state, legal, match.rng),
            args.games,
            args.max_moves,
            0.0,
            args.ghosts,
            args.chase,
        )
        return 0
    jev = load_model(args)
    info = jev.model_info
    print(f"Model: {info.name} ({info.dtype}) on {info.device}")
    warm_up(jev)

    def model_decider(
        match: Match,
    ) -> Callable[[dict[str, Any], list[str]], Decision]:
        return lambda state, legal: decide_with_model(jev, state, legal, match.rng)

    if args.regimes in ("both", "back-to-back"):
        run_regime(
            "one call right after the other",
            model_decider,
            args.games,
            args.max_moves,
            0.0,
            args.ghosts,
            args.chase,
        )
    if args.regimes in ("both", "pause"):
        run_regime(
            f"{args.pause:g} s pause after every decision",
            model_decider,
            args.games,
            args.max_moves,
            args.pause,
            args.ghosts,
            args.chase,
        )
    return 0


def load_model(args: argparse.Namespace) -> Jev:
    """The model of the registry (downloaded once and verified) or a local folder."""
    print("Loading the model (the first run downloads it)...")
    name: str | Path = Path(args.model) if Path(args.model).is_dir() else args.model
    return Jev.from_pretrained(name, device=args.device)


def warm_up(jev: Jev) -> None:
    """The very first call is slower than the others (caches and threads are cold)."""
    match = Match(FIRST_SEED)
    for _ in range(40):
        state, legal = match.observe()
        if len(legal) > 1:
            decide_with_model(jev, state, legal, match.rng)
            return
        match.apply(Decision(legal[0], forced=True))


# ======================================================================================
# 4. TALKING TO THE MODEL WITHOUT FREEZING THE WINDOW
# ======================================================================================
# Tkinter must only be touched from the main thread and a decision takes from a few tens of
# milliseconds (GPU) to more than half a second (CPU): the decision runs in a worker thread.
class Worker:
    """A background thread that takes one decision at a time."""

    def __init__(self, decider: Callable[[dict[str, Any], list[str]], Decision]) -> None:
        self._decider = decider
        self._requests: queue.Queue[tuple[dict[str, Any], list[str]] | None] = queue.Queue()
        self.results: queue.Queue[Decision | Exception] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, state: dict[str, Any], legal: list[str]) -> None:
        self._requests.put((state, legal))

    def close(self) -> None:
        self._requests.put(None)

    def _loop(self) -> None:
        while (request := self._requests.get()) is not None:
            try:
                self.results.put(self._decider(*request))
            except Exception as exc:  # JevError subclasses; never kill the thread
                self.results.put(exc)


# ======================================================================================
# 5. THE WINDOW
# ======================================================================================
CELL = 48
MX, MY = 16, 64  # top-left corner of the maze on the canvas
W, H = 960, 640
FONT = "Courier New"
BG = "#070714"
PANEL = "#0d0d2b"
WALL = "#0d1b5e"
WALL_EDGE = "#2b44ff"
CYAN = "#00e5ff"
MAGENTA = "#ff4fd8"
YELLOW = "#ffe600"
GREEN = "#39ff14"
RED = "#ff2e63"
ORANGE = "#ff9f1c"
WHITE = "#e8f6ff"
DIM = "#3a3f6b"
GHOST_COLORS = (RED, MAGENTA, CYAN, ORANGE)
MOUTH = {"right": 0, "up": 90, "left": 180, "down": 270}
LIVES = 3
WINDOW_MAX_MOVES = 600  # a game that wanders for ever stops here


def font(size: int) -> tuple[str, int, str]:
    """A bold font sized in screen pixels (a negative size is pixels in Tk)."""
    return (FONT, -round(size * 4 / 3), "bold")


class _LogToQueue(logging.Handler):
    """Collects the library's log lines so the loading screen can show the download."""

    def __init__(self, sink: queue.Queue[str]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.put(record.getMessage())


class Window:
    """Owns the window, the current match and the model."""

    def __init__(self, args: argparse.Namespace) -> None:
        if tk is None:
            raise SystemExit("tkinter is not available; try --headless instead")
        self.args = args
        try:  # Windows: ask for crisp pixels instead of a blurry, upscaled window
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (ImportError, AttributeError, OSError):
            pass
        self.root = tk.Tk()
        self.root.title("Pellet maze - played by archai-jev")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=W, height=H, bg=BG, highlightthickness=0)
        self.canvas.pack()
        self.root.bind("<space>", lambda _e: self._on_space())
        self.root.bind("<Escape>", lambda _e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.mode = "loading"  # loading -> title -> play -> over | error
        self.jev: Jev | None = None
        self.worker: Worker | None = None
        self.match: Match | None = None
        self.load_error: str | None = None
        self.load_text = "Starting"
        self.load_percent = 0.0
        self.log_lines: queue.Queue[str] = queue.Queue()
        self.games_played = 0
        self.high_score = 0
        self.busy = False  # a decision is being taken by the worker
        self.hold_until = 0.0  # next decision not before this time (--tick-ms, and to be seen)
        self.shown: tuple[tuple[int, int], Decision] | None = None  # the overlay of probabilities
        self.shown_until = 0.0
        self.flash_until = 0.0
        self.stamps: list[float] = []  # when each model decision arrived (live rate)
        self.peak_rate = 0.0
        self.last_move = "right"
        self.play_start = 0.0
        self.fps = 30.0
        self._last_frame = time.perf_counter()
        self._ids = itertools.count(1)

        self._start_loading()
        if args.max_seconds:
            self.root.after(int(args.max_seconds * 1000), self._quit)

    # ---------------------------------------------------------------- loading
    def _start_loading(self) -> None:
        if self.args.oracle:
            self.load_text = "The oracle plays"
            self.worker = Worker(
                lambda state, legal: decide_with_oracle(state, legal, self._current_rng())
            )
            return
        logger = logging.getLogger("archai_jev")
        logger.setLevel(logging.INFO)
        logger.addHandler(_LogToQueue(self.log_lines))

        def load() -> None:
            try:
                jev = load_model(self.args)
                self.load_text = "Warming up the model"
                warm_up(jev)
                self.jev = jev
            except Exception as exc:  # shown on the screen, never replaced by anything else
                self.load_error = f"{type(exc).__name__}: {exc}"

        threading.Thread(target=load, daemon=True).start()

    def _current_rng(self) -> random.Random:
        assert self.match is not None
        return self.match.rng

    def _drain_log_lines(self) -> None:
        progress = re.compile(r"(\d+) of (\d+) bytes")
        while True:
            try:
                line = self.log_lines.get_nowait()
            except queue.Empty:
                return
            m = progress.search(line)
            if m:
                self.load_percent = int(m.group(1)) / max(1, int(m.group(2)))
                self.load_text = "Downloading the model (once)"
            else:
                self.load_text = line[:70].rstrip(".")

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        self.root.after(16, self._tick)
        self.root.mainloop()

    def _quit(self) -> None:
        if self.worker:
            self.worker.close()
        if self.mode == "play":  # a finished game has already printed its summary
            self._print_summary()
        self.root.destroy()

    def _tick(self) -> None:
        now = time.perf_counter()
        self._drain_log_lines()
        # The rules and the model are looked after every few milliseconds (so that the window
        # never makes the model wait), but the screen is redrawn only about 30 times a second.
        draw = now - self._last_frame >= 1 / 30
        if draw:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last_frame, 1e-3)
            self._last_frame = now
        if self.mode == "loading":
            if draw:
                self._tick_loading(now)
        elif self.mode == "play":
            self._tick_play(now, draw)
        self.root.after(5, self._tick)

    def _tick_loading(self, now: float) -> None:
        c = self.canvas
        c.delete("ui")
        self._text(W // 2, 220, "PELLET MAZE", 44, CYAN)
        self._text(W // 2, 320, self.load_text + "." * (int(now * 3) % 4), 12, WHITE)
        for i in range(20):
            c.create_rectangle(
                180 + i * 30,
                360,
                180 + i * 30 + 24,
                380,
                fill=CYAN if i < self.load_percent * 20 else "",
                outline=DIM,
                tags=("ui",),
            )
        self._text(W // 2, 430, "First run: ~1.6 GB are downloaded once and verified.", 10, DIM)
        if self.load_error:
            self.mode = "error"
            self._show_error(self.load_error)
        elif self.args.oracle or self.jev is not None:
            if self.jev is not None:
                self.worker = Worker(self._model_decide)
            self.mode = "title"
            self._show_title()
            if self.args.autostart:
                self._start_game()

    def _model_decide(self, state: dict[str, Any], legal: list[str]) -> Decision:
        assert self.jev is not None
        return decide_with_model(self.jev, state, legal, self._current_rng())

    # ---------------------------------------------------------------- screens
    def _text(
        self, x: float, y: float, text: str, size: int, color: str, anchor: str = "center", tag: str = "ui"
    ) -> int:
        return self.canvas.create_text(
            x, y, text=text, fill=color, font=font(size), anchor=anchor, tags=(tag,), justify="left"
        )

    def _show_title(self) -> None:
        c = self.canvas
        c.delete("ui")
        self._text(W // 2, 130, "PELLET MAZE", 46, CYAN)
        self._text(W // 2, 186, "a game played by a small language model", 14, MAGENTA)
        rules = (
            "Eat every pellet. Avoid the ghosts: three lives.",
            "A power pellet scares the ghosts: then they can be eaten.",
            "At every crossing the MODEL chooses where to go.",
            "The GAME computes the sensors (distances, exits...): the model only picks the move.",
            "The model does not read the board and can be wrong: nothing corrects its moves.",
        )
        for i, line in enumerate(rules):
            self._text(W // 2, 260 + i * 30, line, 12, WHITE)
        device = self.jev.model_info.device if self.jev else "no model"
        self._text(W // 2, 450, "PRESS SPACE TO START", 22, YELLOW)
        self._text(W // 2, 500, f"model on: {device}", 11, DIM)
        self._text(W // 2, 524, "The numbers on screen are measured live on this computer.", 11, DIM)

    def _show_error(self, message: str) -> None:
        self.canvas.delete("ui")
        self._text(W // 2, 220, "SOMETHING WENT WRONG", 30, RED)
        self.canvas.create_text(
            W // 2, 330, text=message, fill=WHITE, font=font(11), width=800, tags=("ui",)
        )
        self._text(W // 2, 480, "Press ESC to quit.", 12, YELLOW)
        print(f"ERROR: {message}")

    def _on_space(self) -> None:
        if self.mode in ("title", "over"):
            self._start_game()

    # ---------------------------------------------------------------- the game
    def _start_game(self) -> None:
        seed = self.args.seed + SEED_STEP * self.games_played
        self.games_played += 1
        self.match = Match(
            seed,
            lives=LIVES,
            max_moves=WINDOW_MAX_MOVES,
            ghosts=self.args.ghosts,
            chase=self.args.chase,
        )
        self.mode = "play"
        self.busy = False
        self.hold_until = 0.0
        self.shown = None
        self.stamps = []
        self.peak_rate = 0.0
        self.play_start = time.perf_counter()
        self.canvas.delete("ui")
        self.canvas.delete("wall")
        g = self.match.game
        for y in range(g.h):
            for x in range(g.w):
                if g.wall[y][x]:
                    self.canvas.create_rectangle(
                        MX + x * CELL + 1,
                        MY + y * CELL + 1,
                        MX + (x + 1) * CELL - 1,
                        MY + (y + 1) * CELL - 1,
                        fill=WALL,
                        outline=WALL_EDGE,
                        tags=("wall",),
                    )
        self._text(24, 16, "", 17, YELLOW, "w", "hud_score")
        self._text(W - 296, 16, "", 17, MAGENTA, "w", "hud_lives")
        self._text(24, 40, "", 11, CYAN, "w", "hud_line")

    def _tick_play(self, now: float, draw: bool) -> None:
        match = self.match
        assert match is not None and self.worker is not None
        self._collect(now)
        if self.mode != "play":
            return
        if not self.busy and now >= self.hold_until and match.outcome is None:
            self._request(now)
        if draw and self.mode == "play":
            self._draw(now)

    def _request(self, now: float) -> None:
        assert self.match is not None and self.worker is not None
        state, legal = self.match.observe()
        if len(legal) == 1:  # nothing to decide: the game plays it, the model is not asked
            before = self.match.game.pos
            self.match.apply(Decision(legal[0], forced=True))
            self.last_move = legal[0]
            self.shown = None
            self.hold_until = now + self.args.tick_ms / 2000.0
            self._after_move(before, now)
            return
        self.busy = True
        self.worker.submit(state, legal)

    def _collect(self, now: float) -> None:
        assert self.match is not None and self.worker is not None
        while True:
            try:
                result = self.worker.results.get_nowait()
            except queue.Empty:
                return
            self.busy = False
            if isinstance(result, Exception):
                self.mode = "error"
                self._show_error(f"{type(result).__name__}: {result}")
                return
            before = self.match.game.pos
            lives = self.match.lives
            self.match.apply(result)
            self.last_move = result.move
            self.shown = (before, result)
            self.shown_until = now + max(0.45, self.args.tick_ms / 1000.0)
            self.hold_until = now + self.args.tick_ms / 1000.0
            if not result.forced and not self.args.oracle:
                self.stamps.append(now)
            if self.match.lives < lives:
                self.flash_until = now + 0.4
            self._after_move(before, now)

    def _after_move(self, _before: tuple[int, int], now: float) -> None:
        assert self.match is not None
        if self.match.outcome is not None:
            self._game_over()
        elif len(self.stamps) and now - self.play_start >= 3.0:
            self.peak_rate = max(self.peak_rate, self._rate(now))

    def _rate(self, now: float) -> float:
        """Decisions per second of the model over the last 4 seconds of play."""
        window = 4.0
        recent = sum(1 for t in self.stamps if now - t <= window)
        return recent / max(1.0, min(window, now - self.play_start))

    # ---------------------------------------------------------------- drawing
    def _draw(self, now: float) -> None:
        assert self.match is not None
        c = self.canvas
        c.delete("dyn")
        g = self.match.game

        def cell(pos: tuple[int, int]) -> tuple[float, float]:
            return MX + (pos[0] + 0.5) * CELL, MY + (pos[1] + 0.5) * CELL

        for p in g.pellets:
            x, y = cell(p)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=WHITE, outline="", tags=("dyn",))
        blink = int(now * 3) % 2 == 0
        for p in g.powers:
            x, y = cell(p)
            r = 10 if blink else 8
            c.create_oval(x - r, y - r, x + r, y + r, fill=YELLOW, outline="", tags=("dyn",))
        for i, ghost in enumerate(g.ghosts):
            x, y = cell(ghost["pos"])
            if ghost["scared"]:
                ending = g.power_left <= 3 and int(now * 6) % 2 == 0
                color = WHITE if ending else "#2b44ff"
            else:
                color = GHOST_COLORS[i % len(GHOST_COLORS)]
            c.create_oval(x - 16, y - 18, x + 16, y + 10, fill=color, outline="", tags=("dyn",))
            c.create_rectangle(x - 16, y - 4, x + 16, y + 18, fill=color, outline="", tags=("dyn",))
            for dx in (-6, 6):
                c.create_oval(x + dx - 4, y - 8, x + dx + 4, y, fill=WHITE, outline="", tags=("dyn",))
                c.create_oval(x + dx - 2, y - 6, x + dx + 2, y - 2, fill=BG, outline="", tags=("dyn",))
        px, py = cell(g.pos)
        mouth = 12 + 24 * abs(math.sin(now * 8))
        c.create_arc(
            px - 18,
            py - 18,
            px + 18,
            py + 18,
            start=MOUTH[self.last_move] + mouth,
            extent=360 - 2 * mouth,
            fill=YELLOW,
            outline="",
            tags=("dyn",),
        )
        self._draw_probabilities(now, cell)
        if now < self.flash_until:
            c.create_rectangle(3, 3, W - 3, H - 3, outline=RED, width=8, tags=("dyn",))
        self._draw_hud(now)

    def _draw_probabilities(
        self, now: float, cell: Callable[[tuple[int, int]], tuple[float, float]]
    ) -> None:
        """The probability of every legal move, drawn on the cell it leads to."""
        c = self.canvas
        if self.busy:
            x, y = cell(self.match.game.pos) if self.match else (0, 0)
            c.create_text(x, y - 34, text="...", fill=YELLOW, font=font(14), tags=("dyn",))
            return
        if self.shown is None or now >= self.shown_until:
            return
        origin, decision = self.shown
        best = decision.move
        for move, p in decision.probabilities.items():
            dx, dy = games.DIRS[move]
            x, y = cell((origin[0] + dx, origin[1] + dy))
            color = GREEN if move == best else CYAN
            top = y + CELL / 2 - 4 - (CELL - 8) * p
            c.create_rectangle(x - CELL / 2 + 4, top, x + CELL / 2 - 4, y + CELL / 2 - 4, outline=color, width=2, tags=("dyn",))
            c.create_text(x, y - CELL / 2 + 12, text=f"{100 * p:.0f}%", fill=color, font=font(12), tags=("dyn",))

    def _draw_hud(self, now: float) -> None:
        assert self.match is not None
        m = self.match
        c = self.canvas
        c.itemconfigure("hud_score", text=f"SCORE {m.score:05d}   HI {max(self.high_score, m.score):05d}")
        c.itemconfigure("hud_lives", text="LIVES " + "O " * m.lives)
        mean = f"{statistics.fmean(m.latencies):.0f}" if m.latencies else "---"
        last = f"{m.latencies[-1]:.0f}" if m.latencies else "---"
        device = self.jev.model_info.device if self.jev else "oracle"
        c.itemconfigure(
            "hud_line",
            text=f"{device.upper()}  {self._rate(now):4.1f} DECISIONS/S | LAST {last} MS | AVG {mean} MS | FPS {self.fps:.0f}",
        )
        c.delete("panel")
        x0 = MX + self.match.game.w * CELL + 24
        lines = (
            ("THE MODEL", MAGENTA),
            (f"decisions   {m.asked}", WHITE),
            (f"forced       {m.forced}", WHITE),
            ("", WHITE),
            ("THE ORACLE", MAGENTA),
            (f"would have chosen", WHITE),
            (f"otherwise  {m.disagreements} of {m.asked}", YELLOW),
            ("", WHITE),
            ("PELLETS LEFT", MAGENTA),
            (f"{len(m.game.pellets) + len(m.game.powers)}", WHITE),
            ("", WHITE),
            ("POWER", MAGENTA),
            (f"{m.game.power_left} ticks", WHITE),
        )
        for i, (text, color) in enumerate(lines):
            if text:
                self._text(x0, MY + 8 + i * 26, text, 12, color, "nw", "panel")
        self._text(
            MX,
            H - 22,
            "The game computes the sensors; the model only chooses the move. A small model: it can be wrong.",
            10,
            DIM,
            "w",
            "panel",
        )

    # ---------------------------------------------------------------- game over
    def _game_over(self) -> None:
        assert self.match is not None
        m = self.match
        self.mode = "over"
        self.high_score = max(self.high_score, m.score)
        c = self.canvas
        c.delete("dyn")
        titles = {"won": "ALL PELLETS EATEN", "died": "THE LAST LIFE IS GONE", "timeout": "TIME IS UP"}
        c.create_rectangle(150, 120, W - 150, 520, fill=PANEL, outline=RED if m.outcome != "won" else GREEN, width=4, tags=("ui",))
        self._text(W // 2, 165, titles[m.outcome or "died"], 28, GREEN if m.outcome == "won" else RED)
        mean = statistics.fmean(m.latencies) if m.latencies else 0.0
        agree = 100 * (1 - m.disagreements / m.asked) if m.asked else 100.0
        lines = (
            f"{'Score':<27}{m.score}",
            f"{'Lives lost':<27}{m.deaths}",
            f"{'Moves played':<27}{m.game.moves}",
            f"{'Decisions of the model':<27}{m.asked}",
            f"{'Moves the oracle accepts':<27}{agree:.0f}%",
            f"{'Average per decision':<27}{mean:.0f} ms",
            f"{'Peak speed':<27}{self.peak_rate:.1f} decisions/s",
        )
        for i, line in enumerate(lines):
            self._text(250, 225 + i * 34, line, 14, WHITE, "w")
        self._text(W // 2, 480, "PRESS SPACE TO PLAY AGAIN", 16, YELLOW)
        self._print_summary()

    def _print_summary(self) -> None:
        m = self.match
        if m is None or not m.asked:
            return
        print(
            f"seed {m.seed}: score {m.score}, {m.outcome or 'quit'}, {m.asked} decisions, "
            f"{m.disagreements} where the oracle would have chosen otherwise, "
            f"mean {statistics.fmean(m.latencies):.0f} ms, peak {self.peak_rate:.1f} decisions/s"
        )


# ======================================================================================
# 6. ENTRY POINT
# ======================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Pellet maze (archai-jev demo)")
    parser.add_argument("--headless", action="store_true", help="no window: whole games, numbers on the console")
    parser.add_argument("--oracle", action="store_true", help="the oracle plays instead of the model")
    parser.add_argument("--autostart", action="store_true", help="skip the title screen")
    parser.add_argument("--seed", type=int, default=FIRST_SEED, help="seed of the first game")
    parser.add_argument("--tick-ms", type=int, default=0, help="minimum pause between moves, in ms, to slow the game down (default 0)")
    parser.add_argument("--device", default="cpu", help="cpu (default), auto, cuda:0, vulkan:1...")
    parser.add_argument("--model", default=MODEL_NAME, help="a registry name or a local folder with a manifest")
    parser.add_argument(
        "--ghosts",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="number of ghosts (default 2, the setting the model was measured on)",
    )
    parser.add_argument(
        "--chase",
        type=float,
        choices=(0.35, 0.5, 0.65),
        default=0.5,
        help="how often a ghost goes straight for you (default 0.5, the one measured)",
    )
    parser.add_argument("--max-seconds", type=float, default=0.0, help="close the window after N seconds")
    parser.add_argument("--games", type=int, default=30, help="--headless: games per regime (default 30)")
    parser.add_argument("--max-moves", type=int, default=200, help="--headless: moves per game (default 200)")
    parser.add_argument(
        "--regimes",
        choices=("both", "back-to-back", "pause"),
        default="both",
        help="--headless: calls one after the other, after a pause, or both (default)",
    )
    parser.add_argument("--pause", type=float, default=0.5, help="--headless: pause in seconds (default 0.5)")
    args = parser.parse_args()
    if args.tick_ms < 0:
        parser.error("--tick-ms must be 0 or more")
    if args.games < 1 or args.max_moves < 1:
        parser.error("--games and --max-moves must be at least 1")
    if args.pause < 0:
        parser.error("--pause must be 0 or more")
    if args.headless:
        return run_headless(args)
    Window(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
