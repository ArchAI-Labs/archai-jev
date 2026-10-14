"""Truth Invaders: a retro arcade game played by a language model, in real time.

What you are looking at
-----------------------
A *briefing* (a short fact) is shown at the top of the screen. Then *claims* fall from
the sky like space invaders. Some claims are supported by the briefing, some are
contradicted by it. The cannon at the bottom is driven by the model:

* the model says the claim is TRUE  -> it lets the invader land (it is a friend),
* the model says the claim is FALSE -> the cannon locks on and shoots it down.

Every wrong call costs a base block: a lie that lands (BREACH), a truth that gets shot
(FRIENDLY FIRE), or a claim the model did not judge before it touched the ground
(OVERRUN). The waves get faster and faster, so the score is really a measure of how many
decisions per second the model can take. The HUD shows that number live.

How the model is used (the whole point of this demo)
-----------------------------------------------------
Each claim is ONE call to ``Jev.ask``:

    state     = the briefing
    question  = YesNo("Is this statement supported by the text? <the claim>")

and the answer is a typed value with a probability: ``answer.probability`` is P(yes).
There is no text generation and no parsing of free text anywhere in this file.

Honest limits (please read before recording a video)
-----------------------------------------------------
* ``Jev.from_pretrained()`` loads a small general model (1.5 billion parameters) that is
  reliable on the kinds of task it was trained on, and not on others: a question it
  cannot score raises ``UnsupportedRequestError``, a deliberate safety feature.
* The model is overconfident outside its training distribution and it can be wrong.
  The game shows its accuracy on the game over screen, mistakes included.
* Speed numbers on screen are measured on YOUR machine, live. Nothing here is a claim.

Run it
------
    python demos/000_truth_invaders.py                # play (first run downloads ~1.6 GB)
    python demos/000_truth_invaders.py --autostart    # skip the title screen
    python demos/000_truth_invaders.py --headless     # no window: measure accuracy/latency

Other options: ``--speed 1.5`` (harder), ``--seed N``, ``--max-seconds N`` (quit by
itself), ``--threads N`` (CPU threads for the model: by default the game leaves a few
cores to the window, see ``configure_threads``).
Keys: SPACE = start / play again, ESC = quit.
"""

from __future__ import annotations

import argparse
import colorsys
import itertools
import logging
import math
import os
import queue
import random
import re
import statistics
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from archai_jev import Jev, YesNo

try:  # tkinter ships with Python, but some Linux distros package it separately.
    import tkinter as tk
except ImportError:  # pragma: no cover - only matters for --headless on such systems
    tk = None  # type: ignore[assignment]


# ======================================================================================
# 1. THE QUESTION WE ASK THE MODEL
# ======================================================================================
# The question is plain English: the model was trained on entailment ("does the text
# support this statement?"), so we ask exactly that.
QUESTION = "Is this statement supported by the text? "


def configure_threads(requested: int | None, reserve_for_ui: bool) -> None:
    """Choose how many CPU threads the model may use (``ARCHAI_JEV_NUM_THREADS``).

    The default of the library is "all cores", which is right for a script that only
    runs the model. In a game the window needs a core too, and the inference threads
    wait for each other at every step: if one of them is delayed because the UI is using
    its core, ALL of them wait. So for the game we keep a few cores free. The
    environment variable is read when the model is loaded; if the user already set it,
    we leave it alone.
    """
    if "ARCHAI_JEV_NUM_THREADS" in os.environ and requested is None:
        return
    cpus = os.cpu_count() or 4
    if requested is None and not reserve_for_ui:
        return  # headless: keep the library default
    threads = requested or max(1, cpus - max(1, cpus // 5))
    os.environ["ARCHAI_JEV_NUM_THREADS"] = str(threads)


def p_claim_is_true(jev: Jev, briefing: str, claim: str) -> float:
    """Ask the model: "is ``claim`` supported by ``briefing``?" and return P(yes).

    ``state`` is the briefing and the question's ``instructions`` carry the claim.
    The answer is a ``YesNoAnswer``; its ``probability`` is a float in [0, 1] that
    already includes the model's declared calibration.
    """
    answers = jev.ask(briefing, {"claim": YesNo(QUESTION + claim)})
    return answers.yes_nos["claim"].probability


# ======================================================================================
# 2. THE GAME DATA: briefings and the claims that fall from the sky
# ======================================================================================
@dataclass(frozen=True)
class Claim:
    """One falling invader: a sentence and whether the briefing really supports it."""

    briefing: str
    text: str
    is_true: bool  # the gold label, written by hand. The model never sees it.


@dataclass(frozen=True)
class Briefing:
    """A fact plus four claims it supports and four claims it contradicts."""

    text: str
    true_claims: tuple[str, ...]
    false_claims: tuple[str, ...]

    def claims(self) -> list[Claim]:
        return [Claim(self.text, c, True) for c in self.true_claims] + [
            Claim(self.text, c, False) for c in self.false_claims
        ]


# The gold labels are deliberately unambiguous: every claim is either implied by the
# briefing or in direct contradiction with it ("maybe" claims would make the game
# unfair, because there would be no right answer).
BRIEFINGS: tuple[Briefing, ...] = (
    Briefing(
        "The wizard drank a red potion and his wounds closed.",
        (
            "The wizard was healed.",
            "The potion helped the wizard.",
            "The wizard had been wounded.",
            "The wizard drank something.",
        ),
        (
            "The wizard was poisoned.",
            "The wizard threw the potion away.",
            "The wizard's wounds got worse.",
            "The wizard never touched the potion.",
        ),
    ),
    Briefing(
        "A pirate ship sailed into the harbor at dawn.",
        (
            "A ship arrived at a port.",
            "The ship came in the early morning.",
            "A vessel entered the harbor.",
            "The ship was a pirate ship.",
        ),
        (
            "The ship left the harbor at midnight.",
            "No ship came near the harbor.",
            "A train arrived at the station.",
            "The ship sank far out at sea.",
        ),
    ),
    Briefing(
        "The robot picked up the key and unlocked the door.",
        (
            "The door is now unlocked.",
            "The robot used a key.",
            "A robot opened a lock.",
            "The robot held the key.",
        ),
        (
            "The door stayed locked.",
            "The robot broke the door down with a hammer.",
            "The robot left the key on the floor.",
            "Nobody touched the key.",
        ),
    ),
    Briefing(
        "The dragon sleeps on a pile of gold inside a mountain cave.",
        (
            "A dragon is inside a cave.",
            "There is gold in the cave.",
            "The dragon is asleep.",
            "The dragon is in a mountain.",
        ),
        (
            "The dragon is flying over the sea.",
            "The cave is empty.",
            "The dragon is wide awake and hunting.",
            "There is no gold anywhere.",
        ),
    ),
    Briefing(
        "Two players ran across the bridge while the guard slept.",
        (
            "The bridge was crossed.",
            "The guard was asleep.",
            "Two people crossed a bridge.",
            "The players were running.",
        ),
        (
            "The guard chased the players.",
            "Only one player crossed.",
            "The guard was wide awake.",
            "The players stayed home all day.",
        ),
    ),
    Briefing(
        "The knight lost his shield but won the duel.",
        (
            "The knight won.",
            "The knight no longer had his shield.",
            "The knight fought a duel.",
            "The knight did not lose the duel.",
        ),
        (
            "The knight lost the duel.",
            "The knight kept his shield the whole time.",
            "There was no fight.",
            "The knight gave up before the duel began.",
        ),
    ),
    Briefing(
        "The chef burned the soup, so the guests left hungry.",
        (
            "The soup was ruined.",
            "The guests did not eat well.",
            "The chef made a mistake.",
            "Someone cooked soup.",
        ),
        (
            "The guests loved the soup.",
            "The soup was perfect.",
            "The guests stayed for seconds.",
            "The chef did not cook anything.",
        ),
    ),
    Briefing(
        "A little robot is watering flowers in a sunny garden.",
        (
            "A robot is in a garden.",
            "The flowers are being watered.",
            "It is a sunny day.",
            "The robot is small.",
        ),
        (
            "The robot is sleeping in a cave.",
            "It is raining heavily.",
            "There are no plants nearby.",
            "The robot is cutting down a forest.",
        ),
    ),
    Briefing(
        "The spaceship landed on a red planet with two moons.",
        (
            "A spaceship reached a planet.",
            "The planet has moons.",
            "The planet is red.",
            "The ship is no longer flying.",
        ),
        (
            "The spaceship exploded before landing.",
            "The planet has no moons.",
            "The ship is still far away in space.",
            "The ship landed on the blue ocean of Earth.",
        ),
    ),
    Briefing(
        "The thief slipped out the window just before the alarm rang.",
        (
            "The thief left the building.",
            "The alarm rang after the thief left.",
            "The thief used a window.",
            "The alarm rang.",
        ),
        (
            "The thief was caught inside.",
            "The alarm rang before the thief moved.",
            "The thief used the front door.",
            "There was no alarm at all.",
        ),
    ),
    Briefing(
        "Nobody in the village had seen rain for three years.",
        (
            "The village had a long drought.",
            "It had not rained there for years.",
            "People there had not seen rain recently.",
            "Rain was rare in the village.",
        ),
        (
            "It rained every day in the village.",
            "The villagers saw rain last week.",
            "The village was flooded every month.",
            "The village had too much water.",
        ),
    ),
    Briefing(
        "She finished the race first and received a golden trophy.",
        (
            "She won the race.",
            "She got a trophy.",
            "The trophy was golden.",
            "Nobody finished before her.",
        ),
        (
            "She came in last.",
            "She dropped out of the race.",
            "She received no prize.",
            "Another runner won the race.",
        ),
    ),
)


# ======================================================================================
# 3. TALKING TO THE MODEL WITHOUT FREEZING THE GAME
# ======================================================================================
# Tkinter must only be touched from the main thread, and ``Jev.ask`` blocks for a few
# hundred milliseconds. So the model lives in a worker thread: the game puts a request
# on one queue and picks up the verdicts from another one, once per frame.
@dataclass
class Verdict:
    """The result of one model call (or the error that stopped it)."""

    uid: int
    p_yes: float = 0.0
    ms: float = 0.0
    error: str | None = None


class Judge:
    """A background thread that answers one "is this claim true?" request at a time."""

    def __init__(self, jev: Jev) -> None:
        self._jev = jev
        self._requests: queue.Queue[tuple[int, str, str] | None] = queue.Queue()
        self.results: queue.Queue[Verdict] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, uid: int, briefing: str, claim: str) -> None:
        self._requests.put((uid, briefing, claim))

    def close(self) -> None:
        self._requests.put(None)

    def _loop(self) -> None:
        while (request := self._requests.get()) is not None:
            uid, briefing, claim = request
            started = time.perf_counter()
            try:
                p_yes = p_claim_is_true(self._jev, briefing, claim)
            except Exception as exc:  # JevError subclasses, but never kill the thread
                self.results.put(Verdict(uid, error=f"{type(exc).__name__}: {exc}"))
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.results.put(Verdict(uid, p_yes=p_yes, ms=elapsed_ms))


# ======================================================================================
# 4. HEADLESS MODE: measure the model on every claim, without a window
# ======================================================================================
def run_headless(seed: int, args_threads: int | None) -> int:
    """Ask every claim once and print accuracy and latency. No window, no game."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    configure_threads(args_threads, reserve_for_ui=False)
    print("Loading the model (the first run downloads it)...")
    jev = Jev.from_pretrained()
    print(f"Model: {jev.model_info.name} ({jev.model_info.dtype})")

    claims = [c for b in BRIEFINGS for c in b.claims()]
    random.Random(seed).shuffle(claims)
    p_claim_is_true(jev, "warm-up", "Warm-up.")  # the very first call is slower

    latencies: list[float] = []
    right = {True: 0, False: 0}
    total = {True: 0, False: 0}
    mistakes: list[str] = []
    started = time.perf_counter()
    for claim in claims:
        t0 = time.perf_counter()
        p = p_claim_is_true(jev, claim.briefing, claim.text)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        total[claim.is_true] += 1
        if (p >= 0.5) == claim.is_true:
            right[claim.is_true] += 1
        else:
            mistakes.append(f"  p(true)={p:.2f} gold={claim.is_true}: {claim.text}")
    wall = time.perf_counter() - started

    latencies.sort()
    n = len(claims)
    ok = right[True] + right[False]
    print(f"\n{n} claims, {len(BRIEFINGS)} briefings (seed {seed})")
    print(f"accuracy         : {ok}/{n} = {100 * ok / n:.1f}%")
    print(f"  true claims    : {right[True]}/{total[True]}")
    print(f"  false claims   : {right[False]}/{total[False]}")
    print(
        f"latency per call : mean {statistics.mean(latencies):.0f} ms, "
        f"p50 {statistics.median(latencies):.0f} ms, "
        f"p95 {latencies[int(0.95 * (n - 1))]:.0f} ms"
    )
    print(f"throughput       : {n / wall:.2f} decisions/s (one call at a time)")
    if mistakes:
        print("\nMistakes:")
        print("\n".join(mistakes))
    return 0


# ======================================================================================
# 5. PIXEL ART
# ======================================================================================
# Sprites are tiny text bitmaps: "." is transparent, any other character is a colour
# key. We draw each pixel as a rectangle on the canvas, so there are no image files.
ALIEN_A = (
    "..#.....#..",
    "...#...#...",
    "..#######..",
    ".##e###e##.",
    "###########",
    "#.#######.#",
    "#.#.....#.#",
    "...##.##...",
)
ALIEN_B = (  # the second animation frame: the legs and the arms move
    "..#.....#..",
    "#..#...#..#",
    "#.#######.#",
    "###e###e###",
    "###########",
    ".#########.",
    "..#.....#..",
    ".#.......#.",
)
CANNON = (
    "......#......",
    ".....###.....",
    ".....###.....",
    ".###########.",
    "#############",
    "#############",
    "#############",
    "#############",
)
BUNKER = (
    "..#########..",
    ".###########.",
    "#############",
    "#############",
    "###.......###",
    "###.......###",
)


def create_sprite(
    canvas: tk.Canvas,
    rows: tuple[str, ...],
    left: float,
    top: float,
    px: int,
    colors: dict[str, str],
    tags_for: Callable[[str], tuple[str, ...]],
) -> list[int]:
    """Draw a text bitmap on the canvas and return the ids of the rectangles.

    Neighbouring pixels of the same kind in a row are merged into one rectangle, which
    keeps the number of canvas items (and so the cost of every frame) low.
    """
    ids: list[int] = []
    for r, row in enumerate(rows):
        c = 0
        while c < len(row):
            kind = row[c]
            if kind == ".":
                c += 1
                continue
            start = c
            while c < len(row) and row[c] == kind:
                c += 1
            ids.append(
                canvas.create_rectangle(
                    left + start * px,
                    top + r * px,
                    left + c * px,
                    top + (r + 1) * px,
                    fill=colors[kind],
                    outline="",
                    tags=tags_for(kind),
                )
            )
    return ids


# ======================================================================================
# 6. LOOK AND FEEL
# ======================================================================================
W, H = 960, 720  # the whole game lives in this fixed canvas
FONT = "Courier New"  # monospaced and bold looks the most "arcade" without extra files


def font(size: int) -> tuple[str, int, str]:
    """A bold font whose size is given in screen pixels (about ``size`` * 4/3).

    Tk measures positive font sizes in points, which Windows scales with the display
    zoom, so the text would grow on a 150% screen while our fixed-size canvas does not.
    A NEGATIVE size means pixels, so the layout looks the same everywhere.
    """
    return (FONT, -round(size * 4 / 3), "bold")


BG = "#070714"  # near-black navy
PANEL = "#0d0d2b"
CYAN = "#00e5ff"
MAGENTA = "#ff4fd8"
YELLOW = "#ffe600"
GREEN = "#39ff14"
RED = "#ff2e63"
WHITE = "#e8f6ff"
DIM = "#3a3f6b"
SHADOW = "#1a0a38"
STAR_COLORS = ("#3a3f6b", "#5a63a8", "#8f9bff", "#e8f6ff")

LANES = 5
LANE_X = [96 + i * 192 for i in range(LANES)]  # 96, 288, 480, 672, 864
SPAWN_Y = 164  # top of a freshly spawned invader group
GROUP_H = 146  # height of an invader group: status + sprite + bar + 3 text lines
CLAIM_CHARS = 17  # a claim is wrapped to lines of this many characters (lane = 192 px)
GROUND_Y = 612  # the "ground": a group whose bottom touches it has landed
LAND_Y = GROUND_Y - GROUP_H  # the same line, expressed as the group's top
CANNON_TOP = 626
SPRITE_PX = 4  # one pixel of the cannon = 4x4 screen pixels
ALIEN_PX = 5  # the invaders are a bit bigger: they are the stars of the show
ALIEN_CENTER = 40  # distance from the top of an invader group to its sprite centre
START_LIVES = LANES  # one base block per lane


@dataclass(eq=False)
class Invader:
    """A falling claim and everything the game draws for it."""

    uid: int
    claim: Claim
    lane: int
    x: float
    y: float
    vy: float
    state: str = "falling"  # falling -> judging -> judged (TRUE) | locked (FALSE)
    verdict: bool | None = None  # what the model said; None = not decided yet
    bar_ids: list[int] = field(default_factory=list)


@dataclass
class Bullet:
    x: float
    y: float
    target: Invader


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: str
    size: int


@dataclass
class Popup:
    x: float
    y: float
    text: str
    color: str
    life: float


class Label:
    """Text with a drop shadow (two canvas items that always change together)."""

    def __init__(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        text: str,
        size: int,
        color: str,
        anchor: str = "center",
        tags: tuple[str, ...] = ("ui",),
        width: int = 0,
    ) -> None:
        self._canvas = canvas
        opts: dict[str, Any] = {
            "font": font(size),
            "anchor": anchor,
            "justify": "center",
            "tags": tags,
        }
        if width:
            opts["width"] = width
        self.shadow = canvas.create_text(x + 2, y + 2, text=text, fill=SHADOW, **opts)
        self.main = canvas.create_text(x, y, text=text, fill=color, **opts)

    def set(self, text: str | None = None, color: str | None = None) -> None:
        for item in (self.shadow, self.main):
            if text is not None:
                self._canvas.itemconfigure(item, text=text)
        if color is not None:
            self._canvas.itemconfigure(self.main, fill=color)

    def show(self, visible: bool) -> None:
        state = "normal" if visible else "hidden"
        for item in (self.shadow, self.main):
            self._canvas.itemconfigure(item, state=state)


def rainbow(t: float) -> str:
    """A neon colour that cycles with time (used for the title)."""
    r, g, b = colorsys.hsv_to_rgb(t % 1.0, 0.65, 1.0)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ======================================================================================
# 7. THE GAME
# ======================================================================================
class _LogToQueue(logging.Handler):
    """Collects the library's log lines so the loading screen can show the download."""

    def __init__(self, sink: queue.Queue[str]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.put(record.getMessage())


class Game:
    """Owns the window, the game state and the model. One instance per run."""

    def __init__(self, args: argparse.Namespace) -> None:
        if tk is None:
            raise SystemExit("tkinter is not available; try --headless instead")
        self.args = args
        self.rng = random.Random(args.seed)
        try:  # Windows: ask for crisp pixels instead of a blurry, upscaled window
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (ImportError, AttributeError, OSError):
            pass
        self.root = tk.Tk()
        self.root.title("Truth Invaders - played by archai-jev")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root, width=W, height=H, bg=BG, highlightthickness=0
        )
        self.canvas.pack()
        self.root.bind("<space>", lambda _e: self._on_space())
        self.root.bind("<Escape>", lambda _e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.mode = "loading"  # loading -> title -> play -> over | error
        self.judge: Judge | None = None
        self.jev: Jev | None = None
        # True while one request is in flight. It is NOT reset between games: if the
        # base fell while the model was thinking, that answer still has to arrive.
        self.awaiting_model = False
        self.fps = 60.0  # smoothed frames per second, shown in the HUD
        self.load_error: str | None = None
        self.log_lines: queue.Queue[str] = queue.Queue()
        self.load_text = "Starting"
        self.load_percent = 0.0
        self.high_score = 0
        self.t_start = time.perf_counter()
        self._last_frame = self.t_start
        self._next_frame = self.t_start
        self._ids = itertools.count(1)

        self._build_static()
        self._start_loading_model()
        self._build_loading_screen()
        if args.max_seconds:
            self.root.after(int(args.max_seconds * 1000), self._quit)

    # ---------------------------------------------------------------- model loading
    def _start_loading_model(self) -> None:
        """Load the model in a thread: the first run downloads 1.6 GB and checks it."""
        logger = logging.getLogger("archai_jev")
        logger.setLevel(logging.INFO)
        logger.addHandler(_LogToQueue(self.log_lines))

        configure_threads(self.args.threads, reserve_for_ui=True)

        def load() -> None:
            try:
                jev = Jev.from_pretrained()
                # The very first call is slower than the others (caches and threads are
                # cold), so we make it here, before the game starts, with a throwaway
                # question. Otherwise the first decision of every game looks slow.
                self.load_text = "Warming up the model"
                p_claim_is_true(jev, "Warm-up.", "This is a warm-up.")
                self.jev = jev
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"

        threading.Thread(target=load, daemon=True).start()

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
            elif "Demo model" not in line:
                self.load_text = line[:70].rstrip(".")

    # ---------------------------------------------------------------- static scenery
    def _build_static(self) -> None:
        c = self.canvas
        # Stars: small squares that drift down slowly, in four brightness levels.
        self.stars: list[list[float]] = []
        for _ in range(90):
            x, y = self.rng.uniform(0, W), self.rng.uniform(0, H)
            size = self.rng.choice((2, 2, 3))
            item = c.create_rectangle(
                x,
                y,
                x + size,
                y + size,
                fill=self.rng.choice(STAR_COLORS),
                outline="",
                tags=("star",),
            )
            self.stars.append([item, x, y, self.rng.uniform(8, 36), size])

        # HUD, first row: score, wave, high score.
        self.hud_score = Label(c, 24, 16, "", 17, YELLOW, "w", ("hud",))
        self.hud_wave = Label(c, W // 2, 16, "", 17, MAGENTA, "center", ("hud",))
        self.hud_high = Label(c, W - 24, 16, "", 17, WHITE, "e", ("hud",))
        # HUD, second row: the live measurements of the model.
        self.hud_ai = Label(c, 24, 42, "", 12, CYAN, "w", ("hud",))

        # The briefing panel.
        c.create_rectangle(
            24, 64, W - 24, 150, fill=PANEL, outline=CYAN, width=3, tags=("panel",)
        )
        c.create_rectangle(30, 70, W - 30, 144, outline=DIM, width=1, tags=("panel",))
        Label(c, 42, 78, "MISSION BRIEFING", 9, MAGENTA, "w", ("panel",))
        self.premise = Label(c, 42, 98, "", 18, WHITE, "nw", ("panel",), width=870)

        # The ground line, bunkers and cannon.
        c.create_line(0, GROUND_Y + 2, W, GROUND_Y + 2, fill=DIM, dash=(6, 6))
        self.bunkers: dict[int, list[int]] = {}
        self.cannon_x = float(LANE_X[LANES // 2])
        self.cannon_target = self.cannon_x
        self._build_base()

        # Bottom credit and the big banner used between waves.
        Label(
            c,
            W // 2,
            706,
            "archai-jev  |  Rust + llama.cpp  |  running locally on your CPU",
            9,
            DIM,
            "center",
            ("credit",),
        )
        self.banner = Label(c, W // 2, 330, "", 40, YELLOW, "center", ("banner",))
        self.banner.show(False)

        # Scanlines: a thin dark line every 6 pixels (the old CRT look), kept above
        # everything else. They are plain lines on purpose: with "stipple" (a see-through
        # pattern) the canvas redraws about twice as slowly on Windows (measured on the
        # author's machine), and the frame rate drops below 60.
        for y in range(0, H, 6):
            c.create_line(0, y, W, y, fill="#04040c", tags=("scan",))

    def _build_base(self) -> None:
        """(Re)draw one bunker per lane and the cannon in the middle."""
        c = self.canvas
        c.delete("bunker")
        c.delete("cannon")
        self.bunkers = {}
        for lane, cx in enumerate(LANE_X):
            self.bunkers[lane] = create_sprite(
                c, BUNKER, cx - 13 * 5 / 2, 664, 5, {"#": CYAN}, lambda _k: ("bunker",)
            )
        self.cannon_x = float(LANE_X[LANES // 2])
        self.cannon_target = self.cannon_x
        create_sprite(
            c,
            CANNON,
            self.cannon_x - 13 * SPRITE_PX / 2,
            CANNON_TOP,
            SPRITE_PX,
            {"#": GREEN},
            lambda _k: ("cannon",),
        )

    # ---------------------------------------------------------------- screens (overlays)
    def _clear_ui(self) -> None:
        self.canvas.delete("ui")

    def _show_game_ui(self, visible: bool) -> None:
        """Show or hide the HUD and the briefing panel (only wanted while playing)."""
        state = "normal" if visible else "hidden"
        for tag in ("hud", "panel"):
            self.canvas.itemconfigure(tag, state=state)

    def _build_loading_screen(self) -> None:
        self._clear_ui()
        self._show_game_ui(False)
        c = self.canvas
        Label(c, W // 2, 250, "TRUTH INVADERS", 44, CYAN)
        self.load_label = Label(c, W // 2, 380, "", 12, WHITE)
        self.load_hint = Label(
            c,
            W // 2,
            430,
            "First run: ~1.6 GB are downloaded once and verified.",
            10,
            DIM,
        )
        self.load_bar = [
            c.create_rectangle(
                180 + i * 30,
                330,
                180 + i * 30 + 24,
                350,
                fill="",
                outline=DIM,
                tags=("ui",),
            )
            for i in range(20)
        ]

    def _build_title_screen(self) -> None:
        self._clear_ui()
        self._show_game_ui(False)
        c = self.canvas
        self.title_main = Label(c, W // 2, 190, "TRUTH INVADERS", 46, CYAN)
        Label(c, W // 2, 246, "a game played by a language model", 14, MAGENTA)
        # Decorative aliens on both sides of the title.
        for cx in (150, W - 150):
            create_sprite(
                c,
                ALIEN_A,
                cx - 33,
                160,
                6,
                {"#": MAGENTA, "e": BG},
                lambda _k: ("ui",),
            )
        rules = (
            "A fact appears on top. Claims fall from the sky.",
            "The MODEL reads each claim and decides: TRUE or FALSE.",
            "TRUE  -> it lands as a friend.      FALSE -> the cannon shoots it.",
            "Wrong calls and slow decisions cost base blocks.",
            "Waves get faster: how far can the model go?",
        )
        for i, line in enumerate(rules):
            Label(c, W // 2, 320 + i * 30, line, 12, WHITE)
        self.press_space = Label(c, W // 2, 490, "PRESS SPACE TO START", 22, YELLOW)
        notes = (
            "SMALL MODEL: reliable on the tasks it was trained on,",
            "overconfident outside them. It can be wrong.",
            "The numbers on screen are measured live on this computer.",
        )
        for i, note in enumerate(notes):
            Label(c, W // 2, 548 + i * 22, note, 11, DIM)

    # ---------------------------------------------------------------- starting a game
    def _on_space(self) -> None:
        if self.mode in ("title", "over"):
            self._start_game()

    def _start_game(self) -> None:
        self._clear_ui()
        self._show_game_ui(True)
        for tag in ("inv", "fx"):
            self.canvas.delete(tag)
        self._build_base()
        self.mode = "play"
        self.score = 0
        self.lives = START_LIVES
        self.streak = 0
        self.wave = 0
        self.briefing_order: list[Briefing] = []
        self.invaders: list[Invader] = []
        self.by_id: dict[int, Invader] = {}
        self.gold: dict[int, bool] = {}
        self.bullets: list[Bullet] = []
        self.particles: list[Particle] = []
        self.popups: list[Popup] = []
        self.flash_until = 0.0
        # Measurements: (time finished, milliseconds, was the decision right?)
        self.decisions: list[tuple[float, float, bool]] = []
        self.peak_rate = 0.0
        self.play_start = time.perf_counter()
        self._begin_wave()

    def _begin_wave(self) -> None:
        if not self.briefing_order:
            self.briefing_order = list(BRIEFINGS)
            self.rng.shuffle(self.briefing_order)
        self.briefing = self.briefing_order.pop()
        self.to_spawn = self.briefing.claims()
        self.rng.shuffle(self.to_spawn)
        # Difficulty ramp: invaders fall faster and spawn closer together every wave.
        # The floors are low on purpose so a faster backend can keep going further.
        speed = self.args.speed
        self.fall_seconds = max(0.7, 7.5 - 0.6 * self.wave) / speed
        self.spawn_gap = max(0.12, 1.6 - 0.14 * self.wave) / speed
        self.spawn_timer = 0.0
        self.phase = "intro"
        self.phase_t0 = time.perf_counter()
        self.premise.set("")
        self.banner.set(f"WAVE {self.wave + 1}")
        self.banner.show(True)

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        self.root.after(16, self._tick)
        self.root.mainloop()

    def _quit(self) -> None:
        if self.judge:
            self.judge.close()
        if self.mode == "play":  # a finished game has already printed its summary
            self._print_summary()
        self.root.destroy()

    def _tick(self) -> None:
        now = time.perf_counter()
        frame_time = now - self._last_frame
        dt = min(frame_time, 0.05)  # never jump after a long stall
        self._last_frame = now
        self.fps = 0.9 * self.fps + 0.1 / max(frame_time, 1e-3)  # smoothed
        self._drain_log_lines()

        if self.mode == "loading":
            self._tick_loading(now)
        elif self.mode == "title":
            self.title_main.set(color=rainbow(now * 0.15))
            self.press_space.show(int(now * 2.5) % 2 == 0)
        elif self.mode == "play":
            self._update(dt, now)
        self._move_stars(dt)
        self._draw_fx(dt, now)
        self.canvas.tag_raise("ui")
        self.canvas.tag_raise("scan")
        # Aim for a fixed number of frames per second with a deadline per frame instead
        # of a fixed pause: Windows timers only tick every ~15.6 ms, so "wait 16 ms"
        # would alternate between 16 and 31 ms. A late frame shortens the next wait.
        self._next_frame += 1 / self.args.fps
        if self._next_frame < time.perf_counter() - 0.1:  # far behind: do not catch up
            self._next_frame = time.perf_counter()
        wait_ms = int((self._next_frame - time.perf_counter()) * 1000)
        self.root.after(max(1, wait_ms), self._tick)

    def _tick_loading(self, now: float) -> None:
        self.load_label.set(self.load_text + "." * (int(now * 3) % 4))
        for i, item in enumerate(self.load_bar):
            self.canvas.itemconfigure(
                item, fill=CYAN if i < self.load_percent * 20 else ""
            )
        if self.load_error:
            self.mode = "error"
            self._show_error(self.load_error)
        elif self.jev is not None:
            self.judge = Judge(self.jev)
            self.mode = "title"
            self._build_title_screen()
            if self.args.autostart:
                self._start_game()

    def _show_error(self, message: str) -> None:
        self._clear_ui()
        Label(self.canvas, W // 2, 250, "SOMETHING WENT WRONG", 30, RED)
        Label(self.canvas, W // 2, 340, message, 11, WHITE, width=800)
        Label(self.canvas, W // 2, 480, "Press ESC to quit.", 12, YELLOW)
        print(f"ERROR: {message}")

    def _move_stars(self, dt: float) -> None:
        for star in self.stars:
            item, x, y, speed, size = star
            y = (y + speed * dt) % H
            star[2] = y
            self.canvas.coords(int(item), [x, y, x + size, y + size])

    # ---------------------------------------------------------------- the game rules
    def _update(self, dt: float, now: float) -> None:
        self._collect_verdicts(now)
        self._advance_wave(dt, now)
        self._move_invaders(dt)
        self._dispatch_to_model()
        self._move_bullets(dt)
        self._move_cannon(dt)
        self._animate_invaders(now)
        self._update_hud(now)
        if self.lives <= 0 and self.mode == "play":
            self._game_over()

    # -- waves --------------------------------------------------------------------
    def _advance_wave(self, dt: float, now: float) -> None:
        elapsed = now - self.phase_t0
        if self.phase == "intro":
            # Typewriter effect for the briefing, then a short pause.
            shown = int(elapsed * 60)
            text = self.briefing.text
            self.premise.set(text[:shown] + ("_" if shown < len(text) else ""))
            if elapsed > 1.0:
                self.banner.show(False)
            if shown >= len(text) and elapsed > len(text) / 60 + 0.8:
                self.premise.set(text)
                self.banner.show(False)
                self.phase = "spawn"
        elif self.phase == "spawn":
            self.spawn_timer -= dt
            if self.to_spawn and self.spawn_timer <= 0 and self._spawn():
                self.spawn_timer = self.spawn_gap
            if not self.to_spawn:
                self.phase = "drain"
        elif self.phase == "drain":
            if not self.invaders:
                self.phase = "clear"
                self.phase_t0 = now
                self.banner.set("WAVE CLEAR")
                self.banner.show(True)
        elif self.phase == "clear" and elapsed > 1.4:
            self.wave += 1
            self._begin_wave()

    def _spawn(self) -> bool:
        """Create the next invader in a lane that has room. False if every lane is busy."""
        free = [
            lane
            for lane in range(LANES)
            if not any(
                inv.lane == lane and inv.y < SPAWN_Y + GROUP_H + 6
                for inv in self.invaders
            )
        ]
        if not free:
            return False
        claim = self.to_spawn.pop()
        lane = self.rng.choice(free)
        inv = Invader(
            uid=next(self._ids),
            claim=claim,
            lane=lane,
            x=float(LANE_X[lane]),
            y=float(SPAWN_Y),
            vy=(LAND_Y - SPAWN_Y) / self.fall_seconds,
        )
        self._draw_invader(inv)
        self.invaders.append(inv)
        self.by_id[inv.uid] = inv
        # Kept for the whole game, so an answer that arrives late can still be scored.
        self.gold[inv.uid] = claim.is_true
        return True

    def _draw_invader(self, inv: Invader) -> None:
        """Create the canvas items of one invader. They all share the tag ``inv<uid>``
        so a single ``move`` call moves the whole group."""
        c, uid, cx, y = self.canvas, inv.uid, inv.x, inv.y
        group = f"inv{uid}"
        colors = {"#": MAGENTA, "e": BG}

        def tags(frame: str) -> Callable[[str], tuple[str, ...]]:
            # Only the body ("#") changes colour when the model decides.
            return lambda kind: ("inv", group, f"{frame}{uid}") + (
                (f"body{uid}",) if kind == "#" else ()
            )

        # Layout of one group, from the top: status text, sprite (8 rows of ALIEN_PX),
        # confidence bar, claim text (up to 3 lines). GROUP_H is the total height.
        left = cx - 11 * ALIEN_PX / 2
        c.create_text(
            cx,
            y + 9,
            text="",
            font=font(11),
            fill=DIM,
            tags=("inv", group, f"status{uid}"),
        )
        create_sprite(c, ALIEN_A, left, y + 20, ALIEN_PX, colors, tags("fa"))
        create_sprite(c, ALIEN_B, left, y + 20, ALIEN_PX, colors, tags("fb"))
        c.itemconfigure(f"fb{uid}", state="hidden")
        # A 10-block bar that will show how sure the model is.
        for i in range(10):
            inv.bar_ids.append(
                c.create_rectangle(
                    cx - 69 + i * 14,
                    y + 66,
                    cx - 69 + i * 14 + 12,
                    y + 75,
                    fill="",
                    outline=DIM,
                    tags=("inv", group),
                )
            )
        wrapped = "\n".join(textwrap.wrap(inv.claim.text, CLAIM_CHARS))
        c.create_text(
            cx,
            y + 83,
            text=wrapped,
            anchor="n",
            justify="center",
            font=font(13),
            fill=WHITE,
            tags=("inv", group),
        )

    def _move_invaders(self, dt: float) -> None:
        for inv in list(self.invaders):
            if inv.state == "locked":  # target lock: wait for the bullet
                continue
            dy = inv.vy * dt
            inv.y += dy
            self.canvas.move(f"inv{inv.uid}", 0, dy)
            if inv.y >= LAND_Y:
                self._land(inv)

    def _land(self, inv: Invader) -> None:
        """An invader touched the ground: decide who was right."""
        self._remove(inv)
        x, y = inv.x, GROUND_Y - 10
        if inv.verdict is None:  # the model was too slow
            self._burst(x, y, RED)
            self._hurt(inv.lane, "OVERRUN!")
        elif inv.claim.is_true:  # a truth was let through: perfect
            self._burst(x, y, GREEN, 10)
            self._score(50, x, y)
        else:  # a lie was let through
            self._burst(x, y, RED)
            self._hurt(inv.lane, "BREACH!")

    def _score(self, base: int, x: float, y: float) -> None:
        self.streak += 1
        multiplier = min(5, 1 + self.streak // 3)
        self.score += base * multiplier
        label = f"+{base * multiplier}" + (f" x{multiplier}" if multiplier > 1 else "")
        self.popups.append(Popup(x, y - 20, label, YELLOW, 0.9))

    def _hurt(self, lane: int, why: str) -> None:
        """Lose one base block: the one in this lane, or the nearest that is left."""
        self.streak = 0
        self.lives -= 1
        self.flash_until = time.perf_counter() + 0.35
        alive = [ln for ln in self.bunkers if self.bunkers[ln]]
        if alive:
            nearest = min(alive, key=lambda ln: abs(ln - lane))
            for item in self.bunkers[nearest]:
                self.canvas.delete(item)
            self.bunkers[nearest] = []
            self._burst(LANE_X[nearest], 680, CYAN, 26)
        self.popups.append(Popup(LANE_X[lane], GROUND_Y - 70, why, RED, 1.2))

    def _remove(self, inv: Invader) -> None:
        self.canvas.delete(f"inv{inv.uid}")
        if inv in self.invaders:
            self.invaders.remove(inv)
        self.by_id.pop(inv.uid, None)

    # -- the model ------------------------------------------------------------------
    def _dispatch_to_model(self) -> None:
        """Send the invader closest to the ground to the model, if the model is idle."""
        if self.awaiting_model or self.judge is None:
            return
        waiting = [i for i in self.invaders if i.state == "falling"]
        if not waiting:
            return
        inv = max(waiting, key=lambda i: i.y)
        inv.state = "judging"
        self.judge.submit(inv.uid, inv.claim.briefing, inv.claim.text)
        self.awaiting_model = True
        self.canvas.itemconfigure(f"status{inv.uid}", text="SCANNING", fill=YELLOW)

    def _collect_verdicts(self, now: float) -> None:
        assert self.judge is not None
        while True:
            try:
                verdict = self.judge.results.get_nowait()
            except queue.Empty:
                return
            self.awaiting_model = False
            if verdict.error:
                self.mode = "error"
                self._show_error(verdict.error)
                return
            gold = self.gold.get(verdict.uid)
            if gold is None:  # an answer left over from the previous game: ignore
                continue
            says_true = verdict.p_yes >= 0.5
            inv = self.by_id.get(verdict.uid)
            if inv is not None:
                self._apply_verdict(inv, says_true, verdict.p_yes)
            # Count the decision even if the invader landed while the model thought.
            self.decisions.append((now, verdict.ms, says_true == gold))

    def _apply_verdict(self, inv: Invader, says_true: bool, p_yes: float) -> None:
        c = self.canvas
        inv.verdict = says_true
        confidence = p_yes if says_true else 1.0 - p_yes
        color = GREEN if says_true else RED
        word = "TRUE" if says_true else "FALSE"
        c.itemconfigure(f"body{inv.uid}", fill=color)
        c.itemconfigure(
            f"status{inv.uid}", text=f"{word} {confidence * 100:.0f}%", fill=color
        )
        for i, item in enumerate(inv.bar_ids):
            c.itemconfigure(item, fill=color if i < round(confidence * 10) else "")
        if says_true:
            inv.state = "judged"  # let it fall: a friend
        else:
            inv.state = "locked"  # stop it and shoot
            self.cannon_target = inv.x
            self.bullets.append(Bullet(self.cannon_x, CANNON_TOP, inv))

    # -- shooting ---------------------------------------------------------------------
    def _move_cannon(self, dt: float) -> None:
        dx = self.cannon_target - self.cannon_x
        step = max(-1800 * dt, min(1800 * dt, dx))
        self.cannon_x += step
        self.canvas.move("cannon", step, 0)

    def _move_bullets(self, dt: float) -> None:
        for bullet in list(self.bullets):
            tx, ty = bullet.target.x, bullet.target.y + ALIEN_CENTER
            dx, dy = tx - bullet.x, ty - bullet.y
            dist = math.hypot(dx, dy)
            step = 1900 * dt
            if dist <= step + 8:
                self.bullets.remove(bullet)
                self._shot_hit(bullet.target)
            else:
                bullet.x += dx / dist * step
                bullet.y += dy / dist * step

    def _shot_hit(self, inv: Invader) -> None:
        x, y = inv.x, inv.y + ALIEN_CENTER
        self._remove(inv)
        if inv.claim.is_true:  # the model shot a friend
            self._burst(x, y, RED)
            self._hurt(inv.lane, "FRIENDLY FIRE!")
        else:
            self._burst(x, y, YELLOW, 22)
            self._score(100, x, y)

    # -- effects ----------------------------------------------------------------------
    def _burst(self, x: float, y: float, color: str, count: int = 16) -> None:
        for _ in range(count):
            angle = self.rng.uniform(0, math.tau)
            speed = self.rng.uniform(60, 280)
            self.particles.append(
                Particle(
                    x,
                    y,
                    math.cos(angle) * speed,
                    math.sin(angle) * speed,
                    self.rng.uniform(0.3, 0.7),
                    color,
                    self.rng.choice((3, 4, 5)),
                )
            )

    def _animate_invaders(self, now: float) -> None:
        """Flip the two animation frames and make the scanned invader blink."""
        frame_b = int(now * 2.2) % 2 == 1
        for inv in self.invaders:
            uid = inv.uid
            self.canvas.itemconfigure(
                f"fa{uid}", state="hidden" if frame_b else "normal"
            )
            self.canvas.itemconfigure(
                f"fb{uid}", state="normal" if frame_b else "hidden"
            )
            if inv.state == "judging":
                blink = YELLOW if int(now * 8) % 2 == 0 else MAGENTA
                self.canvas.itemconfigure(f"body{uid}", fill=blink)

    def _draw_fx(self, dt: float, now: float) -> None:
        """Bullets, sparks, floating texts and the red flash: redrawn every frame."""
        c = self.canvas
        c.delete("fx")
        if self.mode != "play":
            return
        for bullet in self.bullets:
            c.create_rectangle(
                bullet.x - 2,
                bullet.y - 8,
                bullet.x + 2,
                bullet.y + 8,
                fill=YELLOW,
                outline="",
                tags=("fx",),
            )
        for p in list(self.particles):
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
            if p.life <= 0:
                self.particles.remove(p)
                continue
            c.create_rectangle(
                p.x,
                p.y,
                p.x + p.size,
                p.y + p.size,
                fill=p.color,
                outline="",
                tags=("fx",),
            )
        for pop in list(self.popups):
            pop.y -= 40 * dt
            pop.life -= dt
            if pop.life <= 0:
                self.popups.remove(pop)
                continue
            for offset, color in ((2, SHADOW), (0, pop.color)):  # shadow, then text
                c.create_text(
                    pop.x + offset,
                    pop.y + offset,
                    text=pop.text,
                    font=font(16),
                    fill=color,
                    tags=("fx",),
                )
        if now < self.flash_until:
            c.create_rectangle(3, 3, W - 3, H - 3, outline=RED, width=8, tags=("fx",))

    # -- HUD and measurements -------------------------------------------------------------
    def _rate(self, now: float) -> float:
        """Decisions per second over the last 4 seconds of play."""
        window = 4.0
        recent = sum(1 for t, _ms, _ok in self.decisions if now - t <= window)
        span = max(1.0, min(window, now - self.play_start))
        return recent / span

    def _update_hud(self, now: float) -> None:
        rate = self._rate(now)
        if now - self.play_start >= 3.0:
            self.peak_rate = max(self.peak_rate, rate)
        n = len(self.decisions)
        hit = sum(1 for _t, _ms, ok in self.decisions if ok)
        last = f"{self.decisions[-1][1]:.0f}" if n else "---"
        mean = (
            f"{statistics.mean(ms for _t, ms, _ok in self.decisions):.0f}"
            if n
            else "---"
        )
        acc = f"{100 * hit / n:.0f}%" if n else "--"
        queued = sum(1 for i in self.invaders if i.state == "falling")
        self.hud_score.set(f"SCORE {self.score:06d}")
        self.hud_wave.set(f"WAVE {self.wave + 1:02d}")
        self.hud_high.set(f"HI {max(self.high_score, self.score):06d}")
        self.hud_ai.set(
            f"AI  {rate:4.1f} DECISIONS/S | LAST {last} MS | AVG {mean} MS | "
            f"WAITING {queued} | RIGHT {acc} | FPS {self.fps:.0f}"
        )

    # -- game over ------------------------------------------------------------------------
    def _game_over(self) -> None:
        self.mode = "over"
        self.high_score = max(self.high_score, self.score)
        for tag in ("inv", "fx"):
            self.canvas.delete(tag)
        self.invaders, self.bullets = [], []
        self.banner.show(False)
        c = self.canvas
        c.create_rectangle(
            150, 150, W - 150, 600, fill=PANEL, outline=RED, width=4, tags=("ui",)
        )
        Label(c, W // 2, 195, "THE BASE HAS FALLEN", 30, RED)
        n = len(self.decisions)
        hit = sum(1 for _t, _ms, ok in self.decisions if ok)
        mean = statistics.mean(ms for _t, ms, _ok in self.decisions) if n else 0.0
        lines = (
            f"{'Waves reached':<23}{self.wave + 1}",
            f"{'Score':<23}{self.score}",
            f"{'Decisions taken':<23}{n}",
            f"{'Decisions right':<23}{hit} ({100 * hit / max(1, n):.0f}%)",
            f"{'Average per decision':<23}{mean:.0f} ms",
            f"{'Peak speed':<23}{self.peak_rate:.1f} decisions/s",
        )
        for i, line in enumerate(lines):
            Label(c, 270, 270 + i * 34, line, 14, WHITE, "w")
        Label(c, W // 2, 520, "PRESS SPACE TO PLAY AGAIN", 16, YELLOW)
        Label(c, W // 2, 560, "Demo model: mistakes are part of the game.", 9, DIM)
        self._print_summary()

    def _print_summary(self) -> None:
        n = len(self.decisions)
        if not n:
            return
        hit = sum(1 for _t, _ms, ok in self.decisions if ok)
        mean = statistics.mean(ms for _t, ms, _ok in self.decisions)
        print(
            f"wave {self.wave + 1}, score {self.score}, {n} decisions, "
            f"{100 * hit / n:.0f}% right, mean {mean:.0f} ms, "
            f"peak {self.peak_rate:.1f} decisions/s"
        )


# ======================================================================================
# 8. ENTRY POINT
# ======================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Truth Invaders (archai-jev demo)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no window: measure the model on every claim and exit",
    )
    parser.add_argument(
        "--autostart", action="store_true", help="skip the title screen"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="difficulty multiplier (default 1.0)"
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="frames per second to draw (default 60); fewer frames leave more CPU "
        "to the model",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads for the model (default: all cores but a few for the window)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="close the window after N seconds",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    if not 5 <= args.fps <= 240:
        parser.error("--fps must be between 5 and 240")
    if args.headless:
        return run_headless(args.seed, args.threads)
    Game(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
