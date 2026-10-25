"""Snake and a pellet maze, with SENSORS: the game computes them, the model only chooses the move.

Pure Python, no third-party package. This module is the single source of truth for

* the two game engines (``Snake``, ``Maze``),
* the sensors shown to the model (``sensors()`` of each game), versioned as ``SENSORS_VERSION``,
* the oracle policies ``snake_optimal`` / ``maze_optimal``: a pure function of the sensors AS SHOWN
  (the rendered state alone is enough to recompute the label: no hidden information),
* the decision records (state + question + legal move keys + optimal set) used for training,
* closed-loop play (``play``) with any controller, to measure a model by the score it gets.

Conventions: ``x`` grows to the right and ``y`` grows downwards; moves are "up", "down", "left",
"right". The game is deliberately simple: it is a vehicle to show fast typed decisions, not a claim
that the model understands space (the sensors carry the geometry).

Run ``python games.py --selftest`` for the checks and the measured ceilings of the oracle policies.
"""

from __future__ import annotations

import json
import random
import sys
from collections import deque
from typing import Any, Callable, Iterable

SENSORS_VERSION = "game-sensors-v1"
DIRS: dict[str, tuple[int, int]] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
INF = 99  # "no such thing" in a comparison (the sensors themselves use null)

SNAKE_QUESTION = (
    "Snake: which move should the snake make? It must not trap itself (reaches_tail); "
    "otherwise eat the food or get closer to it."
)
MAZE_QUESTION = (
    "Pellet maze: which move should the player make? Avoid the ghosts (ghost_distance), "
    "eat pellets, and chase scared ghosts while power_left lasts."
)


# ----------------------------------------------------------------------------------------
# grid helpers
# ----------------------------------------------------------------------------------------
def bfs(
    passable: Callable[[tuple[int, int]], bool],
    start: tuple[int, int],
    goal: Callable[[tuple[int, int]], bool],
) -> int | None:
    """Fewest moves from ``start`` to a cell where ``goal`` holds, through cells where
    ``passable`` holds (the goal cell itself may be impassable). ``None`` if unreachable."""
    if goal(start):
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y), d = queue.popleft()
        for dx, dy in DIRS.values():
            c = (x + dx, y + dy)
            if c in seen:
                continue
            if goal(c):
                return d + 1
            if passable(c):
                seen.add(c)
                queue.append((c, d + 1))
    return None


def flood(passable: Callable[[tuple[int, int]], bool], start: tuple[int, int]) -> int:
    """Number of cells reachable from ``start`` (itself included) through passable cells."""
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRS.values():
            c = (x + dx, y + dy)
            if c not in seen and passable(c):
                seen.add(c)
                stack.append(c)
    return len(seen)


# ----------------------------------------------------------------------------------------
# Snake
# ----------------------------------------------------------------------------------------
class Snake:
    """A snake on a ``w`` x ``h`` board. ``body[0]`` is the head, ``body[-1]`` the tail."""

    def __init__(self, w: int, h: int, rng: random.Random, body: list[tuple[int, int]] | None = None,
                 length: int = 3) -> None:
        self.w, self.h, self.rng = w, h, rng
        self.body = body or self._random_body(length)
        self.food = self._new_food()
        self.alive = True
        self.eaten = 0
        self.moves = 0

    def _inside(self, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < self.w and 0 <= c[1] < self.h

    def _random_body(self, length: int) -> list[tuple[int, int]]:
        """A random self-avoiding walk of ``length`` cells, head first."""
        while True:
            body = [(self.rng.randrange(self.w), self.rng.randrange(self.h))]
            while len(body) < length:
                x, y = body[-1]
                options = [(x + dx, y + dy) for dx, dy in DIRS.values()]
                options = [c for c in options if self._inside(c) and c not in body]
                if not options:
                    break
                body.append(self.rng.choice(options))
            if len(body) == length:
                return body

    def _new_food(self) -> tuple[int, int] | None:
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in self.body]
        return self.rng.choice(free) if free else None

    def legal(self) -> list[str]:
        """Moves that do not end the game at once: not a wall, not the body (the tail moves away)."""
        head = self.body[0]
        out = []
        for name, (dx, dy) in DIRS.items():
            c = (head[0] + dx, head[1] + dy)
            if self._inside(c) and c not in self.body[:-1]:
                out.append(name)
        return out

    def sensors(self) -> dict[str, Any]:
        """What the model sees. One entry per legal move, computed on the board AFTER that move."""
        head = self.body[0]
        moves: dict[str, dict[str, Any]] = {}
        for name in self.legal():
            dx, dy = DIRS[name]
            c = (head[0] + dx, head[1] + dy)
            eats = c == self.food
            nb = [c] + (self.body if eats else self.body[:-1])
            wall = set(nb[1:])
            free = lambda p, wall=wall: self._inside(p) and p not in wall  # noqa: E731
            steps = 0 if eats else bfs(free, c, lambda p: p == self.food)
            tail = nb[-1]
            blocked_but_tail = set(nb[1:-1])
            reaches = bfs(
                lambda p, b=blocked_but_tail: self._inside(p) and p not in b, c, lambda p: p == tail
            )
            moves[name] = {
                "next_cell": "food" if eats else "empty",
                "steps_to_food": steps,
                "free_cells": flood(free, c),
                "reaches_tail": reaches is not None,
            }
        return {"game": "snake", "length": len(self.body), "moves": moves}

    def step(self, move: str) -> None:
        dx, dy = DIRS[move]
        c = (self.body[0][0] + dx, self.body[0][1] + dy)
        self.moves += 1
        if not self._inside(c) or c in self.body[:-1]:
            self.alive = False
            return
        if c == self.food:
            self.body.insert(0, c)
            self.eaten += 1
            self.food = self._new_food()
            if self.food is None:  # the board is full: the game is won
                self.alive = False
        else:
            self.body = [c] + self.body[:-1]

    def score(self) -> int:
        return self.eaten


def snake_optimal(state: dict[str, Any]) -> list[str]:
    """The moves the oracle accepts, computed from the sensors alone (policy ``snake-v1``).

    1. keep the moves that can still reach the tail (``reaches_tail``): they cannot trap the snake;
       if there are none, take the one with the most ``free_cells``;
    2. among the kept ones: eat if possible, else the fewest ``steps_to_food``, then the most
       ``free_cells``. Every move that ties for best is accepted.
    """
    moves = state["moves"]
    safe = [m for m, s in moves.items() if s["reaches_tail"]]
    if not safe:
        best = max(s["free_cells"] for s in moves.values())
        return [m for m, s in moves.items() if s["free_cells"] == best]

    def key(m: str) -> tuple[int, int, int]:
        s = moves[m]
        steps = INF if s["steps_to_food"] is None else s["steps_to_food"]
        return (s["next_cell"] != "food", steps, -s["free_cells"])

    best_key = min(key(m) for m in safe)
    return [m for m in safe if key(m) == best_key]


# ----------------------------------------------------------------------------------------
# Pellet maze
# ----------------------------------------------------------------------------------------
def carve_maze(w: int, h: int, rng: random.Random, loops: float) -> list[list[bool]]:
    """``True`` = wall. Odd ``w`` and ``h``. A random spanning tree of corridors, then ``loops`` of
    the walls between two open cells are knocked down, so there are several routes."""
    wall = [[True] * w for _ in range(h)]
    stack = [(1, 1)]
    wall[1][1] = False
    while stack:
        x, y = stack[-1]
        options = [(dx, dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                   if 0 < x + dx < w - 1 and 0 < y + dy < h - 1 and wall[y + dy][x + dx]]
        if not options:
            stack.pop()
            continue
        dx, dy = rng.choice(options)
        wall[y + dy // 2][x + dx // 2] = False
        wall[y + dy][x + dx] = False
        stack.append((x + dx, y + dy))
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if wall[y][x] and (
                (not wall[y][x - 1] and not wall[y][x + 1]) or (not wall[y - 1][x] and not wall[y + 1][x])
            ) and rng.random() < loops:
                wall[y][x] = False
    # no dead ends (as in the classic mazes): open a wall next to every cell with a single exit
    changed = True
    while changed:
        changed = False
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if wall[y][x]:
                    continue
                exits = [(x + dx, y + dy) for dx, dy in DIRS.values() if not wall[y + dy][x + dx]]
                if len(exits) > 1:
                    continue
                # a wall cell next to this one whose far side is open (never the border)
                doors = [(x + dx, y + dy) for dx, dy in DIRS.values()
                         if wall[y + dy][x + dx] and 0 < x + dx < w - 1 and 0 < y + dy < h - 1
                         and (x + 2 * dx, y + 2 * dy) not in exits
                         and 0 <= x + 2 * dx < w and 0 <= y + 2 * dy < h and not wall[y + 2 * dy][x + 2 * dx]]
                if doors:
                    wx, wy = rng.choice(doors)
                    wall[wy][wx] = False
                    changed = True
    return wall


class Maze:
    """A maze with pellets, power pellets and ghosts. The player moves first, then every ghost."""

    POWER_TICKS = 12

    def __init__(self, rng: random.Random, w: int = 13, h: int = 11, ghosts: int = 2,
                 chase: float = 0.5, loops: float = 0.25, speed: float = 0.7) -> None:
        self.rng, self.w, self.h, self.chase, self.speed = rng, w, h, chase, speed
        self.wall = carve_maze(w, h, rng, loops)
        self.open = [(x, y) for y in range(h) for x in range(w) if not self.wall[y][x]]
        self.pos = rng.choice(self.open)
        far = [c for c in self.open if bfs(self.passable, self.pos, lambda p, c=c: p == c) is not None
               and abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]) >= 6]
        homes = rng.sample(far, ghosts)
        self.ghosts = [{"pos": g, "home": g, "scared": False} for g in homes]
        taken = {self.pos, *homes}
        cells = [c for c in self.open if c not in taken]
        self.powers = set(rng.sample(cells, 3))
        self.pellets = {c for c in cells if c not in self.powers}
        self.power_left = 0
        self.alive = True
        self.eaten = 0  # pellets (power pellets included)
        self.ghosts_eaten = 0
        self.moves = 0

    def passable(self, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < self.w and 0 <= c[1] < self.h and not self.wall[c[1]][c[0]]

    def legal(self) -> list[str]:
        x, y = self.pos
        return [n for n, (dx, dy) in DIRS.items() if self.passable((x + dx, y + dy))]

    def _nearest(self, start: tuple[int, int], cells: set[tuple[int, int]]) -> int | None:
        if not cells:
            return None
        return bfs(self.passable, start, lambda p: p in cells)

    def sensors(self) -> dict[str, Any]:
        """What the model sees. One entry per legal move, computed from the cell it leads to."""
        danger = {g["pos"] for g in self.ghosts if not g["scared"]}
        scared = {g["pos"] for g in self.ghosts if g["scared"]}
        food = self.pellets | self.powers
        moves: dict[str, dict[str, Any]] = {}
        for name in self.legal():
            dx, dy = DIRS[name]
            c = (self.pos[0] + dx, self.pos[1] + dy)
            if c in danger:
                cell = "ghost"
            elif c in scared:
                cell = "scared_ghost"
            elif c in self.powers:
                cell = "power_pellet"
            elif c in self.pellets:
                cell = "pellet"
            else:
                cell = "empty"
            moves[name] = {
                "next_cell": cell,
                "ghost_distance": self._nearest(c, danger),
                "scared_distance": self._nearest(c, scared),
                "pellet_distance": self._nearest(c, food),
                "exits": sum(self.passable((c[0] + ex, c[1] + ey)) for ex, ey in DIRS.values()),
            }
        return {"game": "maze", "power_left": self.power_left, "pellets_left": len(food), "moves": moves}

    # --- dynamics
    def _hit(self) -> None:
        """Resolve a player/ghost meeting on the same cell."""
        for g in self.ghosts:
            if g["pos"] == self.pos:
                if g["scared"]:
                    g["pos"], g["scared"] = g["home"], False
                    self.ghosts_eaten += 1
                else:
                    self.alive = False

    def _move_ghost(self, g: dict[str, Any]) -> None:
        x, y = g["pos"]
        options = [(x + dx, y + dy) for dx, dy in DIRS.values() if self.passable((x + dx, y + dy))]
        if g["scared"]:
            if self.rng.random() < 0.5:
                return  # scared ghosts are slow
            dist = {c: bfs(self.passable, c, lambda p: p == self.pos) for c in options}
            best = max(INF if d is None else d for d in dist.values())
            g["pos"] = self.rng.choice([c for c, d in dist.items() if (INF if d is None else d) == best])
        elif self.rng.random() < self.chase:
            dist = {c: bfs(self.passable, c, lambda p: p == self.pos) for c in options}
            best = min(INF if d is None else d for d in dist.values())
            g["pos"] = self.rng.choice([c for c, d in dist.items() if (INF if d is None else d) == best])
        else:
            g["pos"] = self.rng.choice(options)

    def step(self, move: str) -> None:
        dx, dy = DIRS[move]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        self.moves += 1
        if self.pos in self.pellets:
            self.pellets.discard(self.pos)
            self.eaten += 1
        elif self.pos in self.powers:
            self.powers.discard(self.pos)
            self.eaten += 1
            self.power_left = self.POWER_TICKS
            for g in self.ghosts:
                g["scared"] = True
        self._hit()
        if not self.alive:
            return
        for g in self.ghosts:
            if self.rng.random() < self.speed:  # the ghosts are a little slower than the player
                self._move_ghost(g)
        self._hit()
        if self.power_left > 0:
            self.power_left -= 1
            if self.power_left == 0:
                for g in self.ghosts:
                    g["scared"] = False
        if not (self.pellets or self.powers):
            self.alive = False  # all eaten: the level is won

    def won(self) -> bool:
        return not (self.pellets or self.powers)

    def score(self) -> int:
        return self.eaten + 5 * self.ghosts_eaten


def maze_optimal(state: dict[str, Any]) -> list[str]:
    """The moves the oracle accepts, from the sensors alone (policy ``maze-v2``).

    * lethal = stepping on a ghost, or ending next to one (``ghost_distance`` <= 1);
    * if every move is lethal: go where the nearest ghost is farthest;
    * while power lasts and a scared ghost can be reached in time: chase it;
    * if no move leaves the nearest ghost more than 4 steps away (danger zone): flee, that is the
      farthest ghost first, then the most exits;
    * otherwise: keep a margin of 3 steps, avoid dead ends when a ghost is within 6, eat (a power
      pellet is preferred when a ghost is within 6), else go to the nearest pellet, then keep away
      from the ghosts and keep the exits open.
    Every move that ties for best is accepted.
    """
    moves, power = state["moves"], state["power_left"]

    def d(v: int | None) -> int:
        return INF if v is None else v

    def gd(m: str) -> int:
        return min(d(moves[m]["ghost_distance"]), 10)

    def lethal(m: str) -> bool:
        return moves[m]["next_cell"] == "ghost" or (moves[m]["ghost_distance"] is not None and gd(m) <= 1)

    def pick(key: Callable[[str], tuple[int, ...]]) -> list[str]:
        best = min(key(m) for m in moves)
        return [m for m in moves if key(m) == best]

    if all(lethal(m) for m in moves):
        return pick(lambda m: (-gd(m), -moves[m]["exits"]))

    chase = power >= 2 and any(
        s["scared_distance"] is not None and s["scared_distance"] <= power for s in moves.values()
    )
    if chase:
        def chasing(m: str) -> tuple[int, ...]:
            sd = moves[m]["scared_distance"]
            return (lethal(m), INF if sd is None or sd > power else sd, d(moves[m]["pellet_distance"]))
        return pick(chasing)

    if max(gd(m) for m in moves if not lethal(m)) <= 4:
        return pick(lambda m: (lethal(m), -gd(m), -moves[m]["exits"], d(moves[m]["pellet_distance"])))

    def eating(m: str) -> tuple[int, ...]:
        s = moves[m]
        near = s["ghost_distance"] is not None and gd(m) <= 6
        pellet = d(s["pellet_distance"])
        if s["next_cell"] == "power_pellet" and near:
            pellet = -1
        return (lethal(m), gd(m) < 3, s["exits"] <= 1 and near, pellet, -gd(m), -s["exits"])
    return pick(eating)


# ----------------------------------------------------------------------------------------
# decision records
# ----------------------------------------------------------------------------------------
def question_of(game: str, keys: list[str]) -> dict[str, Any]:
    """The Decision Index question (a ``choice`` over the legal moves)."""
    text = SNAKE_QUESTION if game == "snake" else MAZE_QUESTION
    return {"type": "choice", "instructions": text, "criteria": {k: None for k in keys}}


def record_of(state: dict[str, Any], optimal: Iterable[str], rng: random.Random) -> dict[str, Any]:
    """One decision: the order of the moves is shuffled (as other game agents do) to remove position
    bias; ``optimal`` are the indices of every accepted move."""
    names = list(state["moves"])
    rng.shuffle(names)
    shown = {**state, "moves": {m: state["moves"][m] for m in names}}
    ok = set(optimal)
    return {
        "game": state["game"],
        "state": shown,
        "question": question_of(state["game"], names),
        "keys": names,
        "optimal": [i for i, m in enumerate(names) if m in ok],
    }


def reshuffled(rec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """The same decision with the moves in another order (a new epoch sees a new order)."""
    ok = [rec["keys"][i] for i in rec["optimal"]]
    return record_of(rec["state"], ok, rng)


def policy_of(game: str) -> Callable[[dict[str, Any]], list[str]]:
    return snake_optimal if game == "snake" else maze_optimal


def recompute_optimal(rec: dict[str, Any]) -> list[int]:
    """The optimal set from the RENDERED state alone (round trip through JSON): the proof that no
    label depends on information the model does not see."""
    state = json.loads(json.dumps(rec["state"], ensure_ascii=False, separators=(",", ":")))
    ok = set(policy_of(rec["game"])(state))
    return [i for i, m in enumerate(rec["keys"]) if m in ok]


# ----------------------------------------------------------------------------------------
# play (closed loop) and data generation
# ----------------------------------------------------------------------------------------
Controller = Callable[[dict[str, Any], list[str], random.Random], str]


def oracle_controller(state: dict[str, Any], legal: list[str], rng: random.Random) -> str:
    return rng.choice(policy_of(state["game"])(state))


def random_controller(state: dict[str, Any], legal: list[str], rng: random.Random) -> str:
    return rng.choice(legal)


def new_game(game: str, seed: int, big: bool = False) -> Snake | Maze:
    rng = random.Random(seed)
    if game == "snake":
        size = rng.choice([8, 10, 12]) if big else 10
        return Snake(size, size, rng, length=3)
    return Maze(rng, 13, 11, ghosts=2)


def play(game: str, seed: int, controller: Controller, max_moves: int = 300, big: bool = False,
         on_decision: Callable[[dict[str, Any], str], None] | None = None) -> dict[str, Any]:
    """Play one game to its end (death, win, or ``max_moves``). Returns the final numbers."""
    g = new_game(game, seed, big)
    crng = random.Random(seed * 7919 + 1)
    stall = 0
    while g.alive and g.moves < max_moves:
        legal = g.legal()
        if not legal:
            break
        state = g.sensors()
        if len(legal) == 1:
            move = legal[0]  # nothing to decide: the game plays it, the model is not asked
        else:
            move = controller(state, legal, crng)
            if on_decision:
                on_decision(state, move)
        before = g.score()
        g.step(move)
        stall = 0 if g.score() > before else stall + 1
        if game == "snake" and stall > 120:
            break  # wandering for ever without eating
    won = game == "maze" and g.won()
    return {"score": g.score(), "moves": g.moves, "died": not g.alive and not won, "won": won}


def decisions(game: str, seed: int, n: int, explore: float = 0.2, start_random: float = 0.5) -> list[dict[str, Any]]:
    """``n`` decision records. States come from games played by the oracle with a chance
    (``explore``) of a random move, so that they include the situations a mistaken player reaches;
    for Snake, half of the games start from a random long body."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    while len(out) < n:
        gseed = rng.randrange(10**9)
        grng = random.Random(gseed)
        if game == "snake":
            size = grng.choice([8, 10, 12])
            long_start = grng.random() < start_random
            g: Snake | Maze = Snake(size, size, grng, length=grng.randint(4, max(5, size * size // 4)) if long_start else 3)
        else:
            g = Maze(grng, grng.choice([11, 13]), grng.choice([9, 11]), ghosts=grng.choice([1, 2, 3]),
                     chase=grng.choice([0.35, 0.5, 0.65]))
        stall = 0
        while g.alive and g.moves < 250 and len(out) < n:
            legal = g.legal()
            if not legal:
                break
            state = g.sensors()
            optimal = policy_of(game)(state)
            if len(legal) >= 2 and (rng.random() < 0.5 or len(optimal) < len(state["moves"])):
                out.append(record_of(state, optimal, rng))  # half of the trivial states are skipped
            move = grng.choice(legal) if grng.random() < explore else grng.choice(optimal)
            before = g.score()
            g.step(move)
            stall = 0 if g.score() > before else stall + 1
            if stall > 150:
                break
    return out


# ----------------------------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------------------------
def selftest(n_games: int = 60) -> None:
    import statistics

    # 1. sensors against a naive recomputation, on random positions
    rng = random.Random(0)
    for _ in range(300):
        s = Snake(8, 8, rng, length=rng.randint(3, 20))
        st = s.sensors()
        for m, v in st["moves"].items():
            dx, dy = DIRS[m]
            c = (s.body[0][0] + dx, s.body[0][1] + dy)
            assert c not in s.body[:-1]
            eats = c == s.food
            nb = [c] + (s.body if eats else s.body[:-1])
            assert v["next_cell"] == ("food" if eats else "empty")
            # naive reachability: iterate to a fixed point
            reach = {c}
            changed = True
            while changed:
                changed = False
                for (x, y) in list(reach):
                    for ex, ey in DIRS.values():
                        p = (x + ex, y + ey)
                        if 0 <= p[0] < 8 and 0 <= p[1] < 8 and p not in nb[1:] and p not in reach:
                            reach.add(p)
                            changed = True
            assert v["free_cells"] == len(reach), (v, len(reach))
            tail = nb[-1]
            near_tail = any((tail[0] + ex, tail[1] + ey) in reach for ex, ey in DIRS.values()) or c == tail
            reach_tail = near_tail  # a free cell next to the tail (or the head next to it) is enough
            assert v["reaches_tail"] == reach_tail, (v, reach_tail)
            if not eats and v["steps_to_food"] is not None:
                assert v["steps_to_food"] >= abs(c[0] - s.food[0]) + abs(c[1] - s.food[1])
    print("ok: snake sensors == naive recomputation (300 positions)")

    # 1b. maze sensors against an independent recomputation (distances by relaxation, not BFS)
    def relax(m: Maze, source: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
        dist = {c: 0 for c in source}
        changed = True
        while changed:
            changed = False
            for c in m.open:
                best = min((dist[(c[0] + dx, c[1] + dy)] + 1 for dx, dy in DIRS.values()
                            if (c[0] + dx, c[1] + dy) in dist), default=None)
                if best is not None and best < dist.get(c, 10**6):
                    dist[c] = best
                    changed = True
        return dist

    rng = random.Random(1)
    checked = 0
    for _ in range(120):
        m = Maze(rng, rng.choice([11, 13]), rng.choice([9, 11]), ghosts=rng.choice([1, 2, 3]))
        for _ in range(rng.randint(0, 25)):  # a few moves, so there are eaten pellets and power
            if not m.alive:
                break
            m.step(rng.choice(m.legal()))
        if not m.alive:
            continue
        st = m.sensors()
        danger = {g["pos"] for g in m.ghosts if not g["scared"]}
        scared = {g["pos"] for g in m.ghosts if g["scared"]}
        dd, ds, dp = relax(m, danger), relax(m, scared), relax(m, m.pellets | m.powers)
        for name, v in st["moves"].items():
            c = (m.pos[0] + DIRS[name][0], m.pos[1] + DIRS[name][1])
            assert v["ghost_distance"] == (dd.get(c) if danger else None), (name, v)
            assert v["scared_distance"] == (ds.get(c) if scared else None), (name, v)
            assert v["pellet_distance"] == (dp.get(c) if (m.pellets | m.powers) else None), (name, v)
            assert v["exits"] == sum((c[0] + ex, c[1] + ey) in set(m.open) for ex, ey in DIRS.values())
            checked += 1
    print(f"ok: maze sensors == independent recomputation ({checked} moves in 120 mazes)")

    # 2. every label is a function of the rendered state alone
    for game in ("snake", "maze"):
        recs = decisions(game, 11, 400)
        bad = [r for r in recs if recompute_optimal(r) != r["optimal"]]
        assert not bad, bad[:1]
        sizes = {len(r["keys"]) for r in recs}
        rate = statistics.mean(len(r["optimal"]) / len(r["keys"]) for r in recs)
        print(f"ok: {game}: 400 labels recomputed from the rendered state; options {sorted(sizes)}; "
              f"mean share of accepted moves {rate:.2f}")

    # 3. the ceiling of each policy, against a random player (same seeds)
    for game, cap in (("snake", 300), ("maze", 300)):
        for name, ctl in (("oracle", oracle_controller), ("random", random_controller)):
            res = [play(game, 1000 + i, ctl, max_moves=cap) for i in range(n_games)]
            sc = [r["score"] for r in res]
            print(f"{game:5s} {name:6s} score mean {statistics.mean(sc):6.1f} median {statistics.median(sc):5.1f} "
                  f"died {sum(r['died'] for r in res)}/{n_games}"
                  + (f" won {sum(r['won'] for r in res)}" if game == "maze" else "")
                  + f" moves mean {statistics.mean(r['moves'] for r in res):5.0f}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
