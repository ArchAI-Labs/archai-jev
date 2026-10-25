"""Shared builders for the evaluation tests: questions, records, scripted models."""

from __future__ import annotations

import random
from typing import Any

from archai_jev import Choice, Jev, Score, YesNo, calibrated_softmax
from archai_jev.evaluation import LabelledRecord
from archai_jev.testing import MockScorer

QUESTIONS = {
    "team": Choice(
        "Which team?", {"returns": "Exchanges", "billing": None, "other": None}
    ),
    "urgent": YesNo("Is it urgent?"),
    "prio": Score("How important?", ["low", "medium", "high"]),
}
KEYS = ["returns", "billing", "other"]
# What the scripted model answers for every record.
LOGITS = [[2.0, 1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 2.0]]


def scripted(
    temperature: float = 1.0,
    calibrated: bool = False,
    logits: list[list[float]] | None = None,
) -> Jev:
    return Jev.from_scorer(
        MockScorer.scripted(
            LOGITS if logits is None else logits,
            temperature=temperature,
            calibrated=calibrated,
        )
    )


def record(
    i: int,
    team: str = "returns",
    urgent: bool = True,
    prio: int = 2,
    group: str | None = None,
    questions: dict[str, Any] | None = None,
) -> LabelledRecord:
    return LabelledRecord(
        f"ticket {i}",
        QUESTIONS if questions is None else questions,
        {"team": team, "urgent": urgent, "prio": prio},
        group=group,
    )


def as_dict(r: LabelledRecord) -> dict[str, Any]:
    """The dict form of a record (the format of the JSONL file)."""
    out: dict[str, Any] = {"state": r.state, "questions": {}}
    for name, q in r.questions.items():
        spec: dict[str, Any] = {"instructions": q.instructions}
        if isinstance(q, Choice):
            spec |= {"type": "choice", "criteria": dict(q.criteria)}
        elif isinstance(q, Score):
            spec |= {"type": "score", "criteria": list(q.criteria)}
        else:
            spec["type"] = "noul"
        spec["label"] = r.labels[name]
        out["questions"][name] = spec
    if r.id is not None:
        out["id"] = r.id
    if r.group is not None:
        out["group"] = r.group
    return out


def drawn_records(
    n: int, true_t: float, seed: int, group_size: int = 1
) -> list[LabelledRecord]:
    """Records whose labels are drawn from ``softmax(LOGITS / true_t)`` (for ``scripted``)."""
    rng = random.Random(seed)
    weights = [calibrated_softmax(z, true_t) for z in LOGITS]
    rows = []
    for i in range(n):
        team = rng.choices(range(3), weights[0])[0]
        urgent = rng.choices(range(2), weights[1])[0]
        prio = rng.choices(range(3), weights[2])[0]
        rows.append(
            record(i, KEYS[team], bool(urgent), prio, group=f"g{i // group_size}")
        )
    return rows
