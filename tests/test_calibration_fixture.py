"""The formula vectors used by the calibration tests are the 38 of the reference dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data" / "formulas"


def load_formula_vectors() -> list[dict[str, Any]]:
    text = (DATA / "formulas.jsonl").read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = [json.loads(line) for line in text.splitlines()]
    return rows


def test_formula_vectors_are_the_38() -> None:
    rows = load_formula_vectors()
    assert len(rows) == 38
    assert Counter(r["kind"] for r in rows) == {"choice": 13, "score": 17, "noul": 8}
    for row in rows:
        if row["kind"] == "noul":
            assert {"logits_raw", "temperature", "p_true"} <= row.keys()
        elif row["kind"] == "choice":
            assert {"p", "confidence", "argmax"} <= row.keys()
        else:
            assert {"p", "value", "confidence", "mode"} <= row.keys()


def test_checksum_is_the_one_in_the_readme() -> None:
    digest = hashlib.sha256((DATA / "formulas.jsonl").read_bytes()).hexdigest()
    readme = (DATA / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"`{digest}`", readme)
    assert "Apache" in (DATA / "NOTICE").read_text(encoding="utf-8")
