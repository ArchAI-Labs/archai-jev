"""Shared builders for the quantization tests: the gate module and a report that passes it."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import quantization_gate

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "src" / "models" / "registry"
#: The archived report of the measurements (aggregates only). It exists once a quantized
#: variant has been measured; without it the registry may hold no quantized candidate.
REPORT = ROOT / "tests" / "data" / "quantization" / "report.json"

TASKS = ("safety", "intent", "entailment", "similarity")


def load_gate() -> ModuleType:
    """``tests/quantization_gate.py``: the admission rule, versioned with the tests."""
    return quantization_gate


def _quality(ece: float) -> dict[str, Any]:
    return {
        "overall": {
            "n": 1449,
            "accuracy": 0.92,
            "ece": ece,
            "brier": 0.12,
            "log_loss": 0.21,
        },
        "by_task": {
            t: {
                "n": 300,
                "accuracy": 0.92,
                "ece": 0.03,
                "brier": 0.12,
                "log_loss": 0.21,
            }
            for t in TASKS
        },
    }


def _paired(accuracy: float, brier: float) -> dict[str, Any]:
    def diff(x: float) -> dict[str, Any]:
        return {"estimate": x, "ci95": [x - 0.01, x + 0.01]}

    return {
        "overall": {"accuracy_diff": diff(accuracy), "brier_diff": diff(brier)},
        "by_task": {t: {"accuracy_diff": diff(accuracy)} for t in TASKS},
    }


def good_variant(sha: str = "a" * 64) -> dict[str, Any]:
    """A variant that passes every criterion, with a margin, and that has a latency advantage but
    none in memory or size (so a test can break the latency alone)."""
    return {
        "sha256": sha,
        "size_bytes": 990_000_000,
        "size_ratio": 0.80,
        "calibration": {"temperature": 2.1, "fit_n": 1447, "test_n": 1449},
        "quality": _quality(0.020),
        "paired": _paired(-0.004, 0.004),
        "latency": {
            "aa_median_ratio": 1.01,
            "consecutive": {"W1": 0.56, "W2": 0.57, "W3": 0.56},
            "spaced": {"W1": 0.62, "W2": 0.63, "W3": 0.61},
        },
        "memory": {"peak_ratio": 0.98, "steady_ratio": 0.95},
        "selfcheck": {
            "vectors": 10,
            "mutants_total": 12,
            "mutants_caught": 12,
            "platforms": 2,
            "tol_p": 0.03,
            "max_cross_platform_p": 0.004,
        },
    }


def good_report(dtype: str = "q4_k_m") -> dict[str, Any]:
    return {
        "schema": "archai-jev.quantization-report/1",
        "baseline": {"dtype": "q8_0", "quality": _quality(0.0129)},
        "variants": {dtype: good_variant()},
    }


def changed(
    report: dict[str, Any], path: str, value: Any, dtype: str = "q4_k_m"
) -> dict[str, Any]:
    """A deep copy of ``report`` with the figure at ``path`` (dotted, under the variant, or under
    ``baseline.`` when it starts with that) replaced; ``value`` of ``DELETE`` removes it.
    """
    out = copy.deepcopy(report)
    node: Any = out if path.startswith("baseline.") else out["variants"][dtype]
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    if value is DELETE:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    return out


class _Delete:
    def __repr__(self) -> str:
        return "DELETE"


DELETE = _Delete()


def registry_manifests() -> list[dict[str, Any]]:
    """The manifests of the built-in registry, as plain JSON."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(REGISTRY.glob("*.json"))
        if p.name != "index.json"
    ]


def candidate_manifest(
    *,
    sha: str = "a" * 64,
    declared: bool = True,
    temperature: float | None = 2.1,
    reference: str | None = "engine",
) -> dict[str, Any]:
    """A registry entry with the baseline q8_0 and a q4_k_m candidate."""
    calibration: dict[str, Any] = {"declared": declared}
    if declared:
        calibration.update(
            temperature=temperature,
            evidence="variant q4_k_m, n=1447, n=1449, ECE 0.1 -> 0.02",
        )
    selfcheck: dict[str, Any] = {"vectors": []}
    if reference is not None:
        selfcheck["reference"] = reference
    return {
        "name": "someone/model",
        "revision": "1" * 40,
        "default_dtype": "q8_0",
        "variants": [
            {
                "dtype": "q8_0",
                "source": {"kind": "gguf", "file": {"sha256": "b" * 64}},
                "calibration": {
                    "declared": True,
                    "temperature": 2.09,
                    "evidence": "q8_0",
                },
                "selfcheck": {"vectors": []},
            },
            {
                "dtype": "q4_k_m",
                "source": {"kind": "gguf", "file": {"sha256": sha}},
                "calibration": calibration,
                "selfcheck": selfcheck,
            },
        ],
    }
