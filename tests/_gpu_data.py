"""Shared builders for the GPU gate tests: the gate module and a report that passes it."""

from __future__ import annotations

import copy
from pathlib import Path
from types import ModuleType
from typing import Any

import gpu_gate

ROOT = Path(__file__).resolve().parent.parent
#: The archived report of the measurements (aggregates only, spec 011 T3/T4).
REPORT = ROOT / "tests" / "data" / "gpu" / "report.json"

TASKS = ("toxicchat", "rte", "stsb")


def load_gate() -> ModuleType:
    """``tests/gpu_gate.py``: the admission rule, versioned with the tests."""
    return gpu_gate


def _calibration(overall: float, task: float) -> dict[str, Any]:
    return {
        "overall": {"n": 1500, "ece": overall},
        "by_task": {t: {"n": 500, "ece": task} for t in TASKS},
    }


def _latency(ratio: float, aa: float = 1.01) -> dict[str, Any]:
    return {
        regime: {
            workload: {"ratio_to_tuned_cpu": ratio, "aa_median_ratio": aa}
            for workload in ("W1", "W2", "W3", "Q10")
        }
        for regime in ("consecutive", "pause05", "pause15")
    }


def good_backend() -> dict[str, Any]:
    """A backend that passes every criterion, with a margin."""
    return {
        "kind": "cuda",
        "device_type": "discrete",
        "fidelity": {
            "golden": {"questions": 87, "max_dp": 0.05},
            "context": {"questions": 10, "max_dp": 0.04},
            "cost": {"questions": 8, "max_dp": 0.05},
        },
        "calibration": _calibration(0.011, 0.03),
        "latency": _latency(0.2),
        "repeatability": {
            name: {"runs": 20, "distinct_results": 1, "max_spread": 0.0}
            for name in gpu_gate.GP4_CELLS
        },
        "first_use": {
            "process_to_first_answer_s": 1.6,
            "process_to_first_answer_without_cache_s": 1.7,
        },
        "install_safe": {"targets": 3, "passed": 3},
    }


def good_report(backend: str = "cuda:0") -> dict[str, Any]:
    return {
        "schema": "archai-jev.gpu-report/1",
        "cpu": {"calibration": _calibration(0.0111, 0.03)},
        "backends": {backend: good_backend()},
    }


def changed(
    report: dict[str, Any], path: str, value: Any, backend: str = "cuda:0"
) -> dict[str, Any]:
    """A deep copy of ``report`` with the figure at ``path`` (dotted, under the backend, or under
    ``cpu.`` when it starts with that) replaced; ``value`` of ``DELETE`` removes it.
    """
    out = copy.deepcopy(report)
    node: Any = out if path.startswith("cpu.") else out["backends"][backend]
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
