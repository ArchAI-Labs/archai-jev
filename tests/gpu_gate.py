"""The admission gate of a GPU backend (spec 011, section 5), as code.

A backend gets tolerance rows if GP1 and GP2 hold, and an artefact is published only if GP3-GP6 hold
too. The thresholds below were written BEFORE the measurements were made and are not to be touched
after seeing them: a backend that fails is a result, not a bug. Changing a number is a decision.

    python tests/gpu_gate.py tests/data/gpu/report.json

The input is the report written by ``benches/gpu/summarize.py report`` (schema
``archai-jev.gpu-report/1``). A missing figure fails its check: no measure, no admission.

This file lives with the tests, not with the (local, unversioned) benchmark scripts, because the rule
that admits a backend must be reviewed and run by CI like any other code.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "archai-jev.gpu-report/1"

# ---- thresholds (spec 011, section 5; accepted by the user on 2026-10-19, D63) -----------------

#: GP1: the largest |dp| against the fp32 oracle may not exceed the CPU row of (family, dtype).
GP1_MAX_DP = 0.1
#: GP2: overall ECE on the device may exceed the CPU's by this ...
GP2_OVERALL_ECE_MARGIN = 0.01
#: ... and per task by this (as G1 of the quantization gate, after D57).
GP2_TASK_ECE_MARGIN = 0.02
#: GP3: p50 of the device over p50 of the tuned CPU, in BOTH regimes, on at least one workload.
GP3_LATENCY_RATIO_MAX = 0.85
#: GP4: when the same request is not bit-identical, the largest spread that may be declared.
GP4_MAX_SPREAD = 1e-3
#: GP5: from process start to the first answer; above this the distribution is blocked.
GP5_FIRST_USE_MAX_S = 60.0

GP3_WORKLOADS = ("W1", "W3", "Q10")
GP3_REGIMES = ("consecutive", "pause05")
#: The A/A control (the same device against itself) must have its median ratio inside this interval,
#: else the latency cell is discarded.
AA_INTERVAL = (0.95, 1.05)
#: Repeatability is asked on these requests (cold = first call of the process, warm = a later one).
GP4_CELLS = (
    "W1_cold",
    "W1_warm",
    "W3_cold",
    "W3_warm",
    "Q10_cold",
    "Q10_warm",
)
#: Fidelity is asked on the golden of the default v2 and on the wide set of the quantization study.
GP1_SETS = ("golden", "context", "cost")

THRESHOLDS: dict[str, Any] = {
    "GP1": {"max_dp": GP1_MAX_DP},
    "GP2": {
        "overall_ece_margin": GP2_OVERALL_ECE_MARGIN,
        "task_ece_margin": GP2_TASK_ECE_MARGIN,
    },
    "GP3": {"latency_ratio_max": GP3_LATENCY_RATIO_MAX},
    "GP4": {"max_spread": GP4_MAX_SPREAD},
    "GP5": {"first_use_max_s": GP5_FIRST_USE_MAX_S},
}


@dataclass(frozen=True)
class Check:
    """One criterion: whether it holds, what was measured, what it was compared with, and why."""

    passed: bool
    measured: Any
    threshold: Any
    detail: str


@dataclass(frozen=True)
class GateResult:
    """The verdict on one backend: a check per criterion (GP1..GP6)."""

    backend: str
    checks: dict[str, Check] = field(default_factory=dict)

    def _all(self, names: tuple[str, ...]) -> bool:
        return all(n in self.checks and self.checks[n].passed for n in names)

    @property
    def tolerance_rows(self) -> bool:
        """GP1 and GP2: the backend may have tolerance rows (and so answer at all)."""
        return self._all(("GP1", "GP2"))

    @property
    def distribution_ready(self) -> bool:
        """GP1-GP5: everything that can be measured before an artefact exists."""
        return self._all(("GP1", "GP2", "GP3", "GP4", "GP5"))

    @property
    def publishable(self) -> bool:
        """GP1-GP6: an artefact for this backend may be published."""
        return self._all(("GP1", "GP2", "GP3", "GP4", "GP5", "GP6"))

    def failures(self) -> list[str]:
        return [
            f"{name}: {c.detail}" for name, c in self.checks.items() if not c.passed
        ]


class _MissingError(Exception):
    """A figure the gate needs is not in the report."""


def _get(node: Any, *path: str) -> Any:
    here = node
    for key in path:
        if not isinstance(here, dict) or key not in here:
            raise _MissingError("/".join(path))
        here = here[key]
    return here


def _number(node: Any, *path: str) -> float:
    value = _get(node, *path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise _MissingError("/".join(path) + " (not a finite number)")
    return float(value)


def _guard(fn: Any) -> Check:
    """Run one criterion; a missing figure is a failed check, never an exception."""
    try:
        result: Check = fn()
    except _MissingError as missing:
        return Check(False, None, None, f"missing from the report: {missing}")
    return result


# ---- the six criteria ---------------------------------------------------------------------------


def _gp1(report: dict[str, Any], b: dict[str, Any]) -> Check:
    del report
    worst = 0.0
    for name in GP1_SETS:  # every set must have been measured
        worst = max(worst, _number(b, "fidelity", name, "max_dp"))
    if worst > GP1_MAX_DP:
        return Check(
            False,
            worst,
            GP1_MAX_DP,
            f"max |dp| {worst:.4f} above the CPU row {GP1_MAX_DP}",
        )
    return Check(True, worst, GP1_MAX_DP, f"max |dp| {worst:.4f}")


def _gp2(report: dict[str, Any], b: dict[str, Any]) -> Check:
    base = _number(report, "cpu", "calibration", "overall", "ece")
    got = _number(b, "calibration", "overall", "ece")
    limit = base + GP2_OVERALL_ECE_MARGIN
    if got > limit:
        return Check(
            False,
            got,
            limit,
            f"overall ECE {got:.4f} exceeds the CPU's {base:.4f} + {GP2_OVERALL_ECE_MARGIN}",
        )
    tasks = _get(b, "calibration", "by_task")
    if not isinstance(tasks, dict) or not tasks:
        raise _MissingError("calibration/by_task")
    for task in tasks:
        ece = _number(b, "calibration", "by_task", task, "ece")
        base_task = _number(report, "cpu", "calibration", "by_task", task, "ece")
        task_limit = base_task + GP2_TASK_ECE_MARGIN
        if ece > task_limit:
            return Check(
                False,
                ece,
                task_limit,
                f"ECE of task {task!r} is {ece:.4f}, above the CPU's {base_task:.4f} + {GP2_TASK_ECE_MARGIN}",
            )
    return Check(
        True, got, limit, f"overall ECE {got:.4f} against the CPU's {base:.4f}"
    )


def _cell(b: dict[str, Any], regime: str, workload: str) -> tuple[bool, str]:
    """One latency cell: (advantage?, why).

    A claimed advantage needs a valid A/A control (the device against itself); a cell that shows no
    advantage does not, the claim it refuses is not made.
    """
    ratio = _number(b, "latency", regime, workload, "ratio_to_tuned_cpu")
    if ratio > GP3_LATENCY_RATIO_MAX:
        return False, f"{regime}: ratio {ratio:.3f} above {GP3_LATENCY_RATIO_MAX}"
    aa = _number(b, "latency", regime, workload, "aa_median_ratio")
    if not AA_INTERVAL[0] <= aa <= AA_INTERVAL[1]:
        return (
            False,
            f"{regime}: A/A control {aa:.3f} outside {AA_INTERVAL}, the cell is discarded",
        )
    return True, f"{regime} {ratio:.3f}"


def _gp3(report: dict[str, Any], b: dict[str, Any]) -> Check:
    del report
    wins: list[str] = []
    reasons: list[str] = []
    for workload in GP3_WORKLOADS:
        details: list[str] = []
        ok = True
        for regime in GP3_REGIMES:
            try:
                good, why = _cell(b, regime, workload)
            except _MissingError as missing:
                ok = False
                details.append(f"{regime}: not measured ({missing})")
                continue
            ok = ok and good
            details.append(why)
        if ok:
            wins.append(workload)
        else:
            reasons.append(f"{workload}: " + "; ".join(details))
    if wins:
        return Check(
            True,
            wins,
            GP3_LATENCY_RATIO_MAX,
            "advantage in both regimes on " + ", ".join(wins),
        )
    return Check(
        False,
        None,
        GP3_LATENCY_RATIO_MAX,
        "no measured advantage; " + " | ".join(reasons),
    )


def _gp4(report: dict[str, Any], b: dict[str, Any]) -> Check:
    del report
    spread = 0.0
    distinct = 0
    for name in GP4_CELLS:
        runs = _number(b, "repeatability", name, "runs")
        if runs < 20:
            return Check(False, runs, 20, f"{name}: {runs:g} runs, 20 are asked")
        distinct = max(
            distinct, int(_number(b, "repeatability", name, "distinct_results"))
        )
        spread = max(spread, _number(b, "repeatability", name, "max_spread"))
    if distinct == 1 and spread == 0.0:
        return Check(True, 0.0, 0.0, "bit-identical in every cell")
    if spread <= GP4_MAX_SPREAD:
        return Check(
            True,
            spread,
            GP4_MAX_SPREAD,
            f"not bit-identical: spread {spread:g}; the guide lists the 007 identity tests that hold",
        )
    return Check(
        False, spread, GP4_MAX_SPREAD, f"spread {spread:g} above {GP4_MAX_SPREAD}"
    )


def _gp5(report: dict[str, Any], b: dict[str, Any]) -> Check:
    del report
    worst = _number(b, "first_use", "process_to_first_answer_s")
    without_cache = _number(b, "first_use", "process_to_first_answer_without_cache_s")
    worst = max(worst, without_cache)
    if worst > GP5_FIRST_USE_MAX_S:
        return Check(
            False,
            worst,
            GP5_FIRST_USE_MAX_S,
            f"first use takes {worst:.1f} s: shader/PTX compilation blocks the distribution",
        )
    return Check(True, worst, GP5_FIRST_USE_MAX_S, f"first use {worst:.1f} s")


def _gp6(report: dict[str, Any], b: dict[str, Any]) -> Check:
    del report
    targets = int(_number(b, "install_safe", "targets"))
    passed = int(_number(b, "install_safe", "passed"))
    if targets <= 0:
        return Check(False, targets, ">0", "no target was tried")
    if passed != targets:
        return Check(
            False, passed, targets, f"{passed} of {targets} targets import and answer"
        )
    return Check(True, passed, targets, f"all {targets} targets are safe to install")


_CRITERIA = (
    ("GP1", _gp1),
    ("GP2", _gp2),
    ("GP3", _gp3),
    ("GP4", _gp4),
    ("GP5", _gp5),
    ("GP6", _gp6),
)


def evaluate_gate(report: dict[str, Any], backend: str) -> GateResult:
    """The verdict of the gate on ``backend`` (``"cuda:0"``, ``"vulkan:1"``, ...) of ``report``.

    A criterion whose figures are missing from the report fails (fail-closed); a report that is not
    a dict of the right schema, or a backend that is not in it, is a ``ValueError``.
    """
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError(f"not a GPU report (schema {SCHEMA!r} expected)")
    backends = report.get("backends")
    if (
        not isinstance(backends, dict)
        or backend not in backends
        or not isinstance(backends[backend], dict)
    ):
        raise ValueError(f"the report has no backend {backend!r}")
    b = backends[backend]
    checks = {name: _guard(lambda fn=fn: fn(report, b)) for name, fn in _CRITERIA}
    return GateResult(backend=backend, checks=checks)


def _ms(cell: dict[str, Any]) -> str:
    return f"{cell['p50_ms']:.0f} ms ({cell['ratio_to_tuned_cpu']:.2f}x)"


def render_table(report: dict[str, Any]) -> str:
    """The tables of ``docs/gpu.md``, generated from the archived report (never written by hand).

    One table per regime (calls back to back, 0.5 s of pause between calls), a column per workload
    (p50 of the device and its ratio to the best setting of the CPU in the same blocks), plus the
    fidelity and the calibration that decide whether the backend has tolerance rows.
    """
    backends: dict[str, Any] = report["backends"]
    cpu_ece = report["cpu"]["calibration"]["overall"]["ece"]
    out: list[str] = []
    for regime, title in (
        ("consecutive", "Calls back to back"),
        ("pause05", "A pause of 0.5 s between calls"),
    ):
        out += [
            f"**{title}** (p50 of the device, ratio to the tuned CPU in brackets; below 1 is faster)",
            "",
            "| Device | Kind | W1 (94 tokens) | W3 (365 tokens) | Q10 (10 questions, 1537 tokens) |",
            "|---|---|---|---|---|",
        ]
        for name, b in backends.items():
            lat = b["latency"][regime]
            out.append(
                f"| `{name}` {b['device']} | {b['device_type']} | {_ms(lat['W1'])} | {_ms(lat['W3'])} | {_ms(lat['Q10'])} |"
            )
        out.append("")
    out += [
        "**Fidelity and calibration** (default model, fp32 reference; ECE of the CPU: "
        f"{cpu_ece:.4f})",
        "",
        "| Device | Largest abs. difference in probability | ECE | Tolerance row |",
        "|---|---|---|---|",
    ]
    for name, b in backends.items():
        dp = max(b["fidelity"][k]["max_dp"] for k in b["fidelity"])
        ece = b.get("calibration", {}).get("overall", {}).get("ece")
        rows = "yes" if evaluate_gate(report, name).tolerance_rows else "no"
        out.append(
            f"| `{name}` | {dp:.3f} | {'not measured' if ece is None else f'{ece:.4f}'} | {rows} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    path = (
        Path(argv[1])
        if len(argv) > 1
        else Path(__file__).parent / "data" / "gpu" / "report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    for backend in report.get("backends", {}):
        result = evaluate_gate(report, backend)
        rows = "tolerance rows: " + ("yes" if result.tolerance_rows else "no")
        dist = "GP1-GP5: " + ("pass" if result.distribution_ready else "no")
        print(f"{backend}: {rows}; {dist}; publishable: {result.publishable}")
        for name, check in result.checks.items():
            print(f"  {name} {'ok  ' if check.passed else 'FAIL'} {check.detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
