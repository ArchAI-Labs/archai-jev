"""The admission gate of a quantized variant (spec 009, section 5), as code.

A quantized variant enters the registry only if ALL of G1-G5 hold. The thresholds below were
written BEFORE the measurements were made and are not to be touched after seeing them: a variant
that fails is a result, not a bug (spec 009; D50). Changing a number is a decision.

    python tests/quantization_gate.py tests/data/quantization/report.json

The input is the report written by ``benches/quantization/summarize.py`` (schema
``archai-jev.quantization-report/1``, described in ``benches/quantization/README.md``). A missing
figure fails its check: no measure, no admission.

This file lives with the tests, not with the (local, unversioned) benchmark scripts, because the
rule that admits a variant to the registry must be reviewed and run by CI like any other code.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "archai-jev.quantization-report/1"

# ---- thresholds (spec 009, section 5; confirmed by the user on 2026-10-13, D50) ----------------

#: G1: overall ECE (test split, T fitted on the validation split) may exceed the baseline by this.
G1_OVERALL_ECE_MARGIN = 0.01
#: G1: per task, the ECE may exceed the baseline's ECE of the same task by this. (Amended 2026-10-18,
#: D57, before any v2 variant was measured: the first rule, an absolute 0.06, was tuned on the v1 and
#: the v2 baseline itself is above it on two tasks, which would have refused every variant.)
G1_TASK_ECE_MARGIN = 0.02
#: G2: paired accuracy difference (variant - baseline), overall: at least this (-1.0 point).
G2_OVERALL_ACCURACY_MIN = -0.010
#: G2: ... and for each task (-3.0 points).
G2_TASK_ACCURACY_MIN = -0.030
#: G3: paired Brier difference (variant - baseline), overall: at most this.
G3_BRIER_MAX = 0.01
#: G4: p50 latency ratio (variant / baseline) must be at most this on W1 and W3, in both regimes.
G4_LATENCY_RATIO_MAX = 0.85
#: G4 (alternative): steady-state memory ratio at most this.
G4_MEMORY_RATIO_MAX = 0.85
#: G4 (alternative): file size ratio at most this.
G4_SIZE_RATIO_MAX = 0.70
#: G5: the tolerance of the engine-recorded vectors may not exceed this ...
G5_TOL_P_MAX = 0.05
#: G5: ... and must be at least this many times the largest difference seen between platforms.
G5_TOL_P_MIN_MARGIN = 2.0
#: G5: platforms the vectors were reproduced on.
G5_MIN_PLATFORMS = 2

LATENCY_REGIMES = ("consecutive", "spaced")
LATENCY_WORKLOADS = ("W1", "W3")
#: The A/A control (baseline against itself) must have its median ratio inside this interval.
AA_INTERVAL = (0.95, 1.05)

THRESHOLDS: dict[str, Any] = {
    "G1": {
        "overall_ece_margin": G1_OVERALL_ECE_MARGIN,
        "task_ece_margin": G1_TASK_ECE_MARGIN,
    },
    "G2": {
        "overall_accuracy_min": G2_OVERALL_ACCURACY_MIN,
        "task_accuracy_min": G2_TASK_ACCURACY_MIN,
    },
    "G3": {"brier_max": G3_BRIER_MAX},
    "G4": {
        "latency_ratio_max": G4_LATENCY_RATIO_MAX,
        "memory_ratio_max": G4_MEMORY_RATIO_MAX,
        "size_ratio_max": G4_SIZE_RATIO_MAX,
    },
    "G5": {
        "tol_p_max": G5_TOL_P_MAX,
        "tol_p_min_margin": G5_TOL_P_MIN_MARGIN,
        "min_platforms": G5_MIN_PLATFORMS,
    },
}

#: dtypes that are not quantized (nothing to admit)
FULL_PRECISION = ("f32", "bf16")


@dataclass(frozen=True)
class Check:
    """One criterion: whether it holds, what was measured, what it was compared with, and why."""

    passed: bool
    measured: Any
    threshold: Any
    detail: str


@dataclass(frozen=True)
class GateResult:
    """The verdict on one variant: a check per criterion (G1..G5) and the sub-checks of each."""

    dtype: str
    checks: dict[str, Check] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks.values())

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


def _ci(node: Any, *path: str) -> tuple[float, float]:
    value = _get(node, *path)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        )
        or value[0] > value[1]
    ):
        raise _MissingError("/".join(path) + " (needs a 95% interval [low, high])")
    return float(value[0]), float(value[1])


def _guard(fn: Any) -> Check:
    """Run one criterion; a missing figure is a failed check, never an exception."""
    try:
        result: Check = fn()
    except _MissingError as missing:
        return Check(False, None, None, f"missing from the report: {missing}")
    return result


# ---- the five criteria --------------------------------------------------------------------------


def _g1(report: dict[str, Any], v: dict[str, Any]) -> Check:
    base = _number(report, "baseline", "quality", "overall", "ece")
    got = _number(v, "quality", "overall", "ece")
    limit = base + G1_OVERALL_ECE_MARGIN
    if got > limit:
        return Check(
            False,
            got,
            limit,
            f"overall ECE {got:.4f} exceeds baseline {base:.4f} + {G1_OVERALL_ECE_MARGIN}",
        )
    tasks = _get(v, "quality", "by_task")
    if not isinstance(tasks, dict) or not tasks:
        raise _MissingError("quality/by_task")
    for task in tasks:
        ece = _number(v, "quality", "by_task", task, "ece")
        base_task = _number(report, "baseline", "quality", "by_task", task, "ece")
        task_limit = base_task + G1_TASK_ECE_MARGIN
        if ece > task_limit:
            return Check(
                False,
                ece,
                task_limit,
                f"ECE of task {task!r} is {ece:.4f}, above the baseline's {base_task:.4f} + {G1_TASK_ECE_MARGIN}",
            )
    return Check(True, got, limit, "calibration within the margin")


def _g2(report: dict[str, Any], v: dict[str, Any]) -> Check:
    del report
    overall = _number(v, "paired", "overall", "accuracy_diff", "estimate")
    _ci(
        v, "paired", "overall", "accuracy_diff", "ci95"
    )  # the interval must be reported
    if overall < G2_OVERALL_ACCURACY_MIN:
        return Check(
            False,
            overall,
            G2_OVERALL_ACCURACY_MIN,
            f"accuracy {overall * 100:+.2f} points against the baseline",
        )
    tasks = _get(v, "paired", "by_task")
    if not isinstance(tasks, dict) or not tasks:
        raise _MissingError("paired/by_task")
    for task in tasks:
        diff = _number(v, "paired", "by_task", task, "accuracy_diff", "estimate")
        _ci(v, "paired", "by_task", task, "accuracy_diff", "ci95")
        if diff < G2_TASK_ACCURACY_MIN:
            return Check(
                False,
                diff,
                G2_TASK_ACCURACY_MIN,
                f"task {task!r}: accuracy {diff * 100:+.2f} points",
            )
    return Check(True, overall, G2_OVERALL_ACCURACY_MIN, "accuracy within the margin")


def _g3(report: dict[str, Any], v: dict[str, Any]) -> Check:
    del report
    diff = _number(v, "paired", "overall", "brier_diff", "estimate")
    _ci(v, "paired", "overall", "brier_diff", "ci95")
    if diff > G3_BRIER_MAX:
        return Check(
            False, diff, G3_BRIER_MAX, f"Brier {diff:+.4f} against the baseline"
        )
    return Check(True, diff, G3_BRIER_MAX, "Brier within the margin")


def _latency_advantage(v: dict[str, Any]) -> tuple[bool, str]:
    """p50 ratio <= the limit on W1 and W3, in both regimes, with a valid A/A control."""
    aa = _get(v, "latency", "aa_median_ratio")
    if (
        isinstance(aa, bool)
        or not isinstance(aa, (int, float))
        or not AA_INTERVAL[0] <= aa <= AA_INTERVAL[1]
    ):
        return (
            False,
            f"A/A control {aa!r} outside {AA_INTERVAL}: the latency bench is discarded",
        )
    worst = 0.0
    for regime in LATENCY_REGIMES:
        for workload in LATENCY_WORKLOADS:
            worst = max(worst, _number(v, "latency", regime, workload))
    if worst > G4_LATENCY_RATIO_MAX:
        return False, f"worst latency ratio {worst:.3f} above {G4_LATENCY_RATIO_MAX}"
    return True, f"latency ratio at most {worst:.3f} in both regimes"


def _memory_advantage(v: dict[str, Any]) -> tuple[bool, str]:
    ratio = _number(v, "memory", "steady_ratio")
    return (
        ratio <= G4_MEMORY_RATIO_MAX,
        f"steady memory ratio {ratio:.3f}, limit {G4_MEMORY_RATIO_MAX}",
    )


def _size_advantage(v: dict[str, Any]) -> tuple[bool, str]:
    ratio = _number(v, "size_ratio")
    return (
        ratio <= G4_SIZE_RATIO_MAX,
        f"file size ratio {ratio:.3f}, limit {G4_SIZE_RATIO_MAX}",
    )


def _g4(report: dict[str, Any], v: dict[str, Any]) -> Check:
    del report
    reasons: list[str] = []
    for name, measure in (
        ("latency", _latency_advantage),
        ("memory", _memory_advantage),
        ("size", _size_advantage),
    ):
        try:
            ok, why = measure(v)
        except _MissingError as missing:
            reasons.append(f"{name}: not measured ({missing})")
            continue
        if ok:
            return Check(True, name, None, f"advantage: {why}")
        reasons.append(f"{name}: {why}")
    return Check(False, None, None, "no measured advantage; " + "; ".join(reasons))


def _g5(report: dict[str, Any], v: dict[str, Any]) -> Check:
    del report
    total = int(_number(v, "selfcheck", "mutants_total"))
    caught = int(_number(v, "selfcheck", "mutants_caught"))
    if total <= 0:
        return Check(False, total, ">0", "no mutants were tried")
    if caught != total:
        return Check(
            False, caught, total, f"only {caught} of {total} mutants were caught"
        )
    platforms = int(_number(v, "selfcheck", "platforms"))
    if platforms < G5_MIN_PLATFORMS:
        return Check(
            False,
            platforms,
            G5_MIN_PLATFORMS,
            f"vectors reproduced on {platforms} platform(s)",
        )
    tol = _number(v, "selfcheck", "tol_p")
    seen = _number(v, "selfcheck", "max_cross_platform_p")
    if tol > G5_TOL_P_MAX:
        return Check(False, tol, G5_TOL_P_MAX, f"tol_p {tol} above {G5_TOL_P_MAX}")
    if tol < G5_TOL_P_MIN_MARGIN * seen:
        return Check(
            False,
            tol,
            G5_TOL_P_MIN_MARGIN * seen,
            f"tol_p {tol} below {G5_TOL_P_MIN_MARGIN} x {seen} (largest platform difference)",
        )
    return Check(True, tol, G5_TOL_P_MAX, "vectors reproduced and every mutant caught")


_CRITERIA = (("G1", _g1), ("G2", _g2), ("G3", _g3), ("G4", _g4), ("G5", _g5))


def evaluate_gate(report: dict[str, Any], dtype: str) -> GateResult:
    """The verdict of the gate on the variant ``dtype`` of ``report``.

    A criterion whose figures are missing from the report fails (fail-closed); a report that is
    not a dict of the right schema, or a variant that is not in it, is a ``ValueError``.
    """
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError(f"not a quantization report (schema {SCHEMA!r} expected)")
    variants = report.get("variants")
    if (
        not isinstance(variants, dict)
        or dtype not in variants
        or not isinstance(variants[dtype], dict)
    ):
        raise ValueError(f"the report has no variant {dtype!r}")
    v = variants[dtype]
    checks = {name: _guard(lambda fn=fn: fn(report, v)) for name, fn in _CRITERIA}
    return GateResult(dtype=dtype, checks=checks)


# ---- the registry must agree with the reports ---------------------------------------------------


def candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The quantized variants of a registry entry that are not its baseline (the default dtype)."""
    return [
        v
        for v in manifest.get("variants", [])
        if v["dtype"] not in FULL_PRECISION
        and v["dtype"] != manifest.get("default_dtype")
    ]


def lint_registry(
    manifests: list[dict[str, Any]], report: dict[str, Any] | None
) -> list[str]:
    """What is wrong with the quantized candidates of a registry (spec 009, AC-7 and AC-14).

    Every quantized variant that is not the baseline of its entry must (1) declare a calibration,
    (2) use engine-recorded vectors, (3) have an archived report whose GGUF SHA-256 is the
    registry's, (4) pass the gate, and (5) declare the temperature the report measured. With no
    candidate there is nothing to check and the answer is an empty list.
    """
    problems: list[str] = []
    for m in manifests:
        for v in candidates(m):
            tag = f"{m['name']}@{m['revision'][:8]} {v['dtype']}"
            calibration = v.get("calibration", {})
            if not calibration.get("declared"):
                problems.append(
                    f"{tag}: a quantized variant enters the registry with a measured, declared calibration"
                )
            if v.get("selfcheck", {}).get("reference") != "engine":
                problems.append(
                    f'{tag}: needs vectors of its own (selfcheck.reference = "engine")'
                )
            if report is None:
                problems.append(f"{tag}: no archived report")
                continue
            try:
                entry = report["variants"][v["dtype"]]
            except (KeyError, TypeError):
                problems.append(f"{tag}: the report has no such variant")
                continue
            if entry.get("sha256") != v.get("source", {}).get("file", {}).get("sha256"):
                problems.append(
                    f"{tag}: the report measured another file (GGUF SHA-256 differs from the registry's)"
                )
            result = evaluate_gate(report, v["dtype"])
            if not result.passed:
                problems.extend(f"{tag}: gate {f}" for f in result.failures())
            measured = entry.get("calibration", {}).get("temperature")
            declared = calibration.get("temperature")
            if (
                calibration.get("declared")
                and isinstance(measured, (int, float))
                and isinstance(declared, (int, float))
                and abs(round(measured, 2) - declared) > 0.005
            ):
                problems.append(
                    f"{tag}: declares T = {declared} but the report measured {round(measured, 2)}"
                )
    return problems


def main(argv: list[str]) -> int:
    path = (
        Path(argv[1])
        if len(argv) > 1
        else Path(__file__).parent / "data" / "quantization" / "report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    code = 0
    for dtype in report.get("variants", {}):
        result = evaluate_gate(report, dtype)
        print(f"{dtype}: {'ADMITTED' if result.passed else 'REFUSED'}")
        for name, check in result.checks.items():
            print(f"  {name} {'ok  ' if check.passed else 'FAIL'} {check.detail}")
        code = code or (0 if result.passed else 1)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
