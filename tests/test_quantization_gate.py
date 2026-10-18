"""The admission gate of spec 009 (G1-G5) as executable rules.

Every threshold has a test that stops just short of it (accepted) and just beyond it (refused),
so moving a number in ``tests/quantization_gate.py`` makes a test fail: a threshold is a
decision, not a knob. All figures here are synthetic (no model, no weights).
"""

from __future__ import annotations

from typing import Any

import pytest

from _quantization_data import DELETE, changed, good_report, load_gate

gate = load_gate()
EPS = 1e-9
REPORT = good_report()
# a variant with no advantage in memory or size, so that a test can break the latency alone
NO_OTHER_ADVANTAGE = {"memory.steady_ratio": 0.99, "size_ratio": 0.99}


def run(report: dict[str, Any]) -> Any:
    return gate.evaluate_gate(report, "q4_k_m")


def only_latency(report: dict[str, Any]) -> dict[str, Any]:
    for path, value in NO_OTHER_ADVANTAGE.items():
        report = changed(report, path, value)
    return report


def test_a_good_report_is_admitted() -> None:
    result = run(REPORT)
    assert result.passed, result.failures()
    assert list(result.checks) == ["G1", "G2", "G3", "G4", "G5"]


def test_the_thresholds_are_the_ones_the_user_confirmed() -> None:
    # spec 009 section 5 and D50. Moving one of these is a decision: change it here on purpose.
    assert gate.THRESHOLDS == {
        "G1": {"overall_ece_margin": 0.01, "task_ece_margin": 0.02},
        "G2": {"overall_accuracy_min": -0.010, "task_accuracy_min": -0.030},
        "G3": {"brier_max": 0.01},
        "G4": {
            "latency_ratio_max": 0.85,
            "memory_ratio_max": 0.85,
            "size_ratio_max": 0.70,
        },
        "G5": {"tol_p_max": 0.05, "tol_p_min_margin": 2.0, "min_platforms": 2},
    }
    assert gate.LATENCY_REGIMES == ("consecutive", "spaced")
    assert gate.LATENCY_WORKLOADS == ("W1", "W3")
    assert gate.AA_INTERVAL == (0.95, 1.05)


BASE_ECE = REPORT["baseline"]["quality"]["overall"]["ece"]
BASE_TASK_ECE = REPORT["baseline"]["quality"]["by_task"]["intent"]["ece"]

# (criterion, figure, value that just passes, value that just fails, extra changes)
EDGES = [
    ("G1", "quality.overall.ece", BASE_ECE + 0.01 - EPS, BASE_ECE + 0.01 + EPS, {}),
    (
        "G1",
        "quality.by_task.intent.ece",
        BASE_TASK_ECE + 0.02 - EPS,
        BASE_TASK_ECE + 0.02 + EPS,
        {},
    ),
    ("G2", "paired.overall.accuracy_diff.estimate", -0.010 + EPS, -0.010 - EPS, {}),
    (
        "G2",
        "paired.by_task.similarity.accuracy_diff.estimate",
        -0.030 + EPS,
        -0.030 - EPS,
        {},
    ),
    ("G3", "paired.overall.brier_diff.estimate", 0.01 - EPS, 0.01 + EPS, {}),
    ("G4", "latency.consecutive.W1", 0.85, 0.85 + EPS, NO_OTHER_ADVANTAGE),
    ("G4", "latency.consecutive.W3", 0.85, 0.85 + EPS, NO_OTHER_ADVANTAGE),
    ("G4", "latency.spaced.W1", 0.85, 0.85 + EPS, NO_OTHER_ADVANTAGE),
    ("G4", "latency.spaced.W3", 0.85, 0.85 + EPS, NO_OTHER_ADVANTAGE),
    ("G4", "latency.aa_median_ratio", 1.05, 1.05 + EPS, NO_OTHER_ADVANTAGE),
    ("G4", "latency.aa_median_ratio", 0.95, 0.95 - EPS, NO_OTHER_ADVANTAGE),
    ("G5", "selfcheck.tol_p", 0.05, 0.05 + EPS, {}),
    ("G5", "selfcheck.platforms", 2, 1, {}),
    ("G5", "selfcheck.mutants_caught", 12, 11, {}),
]


def apply_all(report: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for path, value in extra.items():
        report = changed(report, path, value)
    return report


@pytest.mark.parametrize(("criterion", "path", "ok", "bad", "extra"), EDGES)
def test_each_threshold_accepts_at_the_edge_and_refuses_beyond_it(
    criterion: str, path: str, ok: Any, bad: Any, extra: dict[str, Any]
) -> None:
    base = apply_all(REPORT, extra)
    accepted = run(changed(base, path, ok))
    assert accepted.checks[criterion].passed, accepted.failures()
    result = run(changed(base, path, bad))
    assert not result.checks[criterion].passed, f"{path}={bad} should fail {criterion}"
    assert not result.passed
    # a refusal is local: the other criteria still read their own figures
    others = [c for name, c in result.checks.items() if name != criterion]
    assert all(c.passed for c in others), result.failures()


def test_zero_mutants_tried_is_refused_even_though_none_escaped() -> None:
    none_tried = changed(
        changed(REPORT, "selfcheck.mutants_total", 0), "selfcheck.mutants_caught", 0
    )
    assert not run(none_tried).checks["G5"].passed
    assert "no mutants" in run(none_tried).checks["G5"].detail


def test_the_cross_platform_margin_of_the_selfcheck() -> None:
    # tol_p must be at least 2 x the largest difference seen between platforms
    base = changed(REPORT, "selfcheck.max_cross_platform_p", 0.01)
    assert run(changed(base, "selfcheck.tol_p", 0.02)).checks["G5"].passed
    assert not run(changed(base, "selfcheck.tol_p", 0.02 - EPS)).checks["G5"].passed


def test_any_one_advantage_is_enough_but_one_is_needed() -> None:
    nothing = changed(only_latency(REPORT), "latency.consecutive.W1", 0.99)
    assert not run(nothing).checks["G4"].passed
    assert "no measured advantage" in run(nothing).checks["G4"].detail
    # memory alone
    assert run(changed(nothing, "memory.steady_ratio", 0.85)).checks["G4"].passed
    assert (
        not run(changed(nothing, "memory.steady_ratio", 0.85 + EPS)).checks["G4"].passed
    )
    # file size alone
    assert run(changed(nothing, "size_ratio", 0.70)).checks["G4"].passed
    assert not run(changed(nothing, "size_ratio", 0.70 + EPS)).checks["G4"].passed
    # the peak is not the criterion: memory is judged at steady state
    assert not run(changed(nothing, "memory.peak_ratio", 0.10)).checks["G4"].passed


def test_latency_counts_only_in_both_regimes_on_both_workloads() -> None:
    base = only_latency(REPORT)
    for path in (
        "latency.consecutive.W1",
        "latency.consecutive.W3",
        "latency.spaced.W1",
        "latency.spaced.W3",
    ):
        assert not run(changed(base, path, 0.9)).checks["G4"].passed, path
    # W2 is reported but is not a criterion
    assert run(changed(base, "latency.consecutive.W2", 0.99)).checks["G4"].passed


def test_a_latency_bench_with_a_bad_aa_control_is_discarded() -> None:
    result = run(changed(only_latency(REPORT), "latency.aa_median_ratio", 1.16))
    assert not result.checks["G4"].passed
    assert "A/A" in result.checks["G4"].detail


def required_figures() -> list[tuple[str, str]]:
    return [
        ("G1", "quality.overall.ece"),
        ("G1", "quality.by_task"),
        ("G1", "baseline.quality.by_task.intent.ece"),
        ("G2", "paired.overall.accuracy_diff.estimate"),
        ("G2", "paired.overall.accuracy_diff.ci95"),
        ("G2", "paired.by_task.safety.accuracy_diff.ci95"),
        ("G3", "paired.overall.brier_diff.estimate"),
        ("G3", "paired.overall.brier_diff.ci95"),
        ("G5", "selfcheck.mutants_total"),
        ("G5", "selfcheck.platforms"),
        ("G5", "selfcheck.tol_p"),
        ("G5", "selfcheck.max_cross_platform_p"),
    ]


@pytest.mark.parametrize(("criterion", "path"), required_figures())
def test_a_missing_figure_fails_its_criterion_and_only_that_one(
    criterion: str, path: str
) -> None:
    result = run(changed(REPORT, path, DELETE))
    assert not result.checks[criterion].passed
    assert "missing from the report" in result.checks[criterion].detail
    assert [n for n, c in result.checks.items() if not c.passed] == [criterion]


def test_the_baseline_ece_is_a_required_figure_too() -> None:
    result = run(changed(REPORT, "baseline.quality.overall.ece", DELETE))
    assert not result.checks["G1"].passed


def test_nothing_measured_is_not_admitted() -> None:
    nothing = changed(
        changed(changed(REPORT, "latency", DELETE), "memory", DELETE),
        "size_ratio",
        DELETE,
    )
    assert not run(nothing).checks["G4"].passed
    assert not run(nothing).passed


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "something-else"}])
def test_a_report_of_the_wrong_kind_is_a_value_error(bad: Any) -> None:
    with pytest.raises(ValueError, match="not a quantization report"):
        gate.evaluate_gate(bad, "q4_k_m")


def test_a_variant_that_is_not_in_the_report_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="no variant 'q6_k'"):
        gate.evaluate_gate(REPORT, "q6_k")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.01", True, None])
def test_a_figure_that_is_not_a_finite_number_counts_as_missing(bad: Any) -> None:
    assert (
        not run(changed(REPORT, "paired.overall.brier_diff.estimate", bad))
        .checks["G3"]
        .passed
    )
    assert not run(changed(REPORT, "quality.overall.ece", bad)).checks["G1"].passed


@pytest.mark.parametrize("ci", [[0.1], [0.2, 0.1], "wide", None, [0.0, float("nan")]])
def test_a_malformed_interval_counts_as_missing(ci: Any) -> None:
    assert (
        not run(changed(REPORT, "paired.overall.accuracy_diff.ci95", ci))
        .checks["G2"]
        .passed
    )
