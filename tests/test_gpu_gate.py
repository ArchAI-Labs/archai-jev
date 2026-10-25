"""The admission gate of spec 011 (GP1-GP6) as executable rules.

Every threshold has a test that stops just short of it (accepted) and just beyond it (refused), so
moving a number in ``tests/gpu_gate.py`` makes a test fail: a threshold is a decision, not a knob.
All figures here are synthetic (no model, no GPU), except the last tests, which read the archived
report of the measurements.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from _gpu_data import DELETE, REPORT, changed, good_report, load_gate

gate = load_gate()
EPS = 1e-9
GOOD = good_report()
CPU_ECE = GOOD["cpu"]["calibration"]["overall"]["ece"]
CPU_TASK_ECE = GOOD["cpu"]["calibration"]["by_task"]["rte"]["ece"]


def run(report: dict[str, Any]) -> Any:
    return gate.evaluate_gate(report, "cuda:0")


def no_advantage_but(report: dict[str, Any], workload: str) -> dict[str, Any]:
    """A report where only ``workload`` has a latency advantage, so a test can break it alone."""
    for regime in ("consecutive", "pause05", "pause15"):
        for other in ("W1", "W3", "Q10"):
            if other != workload:
                report = changed(
                    report, f"latency.{regime}.{other}.ratio_to_tuned_cpu", 2.0
                )
    return report


def test_a_good_report_is_admitted_on_every_criterion() -> None:
    result = run(GOOD)
    assert result.publishable, result.failures()
    assert list(result.checks) == ["GP1", "GP2", "GP3", "GP4", "GP5", "GP6"]
    assert result.tolerance_rows and result.distribution_ready


def test_the_thresholds_are_the_ones_the_user_accepted() -> None:
    # spec 011 section 5 and D63. Moving one of these is a decision: change it here on purpose.
    assert gate.THRESHOLDS == {
        "GP1": {"max_dp": 0.1},
        "GP2": {"overall_ece_margin": 0.01, "task_ece_margin": 0.02},
        "GP3": {"latency_ratio_max": 0.85},
        "GP4": {"max_spread": 1e-3},
        "GP5": {"first_use_max_s": 60.0},
    }
    assert gate.GP3_WORKLOADS == ("W1", "W3", "Q10")
    assert gate.GP3_REGIMES == ("consecutive", "pause05")
    assert gate.AA_INTERVAL == (0.95, 1.05)


# (criterion, figure, value that just passes, value that just fails, extra changes)
EDGES = [
    ("GP1", "fidelity.golden.max_dp", 0.1, 0.1 + EPS),
    ("GP1", "fidelity.context.max_dp", 0.1, 0.1 + EPS),
    ("GP1", "fidelity.cost.max_dp", 0.1, 0.1 + EPS),
    ("GP2", "calibration.overall.ece", CPU_ECE + 0.01 - EPS, CPU_ECE + 0.01 + EPS),
    (
        "GP2",
        "calibration.by_task.rte.ece",
        CPU_TASK_ECE + 0.02 - EPS,
        CPU_TASK_ECE + 0.02 + EPS,
    ),
    ("GP5", "first_use.process_to_first_answer_s", 60.0, 60.0 + EPS),
    ("GP5", "first_use.process_to_first_answer_without_cache_s", 60.0, 60.0 + EPS),
    ("GP6", "install_safe.passed", 3, 2),
]


@pytest.mark.parametrize(("criterion", "path", "ok", "bad"), EDGES)
def test_each_threshold_accepts_at_the_edge_and_refuses_beyond_it(
    criterion: str, path: str, ok: Any, bad: Any
) -> None:
    accepted = run(changed(GOOD, path, ok))
    assert accepted.checks[criterion].passed, accepted.failures()
    result = run(changed(GOOD, path, bad))
    assert not result.checks[criterion].passed, f"{path}={bad} should fail {criterion}"
    assert not result.publishable
    # a refusal is local: the other criteria still read their own figures
    others = [c for name, c in result.checks.items() if name != criterion]
    assert all(c.passed for c in others), result.failures()


LATENCY_EDGES = [
    ("consecutive", "W1"),
    ("pause05", "W1"),
    ("consecutive", "W3"),
    ("pause05", "W3"),
    ("consecutive", "Q10"),
    ("pause05", "Q10"),
]


@pytest.mark.parametrize(("regime", "workload"), LATENCY_EDGES)
def test_latency_accepts_at_the_edge_and_refuses_beyond_it(
    regime: str, workload: str
) -> None:
    base = no_advantage_but(GOOD, workload)
    path = f"latency.{regime}.{workload}.ratio_to_tuned_cpu"
    assert run(changed(base, path, 0.85)).checks["GP3"].passed
    result = run(changed(base, path, 0.85 + EPS))
    assert not result.checks["GP3"].passed
    assert "no measured advantage" in result.checks["GP3"].detail


@pytest.mark.parametrize(("regime", "workload"), LATENCY_EDGES)
def test_the_aa_control_is_an_interval_on_both_sides(
    regime: str, workload: str
) -> None:
    base = no_advantage_but(GOOD, workload)
    path = f"latency.{regime}.{workload}.aa_median_ratio"
    for ok in (0.95, 1.05):
        assert run(changed(base, path, ok)).checks["GP3"].passed
    for bad in (0.95 - EPS, 1.05 + EPS):
        result = run(changed(base, path, bad))
        assert not result.checks["GP3"].passed
        assert "A/A" in result.checks["GP3"].detail


def test_one_workload_is_enough_but_both_regimes_are_needed() -> None:
    only_w3 = no_advantage_but(GOOD, "W3")
    assert run(only_w3).checks["GP3"].passed
    assert run(only_w3).checks["GP3"].measured == ["W3"]
    # the advantage must hold with the 0.5 s pause too, not only back to back
    slow_pause = changed(only_w3, "latency.pause05.W3.ratio_to_tuned_cpu", 1.2)
    assert not run(slow_pause).checks["GP3"].passed
    # the 1.5 s pause is reported but is not a criterion, and neither is W2
    assert (
        run(changed(only_w3, "latency.pause15.W3.ratio_to_tuned_cpu", 3.0))
        .checks["GP3"]
        .passed
    )
    only_w2 = changed(
        no_advantage_but(GOOD, "W3"), "latency.consecutive.W3.ratio_to_tuned_cpu", 2.0
    )
    assert not run(only_w2).checks["GP3"].passed


def test_a_gpu_slower_than_the_cpu_has_no_measured_advantage() -> None:
    slow = GOOD
    for regime in ("consecutive", "pause05"):
        for workload in ("W1", "W3", "Q10"):
            slow = changed(slow, f"latency.{regime}.{workload}.ratio_to_tuned_cpu", 2.2)
    result = run(slow)
    assert not result.checks["GP3"].passed
    assert result.checks["GP3"].detail.startswith("no measured advantage")
    # the other criteria are not affected: such a backend can still hold tolerance rows
    assert result.tolerance_rows
    assert not result.distribution_ready


def test_repeatability_is_bit_identity_or_a_small_declared_spread() -> None:
    assert run(GOOD).checks["GP4"].detail == "bit-identical in every cell"
    wobble = changed(GOOD, "repeatability.W3_warm.distinct_results", 2)
    spread = changed(wobble, "repeatability.W3_warm.max_spread", 1e-3)
    assert run(spread).checks["GP4"].passed
    assert "not bit-identical" in run(spread).checks["GP4"].detail
    assert (
        not run(changed(spread, "repeatability.W3_warm.max_spread", 1e-3 + EPS))
        .checks["GP4"]
        .passed
    )
    assert (
        not run(changed(GOOD, "repeatability.Q10_cold.runs", 19)).checks["GP4"].passed
    )
    assert run(changed(GOOD, "repeatability.Q10_cold.runs", 20)).checks["GP4"].passed


def test_zero_targets_tried_is_refused_even_though_none_failed() -> None:
    none = changed(changed(GOOD, "install_safe.targets", 0), "install_safe.passed", 0)
    assert not run(none).checks["GP6"].passed
    assert "no target" in run(none).checks["GP6"].detail


def required_figures() -> list[tuple[str, str]]:
    return [
        ("GP1", "fidelity.golden"),
        ("GP1", "fidelity.context.max_dp"),
        ("GP1", "fidelity.cost.max_dp"),
        ("GP2", "calibration.overall.ece"),
        ("GP2", "calibration.by_task"),
        ("GP2", "cpu.calibration.overall.ece"),
        ("GP2", "cpu.calibration.by_task.stsb.ece"),
        ("GP3", "latency"),
        ("GP4", "repeatability.W1_cold.max_spread"),
        ("GP4", "repeatability.Q10_warm"),
        ("GP5", "first_use.process_to_first_answer_s"),
        ("GP5", "first_use.process_to_first_answer_without_cache_s"),
        ("GP6", "install_safe"),
        ("GP6", "install_safe.targets"),
    ]


@pytest.mark.parametrize(("criterion", "path"), required_figures())
def test_a_missing_figure_fails_its_criterion_and_only_that_one(
    criterion: str, path: str
) -> None:
    result = run(changed(GOOD, path, DELETE))
    assert not result.checks[criterion].passed
    detail = result.checks[criterion].detail
    assert "missing from the report" in detail or "not measured" in detail
    assert [n for n, c in result.checks.items() if not c.passed] == [criterion]


def test_a_workload_with_no_aa_control_does_not_count() -> None:
    # Q10 only, but its A/A was never measured (the device has no twin process): no advantage
    base = no_advantage_but(GOOD, "Q10")
    for regime in ("consecutive", "pause05"):
        base = changed(base, f"latency.{regime}.Q10.aa_median_ratio", DELETE)
    result = run(base)
    assert not result.checks["GP3"].passed
    assert "not measured" in result.checks["GP3"].detail


def test_a_missing_first_use_in_the_artefact_free_report_blocks_distribution() -> None:
    # GP6 cannot be measured before an artefact exists (T14/T15): it is missing, so nothing is
    # publishable yet, while the tolerance rows and GP3-GP5 are judged on their own
    early = changed(GOOD, "install_safe", DELETE)
    result = run(early)
    assert not result.publishable
    assert result.distribution_ready
    assert result.tolerance_rows


def test_the_tolerance_rows_depend_on_gp1_and_gp2_only() -> None:
    assert not run(changed(GOOD, "fidelity.golden.max_dp", 0.2)).tolerance_rows
    assert not run(changed(GOOD, "calibration.overall.ece", 0.2)).tolerance_rows
    assert run(
        changed(GOOD, "first_use.process_to_first_answer_s", 99.0)
    ).tolerance_rows
    assert not run(
        changed(GOOD, "first_use.process_to_first_answer_s", 99.0)
    ).distribution_ready


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "something-else"}])
def test_a_report_of_the_wrong_kind_is_a_value_error(bad: Any) -> None:
    with pytest.raises(ValueError, match="not a GPU report"):
        gate.evaluate_gate(bad, "cuda:0")


def test_a_backend_that_is_not_in_the_report_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="no backend 'metal'"):
        gate.evaluate_gate(GOOD, "metal")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.05", True, None])
def test_a_figure_that_is_not_a_finite_number_counts_as_missing(bad: Any) -> None:
    assert not run(changed(GOOD, "fidelity.golden.max_dp", bad)).checks["GP1"].passed
    assert not run(changed(GOOD, "calibration.overall.ece", bad)).checks["GP2"].passed
    # W1 stops counting (its figure is missing), the other workloads still do
    counted = run(changed(GOOD, "latency.consecutive.W1.ratio_to_tuned_cpu", bad))
    assert counted.checks["GP3"].measured == ["W3", "Q10"]


# ---- the archived report ------------------------------------------------------------------------


def archived() -> dict[str, Any]:
    report: dict[str, Any] = json.loads(REPORT.read_text(encoding="utf-8"))
    return report


def test_the_archived_report_has_the_right_schema_and_an_environment() -> None:
    report = archived()
    assert report["schema"] == gate.SCHEMA
    env = report["environment"]
    for key in ("gpu", "driver", "cuda_toolkit", "power", "cpu"):
        assert key in env, key
    assert set(report["backends"]) >= {"cuda:0", "vulkan:1", "vulkan:0"}


def test_the_archived_report_holds_no_path_user_name_or_prompt_text() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for needle in ("C:\\\\", "C:/", "/Users/", "nicola", "procopio", "How do I"):
        assert needle not in text, needle


def test_the_archived_verdicts() -> None:
    # What the gate says about the numbers measured on the development machine (spec 011, T3/T4).
    # Moving one of these is a new measurement or a new decision, not a fix.
    report = archived()
    cuda = gate.evaluate_gate(report, "cuda:0")
    vulkan = gate.evaluate_gate(report, "vulkan:1")
    intel = gate.evaluate_gate(report, "vulkan:0")
    assert cuda.tolerance_rows and cuda.distribution_ready, cuda.failures()
    assert vulkan.tolerance_rows and vulkan.distribution_ready, vulkan.failures()
    assert not intel.checks["GP3"].passed
    assert intel.checks["GP3"].detail.startswith("no measured advantage")
    # GP6 is measured on the artefacts (T14/T15): until then nothing is publishable
    for result in (cuda, vulkan, intel):
        assert not result.checks["GP6"].passed
        assert not result.publishable
