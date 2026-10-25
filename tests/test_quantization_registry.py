"""The registry and the archived quantization report must agree (spec 009, AC-7 and AC-14).

Today no quantized variant other than the baseline is in the registry, so the real check is
vacuously green: that is a valid outcome of the spec. The rules are proved on synthetic entries.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from _quantization_data import (
    REPORT,
    candidate_manifest,
    changed,
    good_report,
    load_gate,
    registry_manifests,
)

gate = load_gate()


def lint(manifest: dict[str, Any], report: dict[str, Any] | None) -> list[str]:
    result: list[str] = gate.lint_registry([manifest], report)
    return result


def test_the_real_registry_agrees_with_the_archived_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else None
    assert gate.lint_registry(registry_manifests(), report) == []


def test_the_archived_report_keeps_the_q4_k_m_out_for_one_reason() -> None:
    # D60: the Q4_K_M of the v2 passed the quality and advantage criteria (G1-G4) and was refused by the
    # self-check (G5): the engine's numbers differ between platforms by up to 0.137 in probability, and a
    # check that tolerant (at least 0.27) would notice almost nothing. Its GGUF is not in the registry.
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    result = gate.evaluate_gate(report, "q4_k_m")
    assert not result.passed
    assert [name for name, check in result.checks.items() if not check.passed] == ["G5"]
    assert "platform" in result.checks["G5"].detail
    for m in registry_manifests():
        assert all(v["dtype"] != "q4_k_m" for v in m["variants"])


def test_zero_candidates_is_green() -> None:
    manifests = registry_manifests()
    # the baseline q8_0 of the default and the full-precision variants of Kev are not candidates
    assert all(gate.candidates(m) == [] for m in manifests)
    assert gate.lint_registry(manifests, None) == []


def test_a_variant_with_everything_in_order_is_clean() -> None:
    assert lint(candidate_manifest(), good_report()) == []


def test_a_candidate_without_a_report_is_refused() -> None:
    problems = lint(candidate_manifest(), None)
    assert any("no archived report" in p for p in problems), problems
    other = good_report("q6_k")  # a report that is about another variant
    assert any("no such variant" in p for p in lint(candidate_manifest(), other))


def test_a_report_about_another_file_is_refused() -> None:
    problems = lint(candidate_manifest(sha="c" * 64), good_report())
    assert any("another file" in p for p in problems), problems


def test_a_variant_that_fails_the_gate_is_refused_with_the_reason() -> None:
    bad = changed(good_report(), "paired.overall.accuracy_diff.estimate", -0.05)
    problems = lint(candidate_manifest(), bad)
    assert any("gate G2" in p for p in problems), problems


def test_the_declared_temperature_is_the_one_the_report_measured() -> None:
    report = good_report()  # measured T = 2.1
    assert lint(candidate_manifest(temperature=2.1), report) == []
    assert (
        lint(candidate_manifest(temperature=2.104), report) == []
    )  # same to 2 decimals
    problems = lint(
        candidate_manifest(temperature=2.09), report
    )  # the baseline's T, inherited
    assert any(
        "declares T = 2.09 but the report measured 2.1" in p for p in problems
    ), problems
    assert any(
        "declares T" in p for p in lint(candidate_manifest(temperature=1.0), report)
    )


def test_a_quantized_candidate_needs_a_declared_calibration() -> None:
    problems = lint(candidate_manifest(declared=False), good_report())
    assert any("measured, declared calibration" in p for p in problems), problems


@pytest.mark.parametrize("reference", [None, "full_precision"])
def test_a_quantized_candidate_needs_vectors_of_its_own(reference: str | None) -> None:
    problems = lint(candidate_manifest(reference=reference), good_report())
    assert any('selfcheck.reference = "engine"' in p for p in problems), problems


def test_the_baseline_and_full_precision_variants_are_not_candidates() -> None:
    manifest = candidate_manifest()
    manifest["variants"].append(
        {"dtype": "bf16", "source": {"kind": "gguf", "file": {"sha256": "d" * 64}}}
    )
    assert [v["dtype"] for v in gate.candidates(manifest)] == ["q4_k_m"]
