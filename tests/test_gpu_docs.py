"""``docs/gpu.md`` tells the truth about the archived measurements.

The tables are generated from ``tests/data/gpu/report.json`` and compared with the file; the text
keeps the one-machine warning and the README makes no claim of speed. To regenerate the tables after
a new report::

    python tests/test_gpu_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import gpu_gate

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "gpu.md"
README = ROOT / "README.md"
REPORT = ROOT / "tests" / "data" / "gpu" / "report.json"
BEGIN, END = "<!-- gpu-table:begin -->", "<!-- gpu-table:end -->"


def _generated() -> str:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return f"{BEGIN}\n{gpu_gate.render_table(report)}{END}"


def _embedded(text: str) -> str:
    start, stop = text.index(BEGIN), text.index(END) + len(END)
    return text[start:stop]


def test_the_tables_are_the_ones_generated_from_the_report() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert _embedded(text) == _generated(), (
        "docs/gpu.md differs from tests/data/gpu/report.json: "
        "run `python tests/test_gpu_docs.py` and read the result"
    )


def test_the_generator_refuses_nothing_it_cannot_show() -> None:
    # every backend of the report appears, with the verdict of the gate on its tolerance rows
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    table = gpu_gate.render_table(report)
    for name in report["backends"]:
        assert f"`{name}`" in table
        verdict = "yes" if gpu_gate.evaluate_gate(report, name).tolerance_rows else "no"
        assert re.search(
            rf"`{re.escape(name)}` \| [\d.]+ \| [^|]+ \| {verdict} \|", table
        )


def test_the_guide_says_the_measures_come_from_one_machine() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "one machine" in text
    assert "not a promise of speed" in text or "Nothing here is a promise" in text


def test_the_guide_names_the_exact_build_variables() -> None:
    text = DOC.read_text(encoding="utf-8")
    for needle in (
        "--features cuda",
        "--features vulkan",
        "--features metal",
        "cuda-no-vmm",
        "CMAKE_CUDA_ARCHITECTURES",
        "CUDA_PATH_V12_9",
        "GGML_METAL=ON",
        "CMAKE_GENERATOR",
        "VULKAN_SDK",
    ):
        assert needle in text, needle


def test_the_readme_makes_no_claim_of_gpu_speed() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/gpu.md" in text
    # no figure of speed or ratio next to the GPU in the README
    for line in text.splitlines():
        if re.search(r"\b(GPU|CUDA|Vulkan|Metal)\b", line):
            assert not re.search(
                r"\d+(\.\d+)?\s*(x|times|ms|s/|per second|%\s*faster)", line, re.I
            ), line


if __name__ == "__main__":
    doc = DOC.read_text(encoding="utf-8")
    DOC.write_text(doc.replace(_embedded(doc), _generated()), encoding="utf-8")
    print("tables regenerated:", DOC)
    sys.exit(0)
