"""The pyproject of a GPU wheel differs from the CPU one in two lines (spec 011, T14)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_variant", ROOT / "ci" / "gpu" / "make_variant.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_name_and_the_description_change() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    new = _script().variant_text(text, "vulkan")
    changed = [
        (a, b)
        for a, b in zip(text.splitlines(), new.splitlines(), strict=True)
        if a != b
    ]
    assert len(text.splitlines()) == len(new.splitlines())
    assert [a.split(" = ")[0] for a, _ in changed] == ["name", "description"]
    assert changed[0][1] == 'name = "archai-jev-vulkan"'
    assert changed[1][1].startswith(changed[1][0][:-1])
    assert "Vulkan" in changed[1][1]


def test_no_dependency_or_module_is_touched_by_a_variant() -> None:
    new = _script().variant_text(PYPROJECT.read_text(encoding="utf-8"), "vulkan")
    assert "dependencies = [" not in new.split("[dependency-groups]")[0]
    assert 'module-name = "archai_jev._core"' in new
    assert 'dynamic = ["version"]' in new


def test_a_pyproject_that_is_not_as_expected_is_refused() -> None:
    script = _script()
    with pytest.raises(ValueError, match="unknown variant"):
        script.variant_text(PYPROJECT.read_text(encoding="utf-8"), "cuda")
    with pytest.raises(ValueError, match="exactly one line"):
        script.variant_text('[project]\nname = "other"\n', "vulkan")
    with pytest.raises(ValueError, match="description"):
        script.variant_text('[project]\nname = "archai-jev"\n', "vulkan")
