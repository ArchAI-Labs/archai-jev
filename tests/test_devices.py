"""Devices from Python (spec 011): what a build contains and a machine has, the exceptions, and
what ``ModelInfo`` says. A CPU wheel is the case these tests describe; a build with a GPU
backend is checked by the build jobs of CI.

The real model is used only if ``ARCHAI_JEV_TEST_CACHE`` names a cache that holds it.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import pickle
import subprocess
import sys
import warnings
from typing import Any

import pytest

import archai_jev
from archai_jev import (
    DeviceFallbackWarning,
    DeviceInfo,
    DeviceUnavailableError,
    Jev,
    JevError,
    ModelInfo,
    _core,
    available_devices,
    compiled_backends,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_available_devices_on_a_cpu_wheel() -> None:
    devices = available_devices()
    assert isinstance(devices, tuple) and devices
    first = devices[0]
    assert (first.name, first.kind, first.backend) == ("cpu", "cpu", "cpu")
    assert first.description
    names = [d.name for d in devices]
    assert len(set(names)) == len(names)
    for d in devices:
        assert d.kind in ("cpu", "gpu", "igpu")
        assert d.name == "cpu" or d.name.startswith(d.backend + ":")
        assert d.memory_total_mib is None or d.memory_total_mib > 0
    backends = compiled_backends()
    assert backends[0] == "cpu"
    assert set(backends) <= {"cpu", "cuda", "vulkan", "metal"}
    if backends == ("cpu",):  # the wheels on PyPI: the processor and nothing else
        assert [d.name for d in devices] == ["cpu"]


def test_device_info_is_frozen_hashable_picklable() -> None:
    d = DeviceInfo("cuda:0", "gpu", "cuda", "NVIDIA GeForce", 8151, 7000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.name = "cuda:1"  # type: ignore[misc]
    assert {d: 1}[
        DeviceInfo("cuda:0", "gpu", "cuda", "NVIDIA GeForce", 8151, 7000)
    ] == 1
    assert pickle.loads(pickle.dumps(d)) == d
    real = available_devices()[0]
    assert pickle.loads(pickle.dumps(real)) == real
    assert hash(real) == hash(pickle.loads(pickle.dumps(real)))


def test_import_does_not_enumerate_devices() -> None:
    # listing the devices starts the GPU drivers: a fresh interpreter that only imports must not
    code = (
        "import archai_jev\n"
        "from archai_jev import _core\n"
        "print(_core._device_enumerations())\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0", out
    before = _core._device_enumerations()
    available_devices()
    assert _core._device_enumerations() == before + 1
    compiled_backends()
    assert (
        _core._device_enumerations() == before + 1
    ), "compiled_backends() lists nothing"


def test_no_environment_variable_selects_the_device() -> None:
    # nothing silent may change which device answers (as dtype in 009): no ARCHAI_JEV_* names a
    # device, in the Python layer or in the Rust core
    sources = [
        *(ROOT / "python" / "archai_jev").glob("*.py"),
        *(ROOT / "src").rglob("*.rs"),
    ]
    assert len(sources) > 50
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if (
                "ARCHAI_JEV_" in line
                and "DEVICE" in line.upper().split("ARCHAI_JEV_")[1][:12]
            ):
                assert "TEST" in line, f"{path.name}: {line.strip()}"


def test_the_new_names_are_public() -> None:
    for name in (
        "DeviceInfo",
        "DeviceFallbackWarning",
        "DeviceUnavailableError",
        "available_devices",
        "compiled_backends",
    ):
        assert name in archai_jev.__all__
    assert issubclass(DeviceUnavailableError, (JevError, RuntimeError))
    assert issubclass(DeviceFallbackWarning, UserWarning)
    assert DeviceUnavailableError.__module__ == "archai_jev"
    assert DeviceFallbackWarning.__module__ == "archai_jev"
    assert pickle.loads(pickle.dumps(DeviceUnavailableError("x"))).args == ("x",)


def test_model_info_device_defaults_and_pickles() -> None:
    info = ModelInfo(
        name="m",
        revision=None,
        family="f",
        head="h",
        template="t",
        temperature=1.0,
        calibrated=True,
        license=None,
        max_context=None,
        device="cuda:0",
    )
    assert info.device == "cuda:0"
    assert pickle.loads(pickle.dumps(info)) == info
    assert hash(info) == hash(pickle.loads(pickle.dumps(info)))
    assert dataclasses.replace(info, device=None).device is None
    # the test scorer and the list of models do not run on a device
    from archai_jev.testing import MockScorer

    assert Jev.from_scorer(MockScorer()).model_info.device is None
    assert all(m.device is None for m in archai_jev.list_models())


def test_a_fallback_reaches_the_user_as_a_warning_with_their_stack_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_info = {
        "name": "m",
        "revision": None,
        "family": "f",
        "head": "h",
        "template": "t",
        "temperature": 1.0,
        "calibrated": True,
        "license": None,
        "max_context": None,
    }

    def fake_load(*_args: object) -> tuple[object, dict[str, object]]:
        return _core.MockScorer(), dict(real_info, **fake_load.extra)  # type: ignore[attr-defined]

    monkeypatch.setattr(_core, "load_model", fake_load)
    monkeypatch.setattr(Jev, "_create", classmethod(lambda cls, model, info: info))

    fake_load.extra = {"device": "cpu", "fallback": "using the CPU instead of cuda:0 (no device)"}  # type: ignore[attr-defined]
    with pytest.warns(DeviceFallbackWarning, match="instead of cuda:0") as caught:
        info: Any = Jev.from_pretrained(device="auto")
    assert (
        caught[0].filename == __file__
    ), "the warning points at the caller, not at the library"
    assert info.device == "cpu"

    # no fallback, no warning (the CPU wheel, or a GPU that worked)
    fake_load.extra = {"device": "cuda:0", "fallback": None}  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Jev.from_pretrained(device="cuda")


@pytest.mark.skipif(
    not os.environ.get("ARCHAI_JEV_TEST_CACHE"),
    reason="set ARCHAI_JEV_TEST_CACHE to a cache folder that holds the default model",
)
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_the_real_model_reports_its_device(device: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a CPU wheel has nothing to fall back from
        jev = Jev.from_pretrained(
            device=device, cache_dir=os.environ["ARCHAI_JEV_TEST_CACHE"], offline=True
        )
    assert jev.model_info.device == "cpu"
    assert pickle.loads(pickle.dumps(jev.model_info)) == jev.model_info
