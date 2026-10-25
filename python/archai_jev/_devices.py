"""The devices a model can run on: what this build contains and what this machine has."""

from __future__ import annotations

from dataclasses import dataclass

from . import _core

__all__ = ["DeviceInfo", "available_devices", "compiled_backends"]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One device a model can run on. Immutable, hashable, picklable.

    Attributes:
        name: The name ``device=`` takes for it: ``"cpu"``, ``"cuda:0"``, ``"vulkan:1"``. The
            index counts the devices of that backend.
        kind: ``"cpu"``, ``"gpu"`` (own memory) or ``"igpu"`` (integrated, shares the memory of
            the processor; ``device="auto"`` does not choose an integrated Vulkan GPU).
        backend: ``"cpu"``, ``"cuda"``, ``"vulkan"`` or ``"metal"``.
        description: The name the driver gives it.
        memory_total_mib: Total memory in MiB, ``None`` if unknown.
        memory_free_mib: Free memory in MiB at the time of the call, ``None`` if unknown.
    """

    name: str
    kind: str
    backend: str
    description: str
    memory_total_mib: int | None
    memory_free_mib: int | None


def available_devices() -> tuple[DeviceInfo, ...]:
    """The devices this process can run a model on, the processor first.

    It only lists: no model is loaded. A device appearing here does not mean a given model may
    run on it: that needs a verified tolerance for the model on that backend, which
    ``from_pretrained(device=...)`` checks. ``import archai_jev`` never calls this, because
    listing starts the GPU drivers.
    """
    return tuple(DeviceInfo(**d) for d in _core.available_devices())


def compiled_backends() -> tuple[str, ...]:
    """The backends this build was compiled with, the processor first.

    The wheels on PyPI are CPU-only: ``("cpu",)``. A build from source with a Cargo feature
    (``cuda``, ``vulkan``, ``metal``) adds that backend. A backend can be compiled in and have no
    device on the machine (no driver): see :func:`available_devices`.
    """
    return tuple(_core.compiled_backends())
