"""009 - Run the model on a GPU.

What you will learn
-------------------
* how to see which devices your build of the library has (``compiled_backends``) and which
  ones your machine has (``available_devices``),
* how to choose where the model runs with ``Jev.from_pretrained(device=...)``,
* how to read where it actually runs (``jev.model_info.device``),
* what happens when a GPU is not available (an error or a warning, never a silent change).

Before you start
----------------
The wheels on PyPI are CPU-ONLY. A GPU is used only by a build you make from the source of
the repository, with the backend of your GPU (``cuda``, ``vulkan`` or ``metal``):

    maturin develop --release --features cuda      # NVIDIA, needs the CUDA Toolkit

``docs/gpu.md`` has the full steps for Windows, Linux and macOS. With a CPU wheel this script
still runs: it tells you that no GPU is compiled in and uses the CPU.

Run it
------
    python demos/009_run_on_a_gpu.py                   # device="auto": the best GPU, else the CPU
    python demos/009_run_on_a_gpu.py --device cuda:0   # insist on this device, or fail
    python demos/009_run_on_a_gpu.py --compare         # also time the CPU, side by side

Nothing here is a speed claim: the times are measured on YOUR machine, live.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
import warnings
from pathlib import Path

# On Windows, Python does not search PATH for DLLs: a CUDA build needs the folder of the CUDA
# runtime added by hand BEFORE ``import archai_jev`` (docs/gpu.md).
if hasattr(os, "add_dll_directory") and os.environ.get("CUDA_PATH"):
    cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
    if cuda_bin.is_dir():
        os.add_dll_directory(str(cuda_bin))

from _common import section, show_download_progress, title

from archai_jev import (
    Choice,
    DeviceFallbackWarning,
    DeviceUnavailableError,
    Jev,
    available_devices,
    compiled_backends,
)

TEAM = Choice(
    "Which team should handle this message?",
    {
        "billing": "payments, invoices and refunds",
        "support": "bugs and how-to questions",
        "sales": "prices and new plans",
    },
)
MESSAGE = "I was charged twice for my subscription this month."


def time_calls(jev: Jev, calls: int) -> float:
    """Median milliseconds of one ``ask``, after one warm-up call (the first is slower)."""
    jev.ask(MESSAGE, {"team": TEAM})
    times = []
    for _ in range(calls):
        started = time.perf_counter()
        jev.ask(MESSAGE, {"team": TEAM})
        times.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the model on a GPU")
    parser.add_argument(
        "--device",
        default="auto",
        help='"cpu", "auto" (default), or "cuda:0" / "vulkan:1" / "metal": read the list below',
    )
    parser.add_argument("--calls", type=int, default=20, help="calls to time (default 20)")
    parser.add_argument("--compare", action="store_true", help="also time the CPU")
    args = parser.parse_args()
    show_download_progress()

    title("1. What this build and this machine have")
    # The backends compiled into THIS build of the library: ("cpu",) in every PyPI wheel.
    print(f"compiled backends : {compiled_backends()}")
    section("devices on this machine (the CPU first; listing starts the GPU drivers)")
    for d in available_devices():
        memory = f"{d.memory_total_mib} MiB" if d.memory_total_mib else "shared"
        print(f"  {d.name:10s} {d.kind:5s} {memory:>9s}  {d.description}")

    title(f'2. Load the model with device="{args.device}"')
    # ``device="cpu"`` (the default of the library) never touches a GPU. ``"auto"`` takes the
    # best GPU that is allowed to answer for this model, else the CPU. ``"cuda:0"`` insists:
    # it either uses that device or raises DeviceUnavailableError.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jev = Jev.from_pretrained(device=args.device)
    except DeviceUnavailableError as exc:
        print(f"\nThe device {args.device!r} cannot be used: {exc}")
        print("Build the library with a GPU backend (docs/gpu.md) or try --device cpu.")
        return 1
    for w in caught:
        if issubclass(w.category, DeviceFallbackWarning):
            # A GPU backend is compiled in but could not be used (driver, memory...): the
            # library went on the CPU and TELLS you instead of doing it silently.
            print(f"warning: {w.message}")
    info = jev.model_info
    print(f"model  : {info.name} ({info.dtype})")
    print(f"runs on: {info.device}")

    title("3. One question, the same answer, on any device")
    answer = jev.ask(MESSAGE, {"team": TEAM}).choices["team"]
    print(f'"{MESSAGE}"')
    print(f"-> {answer.value} (confidence {answer.confidence:.2f})")

    title("4. How long does one call take here?")
    here = time_calls(jev, args.calls)
    print(f"{info.device}: {here:.0f} ms per call (median of {args.calls})")
    if args.compare and info.device != "cpu":
        cpu = time_calls(Jev.from_pretrained(device="cpu"), args.calls)
        print(f"cpu   : {cpu:.0f} ms per call (median of {args.calls})")
        print(f"ratio : {here / cpu:.2f}x of the CPU time (below 1 is faster)")
    print(
        "\nOne machine, one model, one request size: measure your own requests. Some models "
        "may not run on a GPU at all (the library needs a measured tolerance for that "
        "model on that backend, see docs/gpu.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
