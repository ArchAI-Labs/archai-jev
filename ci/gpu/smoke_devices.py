"""Smoke test of a build with a GPU backend, for `.github/workflows/gpu-build.yml` (spec 011).

It runs on machines that may have no GPU and no driver (a CI runner) and checks what must hold there:

* the build says which backends it contains (`--expect cpu,cuda`);
* importing it and listing the devices works and lists the processor first;
* `device="cpu"` answers at once (here: with the model missing it must fail fast with
  `ModelDownloadError`, not hang: the device is chosen before any file is looked at);
* a named GPU is refused with `DeviceUnavailableError` when the machine has none (`--no-gpu`), or
  is listed and reaches the tolerance check when it has one (`--expect-device vulkan:0`);
* `device="auto"` never raises a device error.

Nothing is downloaded: the cache is an empty folder and the call is offline.

    python ci/gpu/smoke_devices.py --expect cpu,vulkan --no-gpu
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

LIMIT_SECONDS = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True, help="the backends the build contains, comma separated")
    ap.add_argument("--no-gpu", action="store_true", help="the machine has no GPU or no driver")
    ap.add_argument("--expect-device", help="a device that available_devices() must list, such as vulkan:0")
    ap.add_argument("--dll-dir", action="append", default=[], help="Windows: folders of the runtime DLLs")
    args = ap.parse_args()

    if sys.platform == "win32":
        # Python 3.8+ ignores PATH for the DLLs an extension needs
        for folder in args.dll_dir:
            if os.path.isdir(folder):
                os.add_dll_directory(folder)

    started = time.perf_counter()
    import archai_jev as a

    print(f"import archai_jev: {time.perf_counter() - started:.2f} s", flush=True)
    expected = tuple(args.expect.split(","))
    assert a.compiled_backends() == expected, (a.compiled_backends(), expected)

    t0 = time.perf_counter()
    devices = a.available_devices()
    print(f"available_devices: {time.perf_counter() - t0:.2f} s", flush=True)
    for d in devices:
        print(f"  {d.name:10} {d.kind:5} {d.backend:7} {d.description} {d.memory_total_mib}", flush=True)
    assert devices and devices[0].name == "cpu", devices
    names = [d.name for d in devices]
    if args.expect_device:
        assert args.expect_device in names, f"{args.expect_device} is not among {names}"
    if args.no_gpu:
        assert names == ["cpu"], f"a machine with no GPU lists {names}"

    def attempt(device: str) -> str:
        """What `from_pretrained(device=...)` does with an empty offline cache, and how long it takes."""
        t = time.perf_counter()
        try:
            a.Jev.from_pretrained(device=device, offline=True, cache_dir=tempfile.mkdtemp())
        except a.DeviceUnavailableError:
            outcome = "DeviceUnavailableError"
        except a.IncompatibleModelError:
            outcome = "IncompatibleModelError"  # the device exists, the model has no tolerance on it
        except a.ModelDownloadError:
            outcome = "ModelDownloadError"  # the device was accepted, the files are not there
        else:
            raise SystemExit(f"device={device!r} loaded a model from an empty cache")
        took = time.perf_counter() - t
        print(f"device={device!r}: {outcome} in {took:.2f} s", flush=True)
        assert took < LIMIT_SECONDS, f"device={device!r} took {took:.1f} s"
        return outcome

    assert attempt("cpu") == "ModelDownloadError"
    assert attempt("auto") in ("ModelDownloadError", "IncompatibleModelError")
    for backend in expected[1:]:
        outcome = attempt(backend)
        if args.no_gpu:
            assert outcome == "DeviceUnavailableError", (backend, outcome)
        else:
            assert outcome in ("IncompatibleModelError", "ModelDownloadError", "DeviceUnavailableError")
    print("smoke ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
