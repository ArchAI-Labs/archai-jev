"""Measurement loops shared by ``kev_torch.py`` (Kev on PyTorch) and ``archai_kev.py`` (archai_jev on the same checkpoint).

Both wrappers give the same callable, ``ask(state) -> list[list[float]]`` (the probabilities of the three questions),
so the series below are identical for the two libraries: 3 warm-up calls, ``latency`` consecutive calls, ``paused``
calls with a 0.5 s pause before each (the CPU clock drops after idle), and a closed loop of C = 1, 4, 16 caller
threads (capacity: requests per second). Every state is different: no request repeats.
Memory is read with the same Windows counter on both sides (peak working set of the process).
"""

from __future__ import annotations

import ctypes
import json
import statistics
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

Ask = Callable[[str], list[list[float]]]


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def stats(ms: list[float]) -> dict[str, Any]:
    return {"n": len(ms), "p50": statistics.median(ms), "p95": pct(ms, 95), "min": min(ms), "max": max(ms), "ms": ms}


def process_memory() -> dict[str, float]:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {}
    mib = 1 << 20
    return {"peak_working_set_mib": round(counters.PeakWorkingSetSize / mib, 1),
            "working_set_mib": round(counters.WorkingSetSize / mib, 1), "private_mib": round(counters.PrivateUsage / mib, 1),
            "peak_commit_mib": round(counters.PeakPagefileUsage / mib, 1)}


def keep_awake() -> None:
    if ctypes.windll.kernel32.SetThreadExecutionState(0x80000001) == 0:
        raise SystemExit("SetThreadExecutionState failed: the machine could sleep during the run")


def series(ask: Ask, states: list[str], pause: float = 0.0) -> dict[str, Any]:
    ms = []
    for s in states:
        if pause:
            time.sleep(pause)
        t = time.perf_counter()
        ask(s)
        ms.append((time.perf_counter() - t) * 1000)
    return stats(ms)


def closed_loop(ask: Ask, states: list[str], callers: int, per_thread_setup: Callable[[], None] | None = None) -> dict[str, Any]:
    """``callers`` threads take states from one shared list until it is empty; throughput = requests / wall seconds."""
    lock = threading.Lock()
    todo = list(states)
    lat: list[float] = []
    errors: list[str] = []
    barrier = threading.Barrier(callers + 1)

    def worker() -> None:
        if per_thread_setup:
            per_thread_setup()
        barrier.wait()
        while True:
            with lock:
                if not todo:
                    return
                s = todo.pop()
            t = time.perf_counter()
            try:
                ask(s)
            except Exception as e:  # noqa: BLE001 - recorded
                errors.append(f"{type(e).__name__}: {e}")
                return
            d = (time.perf_counter() - t) * 1000
            with lock:
                lat.append(d)

    threads = [threading.Thread(target=worker) for _ in range(callers)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    out = {"callers": callers, "requests": len(lat), "wall_s": round(wall, 3), "requests_per_s": round(len(lat) / wall, 3) if wall else 0.0, "errors": errors}
    if lat:
        out["latency"] = stats(lat)
    return out


def write(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)


def load_inputs(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_sections(ask: Ask, inputs: dict[str, Any], result: dict[str, Any], per_thread_setup: Callable[[int], Callable[[], None] | None] | None = None) -> None:
    st = inputs["states"]
    for s in st["warm"]:
        ask(s)
    parity = [ask(s) for s in st["parity"]]
    result["parity_probs"] = parity
    result["latency_consecutive"] = series(ask, st["latency"])
    print(f"  consecutive p50 {result['latency_consecutive']['p50']:.0f} ms p95 {result['latency_consecutive']['p95']:.0f}", flush=True)
    result["latency_paused05"] = series(ask, st["paused"], pause=0.5)
    print(f"  paused05    p50 {result['latency_paused05']['p50']:.0f} ms p95 {result['latency_paused05']['p95']:.0f}", flush=True)
    for callers, key in ((1, "c1"), (4, "c4"), (16, "c16")):
        setup = per_thread_setup(callers) if per_thread_setup else None
        result[f"capacity_{key}"] = closed_loop(ask, st[key], callers, setup)
        print(f"  C{callers}: {result[f'capacity_{key}']['requests_per_s']} req/s", flush=True)
    result["memory"] = process_memory()
    print("  memory", result["memory"], file=sys.stderr, flush=True)
