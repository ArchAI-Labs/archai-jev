"""Latency of the default model in one process.

One model per process, so two contenders never share memory, CPU caches or a GPU context.
``run_blocks.ps1`` starts the contenders in alternating blocks (order reversed in odd blocks) and
``summarize.py`` turns the JSON files into ratios. The device is chosen by the environment of the process
(the CPU wheel ignores ``S14_DEVICE``; it matters only for a GPU build).

    python benches/gpu/bench_latency.py --label cpu-default --out block0_cpu-default.json [--sections W1,W3]

What is measured, per section:

* W1 / W2 / W3   one ``ask`` of a short / medium / long state (see ``common.py`` for the real token counts);
* Q10            one ``ask`` with ten YesNo questions about one state (reuse inside the request);
* C4 / C16       4 and 16 callers on the one model: capacity (requests per second), not latency;

in three regimes: ``consecutive`` (back to back), ``pause05`` and ``pause15`` (a pause of 0.5 / 1.5 s BEFORE each
call: on a laptop the CPU clock drops after idle, and the dGPU has power states).

And once per process: the "first use" figures: seconds from the start of the process to the first answer, split
into import / load / first call (the launcher puts the start time in ``S14_T0``).

Run with a RELEASE build. Needs the default model in the cache (``ARCHAI_JEV_CACHE``, ``ARCHAI_JEV_OFFLINE=1``).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
from typing import Any

import common as c

T_START = time.time()
T_PERF = time.perf_counter()


def timed_ask(jev: Any, name: str, i: int) -> float:
    state, questions = c.workload(name, i)
    started = time.perf_counter()
    jev.ask(state, questions)
    return round((time.perf_counter() - started) * 1000, 2)


FOREIGN = c.ForeignLoad()


def section_latency(jev: Any, name: str, regimes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"prompt_tokens": c.measure_prompt_tokens(jev, name)}
    counter = 0
    for regime in regimes:
        n = c.PROFILE[name][regime]
        gpu_before = c.gpu_state()
        ms: list[float] = []
        t0 = time.time()
        for _ in range(n):
            if c.PAUSE_S[regime]:
                time.sleep(c.PAUSE_S[regime])
            counter += 1
            ms.append(timed_ask(jev, name, counter))
        out[regime] = {"ms": ms, **c.summarize_ms(ms), **FOREIGN.stats(t0, time.time()), "gpu_before": gpu_before, "gpu_after": c.gpu_state()}
        print(f"  {name} {regime:11s} p50 {out[regime]['p50']:8.1f} ms  min {out[regime]['min']:8.1f}  max {out[regime]['max']:8.1f}", flush=True)
    return out


def section_concurrency(jev: Any, callers: int, calls: int) -> dict[str, Any]:
    gpu_before = c.gpu_state()
    latencies: list[list[float]] = [[] for _ in range(callers)]
    errors: list[str] = []
    barrier = threading.Barrier(callers + 1)
    t0 = time.time()

    def worker(k: int) -> None:
        barrier.wait()
        try:
            for j in range(calls):
                latencies[k].append(timed_ask(jev, "W1", 10_000 + k * 100 + j))
        except Exception as e:  # noqa: BLE001 - recorded, the bench goes on
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(callers)]
    for t in threads:
        t.start()
    barrier.wait()
    started = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started
    flat = [x for per in latencies for x in per]
    out = {
        "callers": callers,
        "calls_each": calls,
        "wall_s": round(wall, 3),
        "requests_per_s": round(len(flat) / wall, 3) if wall else 0.0,
        "ms": flat,
        **(c.summarize_ms(flat) if flat else {}),
        "errors": errors,
        **FOREIGN.stats(t0, time.time()),
        "gpu_before": gpu_before,
        "gpu_after": c.gpu_state(),
    }
    print(f"  C{callers:<3d} {len(flat)} calls in {wall:6.2f} s = {out['requests_per_s']:.2f} req/s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default=",".join(c.SECTIONS))
    ap.add_argument("--regimes", default="consecutive,pause05,pause15")
    ap.add_argument("--warmup", type=int, default=3, help="calls before the sections (the first use is recorded apart)")
    a = ap.parse_args()
    c.keep_awake()

    t_launch = float(os.environ.get("S14_T0", T_START))
    from archai_jev import Jev  # noqa: PLC0415 - timed on purpose

    imported = time.time()
    t = time.perf_counter()
    jev = Jev.from_pretrained()
    load_s = time.perf_counter() - t
    state, questions = c.workload("W1", 0)
    t = time.perf_counter()
    jev.ask(state, questions)
    first_call_ms = (time.perf_counter() - t) * 1000
    answered = time.time()
    info = jev.model_info
    result: dict[str, Any] = {
        "label": a.label,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env": c.env_snapshot(),
        "model": {"name": info.name, "dtype": info.dtype, "temperature": info.temperature},
        "first_use": {
            "import_s": round(imported - T_START, 3),
            "load_s": round(load_s, 3),
            "first_call_ms": round(first_call_ms, 1),
            "process_to_first_answer_s": round(answered - t_launch, 3),
        },
        "gpu_idle": c.gpu_state(),
        "sections": {},
    }
    print(f"{a.label}: loaded in {load_s:.2f} s, first call {first_call_ms:.0f} ms, process to answer {answered - t_launch:.2f} s", flush=True)
    FOREIGN.start()
    for i in range(a.warmup):
        timed_ask(jev, "W1", 500 + i)

    regimes = a.regimes.split(",")
    for name in a.sections.split(","):
        if name in c.CONCURRENCY:
            result["sections"][name] = section_concurrency(jev, *c.CONCURRENCY[name])
        else:
            result["sections"][name] = section_latency(jev, name, regimes)
        time.sleep(1.0)  # let the CPU settle between sections
    FOREIGN.stop()
    result["memory"] = c.process_memory()
    result["gpu_end"] = c.gpu_state()
    c.write_json(a.out, result)
    print(f"{a.label} done in {time.perf_counter() - T_PERF:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
