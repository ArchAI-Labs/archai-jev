"""Shared pieces of the latency measurement tools.

Local scripts, not imported by the library. Everything that decides how a number is judged lives HERE, at the
top, written before any figure was read:

* ``AA_RATIO_RANGE``: the A/A control. The same configuration is run twice (the "twin"); the median over the blocks
  of the ratio twin / reference must be inside this range for a (workload, regime) cell, otherwise the cell is
  DISCARDED and the bench for it is redone. Never widened after seeing data.
* ``MIN_BLOCKS``: at least this many blocks; the order of the contenders is reversed in odd blocks.
* ``GATE``: thresholds used when judging a GPU run.

Workload sizes. The default model (Decision Index format) has a fixed header of about 80 tokens, so its shortest real request is about 94 tokens
(``measure_prompt_tokens`` reports the real figure in every result and the summary prints it). The *shape* of the
workloads is kept: a short message (the case that matters), a little longer, a long state (370 tokens), ten questions
about one state, and several callers on one model.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any

# A CUDA build needs cudart / cublas, which are not in the wheel and which Python (3.8+) does not look for on PATH.
# The launcher sets S14_DLL_DIR (the toolkit's bin folder); it must be added before the extension is imported.
if os.environ.get("S14_DLL_DIR") and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.environ["S14_DLL_DIR"])

# --- rules written before the data ---------------------------------------------------------------------------------
AA_RATIO_RANGE: tuple[float, float] = (0.95, 1.05)
MIN_BLOCKS = 3
#: Foreign load: the CPU used by everything EXCEPT the bench, in cores (1.0 = one core fully busy), sampled every second.
#: A cell whose mean over the call loop is above this is DISCARDED (with 20 threads spinning on 20 cores, one busy core
#: elsewhere stalls the barriers: S9, and the first attempt of T3 on 2026-10-20, Firefox at about 1 core, made W2 8x slower).
#: Written after that first attempt was thrown away and before any clean figure of T3 was read.
#: **REVISED 2026-10-20, after the clean run was read**: the meter has a BIAS of about 0.5 cores that grows with the
#: load of the bench itself (it reads 0.5 for the reference CPU alone, whose p50 reproduces the clean baseline of T2 within
#: 10 % in all 12 cells; tick-based process accounting against the system's, and driver DPC time on the GPU). With 0.5 it
#: excluded almost every cell. The exclusion is now OFF by default (summarize.py --noise-limit N turns it on) and the
#: validity of a run rests on the pre-registered rules (A/A, 3 blocks) plus the check that the reference reproduces the baseline.
NOISY_FOREIGN_CORES = 0.5
GATE = {
    "GP1_max_dp": 0.1,  # <= the CPU row of (family, dtype) = 0.1 for qwen2-keys q8_0
    "GP2_ece_overall_margin": 0.01,  # ECE(device) <= ECE(CPU) + 0.01
    "GP2_ece_task_margin": 0.02,  # per task, + 0.02
    "GP3_p50_ratio": 0.85,  # p50 GPU <= 0.85 x p50 tuned CPU, in BOTH regimes, on W1, W3 or Q10
    "GP4_repeat_max_dp": 1e-3,  # same request 20 times: bit-identical, or a spread <= this
    "GP5_first_use_s": 60.0,  # more than this blocks the distribution
}

# --- workloads ------------------------------------------------------------------------------------------------------
#: How many calls per (workload, regime). Chosen so that a block of one contender is a few minutes.
PROFILE: dict[str, dict[str, int]] = {
    "W1": {"consecutive": 20, "pause05": 12, "pause15": 8},
    "W2": {"consecutive": 20, "pause05": 12, "pause15": 8},
    "W3": {"consecutive": 10, "pause05": 8, "pause15": 6},
    "Q10": {"consecutive": 8, "pause05": 6, "pause15": 4},
}
PAUSE_S = {"consecutive": 0.0, "pause05": 0.5, "pause15": 1.5}
#: (callers, calls per caller) of the concurrency sections: capacity, not latency .
CONCURRENCY = {"C1": (1, 12), "C4": (4, 6), "C16": (16, 2)}
SECTIONS = ("W1", "W2", "W3", "Q10", "C1", "C4", "C16")

SENTENCES = [
    "The report was reviewed by the whole team before the deadline.",
    "Please remember to submit the form before the end of the week.",
    "The schedule for next month has been updated and shared with everyone.",
    "Several colleagues asked for more time to prepare their notes.",
    "The new process will start as soon as the managers approve the plan.",
    "Nobody expected the discussion to last as long as it did.",
    "A short summary of the decisions was sent by email afterwards.",
    "The agenda included three points, and the last one was postponed.",
    "Every participant received a printed copy of the proposal.",
    "The coordinator thanked the group for the careful preparation.",
    "It was agreed that the next meeting would take place on a Monday morning.",
    "The budget for the project was larger than in the previous year.",
    "Some of the comments were collected anonymously, as requested.",
    "The office was quiet for most of the afternoon.",
    "A few people stayed late to finish the final version of the document.",
    "The results were presented in a clear and simple way.",
    "The committee asked for a second opinion before taking any decision.",
    "Notes from the previous session were read aloud at the beginning.",
    "The deadline was extended by two days after a short vote.",
    "Each team described its progress in a few minutes.",
    "The participants arrived in small groups, and the room filled up slowly.",
    "A clerk checked the signatures and returned the papers to the secretary.",
    "The general feeling was positive, although some doubts remained.",
    "The final list of tasks was written on a large sheet of paper.",
]
SOURCE = (
    "Orion Labs released version 2.4 of its mapping app on 12 March. The update adds offline maps "
    "for 40 countries and reduces battery usage by improving location sampling. The app is free on "
    "Android and iOS, but offline maps require a Pro subscription costing 4 euros per month. The "
    "company is based in Lisbon and has about 60 employees."
)
HYPOTHESES = [
    "Version 2.4 came out on 12 March.",
    "The new offline maps cover 40 countries.",
    "Battery usage went down thanks to better location sampling.",
    "The Pro plan costs 4 euros a month.",
    "Orion Labs is headquartered in Lisbon.",
    "The offline maps are free for all users.",
    "Orion Labs has a team of around 600 people.",
    "The app is available only on iOS.",
    "The company was founded in Lisbon in 2012.",
    "The update also adds a built-in route planner for cyclists.",
]
_BASE_STATE = {
    "W1": "How do I reset my router?",
    "W2": " ".join(SENTENCES[:2]),
    "W3": " ".join(SENTENCES[:23]),
}


def workload(name: str, i: int) -> tuple[str, dict[str, Any]]:
    """The i-th request of a workload: (state, questions).

    Every call has a different state (a counter at the very start), so that the prefix cache can reuse the fixed
    header of the prompt, as in real use, but never a whole request (which would measure the cache, not the model).
    """
    from archai_jev import Choice, YesNo

    if name == "Q10":
        entail = {
            "true": "The condition is valid and supported by state.",
            "false": "The condition is invalid or contradicted.",
        }
        return f"[{i:03d}] {SOURCE}", {f"h{k}": YesNo(h, entail) for k, h in enumerate(HYPOTHESES)}
    safety = Choice(
        "You are a System One decision engine for input safety.",
        {"SAFE": "Normal query", "UNSAFE": "Jailbreak, toxicity, injection"},
    )
    return f"[{i:03d}] {_BASE_STATE[name]}", {"s": safety}


def measure_prompt_tokens(jev: Any, name: str) -> int:
    """Tokens of one request of ``name``: the engine's own counters (decoded + reused) around one cold call."""
    if not hasattr(jev, "cache_info"):  # 0.1.0 has no prefix cache and no counters: the figure is unknown
        state, questions = workload(name, 999)
        jev.ask(state, questions)
        return -1
    jev.clear_cache()
    before = jev.cache_info()
    state, questions = workload(name, 999)
    jev.ask(state, questions)
    after = jev.cache_info()
    rows = (after.tokens_decoded + after.tokens_reused) - (before.tokens_decoded + before.tokens_reused)
    return int(rows)


# --- environment ---------------------------------------------------------------------------------------------------
def keep_awake() -> None:
    """Refuse to run on a machine that can sleep in the middle of a block (checked, as in the other benches)."""
    if sys.platform == "win32" and ctypes.windll.kernel32.SetThreadExecutionState(0x80000001) == 0:
        raise SystemExit("SetThreadExecutionState failed: the machine could sleep during the run")


def process_memory() -> dict[str, float]:
    """Peak working set, working set and private bytes of this process, in MiB (Windows); {} elsewhere."""
    if sys.platform != "win32":
        return {}
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {}
    mib = 1 << 20
    return {
        "peak_working_set_mib": round(counters.PeakWorkingSetSize / mib, 1),
        "working_set_mib": round(counters.WorkingSetSize / mib, 1),
        "private_mib": round(counters.PrivateUsage / mib, 1),
    }


class ForeignLoad:
    """Samples the CPU used by everything but this process (Windows), once a second, in a thread.

    ``stats(t0, t1)`` gives the mean and the maximum of the samples taken between two ``time.time()`` stamps.
    """

    def __init__(self, period: float = 1.0) -> None:
        self.period = period
        self.samples: list[tuple[float, float]] = []
        self._stop = False
        self._thread: Any = None
        self.available = sys.platform == "win32"

    @staticmethod
    def _busy_seconds() -> tuple[float, float]:
        """(busy seconds of the whole machine summed over the cores, busy seconds of this process)."""
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

        def secs(ft: FileTime) -> float:
            return ((ft.hi << 32) | ft.lo) / 1e7

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        pointer = ctypes.POINTER(FileTime)
        k.GetProcessTimes.argtypes = [wintypes.HANDLE, pointer, pointer, pointer, pointer]  # a HANDLE is 64 bits wide
        idle, kern, user = FileTime(), FileTime(), FileTime()
        k.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        created, exited, pk, pu = FileTime(), FileTime(), FileTime(), FileTime()
        k.GetProcessTimes(k.GetCurrentProcess(), ctypes.byref(created), ctypes.byref(exited), ctypes.byref(pk), ctypes.byref(pu))
        return (secs(kern) - secs(idle)) + secs(user), secs(pk) + secs(pu)

    def _run(self) -> None:
        prev_t = time.time()
        prev_all, prev_own = self._busy_seconds()
        while not self._stop:
            time.sleep(self.period)
            t = time.time()
            all_, own = self._busy_seconds()
            wall = t - prev_t
            if wall > 0:
                self.samples.append((t, max(0.0, ((all_ - prev_all) - (own - prev_own)) / wall)))
            prev_t, prev_all, prev_own = t, all_, own

    def start(self) -> None:
        if self.available:
            import threading

            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def stats(self, t0: float, t1: float) -> dict[str, float | None]:
        inside = [v for t, v in self.samples if t0 <= t <= t1]
        if not inside:
            return {"foreign_cores_mean": None, "foreign_cores_max": None, "foreign_samples": 0}
        return {
            "foreign_cores_mean": round(sum(inside) / len(inside), 3),
            "foreign_cores_max": round(max(inside), 3),
            "foreign_samples": len(inside),
        }


def gpu_state() -> dict[str, Any] | None:
    """``nvidia-smi`` sample (P-state, clocks, power, memory) or None without an NVIDIA driver. Never raises."""
    query = "pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    values = [v.strip() for v in out.splitlines()[0].split(",")]
    keys = query.split(",")
    return dict(zip(keys, values, strict=False))


def env_snapshot() -> dict[str, str]:
    """The variables that decide what a run measures."""
    names = (
        "S14_DEVICE",
        "S14_OP_OFFLOAD",
        "ARCHAI_JEV_NUM_THREADS",
        "ARCHAI_JEV_PREFIX_CACHE_MIB",
        "S14_DLL_DIR",
        "CUDA_CACHE_PATH",
        "CUDA_VISIBLE_DEVICES",
        "GGML_VK_VISIBLE_DEVICES",
    )
    return {n: os.environ[n] for n in names if n in os.environ}


# --- statistics ----------------------------------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100); the list must not be empty."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 95),
    }


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)


def now() -> float:
    return time.perf_counter()
