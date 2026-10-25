"""archai_jev (the wheel on PYTHONPATH) on the Kev checkpoint, with the inputs of the release bench.

    archai_kev.py --label ajev-bf16 --out block0_ajev-bf16.json [--dtype bf16|f32]

The same three questions as ``kev_torch.py`` (one Choice, one YesNo, one Score) on the same 300-token states. Needs the
converted model in ``ARCHAI_JEV_CACHE`` (the first ``from_pretrained('jaredpalmer/kev-0.8b')`` downloads and converts).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevbench as K  # noqa: E402

T_START = time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default=os.path.join(HERE, "inputs.json"))
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args()
    K.keep_awake()
    inputs = K.load_inputs(a.inputs)
    t_launch = float(os.environ.get("KB_T0", T_START))
    from archai_jev import Choice, Jev, Score, YesNo  # noqa: PLC0415 - timed on purpose

    imported = time.time()
    r = inputs["request"]["questions"]
    questions = {
        "department": Choice(r["department"]["instructions"], r["department"]["criteria"]),
        "escalate": YesNo(r["escalate"]["instructions"]),
        "frustration": Score(r["frustration"]["instructions"], r["frustration"]["criteria"]),
    }
    t = time.perf_counter()
    jev = Jev.from_pretrained("jaredpalmer/kev-0.8b", dtype=a.dtype)
    load_s = time.perf_counter() - t

    def ask(state: str) -> list[list[float]]:
        ans = jev.ask(state, questions)
        d, e, f = ans["department"], ans["escalate"], ans["frustration"]
        return [list(d.probabilities.values()) if hasattr(d.probabilities, "values") else list(d.probabilities),
                [1.0 - e.probability, e.probability], list(f.probabilities.values()) if hasattr(f.probabilities, "values") else list(f.probabilities)]

    t = time.perf_counter()
    ask(inputs["states"]["warm"][0])
    first_call_ms = (time.perf_counter() - t) * 1000
    info = jev.model_info
    result: dict = {
        "label": a.label, "library": "archai_jev", "python": sys.version.split()[0], "platform": platform.platform(),
        "model": {"name": info.name, "dtype": info.dtype, "device": info.device},
        "env": {k: os.environ[k] for k in ("ARCHAI_JEV_NUM_THREADS", "ARCHAI_JEV_PREFIX_CACHE_MIB") if k in os.environ},
        "first_use": {"import_s": round(imported - T_START, 3), "load_s": round(load_s, 3), "first_call_ms": round(first_call_ms, 1),
                      "process_to_first_answer_s": round(time.time() - t_launch, 3)},
    }
    print(f"{a.label}: load {load_s:.2f} s, first call {first_call_ms:.0f} ms", flush=True)
    K.run_sections(ask, inputs, result)
    K.write(a.out, result)
    print(f"{a.label} done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
