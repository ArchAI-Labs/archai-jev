"""Kev (PyTorch, fp32, CPU) on the inputs of the release bench. Run with a Python environment that has Kev's dependencies and KEV_SRC set.

    kev_torch.py --label kev-default --out block0_kev-default.json [--intra1 8 --intra4 5 --intra16 1] [--dtype bf16]

The library path (validate -> record -> encode -> probs), the way a user calls Kev from Python. Torch threads: the
torch default (whatever it picks) unless the launcher sets ``OMP_NUM_THREADS``; ``--intraN`` re-sets the intra-op threads
inside each of the N caller threads (torch keeps that setting per thread) for the capacity series.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kev_common as C  # noqa: E402  (loads Kev as a user gets it)
import kevbench as K  # noqa: E402

T_START = time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default=os.path.join(HERE, "inputs.json"))
    ap.add_argument("--dtype", default=None, choices=[None, "bf16", "fp32"])
    ap.add_argument("--intra1", type=int)
    ap.add_argument("--intra4", type=int)
    ap.add_argument("--intra16", type=int)
    a = ap.parse_args()
    K.keep_awake()
    inputs = K.load_inputs(a.inputs)
    t_launch = float(os.environ.get("KB_T0", T_START))
    ctx, tm = C.start(dtype=a.dtype)
    torch, tok, model = ctx["torch"], ctx["tok"], ctx["model"]
    SystemOneRequest, to_record, to_answers = C.kev_api()
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    def ask(state: str) -> list[list[float]]:
        req = dict(inputs["request"], state=state)
        rec, _meta = to_record(SystemOneRequest.model_validate(req))
        enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        return [p.tolist() for p in model.probs(enc)]

    t = time.perf_counter()
    ask(inputs["states"]["warm"][0])
    first_call_ms = (time.perf_counter() - t) * 1000
    result: dict = {
        "label": a.label, "library": "kev-pytorch", "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "dtype": a.dtype or "fp32",
        "first_use": {**tm, "first_call_ms": round(first_call_ms, 1), "process_to_first_answer_s": round(time.time() - t_launch, 3)},
    }
    print(f"{a.label}: startup {tm['startup_total_s']} s (import {tm['import_total_s']}, load {tm['load_model_s']}), threads {torch.get_num_threads()}", flush=True)
    intra = {1: a.intra1, 4: a.intra4, 16: a.intra16}

    def setup_for(callers: int):
        n = intra[callers]
        return (lambda: torch.set_num_threads(n)) if n else None

    result["intra_threads_per_caller"] = {k: v for k, v in intra.items() if v}
    K.run_sections(ask, inputs, result, setup_for)
    K.write(a.out, result)
    print(f"{a.label} done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
