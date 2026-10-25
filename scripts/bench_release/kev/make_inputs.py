"""Write the inputs that Kev (PyTorch) and archai_jev both read in the Kev comparison.

Run with the Kev environment and KEV_SRC set (it needs Kev's tokenizer to build states of EXACTLY 300 tokens):

    python scripts/bench_release/kev/make_inputs.py scripts/bench_release/kev/inputs.json

The workload: a 300-token state (with the start token) plus three questions, one of each type, 402 tokens in all.
Different states for every request, so no request is ever repeated.
"""

from __future__ import annotations

import json
import os
import sys

import kev_common as C  # noqa: E402  (make_state needs Kev's tokenizer)

COUNTS = {"warm": 3, "latency": 50, "paused": 20, "c1": 12, "c4": 24, "c16": 32, "parity": 5}


def main() -> None:
    out = sys.argv[1]
    from kev.checkpoint import Checkpoint, LoadOptions

    ck = Checkpoint(C.RUN)
    tok, _model = ck.load("cpu", LoadOptions())  # the same loader as the bench; the model is not used here
    data: dict[str, list[str]] = {}
    for k, (name, n) in enumerate(COUNTS.items()):
        data[name] = C.states(n, tok, offset=70_000 + 1000 * k)
    request = C.make_request("STATE")
    json.dump({"counts": COUNTS, "states": data, "request": request, "state_tokens": C.STATE_TOKENS}, open(out, "w", encoding="utf-8"), indent=1)
    print("written", out, {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
