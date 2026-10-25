"""Helpers to load Kev (PyTorch, CPU) and build the benchmark workload.

Kev is loaded as a user gets it: fp32, LoRA merged, eager attention, WITHOUT forcing the thread count (the default is
whatever torch picks). Needs a Python environment with Kev's dependencies (torch, transformers, peft) and KEV_SRC set to a
checkout of Kev. Everything here is deterministic given a seed.
"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Kev's source tree (a checkout of github.com/jaredpalmer/kev at the commit in the README) and, optionally, a Hugging Face
# cache that already holds jaredpalmer/kev-0.8b: set KEV_SRC and HF_HOME before running.
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
KEV_SRC = os.environ["KEV_SRC"]
sys.path.insert(0, KEV_SRC)
EXTRA_SITE = os.path.join(HERE, "extra-site")

RUN = "jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e"  # the checkpoint the comparison uses
STATE_TOKENS = 300  # target length of the state (Ls: includes <|fim_prefix|>)
BASE_SEED = 20260930


def rss():
    """(current working set, peak working set, peak private/commit) in MiB of this process (psutil, Windows counters)."""
    import psutil

    m = psutil.Process().memory_info()
    mib = 1024 * 1024
    return {"wset_mib": round(m.rss / mib, 1), "peak_wset_mib": round(getattr(m, "peak_wset", m.rss) / mib, 1),
            "peak_commit_mib": round(getattr(m, "peak_pagefile", 0) / mib, 1)}


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self):
        now = time.perf_counter()
        d, self.t = now - self.t, now
        return d


def timed_import(name, out):
    t = time.perf_counter()
    mod = __import__(name, fromlist=["_"])
    out[f"import_{name}_s"] = round(time.perf_counter() - t, 3)
    return mod


def start(threads=None, dtype=None, with_server=False):
    """Import torch/transformers/peft/kev and load the checkpoint; -> (ctx, timings). ctx has ck, tok, model, torch.

    `threads` None = torch's default (the out-of-the-box Kev behaviour). `dtype` None = fp32 (Kev's CPU default) or "bf16".
    Timings are wall-clock seconds of each stage."""
    tm = {}
    t_all = time.perf_counter()
    torch = timed_import("torch", tm)
    timed_import("transformers", tm)
    timed_import("peft", tm)
    timed_import("kev.api", tm)
    ckmod = timed_import("kev.checkpoint", tm)  # pulls kev.model (and its transformers submodules)
    if with_server:
        sys.path.insert(0, EXTRA_SITE)
        timed_import("kev.serve", tm)  # fastapi + starlette
    tm["import_total_s"] = round(time.perf_counter() - t_all, 3)
    tm["torch_default_threads"] = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    t = time.perf_counter()
    ck = ckmod.Checkpoint(RUN)
    tm["checkpoint_meta_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    opts = ckmod.LoadOptions(dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]) if dtype else ckmod.LoadOptions()
    tok, model = ck.load("cpu", opts)
    tm["load_model_s"] = round(time.perf_counter() - t, 3)
    tm["startup_total_s"] = round(time.perf_counter() - t_all, 3)
    tm["threads"] = torch.get_num_threads()
    return {"torch": torch, "ck": ck, "tok": tok, "model": model}, tm


# ---------------------------------------------------------------- workload

SUBJECTS = ["The customer", "A returning customer", "The account owner", "Our enterprise client", "The user", "A first-time buyer"]
EVENTS = [
    "ordered {n} pairs of running shoes on {d} and the parcel still has not arrived",
    "was charged twice for invoice {n} and wants one of the payments reversed",
    "reports that the mobile app crashes every time the checkout button is pressed",
    "asked to change the delivery address of order {n} after it had already shipped",
    "received a package with a broken screen and is asking for a replacement",
    "cannot log in since the password reset email never reached the inbox",
    "wants to cancel the annual subscription renewed on {d} because the price went up",
    "was promised a callback within {n} hours and nobody has called so far",
    "says the discount code applied at checkout was ignored on the final invoice",
    "complained about a delay of {n} days and mentioned leaving a public review",
]
TAILS = [
    "The agent replied twice but the answer did not address the original problem.",
    "They attached a screenshot and a photo of the label as evidence.",
    "This is the third contact about the same issue this month.",
    "They explicitly asked to speak with a manager.",
    "The tone of the last message was polite but clearly impatient.",
    "No refund has been issued yet and the order status still shows pending.",
    "They mentioned that they have been a customer for more than five years.",
    "The case was reopened after being closed without a resolution.",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def make_state(seed, tok, target=STATE_TOKENS):
    """Deterministic support-ticket text whose encoded state has exactly `target` tokens (Ls, incl. <|fim_prefix|>).

    Sentences are drawn with random.Random(seed) until the state reaches the target, then trailing words are dropped one
    at a time until it fits; the tokenizer is Kev's (kev.model.user_tokens), so the count is the one the model sees."""
    from kev.model import user_tokens

    rng = random.Random(seed)
    sents = []
    while True:
        ev = rng.choice(EVENTS).format(n=rng.randint(2, 9000), d=f"{rng.randint(1, 28)} {rng.choice(MONTHS)}")
        sents.append(f"{rng.choice(SUBJECTS)} {ev}. " if rng.random() < 0.7 else rng.choice(TAILS) + " ")
        text = "".join(sents).strip()
        if len(user_tokens(tok, text)) + 1 >= target:
            break
    words = text.split(" ")
    while len(user_tokens(tok, " ".join(words))) + 1 > target:
        words.pop()
    text = " ".join(words)
    # a long word (a number, a name) may have cost several tokens: top up with one-token filler words to hit the target exactly
    while len(user_tokens(tok, text)) + 1 < target:
        text += " ok"
    assert len(user_tokens(tok, text)) + 1 == target, "could not reach the exact state length"
    return text


def make_request(state):
    """The fixed three questions (one per type) on a given state, in the TypeSafe JSON shape Kev's API validates."""
    return {
        "state": state,
        "model": "kev-latest",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this case?",
                           "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                        "shipping": "Delivery status, delays, lost packages",
                                        "billing": "Charges, invoices, payment problems",
                                        "account": "Login, passwords, subscriptions"}},
            "escalate": {"type": "noul", "instructions": "Does this case need urgent attention from a human manager?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Slightly annoyed", "Frustrated", "Angry", "Furious"]},
        },
    }


def states(n, tok, offset=0):
    """n distinct deterministic states (seeds BASE_SEED+offset+i)."""
    return [make_state(BASE_SEED + offset + i, tok) for i in range(n)]


def kev_api():
    from kev.api import SystemOneRequest, to_answers, to_record

    return SystemOneRequest, to_record, to_answers


def answer_library(ctx, req_dict):
    """One request through the library path, as a user calls Kev: validate -> record -> encode -> probs -> answers.

    Returns (answers, parts) where parts are the seconds spent in each stage and the token count."""
    SystemOneRequest, to_record, to_answers = kev_api()
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    model, tok = ctx["model"], ctx["tok"]
    t0 = time.perf_counter()
    req = SystemOneRequest.model_validate(req_dict)
    rec, meta = to_record(req)
    t1 = time.perf_counter()
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    t2 = time.perf_counter()
    ps = model.probs(enc)
    t3 = time.perf_counter()
    answers = to_answers([p.tolist() for p in ps], meta)
    t4 = time.perf_counter()
    return answers, {"validate_s": t1 - t0, "encode_s": t2 - t1, "forward_s": t3 - t2, "answers_s": t4 - t3,
                     "total_s": t4 - t0, "tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0)}


def pct(xs, p):
    """Percentile with linear interpolation (numpy's default), no numpy needed."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(xs):
    xs = list(xs)
    return {"n": len(xs), "min": min(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "max": max(xs),
            "mean": sum(xs) / len(xs)}


def emit(obj):
    """Print one JSON result line, prefixed so the orchestrator can pick it out of noisy stderr/stdout."""
    print("RESULT " + json.dumps(obj, ensure_ascii=False), flush=True)
