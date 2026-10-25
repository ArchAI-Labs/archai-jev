"""Tables of the 0.2.0 release bench, from the raw ``block<N>_<label>.json`` files.

    python summarize.py default RESULTS_DIR --reference r020 --twin r020-aa
    python summarize.py kev     RESULTS_DIR --reference kev-t8

Numbers per contender and cell: the median over the blocks of the block's p50 (and p95), the range [min, max] of the
block p50 (the dispersion between repetitions: the machine has two speed regimes), and the ratio to the reference
computed block by block (same block = same minutes) and then the median. A/A: the twin of the reference must give a
ratio in 0.95-1.05 (``common.AA_RATIO_RANGE``); a cell outside it is flagged and not to be published.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

AA = (0.95, 1.05)


def load(folder: Path) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for f in sorted(folder.glob("block*_*.json")):
        m = re.match(r"block(\d+)_(.+)\.json$", f.name)
        if m:
            out.setdefault(m.group(2), {})[int(m.group(1))] = json.loads(f.read_text(encoding="utf-8"))
    return out


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def cell_stats(blocks: dict[int, Any], get) -> dict[str, Any] | None:
    """get(block result) -> list of ms (or None). Returns per-block p50/p95 and pooled figures."""
    per = {b: get(r) for b, r in blocks.items()}
    per = {b: v for b, v in per.items() if v}
    if not per:
        return None
    p50 = {b: statistics.median(v) for b, v in per.items()}
    p95 = {b: pct(v, 95) for b, v in per.items()}
    return {"p50": p50, "p95": p95, "n": sum(len(v) for v in per.values())}


def fmt_cell(c: dict[str, Any] | None) -> str:
    if not c:
        return "-"
    p50 = list(c["p50"].values())
    p95 = list(c["p95"].values())
    return f"{statistics.median(p50):.0f} / {statistics.median(p95):.0f} [{min(p50):.0f}-{max(p50):.0f}]"


def ratio(num: dict[int, float], den: dict[int, float]) -> tuple[float, float, float] | None:
    r = [num[b] / den[b] for b in num if b in den and den[b]]
    return (statistics.median(r), min(r), max(r)) if r else None


def ms_of(section: str, regime: str):
    return lambda r: (r.get("sections", {}).get(section, {}).get(regime, {}) or {}).get("ms")


def default_tables(data: dict, ref: str, twin: str | None) -> None:
    labels = list(data)
    print(f"Blocks per contender: { {k: len(v) for k, v in data.items()} }\n")
    for section in ("W1", "W2", "W3", "Q10"):
        for regime in ("consecutive", "pause05", "pause15"):
            cells = {lab: cell_stats(data[lab], ms_of(section, regime)) for lab in labels}
            if not any(cells.values()):
                continue
            toks = {lab: next((r["sections"][section].get("prompt_tokens") for r in data[lab].values() if section in r.get("sections", {})), None) for lab in labels}
            print(f"### {section} {regime}  (tokens: {toks})   p50 / p95 ms [range of block p50]; ratio vs {ref}")
            for lab in labels:
                c = cells[lab]
                rr = ratio(c["p50"], cells[ref]["p50"]) if c and cells.get(ref) and lab != ref else None
                rs = f"  x{rr[0]:.2f} (blocks {rr[1]:.2f}-{rr[2]:.2f})" if rr else ""
                flag = ""
                if twin and lab == twin and rr and not AA[0] <= rr[0] <= AA[1]:
                    flag = "  <-- A/A OUT OF RANGE"
                print(f"- {lab:12s} {fmt_cell(c)}{rs}{flag}")
            print()
    print("### Capacity (requests/s, median over blocks [range])")
    for sec in ("C1", "C4", "C16"):
        row = []
        for lab in labels:
            v = [r["sections"][sec]["requests_per_s"] for r in data[lab].values() if sec in r.get("sections", {})]
            if v:
                row.append(f"{lab}: {statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]")
        print(f"- {sec}: " + "; ".join(row))
    print("\n### First use and memory (median [range] over blocks)")
    for lab in labels:
        fu = [r["first_use"] for r in data[lab].values()]
        mem = [r["memory"].get("peak_working_set_mib") for r in data[lab].values() if r.get("memory")]
        parts = []
        for key in ("import_s", "load_s", "first_call_ms", "process_to_first_answer_s"):
            v = [x[key] for x in fu if key in x]
            parts.append(f"{key} {statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]")
        parts.append(f"peak_ws_mib {statistics.median(mem):.0f} [{min(mem):.0f}-{max(mem):.0f}]" if mem else "")
        print(f"- {lab}: " + "; ".join(parts))


def kev_tables(data: dict, ref: str) -> None:
    labels = list(data)
    print(f"Blocks per contender: { {k: len(v) for k, v in data.items()} }\n")
    for key in ("latency_consecutive", "latency_paused05"):
        cells = {lab: cell_stats(data[lab], lambda r, key=key: (r.get(key) or {}).get("ms")) for lab in labels}
        print(f"### {key}: p50 / p95 ms [range of block p50]; ratio (contender / {ref}; >1 = slower than {ref})")
        for lab in labels:
            c = cells[lab]
            rr = ratio(c["p50"], cells[ref]["p50"]) if c and lab != ref else None
            rs = f"  x{rr[0]:.2f} (blocks {rr[1]:.2f}-{rr[2]:.2f})" if rr else ""
            print(f"- {lab:16s} {fmt_cell(c)}{rs}")
        print()
    print("### Capacity (requests/s, median over blocks [range])")
    for sec in ("capacity_c1", "capacity_c4", "capacity_c16"):
        row = []
        for lab in labels:
            v = [r[sec]["requests_per_s"] for r in data[lab].values() if sec in r]
            if v:
                row.append(f"{lab}: {statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]")
        print(f"- {sec}: " + "; ".join(row))
    print("\n### First use and memory (median [range] over blocks)")
    for lab in labels:
        fu = [r["first_use"] for r in data[lab].values()]
        mem = [r["memory"].get("peak_working_set_mib") for r in data[lab].values() if r.get("memory")]
        keys = sorted({k for x in fu for k in x if k.endswith("_s") or k.endswith("_ms")})
        parts = []
        for key in keys:
            v = [x[key] for x in fu if key in x]
            parts.append(f"{key} {statistics.median(v):.2f}")
        parts.append(f"peak_ws_mib {statistics.median(mem):.0f} [{min(mem):.0f}-{max(mem):.0f}]" if mem else "")
        print(f"- {lab}: " + "; ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("default", "kev"))
    ap.add_argument("folder", type=Path)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--twin")
    a = ap.parse_args()
    data = load(a.folder)
    (default_tables(data, a.reference, a.twin) if a.what == "default" else kev_tables(data, a.reference))


if __name__ == "__main__":
    main()
