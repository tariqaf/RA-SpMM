#!/usr/bin/env python3
"""Per-kernel TF32 decision tables from the A/B sweep CSVs.

For each (base kernel, TF32 variant) pair found in the given CSVs, prints:
  kernel | regime | fp16 geomean vs cuSPARSE | tf32 geomean vs cuSPARSE |
  tf32/fp16 | wins/total
plus a degree-bucket table and a proposed rule evaluation.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

PAIRS = [
    ("TC_DIRECT", "TC_DIRECT_TF32"),
    ("COMMUNITY_TC", "COMMUNITY_TC_TF32"),
    ("SEGMENT_HYBRID", "SEGMENT_HYBRID_TF32"),
]


def gm(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--rule", default="deg<5 or (M<=25000 and deg<9)",
                    help="python expression over deg, M, cv, N")
    args = ap.parse_args()

    rows = []
    for path in args.csvs:
        rows.extend(csv.DictReader(open(path)))

    cfg = defaultdict(dict)
    for r in rows:
        cfg[(r["dataset"], r["N"])][r["kernel"]] = r

    for base, tf in PAIRS:
        recs = []
        for (ds, N), kmap in sorted(cfg.items()):
            a, b = kmap.get(base), kmap.get(tf)
            if not a or not b:
                continue
            if a.get("correct") != "True" or b.get("correct") != "True":
                print(f"  !! gate issue {base}/{tf} {ds} N={N}")
                continue
            recs.append({
                "ds": ds, "cat": a["category"], "N": int(N),
                "deg": float(a["avg_nnz_per_row"]), "cv": float(a["cv_d"]),
                "M": int(a["M"]),
                "r": float(a["ms_warm"]) / float(b["ms_warm"]),
                "f16_cus": float(a["ms_cusparse_warm"]) / float(a["ms_warm"]),
                "t32_cus": float(a["ms_cusparse_warm"]) / float(b["ms_warm"]),
            })
        if not recs:
            continue
        print(f"\n===== {base} vs {tf} ({len(recs)} configs) =====")
        print(f"{'regime':20s} {'n':>3s} {'fp16 vs cuSP':>12s} {'tf32 vs cuSP':>12s} {'tf32/fp16':>10s} {'wins':>9s}")
        cats = sorted(set(r["cat"] for r in recs))
        for cat in cats + ["ALL"]:
            sel = recs if cat == "ALL" else [r for r in recs if r["cat"] == cat]
            wins = sum(1 for r in sel if r["r"] > 1.0)
            print(f"{cat:20s} {len(sel):3d} {gm([r['f16_cus'] for r in sel]):12.4f} "
                  f"{gm([r['t32_cus'] for r in sel]):12.4f} {gm([r['r'] for r in sel]):10.4f} "
                  f"{wins:4d}/{len(sel):3d}")
        print("degree buckets:")
        for lo, hi in [(0, 3), (3, 5), (5, 8), (8, 15), (15, 30), (30, 1e12)]:
            sel = [r for r in recs if lo <= r["deg"] < hi]
            if not sel:
                continue
            wins = sum(1 for r in sel if r["r"] > 1.0)
            hs = "inf" if hi > 1e9 else str(int(hi))
            print(f"  deg [{lo:>3},{hs:>4}): n={len(sel):3d} wins={wins:3d} gm={gm([r['r'] for r in sel]):.3f}")
        # rule evaluation
        expr = compile(args.rule, "<rule>", "eval")
        gains, routed, mispicks = [], 0, 0
        for r in recs:
            pick = bool(eval(expr, {}, {"deg": r["deg"], "M": r["M"], "cv": r["cv"], "N": r["N"]}))
            gains.append(r["r"] if pick else 1.0)
            routed += pick
            mispicks += pick and r["r"] < 0.98
        print(f"rule [{args.rule}]: family geomean x{gm(gains):.4f}, "
              f"routed={routed}/{len(recs)}, mispicks(>2% loss)={mispicks}")


if __name__ == "__main__":
    main()
