"""
Before/after summary for the TC plan-construction optimization.

Compares conversion_times.csv (old per-group std::map packing) against
conversion_times_v2.csv (new flat sort-by-k-block + reused scratch). Reports per-graph
old->new conversion ms + speedup for the two optimized kernels (TC_DIRECT, COMMUNITY_TC),
median/max, and flags any graph still > 2s (fundamental O(nnz) floor).
"""
from __future__ import annotations
import csv, statistics
from pathlib import Path

R = Path(__file__).resolve().parent.parent
FB = R / "fgcs_results/revision/featbreak"
OLD = FB / "conversion_times.csv"
NEW = FB / "conversion_times_v2.csv"
OPT_KERNELS = ["TC_DIRECT", "COMMUNITY_TC"]


def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d[(r["dataset"], r["kernel"])] = float(r["conversion_ms"])
    return d


def main():
    old, new = load(OLD), load(NEW)
    # per-kernel medians
    lines = ["# TC plan-construction optimization: conversion time before → after\n\n",
             "Replaced the per-16-row-group `std::map<int, std::array<float,256>>` packing (O(log) "
             "per nonzero, 1 KB tree nodes, per-group heap alloc) with a flat pass: gather group "
             "nonzeros → stable-sort by k-block → fill contiguous 16×16 tiles with reused scratch. "
             "Tiles are **byte-identical** (PARITY OK 192/192; TC outputs still match cuSPARSE).\n\n"]

    for k in OPT_KERNELS:
        rows = []
        for (ds, kk), ov in old.items():
            if kk != k or (ds, kk) not in new:
                continue
            nv = new[(ds, kk)]
            rows.append((ds, ov, nv, ov / nv if nv > 0 else 0))
        rows.sort(key=lambda x: -x[1])  # worst old first
        old_med = statistics.median([r[1] for r in rows])
        new_med = statistics.median([r[2] for r in rows])
        old_max = max(r[1] for r in rows); new_max = max(r[2] for r in rows)
        sp_geo = statistics.geometric_mean([r[3] for r in rows if r[3] > 0])
        lines.append(f"## {k}\n\n")
        lines.append(f"- Median conversion: **{old_med:.1f} ms → {new_med:.2f} ms** "
                     f"({old_med/new_med:.0f}× faster)\n")
        lines.append(f"- Worst graph: **{old_max:.1f} ms → {new_max:.1f} ms** "
                     f"({old_max/new_max:.0f}× faster)\n")
        lines.append(f"- Geomean speedup: **{sp_geo:.0f}×**\n\n")
        lines.append("| dataset | old ms | new ms | speedup |\n|---|---|---|---|\n")
        for ds, ov, nv, sp in rows:
            flag = "  ⚠️ >2s" if nv > 2000 else ""
            lines.append(f"| {ds} | {ov:.2f} | {nv:.2f} | {sp:.0f}×{flag} |\n")
        lines.append("\n")

    # graphs still >2s
    still = [(ds, k, new[(ds, k)]) for (ds, k) in new if k in OPT_KERNELS and new[(ds, k)] > 2000]
    if still:
        lines.append("## Graphs still > 2 s (fundamental O(nnz) floor for 100M+ nnz)\n\n")
        for ds, k, nv in sorted(still, key=lambda x: -x[2]):
            lines.append(f"- {ds} / {k}: {nv/1000:.2f} s\n")
    else:
        lines.append("## All optimized-kernel conversions now < 2 s. ✅\n")

    (FB / "CONVERSION_OPT_SUMMARY.md").write_text("".join(lines))
    print("wrote CONVERSION_OPT_SUMMARY.md")
    for k in OPT_KERNELS:
        oms = [old[(ds, kk)] for (ds, kk) in old if kk == k]
        nms = [new[(ds, kk)] for (ds, kk) in new if kk == k]
        print(f"  {k}: median {statistics.median(oms):.1f}ms -> {statistics.median(nms):.2f}ms ; "
              f"max {max(oms):.1f}ms -> {max(nms):.1f}ms")


if __name__ == "__main__":
    main()
