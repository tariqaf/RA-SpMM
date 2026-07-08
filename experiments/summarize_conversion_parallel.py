"""
Before (v2, serial) / after (v3, OpenMP-parallel) summary for the TC
plan-construction parallelization. Also computes the geomean-vs-80ms target check.

USAGE NOTE (--omp-threads): pass the OMP_NUM_THREADS value that was actually used
when measuring conversion_times_v3.csv — it is RECORDED into the summary (together
with the auto-detected CPU model) so readers know what hardware/configuration
produced the timings. It is deliberately an explicit flag, not auto-detected: the
summary must state the value used for the RUN, not whatever machine later renders
the report. On a different machine, measure with your own physical-core count, e.g.:

    export OMP_NUM_THREADS=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
    OMP_NUM_THREADS=$OMP_NUM_THREADS python experiments/time_conversion_pipeline.py \
        --conv-out .../conversion_times_v3.csv ...
    python experiments/summarize_conversion_parallel.py --omp-threads $OMP_NUM_THREADS

Correctness does NOT depend on the thread count (output is byte-identical for any
OMP_NUM_THREADS — see CHECKSUM_VERIFICATION.log); only the measured speed does.
"""
from __future__ import annotations
import argparse, csv, statistics, subprocess
from pathlib import Path

R = Path(__file__).resolve().parent.parent
FB = R / "fgcs_results/revision/featbreak"
OPT_KERNELS = ["TC_DIRECT", "COMMUNITY_TC"]
CONTROL_KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR", "SEGMENT_HYBRID"]


def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d[(r["dataset"], r["kernel"])] = float(r["conversion_ms"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omp-threads", required=True)
    ap.add_argument("--out", default=str(FB / "CONVERSION_OPT_V2_SUMMARY.md"))
    args = ap.parse_args()

    old = load(FB / "conversion_times_v2.csv")
    new = load(FB / "conversion_times_v3.csv")
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cpu_model = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if "Model name" in l), "?")

    lines = [f"# OpenMP-parallel TC plan construction: v2 (serial) → v3 (parallel)\n\n",
             f"CPU: {cpu_model} | OMP_NUM_THREADS={args.omp_threads} | build: nvcc -Xcompiler -fopenmp, sm_86\n\n",
             "**Parallelized (all deterministic — output byte-identical for any thread count, "
             "verified by FNV-1a checksums over every format array):**\n"
             "1. Per-row analysis (centroid/signature/span): `omp parallel for`, each row writes its own slot.\n"
             "2. Row-reordering stable sorts: fixed-chunk parallel `std::stable_sort` + stable "
             "`std::inplace_merge` — a stable sort's output is uniquely determined by (input, comparator), "
             "so any stable algorithm is byte-identical.\n"
             "3. Per-16-row-group inner sorts: parallel over disjoint groups.\n"
             "4. Group metrics/gating/tile packing: two-phase — parallel per-group buffers, then serial "
             "in-group-order concatenation (even double-precision diagnostic sums are bit-identical).\n"
             "5. Reordered-CSR build: serial prefix-sum of output offsets, then parallel copy into "
             "disjoint ranges.\n\n"
             "**Left SERIAL (order-dependent by design — parallelizing would change the output):**\n"
             "- COMMUNITY_TC's CSC build (per-column row order feeds label init), column-label init "
             "ordering, the 2-sweep plurality-vote label propagation, and the first-seen community "
             "renumber. These are the community-detection core; they dominate COMMUNITY_TC's remaining "
             "time on 100M+ nnz graphs.\n\n"]

    for k in OPT_KERNELS:
        rows = []
        for (ds, kk), ov in old.items():
            if kk != k or (ds, kk) not in new:
                continue
            nv = new[(ds, kk)]
            rows.append((ds, ov, nv, ov / nv if nv > 0 else 0))
        rows.sort(key=lambda x: -x[1])
        old_med = statistics.median([r[1] for r in rows]); new_med = statistics.median([r[2] for r in rows])
        old_geo = statistics.geometric_mean([max(r[1], 1e-6) for r in rows])
        new_geo = statistics.geometric_mean([max(r[2], 1e-6) for r in rows])
        old_max = max(r[1] for r in rows); new_max = max(r[2] for r in rows)
        sp_geo = statistics.geometric_mean([r[3] for r in rows if r[3] > 0])
        target = "✅ MET" if new_geo <= 80.0 else "❌ not met"
        lines.append(f"## {k}\n\n")
        lines.append(f"- Geomean conversion: **{old_geo:.1f} ms → {new_geo:.1f} ms** — target ≤80 ms: **{target}**\n")
        lines.append(f"- Median: {old_med:.1f} → {new_med:.2f} ms; worst graph: {old_max:.0f} → {new_max:.0f} ms\n")
        lines.append(f"- Geomean speedup v3/v2: **{sp_geo:.1f}×**\n\n")
        lines.append("| dataset | v2 ms | v3 ms | speedup |\n|---|---|---|---|\n")
        for ds, ov, nv, sp in rows:
            lines.append(f"| {ds} | {ov:.2f} | {nv:.2f} | {sp:.1f}× |\n")
        lines.append("\n")

    # control: CSR-family should be unchanged (they were not touched)
    lines.append("## Control: untouched kernels (should be ~unchanged)\n\n| kernel | v2 median | v3 median |\n|---|---|---|\n")
    for k in CONTROL_KERNELS:
        o = [v for (ds, kk), v in old.items() if kk == k]
        n = [v for (ds, kk), v in new.items() if kk == k]
        if o and n:
            lines.append(f"| {k} | {statistics.median(o):.2f} | {statistics.median(n):.2f} |\n")

    Path(args.out).write_text("".join(lines))
    print(f"wrote {args.out}")
    for k in OPT_KERNELS:
        o = [old[(ds, kk)] for (ds, kk) in old if kk == k and (ds, kk) in new]
        n = [new[(ds, kk)] for (ds, kk) in new if kk == k and (ds, kk) in old]
        print(f"  {k}: geomean {statistics.geometric_mean([max(x,1e-6) for x in o]):.1f} -> "
              f"{statistics.geometric_mean([max(x,1e-6) for x in n]):.1f} ms ; "
              f"max {max(o):.0f} -> {max(n):.0f} ms")


if __name__ == "__main__":
    main()
