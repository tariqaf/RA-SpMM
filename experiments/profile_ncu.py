"""
Nsight Compute profiling driver: for each representative (kernel, graph, N) pair,
runs `ncu --set full` over a single NVTX-tagged SpMM launch (via
bench/profile_case_ra.py), writing <pair>.ncu-rep (full report, has roofline) and
<pair>.csv (raw metric dump), then aggregates the five metric families (tensor-core
pipe utilisation, occupancy, DRAM throughput, roofline, top warp stalls) into
PROFILE_SUMMARY.md and profile_summary.csv under fgcs_results/revision/profile/.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import io
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "bench" / "profile_case_ra.py"
NCU = os.environ.get("NCU", "ncu")

# (label, kernel, dataset, N)
PAIRS = [
    ("a_TC_DIRECT_amazon-photo_N128",      "TC_DIRECT",     "amazon-photo",     128),
    ("b_COMMUNITY_TC_com-DBLP_N128",       "COMMUNITY_TC",  "com-DBLP",         128),
    ("c_SEGMENT_HYBRID_amazon-computers_N128", "SEGMENT_HYBRID", "amazon-computers", 128),
    ("d1_cuSPARSE_amazon-photo_N128",      "CUSPARSE",      "amazon-photo",     128),
    ("d2_cuSPARSE_com-DBLP_N128",          "CUSPARSE",      "com-DBLP",         128),
    ("d3_cuSPARSE_amazon-computers_N128",  "CUSPARSE",      "amazon-computers", 128),
    ("e_RODE_ENHANCED_soc-Pokec_N256",     "RODE_ENHANCED", "soc-Pokec",        256),
]

STALL_PREFIX = "smsp__average_warps_issue_stalled_"


def run_ncu(label, kernel, dataset, N, outdir, gpu):
    rep = outdir / f"{label}.ncu-rep"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        NCU, "--nvtx", "--nvtx-include", "SPMM_PROFILE/",
        "--set", "full", "-f", "-o", str(rep.with_suffix("")),
        sys.executable, str(HARNESS),
        "--dataset", dataset, "--kernel", kernel, "--N", str(N),
    ]
    print(f"[ncu] {label} ...", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED ({r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return None
    # Export raw CSV
    csv_path = outdir / f"{label}.csv"
    exp = subprocess.run([NCU, "--import", str(rep), "--csv", "--page", "raw"],
                         capture_output=True, text=True)
    if exp.returncode == 0 and exp.stdout.strip():
        csv_path.write_text(exp.stdout)
        return csv_path
    print(f"  CSV export failed: {exp.stderr[-500:]}")
    return None


def parse_metrics(csv_path):
    """Return list of per-kernel-launch dicts of {metric_name: value} from raw ncu CSV."""
    text = csv_path.read_text()
    reader = csvmod.DictReader(io.StringIO(text))
    launches = []
    for row in reader:
        # raw page: each row is one metric for one kernel launch; columns include
        # 'ID','Kernel Name','Metric Name','Metric Value'. Group by ID.
        launches.append(row)
    # Detect schema
    if launches and "Metric Name" in launches[0]:
        by_id = {}
        for row in launches:
            kid = row.get("ID", "0")
            by_id.setdefault(kid, {"__kernel__": row.get("Kernel Name", "?")})
            val = row.get("Metric Value", "")
            by_id[kid][row["Metric Name"]] = val
        return list(by_id.values())
    # Wide schema fallback: one row per launch, metrics as columns
    return launches


def _f(d, key, default=0.0):
    v = d.get(key, "")
    if v in ("", None, "N/A", "nan"):
        return default
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return default


def summarize(launch, dataset, kernel, N):
    tc = _f(launch, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")
    occ = _f(launch, "sm__warps_active.avg.pct_of_peak_sustained_active")
    dram = _f(launch, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    sol_sm = _f(launch, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    dur_ns = _f(launch, "gpu__time_duration.sum")
    dbytes = _f(launch, "dram__bytes.sum")
    # FLOPs: FP32 fma counts 2, add/mul count 1; FP16 similarly; tensor hmma ops
    ffma = _f(launch, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
    fadd = _f(launch, "sm__sass_thread_inst_executed_op_fadd_pred_on.sum")
    fmul = _f(launch, "sm__sass_thread_inst_executed_op_fmul_pred_on.sum")
    hfma = _f(launch, "sm__sass_thread_inst_executed_op_hfma_pred_on.sum")
    hadd = _f(launch, "sm__sass_thread_inst_executed_op_hadd_pred_on.sum")
    hmul = _f(launch, "sm__sass_thread_inst_executed_op_hmul_pred_on.sum")
    hmma = _f(launch, "sm__inst_executed_pipe_tensor_op_hmma.sum")
    # hmma instruction = 1 m16n8k8 or similar; approximate FLOPs via cycles pct is
    # unreliable, so we report cuda-core FLOPs and note TC separately.
    flops_cuda = 2 * ffma + fadd + fmul + 2 * hfma + hadd + hmul
    achieved_gflops = flops_cuda / (dur_ns * 1e-9) / 1e9 if dur_ns > 0 else 0.0
    ai = flops_cuda / dbytes if dbytes > 0 else 0.0
    # Warp stalls
    stalls = []
    for k, v in launch.items():
        if k.startswith(STALL_PREFIX) and k.endswith("_per_issue_active.ratio"):
            name = k[len(STALL_PREFIX):-len("_per_issue_active.ratio")]
            stalls.append((name, _f(launch, k)))
    stalls.sort(key=lambda x: -x[1])
    top2 = stalls[:2]
    return {
        "dataset": dataset, "kernel": kernel, "N": N,
        "profiled_kernel": launch.get("__kernel__", "?")[:40],
        "TC_pipe_pct": round(tc, 2), "occupancy_pct": round(occ, 2),
        "DRAM_pct": round(dram, 2), "SM_SOL_pct": round(sol_sm, 2),
        "dur_us": round(dur_ns / 1e3, 3), "dram_MB": round(dbytes / 1e6, 2),
        "cuda_GFLOPs": round(achieved_gflops, 1), "AI_flop_per_byte": round(ai, 3),
        "hmma_insts": int(hmma),
        "top_stall_1": f"{top2[0][0]}={top2[0][1]:.2f}" if top2 else "",
        "top_stall_2": f"{top2[1][0]}={top2[1][1]:.2f}" if len(top2) > 1 else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO_ROOT / "fgcs_results/revision/profile"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", default=None, help="substring filter on pair label")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for label, kernel, dataset, N in PAIRS:
        if args.only and args.only not in label:
            continue
        csv_path = run_ncu(label, kernel, dataset, N, outdir, args.gpu)
        if csv_path is None:
            summary_rows.append({"dataset": dataset, "kernel": kernel, "N": N,
                                 "profiled_kernel": "PROFILE_FAILED"})
            continue
        launches = parse_metrics(csv_path)
        if not launches:
            continue
        # pick the longest-duration launch as the primary one for the summary
        primary = max(launches, key=lambda L: _f(L, "gpu__time_duration.sum"))
        summary_rows.append(summarize(primary, dataset, kernel, N))
        for L in launches:
            if L is not primary and _f(L, "gpu__time_duration.sum") > 0:
                s = summarize(L, dataset, kernel, N)
                s["profiled_kernel"] += " (aux)"
                summary_rows.append(s)

    # Write PROFILE_SUMMARY.md
    md = outdir / "PROFILE_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Nsight Compute Profiling Summary (RTX 3090, SM 86)\n\n")
        f.write("Five metric families per (kernel, graph, N). Full `.ncu-rep` reports "
                "(with roofline) are alongside for the UI.\n\n")
        cols = ["dataset", "kernel", "N", "profiled_kernel", "TC_pipe_pct", "occupancy_pct",
                "DRAM_pct", "SM_SOL_pct", "AI_flop_per_byte", "cuda_GFLOPs", "dur_us",
                "top_stall_1", "top_stall_2"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in summary_rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        f.write("\n_Note: AI/GFLOP/s reflect CUDA-core FP ops; tensor-core work is "
                "shown via TC_pipe_pct and hmma_insts. Use the `.ncu-rep` roofline "
                "section for the tensor-core achieved FLOP/s._\n")
    print(f"\nWrote {md}")

    # Also a machine-readable CSV
    if summary_rows:
        keys = ["dataset", "kernel", "N", "profiled_kernel", "TC_pipe_pct", "occupancy_pct",
                "DRAM_pct", "SM_SOL_pct", "AI_flop_per_byte", "cuda_GFLOPs", "dram_MB",
                "dur_us", "hmma_insts", "top_stall_1", "top_stall_2"]
        with open(outdir / "profile_summary.csv", "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote {outdir / 'profile_summary.csv'}")


if __name__ == "__main__":
    main()
