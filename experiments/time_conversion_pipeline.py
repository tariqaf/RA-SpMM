"""
Times the non-compute stages of the SpMM pipeline per graph:
  (1) per-kernel one-time CSR -> kernel-format build (conversion) time — the
      make_*_plan() cost; CSR_DIRECT needs no conversion (~0).
  (2) at N=128, the three-way feature-extraction : conversion : kernel-compute
      split, and its amortization over a ~400-call (100-epoch x 4 SpMM/epoch) run,
      since conversion + feature-extraction are one-time but compute recurs.

Timing protocol: conversion is a one-time build, so we time it as the median of a
few rebuilds (isolating allocation noise). Kernel-compute uses the paper protocol
(50 warmup + 200 timed CUDA-event iters). Feature-extraction time is taken from the
CPU reference pass (matching time_feature_extraction.py).

Outputs:
  fgcs_results/revision/featbreak/conversion_times.csv
  fgcs_results/revision/featbreak/pipeline_proportion.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset, measure_ms  # noqa: E402

KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
           "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"]


def build_plan(kernel, rp_cpu, ci_cpu, v_cpu, M, N):
    """Return the plan object (or None for CSR_DIRECT which has no conversion)."""
    if kernel == "CSR_DIRECT":
        return None
    if kernel == "ZERO_OVERHEAD_CSR":
        return ra_spmm.make_zero_overhead_plan(rp_cpu, M, M)
    if kernel == "RODE_ENHANCED":
        return ra_spmm.make_rode_enhanced_plan(rp_cpu, M, M)
    if kernel == "TC_DIRECT":
        return ra_spmm.make_tc_direct_plan(rp_cpu, ci_cpu, v_cpu, M, M, N)
    if kernel == "COMMUNITY_TC":
        return ra_spmm.make_community_tc_plan(rp_cpu, ci_cpu, v_cpu, M, M, N)
    if kernel == "SEGMENT_HYBRID":
        return ra_spmm.make_segment_hybrid_plan(rp_cpu, ci_cpu, v_cpu, M, M, N)
    raise ValueError(kernel)


def run_plan(kernel, plan, rp, ci, v, B):
    if kernel == "CSR_DIRECT":
        return ra_spmm.spmm_csr_direct(rp, ci, v, B)
    if kernel == "ZERO_OVERHEAD_CSR":
        return ra_spmm.run_zero_overhead_plan(plan, rp, ci, v, B)
    if kernel == "RODE_ENHANCED":
        return ra_spmm.run_rode_enhanced_plan(plan, ci, v, B)
    if kernel == "TC_DIRECT":
        return ra_spmm.run_tc_direct_plan(plan, B)
    if kernel == "COMMUNITY_TC":
        return ra_spmm.run_community_tc_plan(plan, B)
    if kernel == "SEGMENT_HYBRID":
        return ra_spmm.run_segment_hybrid_plan(plan, ci, v, B)
    raise ValueError(kernel)


def time_conversion(kernel, rp_cpu, ci_cpu, v_cpu, M, N, reps=5):
    """One-time CSR->format build time (ms), median of `reps` rebuilds."""
    if kernel == "CSR_DIRECT":
        return 0.0
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = build_plan(kernel, rp_cpu, ci_cpu, v_cpu, M, N)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def cpu_feature_ms(rowptr_np, warmup=50, iters=200):
    def _pass():
        deg = (rowptr_np[1:] - rowptr_np[:-1]).astype(np.float64)
        s = deg.sum(); ss = (deg * deg).sum()
        M = deg.shape[0]
        d = s / max(1, M)
        _ = (max(0.0, ss / max(1, M) - d * d) ** 0.5) / d if d > 0 else 0.0
    for _ in range(warmup):
        _pass()
    t0 = time.perf_counter()
    for _ in range(iters):
        _pass()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "paper_datasets.json"))
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--epochs-calls", type=int, default=400,
                    help="Number of SpMM calls to amortize one-time costs over (100 epochs x 4).")
    ap.add_argument("--conv-out", default=str(REPO_ROOT / "fgcs_results/revision/featbreak/conversion_times.csv"))
    ap.add_argument("--pipe-out", default=str(REPO_ROOT / "fgcs_results/revision/featbreak/pipeline_proportion.csv"))
    args = ap.parse_args()

    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA {torch.version.cuda}  N={args.N}")

    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    conv_rows = []
    pipe_rows = []

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        # respect per-entry allowed Ns
        allowed = [int(n) for n in entry.get("Ns", [64, 128, 256, 512])]
        if args.N not in allowed:
            print(f"  [skip N] {entry['name']}: N={args.N} not in {allowed}")
            continue
        mat = load_dataset(entry)
        if mat is None:
            print(f"  [skip] {entry['name']}: not found")
            continue
        M = mat["M"]
        rp_cpu = mat["rowptr"].contiguous().int()
        ci_cpu = mat["colind"].contiguous().int()
        v_cpu = mat["vals"].contiguous().float()
        rowptr_np = rp_cpu.numpy()
        nnz = int(rowptr_np[-1])
        rp = rp_cpu.cuda(); ci = ci_cpu.cuda(); v = v_cpu.cuda()
        B = torch.randn(M, args.N, device="cuda")

        feat_ms = cpu_feature_ms(rowptr_np)

        per_kernel = {}
        for k in KERNELS:
            try:
                conv_ms = time_conversion(k, rp_cpu, ci_cpu, v_cpu, M, args.N)
                plan = build_plan(k, rp_cpu, ci_cpu, v_cpu, M, args.N)
                compute_ms = measure_ms(lambda: run_plan(k, plan, rp, ci, v, B))
                per_kernel[k] = (conv_ms, compute_ms)
                conv_rows.append({
                    "dataset": entry["name"], "category": entry.get("category", "?"),
                    "M": M, "nnz": nnz, "N": args.N, "kernel": k,
                    "conversion_ms": round(conv_ms, 4),
                    "compute_ms": round(compute_ms, 4),
                })
                # Pipeline proportion for this kernel
                total_once = feat_ms + conv_ms + compute_ms
                amortized = feat_ms + conv_ms + compute_ms * args.epochs_calls
                pipe_rows.append({
                    "dataset": entry["name"], "category": entry.get("category", "?"),
                    "M": M, "nnz": nnz, "N": args.N, "kernel": k,
                    "feature_ms": round(feat_ms, 4),
                    "conversion_ms": round(conv_ms, 4),
                    "compute_ms": round(compute_ms, 4),
                    "single_call_total_ms": round(total_once, 4),
                    "feat_pct_single": round(100 * feat_ms / total_once, 2) if total_once > 0 else 0,
                    "conv_pct_single": round(100 * conv_ms / total_once, 2) if total_once > 0 else 0,
                    "compute_pct_single": round(100 * compute_ms / total_once, 2) if total_once > 0 else 0,
                    "amortized_total_ms_%dcalls" % args.epochs_calls: round(amortized, 4),
                    "compute_pct_amortized": round(100 * compute_ms * args.epochs_calls / amortized, 2) if amortized > 0 else 0,
                })
                print(f"  {entry['name']:<20s} {k:<18s} feat={feat_ms:7.3f} conv={conv_ms:8.3f} compute={compute_ms:7.4f} ms")
            except Exception as e:
                print(f"  {entry['name']:<20s} {k:<18s} ERROR: {e}")
        del B, rp, ci, v
        torch.cuda.empty_cache()

    for path, rows in [(args.conv_out, conv_rows), (args.pipe_out, pipe_rows)]:
        if rows:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
