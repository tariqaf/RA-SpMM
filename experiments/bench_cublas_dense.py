"""
Dense cuBLAS GEMM baseline on small/dense matrices.

Question: on small/dense matrices, does padding A to dense + a cuBLAS GEMM beat a
sparse kernel? For the tiny/dense-small graphs at N in {64,128,256}:
  - materialise A as a dense M x K FP16 matrix (skip + log any that OOM),
  - time cuBLAS GemmEx (FP16 in / FP32 accumulate) for C = A_dense @ B
    (same 50 warmup + 200 timed CUDA-event protocol),
  - report throughput under BOTH FLOP conventions:
        true-nnz : 2 * nnz * N        (useful work only)
        padded   : 2 * M * K * N       (work cuBLAS actually does)
  - speedup vs cuSPARSE and vs the router's chosen sparse kernel,
  - flag any (graph, N) where cuBLAS beats our router's kernel, with the margin.

torch.matmul on FP16 tensors dispatches to cublasGemmEx with FP32 accumulation on
the tensor cores (allow_fp16_reduced_precision_reduction is left at its default off),
matching the standard FP16-in / FP32-accumulate convention.

Output: fgcs_results/revision/cublas/cublas_small.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import ra_spmm  # noqa: E402
from ra_real_graph_eval import (BASE_ATOL, TC_EXTRA_FACTOR, load_dataset,
                                measure_ms, measure_one_ms, population_cv)  # noqa: E402

# The dense-small + tiny graph set.
TINY_GRAPHS = ["ca-GrQc", "ca-HepTh", "ca-CondMat", "amazon-photo",
               "amazon-computers", "Cora", "CiteSeer", "PPI"]
# Ensure FP32 accumulation (no reduced-precision fp16 reduction).
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def dense_from_csr(rp_cpu, ci_cpu, vals_cpu, M, K):
    """Build dense M x K FP16 adjacency on GPU. Returns None if it would OOM."""
    bytes_needed = M * K * 2
    free, total = torch.cuda.mem_get_info()
    # leave headroom for B, C and workspace
    if bytes_needed > 0.6 * free:
        return None
    A = torch.zeros((M, K), device="cuda", dtype=torch.float16)
    rp = rp_cpu.tolist()
    ci = ci_cpu
    rows = torch.repeat_interleave(
        torch.arange(M, device="cuda"),
        (rp_cpu[1:] - rp_cpu[:-1]).to("cuda"))
    cols = ci.to("cuda").long()
    A[rows, cols] = vals_cpu.to("cuda", dtype=torch.float16)
    return A


def time_cublas(A_half, B_half, warmup=50, timed=200):
    return measure_ms(lambda: torch.matmul(A_half, B_half), warmup, timed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "paper_datasets.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "fgcs_results/revision/cublas/cublas_small.csv"))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--cold-iters", type=int, default=10)
    ap.add_argument("--datasets", default="",
                    help="Optional comma-separated graph names")
    ap.add_argument("--Ns", default="64,128,256")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA {torch.version.cuda}")

    manifest = {d["name"]: d for d in json.loads(Path(args.datasets_file).read_text())["datasets"]}
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
    n_values = [int(value) for value in args.Ns.replace(",", " ").split()]
    rows = []
    cublas_wins = []

    for name in TINY_GRAPHS:
        if selected and name not in selected:
            continue
        entry = manifest.get(name)
        if entry is None:
            print(f"  [skip] {name}: not in manifest")
            continue
        mat = load_dataset(entry)
        if mat is None:
            print(f"  [skip] {name}: file not found")
            continue
        M, K = mat["M"], mat["K"]
        rp_cpu = mat["rowptr"].contiguous().int()
        ci_cpu = mat["colind"].contiguous().int()
        v_cpu = mat["vals"].contiguous().float()
        rp = rp_cpu.cuda(); ci = ci_cpu.cuda(); v = v_cpu.cuda()
        nnz = int(rp_cpu[-1].item())
        d_bar = nnz / max(1, M)
        deg = (rp_cpu[1:] - rp_cpu[:-1]).float()
        cv_d = population_cv(deg)
        max_row_nnz = max(1, int(deg.max().item()))

        A_half = dense_from_csr(rp_cpu, ci_cpu, v_cpu, M, K)
        for N in n_values:
            B32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
            cus_warm = ra_spmm.benchmark_cusparse(
                rp, ci, v, B32, args.warmup, args.timed)
            cus_cold = ra_spmm.benchmark_cusparse_cold(
                rp, ci, v, B32, args.cold_iters)

            if A_half is None:
                rows.append({
                    "dataset": name, "category": entry.get("category", "?"),
                    "M": M, "K": K, "nnz": nnz, "N": N, "d_bar": round(d_bar, 3), "cv_d": round(cv_d, 3),
                    "kernel": "cuBLAS",
                    "ms_warm": "OOM", "preprocess_ms": "OOM",
                    "cold_exec_ms": "OOM", "ms_cold": "OOM",
                    "ms_cusparse_warm": round(float(cus_warm["exec_ms"]), 6),
                    "ms_cusparse_cold": round(float(cus_cold["total_ms"]), 6),
                    "gflops_truennz": "", "gflops_padded": "",
                    "speedup_vs_cusparse_warm": "", "speedup_vs_cusparse_cold": "",
                    "correct": False, "error": "dense_allocation_oom",
                })
                print(f"  {name:<18s} N={N:<4d} cuBLAS=OOM (dense {M}x{K} fp16)")
                del B32
                continue

            Bh = B32.half()
            output = torch.matmul(A_half, Bh).float()
            reference = ra_spmm.spmm_cusparse(rp, ci, v, B32)
            max_error = float((output - reference).abs().max().item())
            tolerance = BASE_ATOL * max(1.0, math.sqrt(max_row_nnz)) * TC_EXTRA_FACTOR
            correct = max_error <= tolerance and max_error < 1.0
            ms_cublas = time_cublas(A_half, Bh, args.warmup, args.timed) if correct else float("nan")
            setup_total = 0.0
            cold_exec_total = 0.0
            if correct:
                for _ in range(args.cold_iters):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    cold_A = dense_from_csr(rp_cpu, ci_cpu, v_cpu, M, K)
                    cold_B = B32.half()
                    torch.cuda.synchronize()
                    setup_total += (time.perf_counter() - start) * 1e3
                    cold_output, cold_exec = measure_one_ms(
                        lambda: torch.matmul(cold_A, cold_B))
                    cold_exec_total += cold_exec
                    del cold_output, cold_A, cold_B
                preprocess_ms = setup_total / args.cold_iters
                cold_exec_ms = cold_exec_total / args.cold_iters
                ms_cold = preprocess_ms + cold_exec_ms
            else:
                preprocess_ms = cold_exec_ms = ms_cold = float("nan")
            flops_truennz = 2.0 * nnz * N
            flops_padded = 2.0 * M * K * N
            gflops_true = flops_truennz / (ms_cublas * 1e-3) / 1e9
            gflops_padded = flops_padded / (ms_cublas * 1e-3) / 1e9
            sp_vs_cus_warm = float(cus_warm["exec_ms"]) / ms_cublas if correct else float("nan")
            sp_vs_cus_cold = float(cus_cold["total_ms"]) / ms_cold if correct else float("nan")

            rows.append({
                "dataset": name, "category": entry.get("category", "?"),
                "M": M, "K": K, "nnz": nnz, "N": N, "d_bar": round(d_bar, 3), "cv_d": round(cv_d, 3),
                "kernel": "cuBLAS",
                "ms_warm": round(ms_cublas, 6),
                "preprocess_ms": round(preprocess_ms, 6),
                "cold_exec_ms": round(cold_exec_ms, 6),
                "ms_cold": round(ms_cold, 6),
                "ms_cusparse_warm": round(float(cus_warm["exec_ms"]), 6),
                "ms_cusparse_cold": round(float(cus_cold["total_ms"]), 6),
                "gflops_truennz": round(gflops_true, 2), "gflops_padded": round(gflops_padded, 2),
                "speedup_vs_cusparse_warm": round(sp_vs_cus_warm, 6),
                "speedup_vs_cusparse_cold": round(sp_vs_cus_cold, 6),
                "correct": correct,
                "soft_fail": tolerance < max_error < 1.0,
                "hard_fail": max_error >= 1.0,
                "max_error": max_error,
                "tolerance": tolerance,
                "error": "",
            })
            print(f"  {name:<18s} N={N:<4d} cuBLAS(warm)={ms_cublas:.4f}ms "
                  f"cus(warm)={float(cus_warm['exec_ms']):.4f}ms correct={correct}")
            del output, reference, Bh, B32
        del A_half, rp, ci, v
        torch.cuda.empty_cache()

    if rows:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
