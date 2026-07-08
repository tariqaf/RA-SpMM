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
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset, measure_ms, run_kernel  # noqa: E402
from ra_router_eval import simple_router  # noqa: E402

# The dense-small + tiny graph set.
TINY_GRAPHS = ["ca-GrQc", "ca-HepTh", "ca-CondMat", "amazon-photo",
               "amazon-computers", "Cora", "CiteSeer", "PPI"]
N_VALUES = [64, 128, 256]

# Ensure FP32 accumulation (no reduced-precision fp16 reduction).
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def dense_from_csr(rp_cpu, ci_cpu, M, K):
    """Build dense M x K FP16 adjacency on GPU. Returns None if it would OOM."""
    bytes_needed = M * K * 2
    free, total = torch.cuda.mem_get_info()
    # leave headroom for B, C and workspace
    if bytes_needed > 0.6 * free:
        return None
    A = torch.zeros((M, K), device="cuda", dtype=torch.float16)
    rp = rp_cpu.tolist()
    ci = ci_cpu
    # Scatter ones (unit values, matching how graphs are loaded)
    rows = torch.repeat_interleave(
        torch.arange(M, device="cuda"),
        (rp_cpu[1:] - rp_cpu[:-1]).to("cuda"))
    cols = ci.to("cuda").long()
    A[rows, cols] = 1.0
    return A


def time_cublas(A_half, B_half):
    return measure_ms(lambda: torch.matmul(A_half, B_half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "paper_datasets.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "fgcs_results/revision/cublas/cublas_small.csv"))
    args = ap.parse_args()

    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA {torch.version.cuda}")

    manifest = {d["name"]: d for d in json.loads(Path(args.datasets_file).read_text())["datasets"]}
    rows = []
    cublas_wins = []

    for name in TINY_GRAPHS:
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
        cv_d = float((deg.std() / deg.mean()).item()) if d_bar > 0 else 0.0

        A_half = dense_from_csr(rp_cpu, ci_cpu, M, K)
        for N in N_VALUES:
            B32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
            plan_cache = {}
            # cuSPARSE reference timing
            ms_cusparse = measure_ms(lambda: run_kernel("CUSPARSE", rp, ci, v, B32, plan_cache, "cus"))
            # router's chosen kernel
            router_kernel = simple_router(d_bar, cv_d, M, N, nnz)
            try:
                ms_router = measure_ms(lambda: run_kernel(router_kernel, rp, ci, v, B32, plan_cache, f"{router_kernel}_{N}"))
            except Exception as e:
                router_kernel, ms_router = f"ERR({router_kernel})", float("nan")

            if A_half is None:
                rows.append({
                    "dataset": name, "category": entry.get("category", "?"),
                    "M": M, "K": K, "nnz": nnz, "N": N, "d_bar": round(d_bar, 3), "cv_d": round(cv_d, 3),
                    "cublas_ms": "OOM", "ms_cusparse": round(ms_cusparse, 4),
                    "router_kernel": router_kernel, "ms_router": round(ms_router, 4),
                    "gflops_truennz": "", "gflops_padded": "",
                    "speedup_cublas_vs_cusparse": "", "speedup_cublas_vs_router": "",
                    "cublas_beats_router": "",
                })
                print(f"  {name:<18s} N={N:<4d} cuBLAS=OOM (dense {M}x{K} fp16)")
                continue

            Bh = B32.half()
            ms_cublas = time_cublas(A_half, Bh)
            flops_truennz = 2.0 * nnz * N
            flops_padded = 2.0 * M * K * N
            gflops_true = flops_truennz / (ms_cublas * 1e-3) / 1e9
            gflops_padded = flops_padded / (ms_cublas * 1e-3) / 1e9
            sp_vs_cus = ms_cusparse / ms_cublas if ms_cublas > 0 else 0
            sp_vs_router = ms_router / ms_cublas if ms_cublas > 0 else 0
            beats_router = ms_cublas < ms_router if ms_router == ms_router else False  # nan-safe

            if beats_router:
                cublas_wins.append((name, N, ms_router / ms_cublas, router_kernel))

            rows.append({
                "dataset": name, "category": entry.get("category", "?"),
                "M": M, "K": K, "nnz": nnz, "N": N, "d_bar": round(d_bar, 3), "cv_d": round(cv_d, 3),
                "cublas_ms": round(ms_cublas, 4), "ms_cusparse": round(ms_cusparse, 4),
                "router_kernel": router_kernel, "ms_router": round(ms_router, 4),
                "gflops_truennz": round(gflops_true, 2), "gflops_padded": round(gflops_padded, 2),
                "speedup_cublas_vs_cusparse": round(sp_vs_cus, 3),
                "speedup_cublas_vs_router": round(sp_vs_router, 3),
                "cublas_beats_router": beats_router,
            })
            flag = "  <<< cuBLAS BEATS ROUTER" if beats_router else ""
            print(f"  {name:<18s} N={N:<4d} cuBLAS={ms_cublas:.4f}ms router={router_kernel}={ms_router:.4f}ms "
                  f"cus={ms_cusparse:.4f}ms  vsRouter={sp_vs_router:.2f}x{flag}")
            del Bh, B32
        del A_half, rp, ci, v
        torch.cuda.empty_cache()

    if rows:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.output} ({len(rows)} rows)")
    if cublas_wins:
        print("\n*** cuBLAS BEATS ROUTER on these (dataset, N, margin, router_kernel): ***")
        for name, N, margin, rk in cublas_wins:
            print(f"    {name} N={N}: {margin:.2f}x faster than router's {rk}")
    else:
        print("\nSparse router wins throughout (no cuBLAS-beats-router cases).")


if __name__ == "__main__":
    main()
