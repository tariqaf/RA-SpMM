"""Time the deployed full router extractor and a labeled lightweight variant.

The production measurements call ``ra_spmm.make_router_plan`` and therefore
include the complete tile/locality feature set. CPU-resident and GPU-resident
inputs are reported separately; the latter includes the binding's device-to-host
copies before the CPU extractor. There is currently no GPU implementation of the
full feature set, which is stated in the output instead of being inferred from a
different algorithm.

The custom GPU degree-moment kernel is retained as an explicitly labeled
``lightweight (d_bar,CV_d only)`` experiment.
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
from torch.utils.cpp_extension import load_inline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ra_real_graph_eval import load_dataset  # noqa: E402
import ra_spmm  # noqa: E402

# ---------------------------------------------------------------------------
# One-pass CUDA feature-extraction kernel (d_bar, CV_d from rowptr)
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// One pass over rowptr[0..M]; degree[i] = rowptr[i+1] - rowptr[i].
// Accumulate sum(deg) and sum(deg^2) in double via block reduction + atomics.
__global__ void deg_moments_kernel(const int* __restrict__ rowptr, int M,
                                   double* __restrict__ g_sum,
                                   double* __restrict__ g_sumsq) {
    extern __shared__ double sdata[];   // [blockDim] sum, [blockDim] sumsq
    double* s_sum = sdata;
    double* s_sq  = sdata + blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double local_sum = 0.0, local_sq = 0.0;
    for (int r = i; r < M; r += stride) {
        double d = (double)(rowptr[r + 1] - rowptr[r]);
        local_sum += d;
        local_sq  += d * d;
    }
    s_sum[tid] = local_sum;
    s_sq[tid]  = local_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid]  += s_sq[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(g_sum, s_sum[0]);
        atomicAdd(g_sumsq, s_sq[0]);
    }
}

// Returns tensor [sum_deg, sumsq_deg] (double, on GPU). Caller derives d_bar/CV.
torch::Tensor deg_moments(torch::Tensor rowptr) {
    TORCH_CHECK(rowptr.is_cuda(), "rowptr must be CUDA");
    TORCH_CHECK(rowptr.dtype() == torch::kInt32, "rowptr must be int32");
    int M = (int)rowptr.numel() - 1;
    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(rowptr.device());
    auto acc = torch::zeros({2}, opts);
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    size_t shmem = 2 * threads * sizeof(double);
    deg_moments_kernel<<<blocks, threads, shmem>>>(
        rowptr.data_ptr<int>(), M,
        acc.data_ptr<double>(), acc.data_ptr<double>() + 1);
    return acc;
}
"""

_CPP_SRC = "torch::Tensor deg_moments(torch::Tensor rowptr);"


def build_ext():
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    return load_inline(
        name="ra_featext",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["deg_moments"],
        extra_cuda_cflags=["-O3", "-arch=sm_86"],
        verbose=False,
    )


def cpu_features(rowptr_np: np.ndarray):
    """Single CPU pass: degree, d_bar, CV_d."""
    deg = (rowptr_np[1:] - rowptr_np[:-1]).astype(np.float64)
    M = deg.shape[0]
    s = deg.sum()
    ss = (deg * deg).sum()
    d_bar = s / max(1, M)
    var = ss / max(1, M) - d_bar * d_bar
    var = max(0.0, var)
    cv = (var ** 0.5) / d_bar if d_bar > 0 else 0.0
    return d_bar, cv


def gpu_features(ext, rowptr_gpu):
    acc = ext.deg_moments(rowptr_gpu)
    s, ss = acc[0].item(), acc[1].item()
    M = rowptr_gpu.numel() - 1
    d_bar = s / max(1, M)
    var = max(0.0, ss / max(1, M) - d_bar * d_bar)
    cv = (var ** 0.5) / d_bar if d_bar > 0 else 0.0
    return d_bar, cv


def time_cpu(rowptr_np, warmup, iters):
    for _ in range(warmup):
        cpu_features(rowptr_np)
    t0 = time.perf_counter()
    for _ in range(iters):
        cpu_features(rowptr_np)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms


def time_gpu_kernel_only(ext, rowptr_gpu, warmup, iters):
    for _ in range(warmup):
        ext.deg_moments(rowptr_gpu)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.deg_moments(rowptr_gpu)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters  # ms


def time_gpu_with_copy(ext, rowptr_cpu, warmup, iters):
    """Include H2D copy of the rowptr array (honest end-to-end if features start on CPU)."""
    for _ in range(warmup):
        rp = rowptr_cpu.cuda(non_blocking=False)
        ext.deg_moments(rp)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        rp = rowptr_cpu.cuda(non_blocking=False)
        ext.deg_moments(rp)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters  # ms


def time_production_full(rowptr, colind, vals, M, K, N, warmup, iters):
    """Wall time of the actual binding, including required residency copies."""
    for _ in range(warmup):
        ra_spmm.make_router_plan(rowptr, colind, vals, M, K, N, "MAIN")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        ra_spmm.make_router_plan(rowptr, colind, vals, M, K, N, "MAIN")
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / max(1, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "paper_datasets.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "fgcs_results/revision/featbreak/feature_extraction_gpu.csv"))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--full-warmup", type=int, default=2)
    ap.add_argument("--full-iters", type=int, default=10)
    ap.add_argument("--N", type=int, default=128)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA {torch.version.cuda}")
    print("Building one-pass feature-extraction CUDA kernel (JIT)...")
    ext = build_ext()
    print("  built OK")

    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    rows = []
    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        mat = load_dataset(entry)
        if mat is None:
            print(f"  [skip] {entry['name']}: file not found")
            continue
        rowptr_cpu = mat["rowptr"].contiguous().int()
        colind_cpu = mat["colind"].contiguous().int()
        vals_cpu = mat["vals"].contiguous().float()
        rowptr_np = rowptr_cpu.numpy()
        rowptr_gpu = rowptr_cpu.cuda()
        colind_gpu = colind_cpu.cuda()
        vals_gpu = vals_cpu.cuda()
        M = rowptr_cpu.numel() - 1
        K = int(mat.get("K", M))
        nnz = int(rowptr_np[-1])

        # Correctness cross-check CPU vs GPU
        d_cpu, cv_cpu = cpu_features(rowptr_np)
        d_gpu, cv_gpu = gpu_features(ext, rowptr_gpu)
        ok = (abs(d_cpu - d_gpu) < 1e-6 * max(1.0, d_cpu)) and (abs(cv_cpu - cv_gpu) < 1e-5 * max(1.0, cv_cpu))

        ms_cpu = time_cpu(rowptr_np, args.warmup, args.iters)
        ms_gpu = time_gpu_kernel_only(ext, rowptr_gpu, args.warmup, args.iters)
        ms_gpu_copy = time_gpu_with_copy(ext, rowptr_cpu, args.warmup, args.iters)
        full_cpu_input_ms = time_production_full(
            rowptr_cpu, colind_cpu, vals_cpu, M, K, args.N,
            args.full_warmup, args.full_iters)
        full_gpu_input_ms = time_production_full(
            rowptr_gpu, colind_gpu, vals_gpu, M, K, args.N,
            args.full_warmup, args.full_iters)
        speedup_kernel = ms_cpu / ms_gpu if ms_gpu > 0 else 0.0
        speedup_with_copy = ms_cpu / ms_gpu_copy if ms_gpu_copy > 0 else 0.0

        rows.append({
            "dataset": entry["name"], "category": entry.get("category", "?"),
            "M": M, "nnz": nnz,
            "d_bar": round(d_cpu, 4), "cv_d": round(cv_cpu, 4),
            "production_full_cpu_input_ms": round(full_cpu_input_ms, 4),
            "production_full_gpu_input_ms": round(full_gpu_input_ms, 4),
            "production_full_backend": "CPU",
            "production_full_gpu_implementation": False,
            "lightweight_cpu_ms": round(ms_cpu, 4),
            "lightweight_gpu_kernel_ms": round(ms_gpu, 5),
            "lightweight_gpu_with_h2d_ms": round(ms_gpu_copy, 5),
            "lightweight_speedup_kernel_only": round(speedup_kernel, 2),
            "lightweight_speedup_with_h2d": round(speedup_with_copy, 2),
            "lightweight_cpu_gpu_match": ok,
        })
        print(f"  {entry['name']:<22s} M={M:>9d}  full(CPU input)={full_cpu_input_ms:9.3f}ms  "
              f"full(GPU input+D2H)={full_gpu_input_ms:9.3f}ms  "
              f"lightweight CPU/GPU={ms_cpu:.4f}/{ms_gpu:.5f}ms match={ok}")

    if rows:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        # Geomean of speedups
        import math
        gk = math.exp(sum(math.log(max(1e-9, r["lightweight_speedup_kernel_only"])) for r in rows) / len(rows))
        gc = math.exp(sum(math.log(max(1e-9, r["lightweight_speedup_with_h2d"])) for r in rows) / len(rows))
        mean_cpu = sum(r["production_full_cpu_input_ms"] for r in rows) / len(rows)
        print(f"\nWrote {args.output}  ({len(rows)} graphs)")
        print(f"Mean production full feature time (CPU input): {mean_cpu:.2f} ms")
        print(f"Lightweight-only GPU speedup: kernel-only={gk:.1f}x, with-H2D-copy={gc:.1f}x")


if __name__ == "__main__":
    main()
