"""
ra_real_graph_eval.py - Evaluate all RA-SpMM kernels on real-world graphs

Loads graphs from paper_datasets.json and tests the six-kernel paper portfolio
against cuSPARSE.
Outputs CSV with per-graph, per-kernel, per-N performance.

Usage:
    python ra_real_graph_eval.py                          # All enabled datasets
    python ra_real_graph_eval.py --category "hub-dominated power-law"  # Specific regime
    python ra_real_graph_eval.py --datasets ogbn-arxiv,Reddit,Cora --N 128
    python ra_real_graph_eval.py --correctness-only       # Correctness only
    python ra_real_graph_eval.py --output real_results.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import ra_spmm
except ImportError:
    print("ERROR: ra_spmm not found. Build first: python setup.py build_ext --inplace")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WARMUP = 50
TIMED_ITERS = 200
COLD_ITERS = 10

# Tolerance follows the existing square-root model for accumulation-order
# differences: BASE_ATOL * sqrt(max row nnz). Tile kernels receive the existing
# factor for configurations that can execute half-precision WMMA groups.
BASE_ATOL = 1e-3
TC_EXTRA_FACTOR = 10.0  # FP16 accumulation is less precise
TC_KERNELS = {"TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID",
              "TC_DIRECT_TF32", "COMMUNITY_TC_TF32", "SEGMENT_HYBRID_TF32"}

# Errors at or above this threshold are classified separately for diagnostics.
HARD_FAIL_THRESHOLD = 1.0

DATASET_FILE = "paper_datasets.json"

# Final kernel roster (6 kernels — dropped VECTORIZED_COARSE and LOCALITY_TILED,
# which contribute 0 oracle wins and are dominated by TC_DIRECT/CSR_DIRECT)
ALL_KERNELS = [
    "CSR_DIRECT",
    "RODE_ENHANCED",
    "ZERO_OVERHEAD_CSR",
    "TC_DIRECT",
    "COMMUNITY_TC",
    "SEGMENT_HYBRID",
]

# Experimental variants selectable via --kernels but not in the default roster.
EXPERIMENTAL_KERNELS = ["TC_DIRECT_TF32", "COMMUNITY_TC_TF32", "SEGMENT_HYBRID_TF32"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_edge_list(path: str, directed: bool, symmetrize: bool,
                   one_indexed: bool, M_hint: int = 0) -> Dict:
    """Load edge-list format (two columns: src dst)."""
    edges = []
    max_node = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                src, dst = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if one_indexed:
                src -= 1
                dst -= 1
            edges.append((src, dst))
            max_node = max(max_node, src, dst)
            if symmetrize and src != dst:
                edges.append((dst, src))

    n = max(max_node + 1, M_hint)
    # Build CSR
    row_counts = [0] * n
    for src, dst in edges:
        if src < n:
            row_counts[src] += 1

    rowptr = [0] * (n + 1)
    for i in range(n):
        rowptr[i + 1] = rowptr[i] + row_counts[i]

    colind = [0] * rowptr[n]
    vals = [1.0] * rowptr[n]
    cursor = list(rowptr[:-1])
    for src, dst in edges:
        if src < n and cursor[src] < rowptr[src + 1]:
            colind[cursor[src]] = min(dst, n - 1)
            cursor[src] += 1

    # Sort and deduplicate column indices per row. This prevents double-counting
    # when symmetrize=true is applied to already-bidirectional input files.
    new_colind = []
    new_vals = []
    new_rowptr = [0]
    for i in range(n):
        start, end = rowptr[i], rowptr[i + 1]
        segment = sorted(set(colind[start:end]))  # deduplicate via set
        new_colind.extend(segment)
        new_vals.extend([1.0] * len(segment))
        new_rowptr.append(len(new_colind))

    return {
        'rowptr': torch.tensor(new_rowptr, dtype=torch.int32),
        'colind': torch.tensor(new_colind, dtype=torch.int32),
        'vals': torch.tensor(new_vals, dtype=torch.float32),
        'M': n, 'K': n,
    }


def load_npz(path: str) -> Dict:
    """Load NPZ format (scipy sparse CSR, custom, or edge_index)."""
    data = np.load(path, allow_pickle=True)
    if 'rowptr' in data:
        rowptr = torch.tensor(data['rowptr'], dtype=torch.int32)
        colind = torch.tensor(data['colind'], dtype=torch.int32)
        vals = torch.tensor(data['vals'], dtype=torch.float32) if 'vals' in data else torch.ones(len(colind))
    elif 'indptr' in data and 'indices' in data:
        # scipy sparse CSR format: indptr=rowptr, indices=colind, data=vals
        rowptr = torch.tensor(data['indptr'].astype(np.int32), dtype=torch.int32)
        colind = torch.tensor(data['indices'].astype(np.int32), dtype=torch.int32)
        if 'data' in data and data['data'] is not None and len(data['data']) > 0:
            vals = torch.tensor(data['data'].astype(np.float32), dtype=torch.float32)
        else:
            vals = torch.ones(len(colind), dtype=torch.float32)
    elif 'edge_index' in data:
        edge_index = data['edge_index']
        src, dst = edge_index[0], edge_index[1]
        n = max(src.max(), dst.max()) + 1
        from collections import Counter
        counts = Counter(src.tolist())
        rowptr = [0]
        for i in range(n):
            rowptr.append(rowptr[-1] + counts.get(i, 0))
        rowptr = torch.tensor(rowptr, dtype=torch.int32)
        order = np.lexsort((dst, src))
        colind = torch.tensor(dst[order], dtype=torch.int32)
        vals = torch.ones(len(colind), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown NPZ format: {list(data.keys())}")
    M = int(rowptr.shape[0] - 1)
    return {'rowptr': rowptr, 'colind': colind, 'vals': vals, 'M': M, 'K': M}


def load_dataset(entry: dict) -> Optional[Dict]:
    """Load dataset based on paper_datasets.json entry."""
    path = entry['path']
    if not os.path.exists(path):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, entry['path'])
    if not os.path.exists(path):
        return None

    fmt = entry.get('format', 'edge')
    if fmt == 'edge':
        return load_edge_list(
            path,
            directed=entry.get('directed', False),
            symmetrize=entry.get('symmetrize', False),
            one_indexed=entry.get('one_indexed', False),
            M_hint=entry.get('M', 0),
        )
    elif fmt == 'npz':
        return load_npz(path)
    else:
        print(f"  Unknown format: {fmt}")
        return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_ms(run_fn, warmup=WARMUP, iters=TIMED_ITERS):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def measure_one_ms(run_fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = run_fn()
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end)


def population_cv(degrees: torch.Tensor) -> float:
    if degrees.numel() == 0:
        return 0.0
    mean = degrees.float().mean()
    if float(mean.item()) == 0.0:
        return 0.0
    return float((degrees.float().std(correction=0) / mean).item())


# ---------------------------------------------------------------------------
# Kernel planning and execution
# ---------------------------------------------------------------------------
def build_kernel_plan(kernel_name: str, rowptr_cpu, colind_cpu, vals_cpu,
                      M: int, K: int, N: int):
    if kernel_name == "CSR_DIRECT":
        return None
    if kernel_name == "ZERO_OVERHEAD_CSR":
        return ra_spmm.make_zero_overhead_plan(rowptr_cpu, M, K)
    if kernel_name == "RODE_ENHANCED":
        return ra_spmm.make_rode_enhanced_plan(rowptr_cpu, M, K)
    if kernel_name in ("TC_DIRECT", "TC_DIRECT_TF32"):
        return ra_spmm.make_tc_direct_plan(rowptr_cpu, colind_cpu, vals_cpu, M, K, N)
    if kernel_name in ("COMMUNITY_TC", "COMMUNITY_TC_TF32"):
        return ra_spmm.make_community_tc_plan(rowptr_cpu, colind_cpu, vals_cpu, M, K, N)
    if kernel_name in ("SEGMENT_HYBRID", "SEGMENT_HYBRID_TF32"):
        return ra_spmm.make_segment_hybrid_plan(rowptr_cpu, colind_cpu, vals_cpu, M, K, N)
    raise ValueError(f"Kernel has no reusable custom plan: {kernel_name}")


def run_planned_kernel(kernel_name: str, plan, rowptr, colind, vals, B):
    if kernel_name == "CSR_DIRECT":
        return ra_spmm.spmm_csr_direct(rowptr, colind, vals, B)
    if kernel_name == "ZERO_OVERHEAD_CSR":
        return ra_spmm.run_zero_overhead_plan(plan, rowptr, colind, vals, B)
    if kernel_name == "RODE_ENHANCED":
        return ra_spmm.run_rode_enhanced_plan(plan, colind, vals, B)
    if kernel_name == "TC_DIRECT":
        return ra_spmm.run_tc_direct_plan(plan, B)
    if kernel_name == "TC_DIRECT_TF32":
        return ra_spmm.run_tc_direct_plan_tf32(plan, B)
    if kernel_name == "COMMUNITY_TC":
        return ra_spmm.run_community_tc_plan(plan, B)
    if kernel_name == "COMMUNITY_TC_TF32":
        return ra_spmm.run_community_tc_plan_tf32(plan, B)
    if kernel_name == "SEGMENT_HYBRID":
        return ra_spmm.run_segment_hybrid_plan(plan, colind, vals, B)
    if kernel_name == "SEGMENT_HYBRID_TF32":
        return ra_spmm.run_segment_hybrid_plan_tf32(plan, colind, vals, B)
    raise ValueError(f"Unknown planned kernel: {kernel_name}")


def run_kernel(kernel_name: str, rowptr, colind, vals, B, plan_cache: dict, cache_key: str):
    M = rowptr.shape[0] - 1
    K = B.shape[0]
    N = B.shape[1]

    if kernel_name == "CUSPARSE":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_cusparse_plan(
                rowptr, colind, vals, B)
        return ra_spmm.run_cusparse_plan(plan_cache[cache_key], B)
    if cache_key not in plan_cache:
        plan_cache[cache_key] = build_kernel_plan(
            kernel_name, rowptr.cpu(), colind.cpu(), vals.cpu(), M, K, N)
    return run_planned_kernel(kernel_name, plan_cache[cache_key], rowptr, colind, vals, B)


def benchmark_custom_cold(kernel_name: str, rowptr_cpu, colind_cpu, vals_cpu,
                          rowptr, colind, vals, B, iters: int):
    M = int(rowptr_cpu.numel() - 1)
    K = int(B.shape[0])
    N = int(B.shape[1])
    plan_total = 0.0
    exec_total = 0.0
    if kernel_name == "CSR_DIRECT":
        for _ in range(max(1, iters)):
            output, exec_ms = measure_one_ms(
                lambda: run_planned_kernel(
                    kernel_name, None, rowptr, colind, vals, B))
            exec_total += exec_ms
            del output
        cold_exec_ms = exec_total / float(max(1, iters))
        return {
            "preprocess_ms": 0.0,
            "cold_exec_ms": cold_exec_ms,
            "ms_cold": cold_exec_ms,
        }
    for _ in range(max(1, iters)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        plan = build_kernel_plan(kernel_name, rowptr_cpu, colind_cpu, vals_cpu, M, K, N)
        torch.cuda.synchronize()
        plan_total += (time.perf_counter() - t0) * 1000.0
        output, exec_ms = measure_one_ms(
            lambda: run_planned_kernel(kernel_name, plan, rowptr, colind, vals, B))
        exec_total += exec_ms
        del output
        del plan
        torch.cuda.synchronize()
    count = float(max(1, iters))
    preprocess_ms = plan_total / count
    cold_exec_ms = exec_total / count
    return {
        "preprocess_ms": preprocess_ms,
        "cold_exec_ms": cold_exec_ms,
        "ms_cold": preprocess_ms + cold_exec_ms,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    global WARMUP, TIMED_ITERS, COLD_ITERS
    parser = argparse.ArgumentParser(description="RA-SpMM Real Graph Evaluation")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by dataset category")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run specific dataset by name")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to run")
    parser.add_argument("--N", "--Ns", dest="Ns", type=str, default=None,
                        help="Comma-separated output feature dimensions")
    parser.add_argument("--output", type=str, default="real_graph_results.csv")
    parser.add_argument("--datasets-file", "--datasets-json", dest="datasets_file",
                        type=str, default=DATASET_FILE)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--timed", type=int, default=TIMED_ITERS)
    parser.add_argument("--cold-iters", type=int, default=COLD_ITERS,
                        help="Independent setup-plus-one-execute repetitions")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--kernels", default=",".join(ALL_KERNELS),
                        help="Comma-separated custom kernels; cuSPARSE is always measured")
    parser.add_argument("--correctness-report", default=None,
                        help="CSV path for per-kernel strict correctness results")
    args = parser.parse_args()
    WARMUP = int(args.warmup)
    TIMED_ITERS = int(args.timed)
    COLD_ITERS = int(args.cold_iters)

    selected_Ns = None
    if args.Ns:
        selected_Ns = [int(x) for x in args.Ns.replace(",", " ").split()]
    selected_kernels = [name.strip() for name in args.kernels.split(",") if name.strip()]
    invalid_kernels = sorted(
        set(selected_kernels) - set(ALL_KERNELS) - set(EXPERIMENTAL_KERNELS))
    if invalid_kernels:
        raise ValueError(f"Unknown kernels: {invalid_kernels}")

    print("=" * 70)
    print("RA-SpMM Real Graph Evaluation — All Kernels")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("ERROR: No CUDA GPU")
        sys.exit(1)

    # Load dataset manifest
    with open(args.datasets_file, 'r') as f:
        manifest = json.load(f)

    datasets = manifest.get('datasets', [])

    # Filter
    if args.category:
        datasets = [d for d in datasets if d.get('category', '') == args.category]
    selected_names = []
    if args.dataset:
        selected_names.append(args.dataset)
    if args.datasets:
        selected_names.extend(x.strip() for x in args.datasets.split(",") if x.strip())
    if selected_names:
        selected = set(selected_names)
        datasets = [d for d in datasets if d.get('name', '') in selected]

    # Only enabled datasets
    datasets = [d for d in datasets if d.get('enabled', True)]

    print(f"\nDatasets to evaluate: {len(datasets)}")
    for d in datasets:
        print(f"  [{d.get('category', '?')}] {d['name']} (M={d.get('M', '?')}, nnz={d.get('nnz', '?')})")

    all_results = []
    correctness_rows = []
    correctness_failures = 0

    for entry in datasets:
        name = entry['name']
        category = entry.get('category', 'unknown')
        declared_Ns = [int(value) for value in entry.get('Ns', [64, 128, 256])]
        Ns = ([value for value in selected_Ns if value in declared_Ns]
              if selected_Ns is not None else declared_Ns)
        max_N = entry.get('max_N', 512)
        Ns = [n for n in Ns if n <= max_N]

        print(f"\n{'='*60}")
        print(f"[{category}] {name}")
        print(f"{'='*60}")

        mat = load_dataset(entry)
        if mat is None:
            print(f"  SKIPPED: dataset file not found ({entry['path']})")
            continue

        M = mat['M']
        K = mat.get('K', M)
        rowptr_cpu = mat['rowptr'].contiguous().int()
        colind_cpu = mat['colind'].contiguous().int()
        vals_cpu = mat['vals'].contiguous().float()
        rowptr = rowptr_cpu.cuda()
        colind = colind_cpu.cuda()
        vals = vals_cpu.cuda()
        nnz = int(rowptr_cpu[-1].item())
        deg = (rowptr_cpu[1:] - rowptr_cpu[:-1]).float()
        d_bar = nnz / max(1, M)
        cv_d = population_cv(deg)
        max_nnz_row = max(1, int(deg.max().item())) if M > 0 else 1
        print(f"  M={M}, nnz={nnz}, avg_deg={d_bar:.1f}, cv_d={cv_d:.3f}")

        # Warn if the loaded nnz differs from the manifest metadata.
        manifest_nnz = entry.get('nnz', None)
        if manifest_nnz is not None:
            ratio = nnz / max(1, manifest_nnz)
            if abs(ratio - 1.0) > 0.01:  # >1% discrepancy
                print(f"  WARNING: loaded nnz={nnz} vs manifest nnz={manifest_nnz} "
                      f"(ratio={ratio:.2f}). Check symmetrize/directed flags!")

        for N in Ns:
            torch.manual_seed(args.seed + M + N)
            B = torch.randn(K, N, device='cuda')
            print(f"\n  N={N}:")

            # The reference call is not used for performance measurement.
            C_ref = ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
            measurements = {}
            for kname in selected_kernels:
                try:
                    plan = build_kernel_plan(
                        kname, rowptr_cpu, colind_cpu, vals_cpu, M, K, N)
                    C_test = run_planned_kernel(kname, plan, rowptr, colind, vals, B)
                    max_err = (C_test - C_ref).abs().max().item()
                    tol = BASE_ATOL * max(1.0, math.sqrt(max_nnz_row))
                    if kname in TC_KERNELS:
                        tol *= TC_EXTRA_FACTOR
                    correct = max_err <= tol and max_err < HARD_FAIL_THRESHOLD
                    soft_fail = tol < max_err < HARD_FAIL_THRESHOLD
                    hard_fail = max_err >= HARD_FAIL_THRESHOLD
                    if not correct:
                        correctness_failures += 1

                    timing = {
                        "correct": correct,
                        "soft_fail": soft_fail,
                        "hard_fail": hard_fail,
                        "max_error": max_err,
                        "tolerance": tol,
                        "error": "",
                    }
                    if args.correctness_only:
                        correctness_rows.append({
                            "dataset": name, "category": category,
                            "synthetic": bool(entry.get("synthetic", False)),
                            "M": M, "K": K, "nnz": nnz, "N": N,
                            "kernel": kname, **timing,
                        })
                    if not args.correctness_only and correct:
                        timing["ms_warm"] = measure_ms(
                            lambda: run_planned_kernel(
                                kname, plan, rowptr, colind, vals, B),
                            WARMUP, TIMED_ITERS)
                        timing.update(benchmark_custom_cold(
                            kname, rowptr_cpu, colind_cpu, vals_cpu,
                            rowptr, colind, vals, B, COLD_ITERS))
                    measurements[kname] = timing
                    status = "PASS" if correct else ("SOFT_FAIL" if soft_fail else "HARD_FAIL")
                    print(f"    [{status}] {kname}: max_error={max_err:.6g} (tol={tol:.6g})")
                    del C_test
                    del plan
                except Exception as exc:
                    correctness_failures += 1
                    measurements[kname] = {
                        "correct": False, "soft_fail": False, "hard_fail": True,
                        "max_error": None, "tolerance": None, "error": str(exc),
                    }
                    if args.correctness_only:
                        correctness_rows.append({
                            "dataset": name, "category": category,
                            "synthetic": bool(entry.get("synthetic", False)),
                            "M": M, "K": K, "nnz": nnz, "N": N,
                            "kernel": kname, **measurements[kname],
                        })
                    print(f"    [ERROR] {kname}: {exc}")

            if not args.correctness_only:
                cus_warm = ra_spmm.benchmark_cusparse(
                    rowptr, colind, vals, B, warmup=WARMUP, iters=TIMED_ITERS)
                cus_cold = ra_spmm.benchmark_cusparse_cold(
                    rowptr, colind, vals, B, max(1, COLD_ITERS))
                measurements["CUSPARSE"] = {
                    "correct": True, "soft_fail": False, "hard_fail": False,
                    "max_error": 0.0, "tolerance": 0.0, "error": "",
                    "ms_warm": float(cus_warm["exec_ms"]),
                    "preprocess_ms": float(cus_cold["plan_ms"]),
                    "cold_exec_ms": float(cus_cold["exec_ms"]),
                    "ms_cold": float(cus_cold["total_ms"]),
                }

                # Precision-matched baseline: same algorithm and timing loops,
                # A/B in fp16, C fp32, compute fp32 (the tile kernels' dtypes).
                cus16_warm = ra_spmm.benchmark_cusparse_fp16(
                    rowptr, colind, vals, B, warmup=WARMUP, iters=TIMED_ITERS)
                cus16_cold = ra_spmm.benchmark_cusparse_fp16_cold(
                    rowptr, colind, vals, B, max(1, COLD_ITERS))
                ms_cusparse_fp16_warm = float(cus16_warm["exec_ms"])
                ms_cusparse_fp16_cold = float(cus16_cold["total_ms"])

                direct = measurements.get("CSR_DIRECT", {})
                cusp = measurements["CUSPARSE"]
                for kname in selected_kernels + ["CUSPARSE"]:
                    timing = measurements[kname]
                    eligible = bool(timing.get("correct"))
                    warm = timing.get("ms_warm") if eligible else None
                    cold = timing.get("ms_cold") if eligible else None
                    all_results.append({
                        "dataset": name,
                        "category": category,
                        "synthetic": bool(entry.get("synthetic", False)),
                        "gpu_name": torch.cuda.get_device_name(0),
                        "cuda_version": str(torch.version.cuda),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "warmup_iters": WARMUP,
                        "timed_iters": TIMED_ITERS,
                        "cold_iters": COLD_ITERS,
                        "M": M, "K": K, "nnz": nnz, "N": N,
                        "avg_nnz_per_row": round(d_bar, 6),
                        "cv_d": round(cv_d, 6),
                        "max_nnz_per_row": max_nnz_row,
                        "kernel": kname,
                        "ms_warm": round(warm, 6) if warm is not None else None,
                        "preprocess_ms": round(timing.get("preprocess_ms", 0.0), 6) if eligible else None,
                        "cold_exec_ms": round(timing.get("cold_exec_ms", 0.0), 6) if eligible else None,
                        "ms_cold": round(cold, 6) if cold is not None else None,
                        "ms_cusparse_warm": round(cusp["ms_warm"], 6),
                        "ms_cusparse_cold": round(cusp["ms_cold"], 6),
                        "ms_csr_direct_warm": round(direct.get("ms_warm", 0.0), 6),
                        "ms_csr_direct_cold": round(direct.get("ms_cold", 0.0), 6),
                        "speedup_vs_cusparse_warm": round(cusp["ms_warm"] / warm, 6) if warm else None,
                        "speedup_vs_cusparse_cold": round(cusp["ms_cold"] / cold, 6) if cold else None,
                        "speedup_vs_csr_direct_warm": round(direct["ms_warm"] / warm, 6)
                            if warm and direct.get("ms_warm") else None,
                        "speedup_vs_csr_direct_cold": round(direct["ms_cold"] / cold, 6)
                            if cold and direct.get("ms_cold") else None,
                        "ms_cusparse_fp16_warm": round(ms_cusparse_fp16_warm, 6),
                        "ms_cusparse_fp16_cold": round(ms_cusparse_fp16_cold, 6),
                        "speedup_precision_matched_warm":
                            round(ms_cusparse_fp16_warm / warm, 6)
                            if warm and kname in TC_KERNELS else None,
                        "speedup_precision_matched_cold":
                            round(ms_cusparse_fp16_cold / cold, 6)
                            if cold and kname in TC_KERNELS else None,
                        "correct": timing["correct"],
                        "soft_fail": timing["soft_fail"],
                        "hard_fail": timing["hard_fail"],
                        "max_error": timing["max_error"],
                        "tolerance": timing["tolerance"],
                        "error": timing["error"],
                    })
                    if kname != "CUSPARSE" and warm is not None:
                        print(f"      {kname}: warm={warm:.4f} ms "
                              f"({cusp['ms_warm']/warm:.3f}x), cold={cold:.4f} ms "
                              f"({cusp['ms_cold']/cold:.3f}x)")

            # Clear GPU memory between N values
            del C_ref
            del B
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    if args.correctness_only:
        if args.correctness_report and correctness_rows:
            report_path = os.path.abspath(args.correctness_report)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(correctness_rows[0].keys()))
                writer.writeheader()
                writer.writerows(correctness_rows)
            print(f"Strict correctness report saved to {args.correctness_report}")
        if correctness_failures:
            print(f"\nStrict correctness failed for {correctness_failures} configurations.")
            sys.exit(1)
        print("\nStrict correctness passed for every loaded configuration.")
        return

    # Save CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output}")

    # Summary: geomean speedup vs cuSPARSE per kernel and matching regime.
    print("\n" + "=" * 60)
    print("SUMMARY: Correct-only geomean speedup vs cuSPARSE")
    print("=" * 60)
    for regime in ("warm", "cold"):
        speed_col = f"speedup_vs_cusparse_{regime}"
        by_kernel = defaultdict(list)
        for row in all_results:
            speedup = row.get(speed_col)
            if row["correct"] and row["kernel"] != "CUSPARSE" and speedup:
                by_kernel[row["kernel"]].append(math.log(speedup))
        print(f"  {regime.upper()}:")
        for kernel in ALL_KERNELS:
            logs = by_kernel.get(kernel, [])
            if logs:
                print(f"    {kernel:23s}: {math.exp(sum(logs) / len(logs)):.3f}x "
                      f"({len(logs)} datapoints)")

    # Summary per category
    print("\n" + "=" * 60)
    print("SUMMARY: Best warm kernel per category (correct rows only)")
    print("=" * 60)
    by_cat_kernel = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        speedup = r.get("speedup_vs_cusparse_warm")
        if r["correct"] and speedup and r["kernel"] != "CUSPARSE":
            by_cat_kernel[r["category"]][r["kernel"]].append(math.log(speedup))

    for cat in sorted(by_cat_kernel.keys()):
        best_k, best_gm = "", 0
        for k, logs in by_cat_kernel[cat].items():
            gm = math.exp(sum(logs) / len(logs))
            if gm > best_gm:
                best_k, best_gm = k, gm
        print(f"  {cat:45s}: {best_k:25s} ({best_gm:.3f}x)")

    print("\nDone.")
    if correctness_failures:
        print(f"WARNING: {correctness_failures} incorrect rows were excluded from all statistics.")


if __name__ == "__main__":
    main()
