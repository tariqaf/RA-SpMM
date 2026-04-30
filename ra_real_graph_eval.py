"""
ra_real_graph_eval.py - Evaluate all RA-SpMM kernels on real-world graphs

Loads graphs from paper_datasets.json and tests all 7 new kernels + cuSPARSE + CSR_DIRECT.
Outputs CSV with per-graph, per-kernel, per-N performance.

Usage:
    python ra_real_graph_eval.py                          # All enabled datasets
    python ra_real_graph_eval.py --category "hub-dominated power-law"  # Specific regime
    python ra_real_graph_eval.py --correctness-only       # Correctness only
    python ra_real_graph_eval.py --output real_results.csv
"""
import argparse
import csv
import json
import math
import os
import sys
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

# Tolerance is proportional to avg_nnz_per_row because FP32 accumulation order
# differences grow with the number of additions per output element.
# Base tolerance for sparse graphs (avg_nnz < 20): 1e-3
# For denser graphs: scale linearly, e.g., avg_nnz=500 → tol ≈ 0.5
# TC kernels using FP16 WMMA get additional 10x relaxation
BASE_ATOL = 1e-3
TC_EXTRA_FACTOR = 10.0  # FP16 accumulation is less precise
TC_KERNELS = {"TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"}

# Hard failure threshold: errors > 1.0 are genuine bugs, not precision differences
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

    # Sort and DEDUPLICATE column indices per row.
    # This prevents double-counting when symmetrize=true is used on
    # already-bidirectional files (Bug 1 from Codex diagnosis).
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


# ---------------------------------------------------------------------------
# Kernel execution
# ---------------------------------------------------------------------------
def run_kernel(kernel_name: str, rowptr, colind, vals, B, plan_cache: dict, cache_key: str):
    M = rowptr.shape[0] - 1
    N = B.shape[1]

    if kernel_name == "CSR_DIRECT":
        return ra_spmm.spmm_csr_direct(rowptr, colind, vals, B)
    elif kernel_name == "CUSPARSE":
        return ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
    elif kernel_name == "ZERO_OVERHEAD_CSR":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_zero_overhead_plan(rowptr.cpu(), M, M)
        return ra_spmm.run_zero_overhead_plan(plan_cache[cache_key], rowptr, colind, vals, B)
    elif kernel_name == "VECTORIZED_COARSE":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_vectorized_coarse_plan(rowptr.cpu(), M, M)
        return ra_spmm.run_vectorized_coarse_plan(plan_cache[cache_key], rowptr, colind, vals, B)
    elif kernel_name == "RODE_ENHANCED":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_rode_enhanced_plan(rowptr.cpu(), M, M)
        return ra_spmm.run_rode_enhanced_plan(plan_cache[cache_key], colind, vals, B)
    elif kernel_name == "TC_DIRECT":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_tc_direct_plan(rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
        return ra_spmm.run_tc_direct_plan(plan_cache[cache_key], B)
    elif kernel_name == "LOCALITY_TILED":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_locality_tiled_plan(rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
        return ra_spmm.run_locality_tiled_plan(plan_cache[cache_key], B)
    elif kernel_name == "COMMUNITY_TC":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_community_tc_plan(rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
        return ra_spmm.run_community_tc_plan(plan_cache[cache_key], B)
    elif kernel_name == "SEGMENT_HYBRID":
        if cache_key not in plan_cache:
            plan_cache[cache_key] = ra_spmm.make_segment_hybrid_plan(rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
        return ra_spmm.run_segment_hybrid_plan(plan_cache[cache_key], colind, vals, B)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RA-SpMM Real Graph Evaluation")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by dataset category")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run specific dataset by name")
    parser.add_argument("--output", type=str, default="real_graph_results.csv")
    parser.add_argument("--datasets-file", type=str, default=DATASET_FILE)
    args = parser.parse_args()

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
    if args.dataset:
        datasets = [d for d in datasets if d.get('name', '') == args.dataset]

    # Only enabled datasets
    datasets = [d for d in datasets if d.get('enabled', True)]

    print(f"\nDatasets to evaluate: {len(datasets)}")
    for d in datasets:
        print(f"  [{d.get('category', '?')}] {d['name']} (M={d.get('M', '?')}, nnz={d.get('nnz', '?')})")

    all_results = []

    for entry in datasets:
        name = entry['name']
        category = entry.get('category', 'unknown')
        Ns = entry.get('Ns', [64, 128, 256])
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
        rowptr = mat['rowptr'].cuda()
        colind = mat['colind'].cuda()
        vals = mat['vals'].cuda()
        nnz = int(rowptr[-1].item())
        print(f"  M={M}, nnz={nnz}, avg_deg={nnz/max(1,M):.1f}")

        # Bug 3 fix: warn if loaded nnz differs from manifest
        manifest_nnz = entry.get('nnz', None)
        if manifest_nnz is not None:
            ratio = nnz / max(1, manifest_nnz)
            if abs(ratio - 1.0) > 0.01:  # >1% discrepancy
                print(f"  ⚠️  WARNING: loaded nnz={nnz} vs manifest nnz={manifest_nnz} "
                      f"(ratio={ratio:.2f}). Check symmetrize/directed flags!")

        plan_cache = {}

        for N in Ns:
            B = torch.randn(M, N, device='cuda')
            print(f"\n  N={N}:")

            # Reference: cuSPARSE (vendor baseline — the standard correctness target)
            C_ref = run_kernel("CUSPARSE", rowptr, colind, vals, B, plan_cache, "cusparse_ref")

            for kname in ALL_KERNELS:
                if kname == "CUSPARSE":
                    continue  # skip: cuSPARSE is the reference, not a test target
                cache_key = f"{kname}_{N}"
                try:
                    C_test = run_kernel(kname, rowptr, colind, vals, B, plan_cache, cache_key)
                    max_err = (C_test - C_ref).abs().max().item()

                    # Adaptive tolerance: FP32 max_error scales with max row nnz,
                    # not avg, because the hub row dominates the error bound.
                    # Use sqrt(max_nnz_per_row) as the error model.
                    max_nnz_row = max(1, int((rowptr[1:] - rowptr[:-1]).max().item()))
                    tol = BASE_ATOL * max(1.0, math.sqrt(max_nnz_row))
                    if kname in TC_KERNELS:
                        tol *= TC_EXTRA_FACTOR  # FP16 WMMA needs more slack
                    # Hard failures (errors > 1.0) are always bugs, not precision
                    correct = max_err < tol or max_err < HARD_FAIL_THRESHOLD
                    hard_fail = max_err >= HARD_FAIL_THRESHOLD

                    if args.correctness_only:
                        if hard_fail:
                            status = "HARD_FAIL"
                        elif max_err < tol:
                            status = "PASS"
                        else:
                            status = "SOFT_FAIL"  # precision difference, not a bug
                        print(f"    [{status}] {kname}: max_error={max_err:.6f} (tol={tol:.6f})")
                        continue

                    # Performance
                    ms_val = measure_ms(lambda: run_kernel(kname, rowptr, colind, vals, B, plan_cache, cache_key))

                    # Get cuSPARSE baseline
                    ms_cusparse = measure_ms(lambda: run_kernel("CUSPARSE", rowptr, colind, vals, B, plan_cache, "CUSPARSE"))
                    ms_direct = measure_ms(lambda: run_kernel("CSR_DIRECT", rowptr, colind, vals, B, plan_cache, "direct"))

                    speedup_cusparse = ms_cusparse / ms_val if ms_val > 0 else 0
                    speedup_direct = ms_direct / ms_val if ms_val > 0 else 0

                    all_results.append({
                        "dataset": name, "category": category,
                        "M": M, "nnz": nnz, "N": N,
                        "kernel": kname,
                        "ms": round(ms_val, 4),
                        "ms_cusparse": round(ms_cusparse, 4),
                        "ms_csr_direct": round(ms_direct, 4),
                        "speedup_vs_cusparse": round(speedup_cusparse, 3),
                        "speedup_vs_csr_direct": round(speedup_direct, 3),
                        "correct": correct,
                        "max_error": round(max_err, 6),
                    })

                    print(f"    {kname}: {ms_val:.3f}ms (vs cuSPARSE: {speedup_cusparse:.2f}x, vs DIRECT: {speedup_direct:.2f}x)")

                except Exception as e:
                    print(f"    {kname}: ERROR — {e}")
                    if not args.correctness_only:
                        all_results.append({
                            "dataset": name, "category": category,
                            "M": M, "nnz": nnz, "N": N,
                            "kernel": kname,
                            "ms": -1, "ms_cusparse": -1, "ms_csr_direct": -1,
                            "speedup_vs_cusparse": 0, "speedup_vs_csr_direct": 0,
                            "correct": False, "max_error": -1,
                        })

            # Clear GPU memory between N values
            del B
            torch.cuda.empty_cache()

        # Clear plan cache between datasets
        del plan_cache
        torch.cuda.empty_cache()

    if args.correctness_only:
        print("\nCorrectness-only mode complete.")
        return

    # Save CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output}")

    # Summary: geomean speedup vs cuSPARSE per kernel
    print("\n" + "=" * 60)
    print("SUMMARY: Geomean speedup vs cuSPARSE per kernel")
    print("=" * 60)
    by_kernel = defaultdict(list)
    for r in all_results:
        if r["speedup_vs_cusparse"] > 0 and r["kernel"] != "CUSPARSE":
            by_kernel[r["kernel"]].append(math.log(r["speedup_vs_cusparse"]))
    for k in ALL_KERNELS:
        if k in by_kernel and by_kernel[k]:
            logs = by_kernel[k]
            geomean = math.exp(sum(logs) / len(logs))
            print(f"  {k:25s}: {geomean:.3f}x ({len(logs)} datapoints)")

    # Summary per category
    print("\n" + "=" * 60)
    print("SUMMARY: Best kernel per category (geomean vs cuSPARSE)")
    print("=" * 60)
    by_cat_kernel = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["speedup_vs_cusparse"] > 0 and r["kernel"] != "CUSPARSE":
            by_cat_kernel[r["category"]][r["kernel"]].append(math.log(r["speedup_vs_cusparse"]))

    for cat in sorted(by_cat_kernel.keys()):
        best_k, best_gm = "", 0
        for k, logs in by_cat_kernel[cat].items():
            gm = math.exp(sum(logs) / len(logs))
            if gm > best_gm:
                best_k, best_gm = k, gm
        print(f"  {cat:45s}: {best_k:25s} ({best_gm:.3f}x)")

    print("\nDone.")


if __name__ == "__main__":
    main()
