"""
ra_eval_extended.py - Evaluation script for RA-SpMM extended kernels

Tests new regime-specific kernels (Wave 1: R6, R2, R1) against cuSPARSE
and existing kernels. Outputs CSV for analysis.

Usage:
    python ra_eval_extended.py                    # Run all tests
    python ra_eval_extended.py --kernel R6        # Test specific kernel
    python ra_eval_extended.py --correctness-only # Correctness check only
"""
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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
ATOL = 1e-3
# TC kernels use FP16 accumulation (WMMA) which has ~3 decimal digits of precision.
# Expected max_error is ~0.005 for typical inputs. Use relaxed tolerance for TC paths.
ATOL_TC = 0.01
TC_KERNELS = {"TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"}  # FP16 WMMA tolerance
DEFAULT_Ns = [64, 128, 256, 512]

@dataclass
class TestCase:
    name: str
    regime: str
    M: int
    K: int
    avg_nnz: float
    generator: str
    gen_kwargs: dict


# ---------------------------------------------------------------------------
# Synthetic test cases per regime
# ---------------------------------------------------------------------------
def build_test_cases() -> List[TestCase]:
    cases = []

    # R1: Hub-dominated power-law
    cases.append(TestCase("powerlaw_100K", "R1_power_law", 100000, 100000, 25.0,
                          "gen_skewed_powerlaw", {"alpha": 2.5, "min_nnz": 1, "max_nnz": 500, "seed": 42}))
    cases.append(TestCase("hub_heavy_50K", "R1_power_law", 50000, 50000, 40.0,
                          "gen_hub_heavy", {"hub_fraction": 0.01, "hub_degree": 2000, "base_degree": 10, "seed": 42}))

    # R2: Ordered sparse / road-network
    cases.append(TestCase("road_like_500K", "R2_road_network", 500000, 500000, 3.0,
                          "gen_road_like", {"avg_degree": 3, "seed": 42}))
    cases.append(TestCase("road_like_1M", "R2_road_network", 1000000, 1000000, 3.0,
                          "gen_road_like", {"avg_degree": 3, "seed": 123}))

    # R6: Dense co-purchase / overhead-sensitive
    cases.append(TestCase("uniform_10K", "R6_overhead_sensitive", 10000, 10000, 8.0,
                          "gen_random_sparse", {"nnz_per_row": 8, "seed": 42}))
    cases.append(TestCase("uniform_50K", "R6_overhead_sensitive", 50000, 50000, 6.0,
                          "gen_random_sparse", {"nnz_per_row": 6, "seed": 42}))

    # R3: Reordered locality (recoverable locality via reordering)
    cases.append(TestCase("locality_50K", "R3_reordered_locality", 50000, 50000, 12.0,
                          "gen_scrambled_locality", {"window_rows": 64, "window_span": 128,
                          "intra_window_density": 0.15, "seed": 42}))
    cases.append(TestCase("block_local_100K", "R3_reordered_locality", 100000, 100000, 8.0,
                          "gen_block_locality", {"block_size": 64, "fill": 0.12, "seed": 42}))

    # R4: Dense block-local / TC-friendly (high tile fill, compact columns)
    cases.append(TestCase("clustered_50K", "R4_tc_friendly", 50000, 50000, 15.0,
                          "gen_clustered_window", {"window_rows": 16, "window_span": 64,
                          "intra_window_density": 0.25, "seed": 42}))
    cases.append(TestCase("community_50K", "R4_tc_friendly", 50000, 50000, 8.0,
                          "gen_community_clustered", {"n_comm": 100, "within_density": 0.08,
                          "between_density": 0.0001, "seed": 42}))

    # R5: Sparse modular community (strong local communities)
    # gen_community_sbm signature: (M, n_comm, within_density, between_density, seed)
    # NOTE: K is not a separate parameter — SBM generates square M×M matrices
    cases.append(TestCase("sbm_100K", "R5_community", 100000, 100000, 7.0,
                          "gen_community_sbm", {"n_comm": 200, "within_density": 0.07,
                          "between_density": 0.00005, "seed": 42}))

    # Additional: mixed regime for comparison
    cases.append(TestCase("random_100K", "R7_mixed", 100000, 100000, 10.0,
                          "gen_random_sparse", {"nnz_per_row": 10, "seed": 42}))

    return cases


def generate_matrix(case: TestCase) -> Dict:
    gen_fn = getattr(ra_spmm, case.generator)
    # Some generators take (M, K, ...) and others take (M, ...) only.
    # Generators producing square matrices (SBM, powerlaw_realistic) don't take K.
    NO_K_GENERATORS = {"gen_community_sbm", "gen_powerlaw_realistic"}
    if case.generator in NO_K_GENERATORS:
        return gen_fn(case.M, **case.gen_kwargs)
    return gen_fn(case.M, case.K, **case.gen_kwargs)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_ms(run_fn, warmup: int = WARMUP, iters: int = TIMED_ITERS) -> float:
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
# Kernel runners
# ---------------------------------------------------------------------------
def run_csr_direct(rowptr, colind, vals, B):
    return ra_spmm.spmm_csr_direct(rowptr, colind, vals, B)


def run_cusparse(rowptr, colind, vals, B):
    return ra_spmm.spmm_cusparse(rowptr, colind, vals, B)


def run_zero_overhead(rowptr, colind, vals, B, plan_cache: dict, key: str):
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_zero_overhead_plan(
            rowptr.cpu(), rowptr.shape[0] - 1, rowptr.shape[0] - 1)
    return ra_spmm.run_zero_overhead_plan(plan_cache[key], rowptr, colind, vals, B)


def run_vectorized_coarse(rowptr, colind, vals, B, plan_cache: dict, key: str):
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_vectorized_coarse_plan(
            rowptr.cpu(), rowptr.shape[0] - 1, rowptr.shape[0] - 1)
    return ra_spmm.run_vectorized_coarse_plan(plan_cache[key], rowptr, colind, vals, B)


def run_rode_enhanced(rowptr, colind, vals, B, plan_cache: dict, key: str):
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_rode_enhanced_plan(
            rowptr.cpu(), rowptr.shape[0] - 1, rowptr.shape[0] - 1)
    return ra_spmm.run_rode_enhanced_plan(plan_cache[key], colind, vals, B)


def run_row_split(rowptr, colind, vals, B, plan_cache: dict, key: str):
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_row_split_plan(
            rowptr.cpu(), rowptr.shape[0] - 1, rowptr.shape[0] - 1)
    return ra_spmm.run_row_split_plan(plan_cache[key], colind, vals, B)


# --- Wave 2 kernel runners ---

def run_tc_direct(rowptr, colind, vals, B, plan_cache: dict, key: str):
    M = rowptr.shape[0] - 1
    N = B.shape[1]
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_tc_direct_plan(
            rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
    return ra_spmm.run_tc_direct_plan(plan_cache[key], B)


def run_locality_tiled(rowptr, colind, vals, B, plan_cache: dict, key: str):
    M = rowptr.shape[0] - 1
    N = B.shape[1]
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_locality_tiled_plan(
            rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
    return ra_spmm.run_locality_tiled_plan(plan_cache[key], B)


# --- Wave 3 kernel runners ---

def run_community_tc(rowptr, colind, vals, B, plan_cache: dict, key: str):
    M = rowptr.shape[0] - 1
    N = B.shape[1]
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_community_tc_plan(
            rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
    return ra_spmm.run_community_tc_plan(plan_cache[key], B)


def run_segment_hybrid(rowptr, colind, vals, B, plan_cache: dict, key: str):
    M = rowptr.shape[0] - 1
    N = B.shape[1]
    if key not in plan_cache:
        plan_cache[key] = ra_spmm.make_segment_hybrid_plan(
            rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
    return ra_spmm.run_segment_hybrid_plan(plan_cache[key], colind, vals, B)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def check_correctness(case: TestCase, Ns: List[int]) -> List[Dict]:
    results = []
    mat = generate_matrix(case)
    rowptr = mat['rowptr'].cuda()
    colind = mat['colind'].cuda()
    vals = mat['vals'].cuda()
    M = mat['M']

    plan_cache = {}

    for N in Ns:
        if N > 512:
            continue
        B = torch.randn(mat['K'], N, device='cuda')
        C_ref = run_csr_direct(rowptr, colind, vals, B)

        kernels = {
            "ZERO_OVERHEAD_CSR": lambda: run_zero_overhead(rowptr, colind, vals, B, plan_cache, f"zo_{N}"),
            "RODE_ENHANCED": lambda: run_rode_enhanced(rowptr, colind, vals, B, plan_cache, f"re_{N}"),
            "TC_DIRECT": lambda: run_tc_direct(rowptr, colind, vals, B, plan_cache, f"ft_{N}"),
            "COMMUNITY_TC": lambda: run_community_tc(rowptr, colind, vals, B, plan_cache, f"ct_{N}"),
            "SEGMENT_HYBRID": lambda: run_segment_hybrid(rowptr, colind, vals, B, plan_cache, f"sh_{N}"),
            # Dropped: VECTORIZED_COARSE, LOCALITY_TILED (0 oracle wins, dominated)
        }

        for kname, kfn in kernels.items():
            try:
                C_test = kfn()
                max_err = (C_test - C_ref).abs().max().item()
                tol = ATOL_TC if kname in TC_KERNELS else ATOL
                passed = max_err < tol
                results.append({
                    "case": case.name, "regime": case.regime, "kernel": kname,
                    "N": N, "correct": passed, "max_error": max_err
                })
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {kname} N={N}: max_error={max_err:.6f}")
            except Exception as e:
                results.append({
                    "case": case.name, "regime": case.regime, "kernel": kname,
                    "N": N, "correct": False, "max_error": float('inf')
                })
                print(f"  [ERROR] {kname} N={N}: {e}")

    return results


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------
def benchmark_case(case: TestCase, Ns: List[int]) -> List[Dict]:
    results = []
    mat = generate_matrix(case)
    rowptr = mat['rowptr'].cuda()
    colind = mat['colind'].cuda()
    vals = mat['vals'].cuda()
    M = mat['M']

    plan_cache = {}

    for N in Ns:
        B = torch.randn(mat['K'], N, device='cuda')

        # Baseline: CSR_DIRECT
        ms_direct = measure_ms(lambda: run_csr_direct(rowptr, colind, vals, B))

        # Baseline: cuSPARSE
        try:
            ms_cusparse = measure_ms(lambda: run_cusparse(rowptr, colind, vals, B))
        except Exception:
            ms_cusparse = float('inf')

        # Baseline: ROW_SPLIT_CUDA
        try:
            ms_row_split = measure_ms(lambda: run_row_split(rowptr, colind, vals, B, plan_cache, f"rs_{N}"))
        except Exception:
            ms_row_split = float('inf')

        # Final kernel roster (6 kernels)
        new_kernels = {
            "ZERO_OVERHEAD_CSR": lambda: run_zero_overhead(rowptr, colind, vals, B, plan_cache, f"zo_{N}"),
            "RODE_ENHANCED": lambda: run_rode_enhanced(rowptr, colind, vals, B, plan_cache, f"re_{N}"),
            "TC_DIRECT": lambda: run_tc_direct(rowptr, colind, vals, B, plan_cache, f"ft_{N}"),
            "COMMUNITY_TC": lambda: run_community_tc(rowptr, colind, vals, B, plan_cache, f"ct_{N}"),
            "SEGMENT_HYBRID": lambda: run_segment_hybrid(rowptr, colind, vals, B, plan_cache, f"sh_{N}"),
        }

        for kname, kfn in new_kernels.items():
            try:
                ms_kernel = measure_ms(kfn)
                speedup_vs_cusparse = ms_cusparse / ms_kernel if ms_kernel > 0 else 0
                speedup_vs_direct = ms_direct / ms_kernel if ms_kernel > 0 else 0

                results.append({
                    "case": case.name,
                    "regime": case.regime,
                    "M": M,
                    "nnz": int(rowptr[-1].item()),
                    "N": N,
                    "kernel": kname,
                    "ms": round(ms_kernel, 4),
                    "ms_cusparse": round(ms_cusparse, 4),
                    "ms_csr_direct": round(ms_direct, 4),
                    "ms_row_split": round(ms_row_split, 4),
                    "speedup_vs_cusparse": round(speedup_vs_cusparse, 3),
                    "speedup_vs_csr_direct": round(speedup_vs_direct, 3),
                })

                print(f"  {kname} N={N}: {ms_kernel:.3f}ms "
                      f"(vs cuSPARSE: {speedup_vs_cusparse:.2f}x, "
                      f"vs CSR_DIRECT: {speedup_vs_direct:.2f}x)")
            except Exception as e:
                print(f"  {kname} N={N}: ERROR - {e}")
                results.append({
                    "case": case.name, "regime": case.regime, "M": M,
                    "nnz": 0, "N": N, "kernel": kname,
                    "ms": -1, "ms_cusparse": ms_cusparse, "ms_csr_direct": ms_direct,
                    "ms_row_split": ms_row_split,
                    "speedup_vs_cusparse": 0, "speedup_vs_csr_direct": 0,
                })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RA-SpMM Extended Evaluation")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Only run correctness checks")
    parser.add_argument("--kernel", type=str, default=None,
                        choices=["R1", "R2", "R3", "R4", "R5", "R6", "all"],
                        help="Test specific regime kernel")
    parser.add_argument("--Ns", type=int, nargs="+", default=DEFAULT_Ns,
                        help="Output dimensions to test")
    parser.add_argument("--output", type=str, default="ra_eval_results.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    print("=" * 70)
    print("RA-SpMM Extended Evaluation - Wave 1 Kernels")
    print("=" * 70)

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    cases = build_test_cases()

    # Filter by regime if specified
    if args.kernel and args.kernel != "all":
        regime_map = {"R1": "R1_power_law", "R2": "R2_road_network", "R3": "R3_reordered_locality",
                      "R4": "R4_tc_friendly", "R5": "R5_community", "R6": "R6_overhead_sensitive"}
        target = regime_map.get(args.kernel, "")
        cases = [c for c in cases if c.regime == target]

    # Correctness
    print("\n--- Correctness Checks ---")
    all_correct = True
    for case in cases:
        print(f"\n[{case.regime}] {case.name} (M={case.M}, avg_nnz={case.avg_nnz}):")
        results = check_correctness(case, args.Ns)
        for r in results:
            if not r["correct"]:
                all_correct = False

    if not all_correct:
        print("\nWARNING: Some correctness checks FAILED!")

    if args.correctness_only:
        print("\nCorrectness-only mode complete.")
        return

    # Performance
    print("\n--- Performance Benchmarks ---")
    all_results = []
    for case in cases:
        print(f"\n[{case.regime}] {case.name} (M={case.M}, avg_nnz={case.avg_nnz}):")
        results = benchmark_case(case, args.Ns)
        all_results.extend(results)

    # Save CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output}")

    # Summary
    print("\n--- Summary (geomean speedup vs cuSPARSE) ---")
    from collections import defaultdict
    import math
    by_kernel = defaultdict(list)
    for r in all_results:
        if r["speedup_vs_cusparse"] > 0:
            by_kernel[r["kernel"]].append(math.log(r["speedup_vs_cusparse"]))
    for k, logs in sorted(by_kernel.items()):
        geomean = math.exp(sum(logs) / len(logs))
        print(f"  {k}: {geomean:.3f}x geomean ({len(logs)} datapoints)")

    print("\nDone.")


if __name__ == "__main__":
    main()
