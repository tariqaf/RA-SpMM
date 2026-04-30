#!/usr/bin/env python3
"""
ra_dtc_single.py - Run DTC on one reordered CSR in an isolated process.

Per (graph, N) this child reports:
  - preprocess_ms
  - selection_variant_ms
  - mean_kernel_ms       (strict kernel-only)
  - std_kernel_ms
  - end_to_end_ms        (permute B + kernel + inverse permute output)
  - correct / max_error  (vs cuSPARSE on original CSR)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dtc_baseline import candidate_variants, is_dtc_available, load_dtc_module, preprocess, run_variant
from dtc_reorder_utils import load_perm, load_reordered_npz
from ra_real_graph_eval import load_dataset

try:
    import ra_spmm
except ImportError:
    ra_spmm = None


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if "error" not in payload else 1


def load_original_entry(path: str) -> Dict[str, object]:
    with open(path) as f:
        entry = json.load(f)
    if not isinstance(entry, dict):
        raise ValueError("dataset_json_entry must contain one dataset-entry object")
    return entry


def time_kernel_only(module, state, B_perm, M: int, nnz: int, use_balance: bool, exeplan: str,
                     warmup_iters: int, timed_iters: int) -> Tuple[float, float]:
    for _ in range(max(0, warmup_iters)):
        _ = run_variant(module, state, B_perm, M, nnz, use_balance, exeplan)
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(1, timed_iters)):
        start.record()
        _ = run_variant(module, state, B_perm, M, nnz, use_balance, exeplan)
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    mean_ms = sum(times_ms) / len(times_ms)
    var = sum((x - mean_ms) ** 2 for x in times_ms) / len(times_ms)
    return mean_ms, math.sqrt(var)


def time_end_to_end(module, state, B_orig, perm_gpu, perm_inv_gpu, M: int, nnz: int,
                    use_balance: bool, exeplan: str, timed_iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(max(1, timed_iters)):
        start.record()
        B_perm = B_orig.index_select(0, perm_gpu)
        out_perm = run_variant(module, state, B_perm, M, nnz, use_balance, exeplan)
        out_orig = torch.empty_like(out_perm)
        out_orig[perm_gpu] = out_perm
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        _ = out_orig
    return sum(times_ms) / len(times_ms)


def candidate_max_error(module, state, B_orig, B_perm, perm_gpu, rowptr_o, colind_o, vals_o,
                        M: int, nnz: int, use_balance: bool, exeplan: str) -> float:
    ref = ra_spmm.spmm_cusparse(rowptr_o, colind_o, vals_o, B_orig)
    out_perm = run_variant(module, state, B_perm, M, nnz, use_balance, exeplan)
    out_orig = torch.empty_like(out_perm)
    out_orig[perm_gpu] = out_perm
    return float((out_orig - ref).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json_entry", required=True,
                        help="Path to JSON with one original dataset entry dict")
    parser.add_argument("--reordered_npz", required=True,
                        help="Path to saved reordered CSR edge-list npz")
    parser.add_argument("--reorder_perm_npz", required=True,
                        help="Path to reorder_id npz where reorder_id[reordered_row]=original_row")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--reorder_ms", type=float, default=0.0)
    parser.add_argument("--reorder_method", default="")
    parser.add_argument("--reorder_version", default="")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--timed_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    if not is_dtc_available():
        return emit({"error": "dtc_unavailable"})
    if ra_spmm is None:
        return emit({"error": "ra_spmm_unavailable"})

    try:
        entry = load_original_entry(args.dataset_json_entry)
        original = load_dataset(entry)
        if original is None:
            raise RuntimeError("original_dataset_load_failed")
        reordered = load_reordered_npz(args.reordered_npz)
        perm = load_perm(args.reorder_perm_npz)

        rowptr_r = reordered["rowptr"].cuda()
        colind_r = reordered["colind"].cuda()
        vals_r = reordered["vals"].cuda()
        M = int(reordered["M"])
        nnz = int(reordered["nnz"])

        rowptr_o = original["rowptr"].cuda()
        colind_o = original["colind"].cuda()
        vals_o = original["vals"].cuda()

        perm_gpu = torch.as_tensor(perm, device="cuda", dtype=torch.long)
        torch.manual_seed(args.seed + int(args.N))
        B_orig = torch.randn((M, int(args.N)), device="cuda", dtype=torch.float32)
        B_perm = B_orig.index_select(0, perm_gpu)
    except Exception as e:
        return emit({"error": f"dataset_prepare_failed: {e}"})

    try:
        module = load_dtc_module()
    except Exception as e:
        return emit({"error": f"dtc_load_failed: {e}"})

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = preprocess(module, rowptr_r, colind_r, M, nnz)
        torch.cuda.synchronize()
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return emit({"error": f"dtc_preprocess_failed: {e}"})

    best_mean = None
    best_std = None
    best_variant = ""
    best_choice = None
    best_error = None
    variants = candidate_variants(int(args.N))

    try:
        torch.cuda.synchronize()
        select_t0 = time.perf_counter()
        for use_balance, exeplan in variants:
            mean_ms, std_ms = time_kernel_only(
                module, state, B_perm, M, nnz, use_balance, exeplan,
                args.warmup_iters, args.timed_iters
            )
            max_error = candidate_max_error(
                module, state, B_orig, B_perm, perm_gpu, rowptr_o, colind_o, vals_o,
                M, nnz, use_balance, exeplan
            )
            tag = f"{'bal' if use_balance else 'nobal'}_{exeplan}"
            if max_error > 1.0:
                continue
            if best_mean is None or mean_ms < best_mean:
                best_mean = mean_ms
                best_std = std_ms
                best_variant = tag
                best_choice = (use_balance, exeplan)
                best_error = max_error
        torch.cuda.synchronize()
        selection_variant_ms = (time.perf_counter() - select_t0) * 1000.0
    except Exception as e:
        return emit({"error": f"dtc_variant_scan_failed: {e}"})

    if best_mean is None or best_choice is None:
        return emit({"error": "dtc_no_valid_variant"})

    use_balance, exeplan = best_choice

    try:
        end_to_end_ms = time_end_to_end(
            module, state, B_orig, perm_gpu, perm_gpu, M, nnz,
            use_balance, exeplan, args.timed_iters
        )
    except Exception as e:
        return emit({"error": f"dtc_end_to_end_failed: {e}"})

    try:
        ref = ra_spmm.spmm_cusparse(rowptr_o, colind_o, vals_o, B_orig)
        out_perm = run_variant(module, state, B_perm, M, nnz, use_balance, exeplan)
        out_orig = torch.empty_like(out_perm)
        out_orig[perm_gpu] = out_perm
        max_error = float((out_orig - ref).abs().max().item())
        correct = bool(max_error <= float(args.atol))
    except Exception as e:
        return emit({"error": f"dtc_correctness_failed: {e}"})

    return emit({
        "dataset": args.dataset_name,
        "category": args.category,
        "M": M,
        "nnz": nnz,
        "N": int(args.N),
        "reorder_method": args.reorder_method,
        "reorder_version": args.reorder_version,
        "reorder_ms": float(args.reorder_ms),
        "preprocess_ms": preprocess_ms,
        "selection_variant_ms": selection_variant_ms,
        "mean_kernel_ms": float(best_mean),
        "std_kernel_ms": float(best_std if best_std is not None else 0.0),
        "end_to_end_ms": float(end_to_end_ms),
        "dtc_ms": float(best_mean),
        "dtc_variant": best_variant,
        "variant_count": len(variants),
        "correct": correct,
        "max_error": max_error,
        "selection_max_error": float(best_error if best_error is not None else max_error),
        "warmup_iters": int(args.warmup_iters),
        "timed_iters": int(args.timed_iters),
    })


if __name__ == "__main__":
    raise SystemExit(main())
