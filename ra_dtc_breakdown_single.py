#!/usr/bin/env python3
"""
ra_dtc_breakdown_single.py - Time one DTC point with a timing breakdown.

Outputs one JSON object on the last stdout line with:
  - preprocess_ms: one-time DTC preprocessing wall time
  - variant_selection_ms: wall time to sweep candidate variants after preprocess
  - tuning_ms: total measured timed-kernel budget across all candidate variants
  - kernel_ms: best steady-state kernel time per call for the winning variant
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dtc_baseline import candidate_variants, is_dtc_available, load_dtc_module, preprocess, run_variant
from ra_real_graph_eval import load_dataset

REORDER_SCRIPT = REPO_ROOT / "external" / "DTC-SpMM_ASPLOS24" / "reordering" / "TCA_reorder.py"
REORDER_METHOD = "DTC_TCA_REORDER"


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if "error" not in payload else 1


def csr_to_src_dst(rowptr: torch.Tensor, colind: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    rowptr_np = rowptr.cpu().numpy().astype(np.int64, copy=False)
    colind_np = colind.cpu().numpy().astype(np.int32, copy=False)
    row_counts = rowptr_np[1:] - rowptr_np[:-1]
    src = np.repeat(np.arange(len(row_counts), dtype=np.int32), row_counts)
    return src, colind_np


def load_reordered_npz(path: str) -> dict:
    data = np.load(path)
    src = np.asarray(data["src_li"], dtype=np.int32)
    dst = np.asarray(data["dst_li"], dtype=np.int32)
    num_nodes = int(data["num_nodes"])
    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]
    counts = np.bincount(src, minlength=num_nodes).astype(np.int32, copy=False)
    rowptr = np.empty(num_nodes + 1, dtype=np.int32)
    rowptr[0] = 0
    np.cumsum(counts, out=rowptr[1:])
    return {
        "rowptr": torch.from_numpy(rowptr),
        "colind": torch.from_numpy(dst.astype(np.int32, copy=False)),
        "vals": torch.ones(int(dst.shape[0]), dtype=torch.float32),
        "M": num_nodes,
        "nnz": int(dst.shape[0]),
    }


def run_reorder(entry: dict, data: dict, threshold: int) -> tuple[dict, float]:
    if not REORDER_SCRIPT.exists():
        raise FileNotFoundError(f"reorder_script_missing: {REORDER_SCRIPT}")

    src, dst = csr_to_src_dst(data["rowptr"], data["colind"])
    with tempfile.TemporaryDirectory(prefix="dtc_reorder_") as tmpdir:
        input_npz = os.path.join(tmpdir, "input.npz")
        output_npz = os.path.join(tmpdir, "output.reorder.npz")
        output_perm = os.path.join(tmpdir, "output.reorder_id.npz")
        np.savez(input_npz, src_li=src, dst_li=dst, num_nodes=int(data["M"]))

        cmd = [
            sys.executable,
            str(REORDER_SCRIPT),
            "--dataset", str(entry.get("name", "graph")),
            "--thres", str(int(threshold)),
            "--input_npz", input_npz,
            "--output_npz", output_npz,
            "--output_perm_npz", output_perm,
        ]

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        reorder_ms = (time.perf_counter() - start) * 1000.0
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail_text = detail[-1] if detail else "reorder child failed"
            raise RuntimeError(f"reorder_failed: rc={proc.returncode} detail={detail_text}")
        if not os.path.exists(output_npz):
            raise RuntimeError("reorder_output_missing")
        reordered = load_reordered_npz(output_npz)
        return reordered, reorder_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json_entry", required=True,
                        help="Path to a JSON file containing exactly one dataset entry dict")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--timed_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use_reorder", action="store_true")
    parser.add_argument("--reorder_threshold", type=int, default=16)
    args = parser.parse_args()

    try:
        with open(args.dataset_json_entry) as f:
            entry = json.load(f)
        if not isinstance(entry, dict):
            return emit({"error": "entry_load_failed"})
    except Exception as e:
        return emit({"error": f"entry_load_failed: {e}"})

    if not is_dtc_available():
        return emit({"error": "dtc_unavailable"})

    try:
        data = load_dataset(entry)
        if data is None:
            return emit({"error": "dataset_load_failed"})

        reorder_ms = 0.0
        reorder_method = ""
        if args.use_reorder:
            data, reorder_ms = run_reorder(entry, data, int(args.reorder_threshold))
            reorder_method = REORDER_METHOD

        rowptr = data["rowptr"].cuda()
        colind = data["colind"].cuda()
        vals = data["vals"].cuda()
        M = int(data["M"])
        nnz = int(colind.numel())
        torch.manual_seed(args.seed + int(args.N))
        B = torch.randn((M, int(args.N)), device="cuda", dtype=torch.float32)
        _ = vals
    except Exception as e:
        return emit({"error": f"dataset_prepare_failed: {e}"})

    try:
        module = load_dtc_module()
    except Exception as e:
        return emit({"error": f"dtc_load_failed: {e}"})

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = preprocess(module, rowptr, colind, M, nnz)
        torch.cuda.synchronize()
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return emit({"error": f"dtc_preprocess_failed: {e}"})

    best_ms = None
    best_variant = ""
    tuning_ms = 0.0
    variants = candidate_variants(int(args.N))

    try:
        torch.cuda.synchronize()
        select_t0 = time.perf_counter()

        for use_balance, exeplan in variants:
            for _ in range(max(0, args.warmup_iters)):
                _ = run_variant(module, state, B, M, nnz, use_balance, exeplan)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(max(1, args.timed_iters)):
                _ = run_variant(module, state, B, M, nnz, use_balance, exeplan)
            end.record()
            end.synchronize()

            avg_ms = float(start.elapsed_time(end) / max(1, args.timed_iters))
            tuning_ms += avg_ms * max(1, args.timed_iters)
            tag = f"{'bal' if use_balance else 'nobal'}_{exeplan}"
            if best_ms is None or avg_ms < best_ms:
                best_ms = avg_ms
                best_variant = tag

        torch.cuda.synchronize()
        variant_selection_ms = (time.perf_counter() - select_t0) * 1000.0
    except Exception as e:
        return emit({"error": f"dtc_variant_scan_failed: {e}"})

    if best_ms is None:
        return emit({"error": "dtc_no_valid_variant"})

    return emit({
        "dataset": entry.get("name", ""),
        "category": entry.get("category", ""),
        "M": M,
        "nnz": nnz,
        "N": int(args.N),
        "reorder_ms": reorder_ms,
        "reorder_method": reorder_method,
        "preprocess_ms": preprocess_ms,
        "variant_selection_ms": variant_selection_ms,
        "tuning_ms": tuning_ms,
        "kernel_ms": float(best_ms),
        "best_variant": best_variant,
        "variant_count": len(variants),
        "warmup_iters": int(args.warmup_iters),
        "timed_iters": int(args.timed_iters),
    })


if __name__ == "__main__":
    raise SystemExit(main())
