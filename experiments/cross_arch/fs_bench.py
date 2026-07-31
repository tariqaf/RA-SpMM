"""
Drives FlashSparse over the RA-SpMM 26-graph suite on the RTX 4090.

FlashSparse (https://github.com/ParCIS/FlashSparse) exposes a CUDA SpMM built for
SM 89/90 tensor cores. Its Python entry points differ across commits, so the call
into FlashSparse is supplied as a `module:function` adapter. The surrounding
driver enforces strict correctness and matching warm/cold lifecycle comparisons.

Emits:
  --out          warm/cold timings, matching cuSPARSE speedups, and correctness
  --preproc-out  preprocessing CSV (format conversion only)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path

import torch

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    candidate for candidate in [THIS_FILE.parent, *THIS_FILE.parents]
    if (candidate / "ra_real_graph_eval.py").exists()
)
sys.path.insert(0, str(REPO_ROOT))
from ra_real_graph_eval import load_dataset, measure_ms  # noqa: E402
import ra_spmm  # noqa: E402  (for cuSPARSE reference)

WARMUP, TIMED = 50, 200


_ADAPTER = None


def failure_row(entry, M, nnz, N, error):
    return {
        "dataset": entry["name"], "category": entry.get("category", "?"),
        "M": M, "nnz": nnz, "N": N, "kernel": "FlashSparse",
        "status": "RUNTIME_ERROR", "ms_warm": None, "preprocess_ms": None,
        "cold_exec_ms": None, "ms_cold": None, "ms_cusparse_warm": None,
        "ms_cusparse_cold": None, "speedup_vs_cusparse_warm": None,
        "speedup_vs_cusparse_cold": None, "correct": False,
        "soft_fail": False, "hard_fail": False, "max_error": None,
        "tolerance": None, "error": error,
    }


def configure_adapter(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--adapter must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"FlashSparse adapter is not callable: {spec}")
    return function


def flashsparse_prepare(rowptr, colind, vals, M, K, N, B):
    """Return `(state, run_fn)` from the configured external adapter."""
    if _ADAPTER is None:
        raise RuntimeError("FlashSparse adapter was not configured")
    state, run_fn = _ADAPTER(rowptr, colind, vals, M, K, N, B)
    if not callable(run_fn):
        raise TypeError("FlashSparse adapter must return (state, callable)")
    return state, run_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default="paper_datasets.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preproc-out", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--cold-iters", type=int, default=10)
    ap.add_argument("--adapter", required=True,
                    help="Import target implementing prepare(...), as module:function")
    args = ap.parse_args()
    global _ADAPTER
    _ADAPTER = configure_adapter(args.adapter)

    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    rows, preproc_rows = [], []
    build_note = []

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        mat = load_dataset(entry)
        if mat is None:
            continue
        M, K = mat["M"], mat["K"]
        rp = mat["rowptr"].cuda().int(); ci = mat["colind"].cuda().int(); v = mat["vals"].cuda().float()
        for N in [int(n) for n in entry.get("Ns", [64, 128, 256, 512])]:
            B = torch.randn(M, N, device="cuda")
            C_ref = ra_spmm.spmm_cusparse(rp, ci, v, B)
            cus_warm = ra_spmm.benchmark_cusparse(
                rp, ci, v, B, args.warmup, args.timed)
            cus_cold = ra_spmm.benchmark_cusparse_cold(
                rp, ci, v, B, args.cold_iters)
            ms_cus = float(cus_warm["exec_ms"])
            try:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                handle, run_fn = flashsparse_prepare(rp, ci, v, M, K, N, B)
                torch.cuda.synchronize(); _warm_setup_ms = (time.perf_counter() - t0) * 1e3
                C = run_fn()
                max_err = (C.float() - C_ref).abs().max().item()
                ms_fs = measure_ms(run_fn, args.warmup, args.timed)
                cold_setup_total = 0.0
                cold_exec_total = 0.0
                for _ in range(max(1, args.cold_iters)):
                    torch.cuda.synchronize()
                    cold_start = time.perf_counter()
                    cold_handle, cold_run = flashsparse_prepare(rp, ci, v, M, K, N, B)
                    torch.cuda.synchronize()
                    cold_setup_total += (time.perf_counter() - cold_start) * 1e3
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(); cold_output = cold_run(); end.record(); end.synchronize()
                    cold_exec_total += start.elapsed_time(end)
                    del cold_output, cold_handle
                cold_count = float(max(1, args.cold_iters))
                cold_preprocess_ms = cold_setup_total / cold_count
                cold_exec_ms = cold_exec_total / cold_count
                ms_fs_cold = cold_preprocess_ms + cold_exec_ms
                max_row_nnz = max(1, int((rp[1:] - rp[:-1]).max().item()))
                tolerance = 1.0e-3 * math.sqrt(max_row_nnz) * 10.0
                correct = max_err <= tolerance and max_err < 1.0
                rows.append({
                    "dataset": entry["name"], "category": entry.get("category", "?"),
                    "M": M, "nnz": int(rp[-1].item()), "N": N,
                    "kernel": "FlashSparse", "status": "OK" if correct else "INCORRECT",
                    "ms_warm": round(ms_fs, 6),
                    "preprocess_ms": round(cold_preprocess_ms, 6),
                    "cold_exec_ms": round(cold_exec_ms, 6),
                    "ms_cold": round(ms_fs_cold, 6),
                    "ms_cusparse_warm": round(ms_cus, 6),
                    "ms_cusparse_cold": round(float(cus_cold["total_ms"]), 6),
                    "speedup_vs_cusparse_warm": round(ms_cus / ms_fs, 6) if correct else "",
                    "speedup_vs_cusparse_cold": round(float(cus_cold["total_ms"]) / ms_fs_cold, 6) if correct else "",
                    "correct": correct, "soft_fail": tolerance < max_err < 1.0,
                    "hard_fail": max_err >= 1.0,
                    "max_error": round(max_err, 8), "tolerance": tolerance,
                    "error": "",
                })
                preproc_rows.append({
                    "dataset": entry["name"], "N": N,
                    "preprocess_ms": round(cold_preprocess_ms, 6),
                    "cold_iters": args.cold_iters,
                })
                print(f"  {entry['name']:<20s} N={N:<4d} FS={ms_fs:.4f}ms preproc={cold_preprocess_ms:.2f}ms "
                      f"({ms_cus/ms_fs:.2f}x vs cuSPARSE)")
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                build_note.append(f"{entry['name']} N={N}: RUNTIME {detail}")
                rows.append(failure_row(entry, M, int(rp[-1].item()), N, detail))
            del B
            torch.cuda.empty_cache()

    if rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if preproc_rows:
        Path(args.preproc_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.preproc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preproc_rows[0].keys())); w.writeheader(); w.writerows(preproc_rows)
    if build_note:
        note_path = Path(args.out).parent / "FLASHSPARSE_BUILD_NOTE.txt"
        note_path.write_text("\n".join(build_note))
        print(f"\n{len(build_note)} graphs not run — see {note_path}")


if __name__ == "__main__":
    main()
