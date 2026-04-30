#!/usr/bin/env python3
"""
Helpers for running DTC's external TCA reordering once per graph, caching the
result on disk, and reusing the reordered CSR + permutation across N values.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
REORDER_SCRIPT = REPO_ROOT / "external" / "DTC-SpMM-ASPLOS24-upstream" / "reordering" / "TCA_reorder.py"
REORDER_METHOD = "DTC_TCA_REORDER"
REORDER_METHOD_NOTE = (
    "Reordering uses DTC's custom TCA reordering tool "
    "(MinHash-LSH/Jaccard clustering followed by cache-aware cluster ordering) "
    "from external/DTC-SpMM-ASPLOS24-upstream/reordering/TCA_reorder.py."
)
DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "dtc_reordered"


def reorder_version() -> str:
    repo_dir = REORDER_SCRIPT.parent.parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def csr_to_src_dst(rowptr: torch.Tensor, colind: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    rowptr_np = rowptr.cpu().numpy().astype(np.int64, copy=False)
    colind_np = colind.cpu().numpy().astype(np.int32, copy=False)
    row_counts = rowptr_np[1:] - rowptr_np[:-1]
    src = np.repeat(np.arange(len(row_counts), dtype=np.int32), row_counts)
    return src, colind_np


def _cache_key(data: Dict[str, object], threshold: int) -> str:
    h = hashlib.sha256()
    h.update(data["rowptr"].cpu().numpy().astype(np.int32, copy=False).tobytes())
    h.update(data["colind"].cpu().numpy().astype(np.int32, copy=False).tobytes())
    h.update(REORDER_METHOD.encode("utf-8"))
    h.update(str(int(threshold)).encode("utf-8"))
    return h.hexdigest()


def _cache_paths(cache_dir: str, key: str) -> Dict[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.join(cache_dir, f"{key}_{REORDER_METHOD.lower()}")
    return {
        "input_npz": stem + ".input.npz",
        "output_npz": stem + ".reorder.npz",
        "output_perm_npz": stem + ".reorder_id.npz",
        "meta_json": stem + ".meta.json",
    }


def save_input_npz(path: str, data: Dict[str, object]) -> None:
    src, dst = csr_to_src_dst(data["rowptr"], data["colind"])
    np.savez(path, src_li=src, dst_li=dst, num_nodes=int(data["M"]))


def load_reordered_npz(path: str) -> Dict[str, object]:
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


def load_perm(path: str) -> np.ndarray:
    data = np.load(path)
    perm = np.asarray(data["reorder_id"], dtype=np.int64)
    return perm


def load_cache_metadata(meta_json: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(meta_json):
        return None
    with open(meta_json) as f:
        return json.load(f)


def save_cache_metadata(meta_json: str, payload: Dict[str, object]) -> None:
    with open(meta_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def reorder_once(
    entry: Dict[str, object],
    data: Dict[str, object],
    threshold: int,
    cache_dir: str | None = None,
    python_exe: str | None = None,
    timeout_s: Optional[int] = None,
) -> Dict[str, object]:
    if not REORDER_SCRIPT.exists():
        raise FileNotFoundError(f"reorder_script_missing: {REORDER_SCRIPT}")

    python_exe = python_exe or sys.executable
    cache_dir = str(cache_dir or DEFAULT_CACHE_DIR)
    dataset = str(entry.get("name", "graph"))
    key = _cache_key(data, threshold)
    paths = _cache_paths(cache_dir, key)
    version = reorder_version()

    meta = load_cache_metadata(paths["meta_json"])
    if (
        meta is not None
        and os.path.exists(paths["output_npz"])
        and os.path.exists(paths["output_perm_npz"])
        and meta.get("reorder_method") == REORDER_METHOD
        and meta.get("reorder_threshold") == int(threshold)
    ):
        return {
            "reordered_npz": paths["output_npz"],
            "reorder_perm_npz": paths["output_perm_npz"],
            "reorder_ms": float(meta.get("reorder_ms", 0.0)),
            "reorder_method": REORDER_METHOD,
            "reorder_version": str(meta.get("reorder_version", version)),
            "cache_key": key,
            "cache_hit": True,
            "M": int(meta.get("M", entry.get("M", 0))),
            "nnz": int(meta.get("nnz", entry.get("nnz", 0))),
        }

    save_input_npz(paths["input_npz"], data)
    cmd = [
        python_exe,
        str(REORDER_SCRIPT),
        "--dataset", dataset,
        "--thres", str(int(threshold)),
        "--input_npz", paths["input_npz"],
        "--output_npz", paths["output_npz"],
        "--output_perm_npz", paths["output_perm_npz"],
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=timeout_s if timeout_s and timeout_s > 0 else None,
    )
    reorder_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        lines = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip().splitlines()
        detail = lines[-1] if lines else "reorder child failed"
        raise RuntimeError(f"reorder_failed: rc={proc.returncode} detail={detail}")
    if not os.path.exists(paths["output_npz"]):
        raise RuntimeError("reorder_output_missing")

    reordered = load_reordered_npz(paths["output_npz"])
    meta_payload = {
        "dataset": dataset,
        "reorder_method": REORDER_METHOD,
        "reorder_version": version,
        "reorder_threshold": int(threshold),
        "reorder_ms": reorder_ms,
        "M": int(reordered["M"]),
        "nnz": int(reordered["nnz"]),
        "cache_key": key,
    }
    save_cache_metadata(paths["meta_json"], meta_payload)
    return {
        "reordered_npz": paths["output_npz"],
        "reorder_perm_npz": paths["output_perm_npz"],
        "reorder_ms": reorder_ms,
        "reorder_method": REORDER_METHOD,
        "reorder_version": version,
        "cache_key": key,
        "cache_hit": False,
        "M": int(reordered["M"]),
        "nnz": int(reordered["nnz"]),
    }
