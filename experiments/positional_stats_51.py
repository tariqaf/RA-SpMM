#!/usr/bin/env python3
"""Task 2: six CSR positional statistics for all 51 suite graphs (CPU-only).

Definition applied EXACTLY as specified (including the common division by M;
all suite matrices are square):
    idx    = A.indices.astype(float64)            # column index of each nonzero
    rowlen = diff(A.indptr)
    row_id = repeat(arange(M), rowlen).astype(float64)
    col_mean_n, col_std_n = idx.mean()/M,    idx.std()/M
    col_min_n,  col_max_n = idx.min()/M,     idx.max()/M
    row_mean_n, row_std_n = row_id.mean()/M, row_id.std()/M
Writes fgcs_results/revision/tf32/positional_stats_51.csv.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ra_real_graph_eval import load_dataset  # noqa: E402

MANIFEST = REPO / "fgcs_results/paper_combined_datasets.json"
OUT = REPO / "fgcs_results/revision/tf32/positional_stats_51.csv"


def main():
    man = json.loads(MANIFEST.read_text())
    entries = [e for e in (man["datasets"] if isinstance(man, dict) else man) if e.get("enabled", True)]
    rows = []
    for e in entries:
        data = load_dataset(e)
        if data is None:
            print(f"[FAIL] {e['name']}: load returned None"); continue
        indptr = np.asarray(data["rowptr"])
        indices = np.asarray(data["colind"])
        M = int(data["M"]); K = int(data.get("K", M)); nnz = int(indices.shape[0])
        idx = indices.astype(np.float64)
        rowlen = np.diff(indptr)
        row_id = np.repeat(np.arange(M), rowlen).astype(np.float64)
        rows.append({
            "dataset": e["name"], "M": M, "K": K, "nnz": nnz,
            "col_mean_n": idx.mean() / M, "col_std_n": idx.std() / M,
            "col_min_n": idx.min() / M, "col_max_n": idx.max() / M,
            "row_mean_n": row_id.mean() / M, "row_std_n": row_id.std() / M,
        })
        print(f"  {e['name']:26s} M={M:>9d} nnz={nnz:>11d} "
              f"col_mean_n={rows[-1]['col_mean_n']:.4f} row_std_n={rows[-1]['row_std_n']:.4f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "M", "K", "nnz", "col_mean_n",
                                          "col_std_n", "col_min_n", "col_max_n",
                                          "row_mean_n", "row_std_n"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)}/51 graphs)")


if __name__ == "__main__":
    main()
