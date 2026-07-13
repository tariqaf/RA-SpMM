#!/usr/bin/env python3
"""E3 Step 0: per-graph ME-BCRS vector fill factors (sizes the ZC-BCRS win).

For every dataset in the manifest, build the TC_DIRECT plan once and report:
  - avg vector fill (nnz / (vectors*8)) and zero-padding share of value bytes
  - plan byte breakdown (values vs indices) and the estimated ZC-BCRS plan
    size: packed nonzero halves + 1-byte fill mask per vector + unchanged atox.

No kernels are timed; this is measurement only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402  (import order: torch before ra_spmm)
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO / "fgcs_results" / "paper_combined_datasets.json"))
    ap.add_argument("--out", default=str(REPO / "fgcs_results" / "revision" / "tf32" / "zc_fill_report.csv"))
    args = ap.parse_args()

    datasets = json.loads(Path(args.datasets_file).read_text())
    if isinstance(datasets, dict):
        datasets = datasets.get("datasets", [])

    rows = []
    for entry in datasets:
        name = entry["name"]
        mat = load_dataset(entry)
        if mat is None:
            print(f"skip {name}: file not found")
            continue
        M = mat["M"]
        K = mat.get("K", M)
        rowptr = mat["rowptr"].contiguous().int()
        colind = mat["colind"].contiguous().int()
        vals = mat["vals"].contiguous().float()
        nnz = int(rowptr[-1].item())

        plan = ra_spmm.make_tc_direct_plan(rowptr, colind, vals, M, K, 128)
        blocks = int(plan.num_tc_tiles)
        vectors = blocks * 8
        fill = float(plan.avg_tc_tile_density)  # nnz / (vectors * 8)
        val_bytes = blocks * 64 * 2
        idx_bytes = vectors * 4
        plan_bytes = int(plan.plan_bytes)
        # ZC-BCRS estimate: packed nonzero halves + 1B fill mask per vector.
        zc_val_bytes = nnz * 2 + vectors * 1
        zc_plan_bytes = plan_bytes - val_bytes + zc_val_bytes
        rows.append({
            "dataset": name,
            "category": entry.get("category", "?"),
            "M": M, "nnz": nnz,
            "windows": int(plan.num_groups),
            "vectors": vectors,
            "avg_vector_fill": round(fill * 8, 3),        # nnz per 8-slot vector
            "fill_fraction": round(fill, 4),              # share of slots nonzero
            "zero_padding_pct": round((1 - fill) * 100, 1),
            "plan_MB": round(plan_bytes / 1e6, 2),
            "value_bytes_pct": round(100 * val_bytes / max(1, plan_bytes), 1),
            "zc_plan_MB": round(zc_plan_bytes / 1e6, 2),
            "zc_shrink_x": round(plan_bytes / max(1, zc_plan_bytes), 2),
        })
        print(f"{name:28s} fill={fill:.3f} plan={plan_bytes/1e6:9.2f}MB "
              f"zc={zc_plan_bytes/1e6:9.2f}MB shrink={plan_bytes/max(1,zc_plan_bytes):.2f}x")
        del plan
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
