#!/usr/bin/env python3
"""Export the full diagnostic C++ feature vector for selector experiments.

The deployed MAIN router intentionally computes only its four rule inputs.
This exporter uses the separate FULL portfolio so learned-selector experiments
retain the documented 34-feature diagnostic input space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch  # Load PyTorch shared libraries before the extension.

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ra_spmm
from ra_real_graph_eval import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--portfolio", choices=("FULL",), default="FULL")
    args = parser.parse_args()

    entries = json.loads(Path(args.datasets_file).read_text())["datasets"]
    rows = []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        matrix = load_dataset(entry)
        if matrix is None:
            continue
        rowptr = matrix["rowptr"].contiguous().int()
        colind = matrix["colind"].contiguous().int()
        vals = matrix["vals"].contiguous().float()
        M, K = int(matrix["M"]), int(matrix["K"])
        plan = ra_spmm.make_router_plan(
            rowptr, colind, vals, M, K, args.N, args.portfolio)
        features = {f"feature_{key}": value for key, value in dict(plan["feature_values"]).items()}
        rows.append({
            "dataset": entry["name"], "category": entry.get("category", ""),
            "M": M, "K": K, "nnz": int(colind.numel()), "feature_N": args.N,
            **features,
        })
        print(f"  {entry['name']}: {len(features)} diagnostic features", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row},
                        key=lambda key: (not key in {"dataset", "category", "M", "K", "nnz", "feature_N"}, key))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} graphs)")


if __name__ == "__main__":
    main()
