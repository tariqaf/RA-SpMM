#!/usr/bin/env python3
"""Evaluate measured dense-GEMM outcomes without adding a fitted router rule.

The script joins the corrected cuBLAS probe to the corrected six-kernel sweep.
It reports where dense GEMM wins or loses against the production router in each
matching lifecycle. It deliberately does not derive thresholds from test points.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_router_eval import KERNELS, simple_router


def truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cublas", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    custom = defaultdict(dict)
    for row in csv.DictReader(Path(args.sweep).open(newline="")):
        if row.get("kernel") in KERNELS and truth(row.get("correct", False)):
            custom[(row["dataset"], int(row["N"]))][row["kernel"]] = row

    output_rows = []
    for dense in csv.DictReader(Path(args.cublas).open(newline="")):
        if not truth(dense.get("correct", False)):
            continue
        key = (dense["dataset"], int(dense["N"]))
        kernels = custom.get(key, {})
        if set(kernels) != set(KERNELS):
            continue
        sample = kernels[KERNELS[0]]
        M, nnz, N = int(sample["M"]), int(sample["nnz"]), key[1]
        routed = simple_router(nnz / max(M, 1), float(sample["cv_d"]), M, N, nnz)
        for regime in ("warm", "cold"):
            dense_ms = float(dense[f"ms_{regime}"])
            sparse_ms = float(kernels[routed][f"ms_{regime}"])
            output_rows.append({
                "dataset": key[0], "N": N, "M": M, "nnz": nnz,
                "regime": regime, "router_kernel": routed,
                "router_ms": sparse_ms, "cublas_ms": dense_ms,
                "cublas_vs_router": sparse_ms / dense_ms,
                "cublas_wins": dense_ms < sparse_ms,
                "selection_note": "measured what-if only; production router unchanged",
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output_rows:
        raise SystemExit("No complete, strictly correct joined configurations")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    wins = sum(truth(row["cublas_wins"]) for row in output_rows)
    print(f"Wrote {output}: {wins}/{len(output_rows)} measured dense wins")


if __name__ == "__main__":
    main()
