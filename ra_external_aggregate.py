#!/usr/bin/env python3
"""Join strict-gated RA-SpMM, cuSPARSE, PyG, and DTC lifecycle results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from ra_router_eval import KERNELS, simple_router


def truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def finite_float(row: dict[str, str], key: str):
    try:
        value = float(row.get(key, ""))
        return value if value > 0.0 else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-results", required=True)
    parser.add_argument("--external-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regime", choices=("warm", "cold"), default="warm")
    parser.add_argument("--expected", type=int, default=192)
    args = parser.parse_args()

    time_column = f"ms_{args.regime}"
    grouped = defaultdict(dict)
    for row in csv.DictReader(Path(args.kernel_results).open(newline="")):
        if truth(row.get("correct", False)):
            grouped[(row["dataset"], int(row["N"]))][row["kernel"]] = row
    complete = {key: rows for key, rows in grouped.items()
                if "CUSPARSE" in rows and all(kernel in rows for kernel in KERNELS)}
    if len(complete) != args.expected:
        raise SystemExit(f"Expected {args.expected} complete strict configurations, got {len(complete)}")

    external = {(row["dataset"], int(row["N"])): row for row in
                csv.DictReader(Path(args.external_results).open(newline=""))}
    output_rows = []
    for (dataset, N), rows in sorted(complete.items()):
        sample = rows[KERNELS[0]]
        M, nnz = int(sample["M"]), int(sample["nnz"])
        times = {kernel: float(rows[kernel][time_column]) for kernel in KERNELS}
        cusp = float(rows["CUSPARSE"][time_column])
        oracle = min(times, key=times.get)
        router = simple_router(nnz / max(M, 1), float(sample["cv_d"]), M, N, nnz)
        row = {
            "dataset": dataset, "category": sample.get("category", ""),
            "M": M, "nnz": nnz, "N": N, "regime": args.regime,
            "router_kernel": router, "router_ms": times[router],
            "router_speedup_vs_cusparse": cusp / times[router],
            "oracle_kernel": oracle, "oracle_ms": times[oracle],
            "oracle_speedup_vs_cusparse": cusp / times[oracle],
            "router_oracle_ratio": times[oracle] / times[router],
        }
        ext = external.get((dataset, N), {})
        ext_cusp = finite_float(ext, f"cusparse_ms_{args.regime}")
        for system in ("pyg", "dtc"):
            elapsed = finite_float(ext, f"{system}_ms_{args.regime}")
            correct = truth(ext.get(f"{system}_correct", False))
            row[f"{system}_correct"] = correct
            row[f"{system}_ms"] = elapsed if correct and elapsed is not None else ""
            row[f"{system}_speedup_vs_cusparse"] = (
                ext_cusp / elapsed if correct and elapsed and ext_cusp else "")
        output_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {output} ({len(output_rows)} {args.regime} configurations)")


if __name__ == "__main__":
    main()
