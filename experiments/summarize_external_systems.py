#!/usr/bin/env python3
"""Summarize strict matching-lifecycle results for external systems."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else 0.0


def add_long(rows, source: Path, system: str, warm_field: str, cold_field: str):
    if not source.exists():
        return
    data = list(csv.DictReader(source.open(newline="")))
    valid = [row for row in data if truth(row.get("correct", False))]
    rows.append({
        "system": system, "strict_correct_points": len(valid), "attempted_points": len(data),
        "warm_geomean_vs_cusparse": geomean([
            float(row[warm_field]) for row in valid if row.get(warm_field, "") not in {"", "nan"}]),
        "cold_geomean_vs_cusparse": geomean([
            float(row[cold_field]) for row in valid if row.get(cold_field, "") not in {"", "nan"}]),
        "scope": "matching warm/warm and cold/cold",
    })


def add_wide(rows, source: Path, system: str):
    if not source.exists():
        return
    data = list(csv.DictReader(source.open(newline="")))
    prefix = system.lower()
    valid = [row for row in data if truth(row.get(f"{prefix}_correct", False))]
    rows.append({
        "system": system, "strict_correct_points": len(valid), "attempted_points": len(data),
        "warm_geomean_vs_cusparse": geomean([
            float(row[f"{prefix}_speedup_vs_cusparse_warm"]) for row in valid
            if row.get(f"{prefix}_speedup_vs_cusparse_warm", "") not in {"", "nan"}]),
        "cold_geomean_vs_cusparse": geomean([
            float(row[f"{prefix}_speedup_vs_cusparse_cold"]) for row in valid
            if row.get(f"{prefix}_speedup_vs_cusparse_cold", "") not in {"", "nan"}]),
        "scope": "matching warm/warm and cold/cold",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    fair = Path(args.fair_dir)
    rows = []
    add_long(rows, fair / "hcspmm.csv", "HC-SpMM",
             "speedup_vs_cusparse_warm", "speedup_vs_cusparse_cold")
    add_long(rows, fair / "mp_spmm.csv", "MP-SpMM",
             "speedup_warm_vs_cusparse", "speedup_cold_vs_cusparse")
    add_long(rows, fair / "cublas_small.csv", "cuBLAS dense",
             "speedup_vs_cusparse_warm", "speedup_vs_cusparse_cold")
    add_long(rows, fair / "flashsparse.csv", "FlashSparse",
             "speedup_vs_cusparse_warm", "speedup_vs_cusparse_cold")
    add_wide(rows, fair / "pyg_dtc.csv", "PyG")
    add_wide(rows, fair / "pyg_dtc.csv", "DTC")
    if not rows:
        raise SystemExit("No external result CSVs found")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# External Systems\n\n",
        "Only strict-correct points contribute. Coverage is reported because native-width, "
        "crash, timeout, and memory limitations differ by system.\n\n",
        "| System | Correct / attempted | Warm vs cuSPARSE | Cold vs cuSPARSE |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['strict_correct_points']} / {row['attempted_points']} | "
            f"{row['warm_geomean_vs_cusparse']:.6f}x | "
            f"{row['cold_geomean_vs_cusparse']:.6f}x |\n")
    Path(args.output_md).write_text("".join(lines))
    print(f"Wrote {output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
