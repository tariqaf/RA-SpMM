#!/usr/bin/env python3
"""Build a strict feature/conversion/compute breakdown from fair measurements."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def load_feature_costs(path: Path, column: str) -> dict[str, float]:
    costs: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get(column, "")
            if value not in {"", None, "nan"}:
                costs[row["dataset"]] = float(value)
    return costs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-column", default="production_full_cpu_input_ms")
    parser.add_argument("--calls", type=int, default=400)
    parser.add_argument("--conv-out", required=True)
    parser.add_argument("--pipe-out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    feature_costs = load_feature_costs(Path(args.features), args.feature_column)
    calls = max(1, args.calls)
    conversion_rows: list[dict[str, object]] = []
    pipeline_rows: list[dict[str, object]] = []
    missing: list[tuple[str, int, str]] = []
    for row in csv.DictReader(Path(args.sweep).open(newline="")):
        if not truth(row.get("correct", False)):
            continue
        dataset, N, kernel = row["dataset"], int(row["N"]), row["kernel"]
        if kernel == "CUSPARSE":
            continue
        if dataset not in feature_costs:
            missing.append((dataset, N, kernel))
            continue
        feature_ms = feature_costs[dataset]
        conversion_ms = float(row["preprocess_ms"])
        first_exec_ms = float(row["cold_exec_ms"])
        warm_ms = float(row["ms_warm"])
        single_total = feature_ms + conversion_ms + first_exec_ms
        lifecycle_total = single_total + (calls - 1) * warm_ms
        conversion_rows.append({
            "dataset": dataset, "category": row["category"],
            "M": row["M"], "nnz": row["nnz"], "N": N, "kernel": kernel,
            "feature_source": args.feature_column,
            "feature_ms": feature_ms, "conversion_ms": conversion_ms,
            "first_exec_ms": first_exec_ms, "warm_exec_ms": warm_ms,
        })
        pipeline_rows.append({
            **conversion_rows[-1],
            "single_call_total_ms": single_total,
            "feature_pct_single": 100.0 * feature_ms / single_total,
            "conversion_pct_single": 100.0 * conversion_ms / single_total,
            "compute_pct_single": 100.0 * first_exec_ms / single_total,
            "calls": calls, "lifecycle_total_ms": lifecycle_total,
            "feature_pct_lifecycle": 100.0 * feature_ms / lifecycle_total,
            "conversion_pct_lifecycle": 100.0 * conversion_ms / lifecycle_total,
            "compute_pct_lifecycle": 100.0 * (
                first_exec_ms + (calls - 1) * warm_ms) / lifecycle_total,
        })

    if missing:
        raise SystemExit(
            f"Missing production feature costs for {len(missing)} rows: {missing[:10]}")
    if not pipeline_rows:
        raise SystemExit("No strict-correct rows available for pipeline breakdown")

    for path, rows in ((Path(args.conv_out), conversion_rows),
                       (Path(args.pipe_out), pipeline_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pipeline_rows:
        grouped[str(row["kernel"])].append(row)
    lines = [
        "# Feature, Conversion, And Compute Breakdown\n\n",
        f"Feature cost column: `{args.feature_column}`. Only strict-correct fair-sweep rows "
        "are included; missing costs terminate the analysis.\n\n",
        "| System | Points | Mean feature % (K=1) | Mean conversion % (K=1) | "
        f"Mean compute % (K={calls}) |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for kernel, rows in sorted(grouped.items()):
        count = len(rows)
        mean_feature = sum(float(row["feature_pct_single"]) for row in rows) / count
        mean_conversion = sum(float(row["conversion_pct_single"]) for row in rows) / count
        mean_compute = sum(float(row["compute_pct_lifecycle"]) for row in rows) / count
        lines.append(
            f"| {kernel} | {count} | {mean_feature:.3f} | "
            f"{mean_conversion:.3f} | {mean_compute:.3f} |\n")
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("".join(lines))
    print(f"Wrote {len(pipeline_rows)} strict pipeline rows and {summary}")


if __name__ == "__main__":
    main()
