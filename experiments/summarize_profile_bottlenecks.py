#!/usr/bin/env python3
"""Aggregate per-pair Nsight metrics into kernel/regime bottlenecks."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

METRICS = [
    "tc_pipe_pct", "achieved_occupancy_pct", "dram_throughput_pct",
    "mem_workload_pct", "cuda_core_achieved_gflops",
    "cuda_core_pct_fp32_peak", "cuda_core_ai_flop_per_byte",
]


def mean(rows, column):
    values = [float(row[column]) for row in rows if row.get(column, "") not in {"", "nan"}]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    source_rows = list(csv.DictReader(Path(args.summary).open(newline="")))
    stall_columns = sorted({key for row in source_rows for key in row if key.startswith("stall_")})
    grouped = defaultdict(list)
    for row in source_rows:
        grouped[(row["kernel"], int(row["N"]), row["category"])].append(row)

    output_rows = []
    for (kernel, N, category), rows in sorted(grouped.items()):
        stalls = sorted(
            ((column.removeprefix("stall_"), mean(rows, column))
             for column in stall_columns), key=lambda item: item[1], reverse=True)
        output_rows.append({
            "kernel": kernel, "N": N, "category": category, "profiles": len(rows),
            **{f"mean_{metric}": mean(rows, metric) for metric in METRICS},
            "hmma_insts_total": sum(int(float(row.get("hmma_insts", 0) or 0)) for row in rows),
            "dominant_stall": stalls[0][0] if stalls else "",
            "dominant_stall_value": stalls[0][1] if stalls else 0.0,
            "second_stall": stalls[1][0] if len(stalls) > 1 else "",
            "second_stall_value": stalls[1][1] if len(stalls) > 1 else 0.0,
            "third_stall": stalls[2][0] if len(stalls) > 2 else "",
            "third_stall_value": stalls[2][1] if len(stalls) > 2 else 0.0,
        })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# Profiling Bottlenecks\n\n",
        "Means are grouped by implementation, feature width, and structural category. ",
        "Stall values preserve Nsight's `warps_issue_stalled_*_per_issue_active` metric ",
        "and are not percentages.\n\n",
        "| Kernel | N | Category | Profiles | Occupancy % | DRAM % | AI | HMMA | Dominant stalls |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in output_rows:
        stalls = (f"{row['dominant_stall']} {row['dominant_stall_value']:.1f}; "
                  f"{row['second_stall']} {row['second_stall_value']:.1f}; "
                  f"{row['third_stall']} {row['third_stall_value']:.1f}")
        lines.append(
            f"| {row['kernel']} | {row['N']} | {row['category']} | {row['profiles']} | "
            f"{row['mean_achieved_occupancy_pct']:.2f} | {row['mean_dram_throughput_pct']:.2f} | "
            f"{row['mean_cuda_core_ai_flop_per_byte']:.3f} | {row['hmma_insts_total']} | {stalls} |\n")
    Path(args.output_md).write_text("".join(lines))
    print(f"Wrote {output_csv} and {args.output_md} ({len(output_rows)} groups)")


if __name__ == "__main__":
    main()
