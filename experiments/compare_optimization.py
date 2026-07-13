#!/usr/bin/env python3
"""Join optimization timing and Nsight data into an auditable before/after log."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, int, str]:
    return row["dataset"], int(row["N"]), row["kernel"]


def valid_number(row: dict[str, str] | None, column: str) -> float | None:
    if not row or row.get(column, "") in {"", "nan", "None"}:
        return None
    return float(row[column])


def ratio(before: float | None, after: float | None) -> float | None:
    return before / after if before is not None and after not in {None, 0.0} else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-before", required=True)
    parser.add_argument("--timing-after", required=True)
    parser.add_argument("--profile-before", required=True)
    parser.add_argument("--profile-after", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    timing_before = {key(row): row for row in load(args.timing_before)}
    timing_after = {key(row): row for row in load(args.timing_after)}
    profile_before = {key(row): row for row in load(args.profile_before)}
    profile_after = {key(row): row for row in load(args.profile_after)}

    output: list[dict[str, object]] = []
    for item in sorted(set(timing_before) & set(timing_after)):
        if item[2] == "CUSPARSE":
            continue
        old_timing, new_timing = timing_before[item], timing_after[item]
        old_profile, new_profile = profile_before.get(item), profile_after.get(item)
        old_ms = valid_number(old_timing, "ms_warm")
        new_ms = valid_number(new_timing, "ms_warm")
        old_duration = valid_number(old_profile, "duration_us")
        new_duration = valid_number(new_profile, "duration_us")
        output.append({
            "dataset": item[0], "N": item[1], "kernel": item[2],
            "strict_correct_before": old_timing["correct"],
            "strict_correct_after": new_timing["correct"],
            "warm_ms_before": old_ms, "warm_ms_after": new_ms,
            "warm_speedup_before_over_after": ratio(old_ms, new_ms),
            "primary_kernel_us_before": old_duration,
            "primary_kernel_us_after": new_duration,
            "kernel_speedup_before_over_after": ratio(old_duration, new_duration),
            "occupancy_pct_before": valid_number(old_profile, "achieved_occupancy_pct"),
            "occupancy_pct_after": valid_number(new_profile, "achieved_occupancy_pct"),
            "dram_pct_before": valid_number(old_profile, "dram_throughput_pct"),
            "dram_pct_after": valid_number(new_profile, "dram_throughput_pct"),
            "barrier_stall_before": valid_number(old_profile, "stall_barrier"),
            "barrier_stall_after": valid_number(new_profile, "stall_barrier"),
            "long_scoreboard_before": valid_number(old_profile, "stall_long_scoreboard"),
            "long_scoreboard_after": valid_number(new_profile, "stall_long_scoreboard"),
        })

    if not output:
        raise SystemExit("No matching optimization rows")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in output:
        grouped[str(row["kernel"])].append(row)
    lines = [
        "# Optimization Before/After\n\n",
        "Ratios above 1.0 favor the after version. Timing includes the corrected asynchronous "
        "Python execution API; kernel duration is the primary Nsight kernel and isolates the "
        "launch-shape change. Stall values retain Nsight's per-issue-active units.\n\n",
        "| Kernel | Timing pairs | Timing geomean | Timing wins | Profile pairs | Kernel geomean | Kernel wins |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for kernel, rows in sorted(grouped.items()):
        timing_ratios = [float(row["warm_speedup_before_over_after"]) for row in rows
                         if row["warm_speedup_before_over_after"] is not None]
        kernel_ratios = [float(row["kernel_speedup_before_over_after"]) for row in rows
                         if row["kernel_speedup_before_over_after"] is not None]
        timing_gm = math.exp(sum(math.log(value) for value in timing_ratios) / len(timing_ratios))
        kernel_gm = (math.exp(sum(math.log(value) for value in kernel_ratios) / len(kernel_ratios))
                     if kernel_ratios else float("nan"))
        lines.append(
            f"| {kernel} | {len(timing_ratios)} | {timing_gm:.6f}x | "
            f"{sum(value > 1.0 for value in timing_ratios)}/{len(timing_ratios)} | "
            f"{len(kernel_ratios)} | {kernel_gm:.6f}x | "
            f"{sum(value > 1.0 for value in kernel_ratios)}/{len(kernel_ratios)} |\n")
    lines.append("\nSee the CSV for per-graph occupancy, DRAM, barrier, and long-scoreboard changes.\n")
    Path(args.output_md).write_text("".join(lines))
    print(f"Wrote {output_path} and {args.output_md} ({len(output)} timing pairs)")


if __name__ == "__main__":
    main()
