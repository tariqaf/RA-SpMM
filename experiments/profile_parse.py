"""Parse Nsight Compute raw exports without inferring FLOP/s from SM%."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_DIR = REPO_ROOT / "fgcs_results" / "revision" / "fair" / "profile"
FP32_PEAK_GFLOPS_RTX3090 = 35_580.0
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
EXCLUDED_KERNEL_FRAGMENTS = (
    "FillFunctor", "vectorized_elementwise", "DeviceRadix", "DeviceScan",
    "DeviceReduce", "distribution_", "elementwise_kernel",
)


def number(row: dict[str, str], units: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name, "")
        if value in ("", None, "N/A", "nan", "no data"):
            continue
        try:
            parsed = float(str(value).replace(",", ""))
            unit = units.get(name, "")
            if unit == "Kbyte":
                parsed *= 1e3
            elif unit == "Mbyte":
                parsed *= 1e6
            elif unit == "Gbyte":
                parsed *= 1e9
            return parsed
        except ValueError:
            continue
    return 0.0


def parse_launch(row: dict[str, str], units: dict[str, str],
                 meta: dict[str, object]) -> dict[str, object]:
    duration_value = number(row, units, "gpu__time_duration.sum")
    duration_unit = units.get("gpu__time_duration.sum", "usecond")
    if duration_unit == "nsecond":
        duration_us = duration_value / 1e3
    elif duration_unit == "msecond":
        duration_us = duration_value * 1e3
    elif duration_unit == "second":
        duration_us = duration_value * 1e6
    else:
        duration_us = duration_value
    dram_bytes = number(row, units, "dram__bytes.sum")
    if dram_bytes == 0.0:
        dram_bytes = (number(row, units, "dram__bytes_read.sum") +
                      number(row, units, "dram__bytes_write.sum"))

    ffma = number(row, units, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
    fadd = number(row, units, "sm__sass_thread_inst_executed_op_fadd_pred_on.sum")
    fmul = number(row, units, "sm__sass_thread_inst_executed_op_fmul_pred_on.sum")
    hfma = number(row, units, "sm__sass_thread_inst_executed_op_hfma_pred_on.sum")
    hadd = number(row, units, "sm__sass_thread_inst_executed_op_hadd_pred_on.sum")
    hmul = number(row, units, "sm__sass_thread_inst_executed_op_hmul_pred_on.sum")
    cuda_flops = 2.0 * (ffma + hfma) + fadd + fmul + hadd + hmul
    hmma_insts = number(row, units, "sm__inst_executed_pipe_tensor_op_hmma.sum")
    achieved_gflops = cuda_flops / duration_us / 1e3 if duration_us > 0.0 else 0.0
    arithmetic_intensity = cuda_flops / dram_bytes if dram_bytes > 0.0 else 0.0

    parsed: dict[str, object] = {
        **meta,
        "launch_id": row.get("ID", ""),
        "profiled_kernel": row.get("Kernel Name", ""),
        "duration_us": duration_us,
        "tc_pipe_pct": number(
            row, units, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
        "achieved_occupancy_pct": number(
            row, units, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "dram_throughput_pct": number(
            row, units, "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_throughput_pct": number(
            row, units, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "mem_workload_pct": number(
            row, units, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_bytes": dram_bytes,
        "cuda_core_flops": cuda_flops,
        "cuda_core_achieved_gflops": achieved_gflops,
        "cuda_core_pct_fp32_peak": achieved_gflops / FP32_PEAK_GFLOPS_RTX3090 * 100.0,
        "cuda_core_ai_flop_per_byte": arithmetic_intensity,
        "hmma_insts": int(hmma_insts),
        "flop_rate_scope": "CUDA-core counters; HMMA reported separately",
    }
    for metric, value in row.items():
        if metric.startswith(STALL_PREFIX) and metric.endswith(STALL_SUFFIX):
            reason = metric[len(STALL_PREFIX):-len(STALL_SUFFIX)]
            parsed[f"stall_{reason}"] = number(row, units, metric)
    return parsed


def write_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    args = parser.parse_args()
    profile_dir = Path(args.profile_dir)

    summary: list[dict[str, object]] = []
    all_stall_columns: set[str] = set()
    parsed_by_pair: list[tuple[Path, list[dict[str, object]]]] = []
    for meta_path in sorted(profile_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        raw_path = profile_dir / str(meta["raw_csv"])
        if not raw_path.exists():
            continue
        with raw_path.open(newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        units = raw_rows[0] if raw_rows and not raw_rows[0].get("ID") else {}
        launches = [parse_launch(row, units, meta) for row in raw_rows
                    if row.get("Kernel Name") and
                    not any(fragment in row["Kernel Name"] for fragment in EXCLUDED_KERNEL_FRAGMENTS)]
        if not launches:
            continue
        primary = max(launches, key=lambda launch: float(launch["duration_us"]))
        for launch in launches:
            launch["is_primary"] = launch is primary
            all_stall_columns.update(key for key in launch if key.startswith("stall_"))
        summary.append(primary)
        parsed_by_pair.append((meta_path.with_suffix("").with_suffix(".metrics.csv"), launches))

    base_columns = [
        "dataset", "category", "synthetic", "kernel", "N", "gpu",
        "timing_regime", "launch_id", "is_primary", "profiled_kernel",
        "duration_us", "tc_pipe_pct", "achieved_occupancy_pct",
        "dram_throughput_pct", "sm_throughput_pct", "mem_workload_pct",
        "dram_bytes", "cuda_core_flops", "cuda_core_achieved_gflops",
        "cuda_core_pct_fp32_peak", "cuda_core_ai_flop_per_byte", "hmma_insts",
        "flop_rate_scope",
    ]
    stall_columns = sorted(all_stall_columns)
    columns = base_columns + stall_columns
    for path, rows in parsed_by_pair:
        write_rows(path, rows, columns)
    if summary:
        write_rows(profile_dir / "profile_summary.csv", summary, columns)

    md_columns = [
        "dataset", "kernel", "N", "profiled_kernel", "tc_pipe_pct",
        "achieved_occupancy_pct", "dram_throughput_pct", "mem_workload_pct",
        "cuda_core_achieved_gflops", "cuda_core_pct_fp32_peak",
        "cuda_core_ai_flop_per_byte", "hmma_insts",
    ]
    lines = [
        "# Nsight Compute Profiling Summary\n\n",
        "Warm execute-only launches; plans are built before the profiled NVTX range. ",
        "CUDA-core FLOP/s is derived from Nsight executed FP instruction counters and kernel duration, ",
        "not from aggregate SM utilization. HMMA instructions and tensor-pipe utilization are reported separately.\n\n",
        "| " + " | ".join(md_columns) + " |\n",
        "|" + "|".join(["---"] * len(md_columns)) + "|\n",
    ]
    for row in summary:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in md_columns) + " |\n")
    lines.append("\nThe machine-readable summary and per-pair CSV files contain every emitted column, including the full warp-stall breakdown.\n")
    (profile_dir / "PROFILE_SUMMARY.md").write_text("".join(lines))
    print(f"Parsed {len(summary)} profile pairs in {profile_dir}")


if __name__ == "__main__":
    main()
