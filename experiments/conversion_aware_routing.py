"""Offline amortization analysis over corrected warm/cold measurements.

This script is intentionally labeled offline: it evaluates every measured,
strictly correct kernel after the sweep. It does not claim that the deployed
router knows the oracle kernel times.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP = REPO_ROOT / "fgcs_results/revision/fair/all_graphs_results.csv"
DEFAULT_FEATURES = REPO_ROOT / "fgcs_results/revision/fair/feature_extraction.csv"
DEFAULT_OUT = REPO_ROOT / "fgcs_results/revision/fair/conversion_aware"
KERNELS = [
    "CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
    "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID",
]
TC_KERNELS = {"TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"}


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def is_true(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def geomean(values: list[float]) -> float:
    valid = [value for value in values if value > 0.0 and math.isfinite(value)]
    return math.exp(sum(math.log(value) for value in valid) / len(valid)) if valid else 0.0


def smallest_strict_crossover(setup_a: float, exec_a: float,
                              setup_b: float, exec_b: float) -> int | None:
    """Smallest integer K>=1 with setup_b + K*exec_b < setup_a + K*exec_a."""
    if exec_b >= exec_a:
        return None
    threshold = (setup_b - setup_a) / (exec_a - exec_b)
    return max(1, math.floor(threshold) + 1)


def load_feature_costs(path: Path, column: str) -> dict[str, float]:
    costs: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = as_float(row, column)
            if value is not None:
                costs[row["dataset"]] = value
    return costs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--feature-column", default="production_full_gpu_input_ms")
    parser.add_argument("--outdir", default=str(DEFAULT_OUT))
    parser.add_argument("--steady-K", type=int, default=1000)
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    feature_path = Path(args.features)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    feature_ms = load_feature_costs(feature_path, args.feature_column)

    timings: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
    metadata: dict[tuple[str, int], dict[str, object]] = {}
    cusparse: dict[tuple[str, int], dict[str, float]] = {}
    with sweep_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not is_true(row.get("correct", False)):
                continue
            key = (row["dataset"], int(row["N"]))
            warm = as_float(row, "ms_warm")
            cold = as_float(row, "ms_cold")
            setup = as_float(row, "preprocess_ms")
            if warm is None or cold is None or setup is None:
                continue
            values = {"warm": warm, "cold": cold, "setup": setup}
            if row["kernel"] == "CUSPARSE":
                cusparse[key] = values
            elif row["kernel"] in KERNELS:
                timings[key][row["kernel"]] = values
            metadata[key] = {
                "category": row.get("category", "?"),
                "M": int(row["M"]), "nnz": int(row["nnz"]),
            }

    cold_rows: list[dict[str, object]] = []
    steady_rows: list[dict[str, object]] = []
    crossover_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    cold_speedups: list[float] = []
    steady_speedups: list[float] = []

    for key in sorted(timings):
        dataset, N = key
        if dataset not in feature_ms:
            missing_rows.append({"dataset": dataset, "N": N,
                                 "reason": f"missing feature column {args.feature_column}"})
            continue
        if key not in cusparse:
            missing_rows.append({"dataset": dataset, "N": N,
                                 "reason": "missing correct cuSPARSE warm/cold row"})
            continue
        candidates = timings[key]
        if len(candidates) != len(KERNELS):
            missing = sorted(set(KERNELS) - set(candidates))
            missing_rows.append({"dataset": dataset, "N": N,
                                 "reason": "missing/incorrect kernels: " + ";".join(missing)})
            continue

        feat = feature_ms[dataset]
        cusp = cusparse[key]

        def total(kernel: str, calls: int) -> float:
            timing = candidates[kernel]
            return feat + timing["cold"] + max(0, calls - 1) * timing["warm"]

        def cusparse_total(calls: int) -> float:
            return cusp["cold"] + max(0, calls - 1) * cusp["warm"]

        cold_kernel = min(candidates, key=lambda kernel: total(kernel, 1))
        cold_total = total(cold_kernel, 1)
        cold_baseline = cusparse_total(1)
        cold_speedup = cold_baseline / cold_total
        cold_speedups.append(cold_speedup)
        cold_rows.append({
            "dataset": dataset, "N": N, **metadata[key],
            "analysis": "offline measured oracle",
            "feature_ms": feat, "kernel": cold_kernel,
            "kernel_preprocess_ms": candidates[cold_kernel]["setup"],
            "kernel_first_exec_ms": candidates[cold_kernel]["cold"] - candidates[cold_kernel]["setup"],
            "ra_total_ms_K1": cold_total,
            "cusparse_total_ms_K1": cold_baseline,
            "speedup_cold_vs_cold": cold_speedup,
        })

        calls = max(1, args.steady_K)
        steady_kernel = min(candidates, key=lambda kernel: total(kernel, calls))
        throughput_kernel = min(candidates, key=lambda kernel: candidates[kernel]["warm"])
        steady_total = total(steady_kernel, calls)
        steady_baseline = cusparse_total(calls)
        steady_speedup = steady_baseline / steady_total
        steady_speedups.append(steady_speedup)
        steady_rows.append({
            "dataset": dataset, "N": N, "K_calls": calls,
            "analysis": "offline measured oracle",
            "kernel": steady_kernel, "throughput_kernel": throughput_kernel,
            "matches_throughput": steady_kernel == throughput_kernel,
            "feature_ms": feat,
            "ra_total_ms": steady_total, "cusparse_total_ms": steady_baseline,
            "ra_amortized_ms_per_call": steady_total / calls,
            "speedup_matching_lifecycle": steady_speedup,
        })

        base_kernel = min(candidates,
                          key=lambda kernel: (candidates[kernel]["setup"],
                                              candidates[kernel]["warm"]))
        tc_candidates = [kernel for kernel in candidates if kernel in TC_KERNELS]
        if tc_candidates:
            best_tc = min(tc_candidates, key=lambda kernel: candidates[kernel]["warm"])
            crossover = smallest_strict_crossover(
                candidates[base_kernel]["cold"] - candidates[base_kernel]["warm"],
                candidates[base_kernel]["warm"],
                candidates[best_tc]["cold"] - candidates[best_tc]["warm"],
                candidates[best_tc]["warm"],
            )
            crossover_rows.append({
                "dataset": dataset, "N": N,
                "base_kernel": base_kernel, "best_tc_kernel": best_tc,
                "base_setup_ms": candidates[base_kernel]["setup"],
                "tc_setup_ms": candidates[best_tc]["setup"],
                "base_warm_ms": candidates[base_kernel]["warm"],
                "tc_warm_ms": candidates[best_tc]["warm"],
                "crossover_K_strict": crossover if crossover is not None else "never",
            })

    write_csv(outdir / "offline_coldstart.csv", cold_rows)
    write_csv(outdir / "offline_steadystate.csv", steady_rows)
    write_csv(outdir / "crossover_K.csv", crossover_rows)
    write_csv(outdir / "missing_measurements.csv", missing_rows)

    summary = [
        "# Offline Conversion-Aware Amortization\n\n",
        "This is a post-hoc analysis over all measured correct kernels, not the deployed router. ",
        "Each lifecycle compares matching costs: first-call setup plus execution, followed by warm execution.\n\n",
        f"- Cold K=1 geomean vs cold cuSPARSE: {geomean(cold_speedups):.6f}x ({len(cold_speedups)} points)\n",
        f"- K={max(1, args.steady_K)} lifecycle geomean vs the matching cuSPARSE lifecycle: ",
        f"{geomean(steady_speedups):.6f}x ({len(steady_speedups)} points)\n",
        f"- Missing or incorrect measurement records: {len(missing_rows)}\n",
    ]
    (outdir / "OFFLINE_AMORTIZATION.md").write_text("".join(summary))
    print("".join(summary), end="")


if __name__ == "__main__":
    main()
