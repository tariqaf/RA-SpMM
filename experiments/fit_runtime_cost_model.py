#!/usr/bin/env python3
"""Fit the disclosed runtime cold/steady cost model from a fair sweep CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNELS = [
    "CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
    "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID",
]
FEATURE_NAMES = [
    "bias", "log1p_M", "log1p_nnz", "log1p_N", "log1p_nnzN",
    "log1p_avg_nnz", "log1p_cv_d",
]


def truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def feature_vector(row: dict[str, str]) -> list[float]:
    M = float(row["M"])
    nnz = float(row["nnz"])
    N = float(row["N"])
    avg = float(row["avg_nnz_per_row"])
    cv = float(row["cv_d"])
    return [
        1.0, math.log1p(M), math.log1p(nnz), math.log1p(N),
        math.log1p(nnz * N), math.log1p(avg), math.log1p(cv),
    ]


def ridge_fit(X: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ target)


def grouped_oof_predictions(X: np.ndarray, target: np.ndarray,
                            groups: list[str], folds: int,
                            alpha: float) -> np.ndarray:
    unique = sorted(set(groups))
    if folds < 2 or folds > len(unique):
        raise ValueError(f"cv folds must be in [2, {len(unique)}]")
    fold_by_group = {group: index % folds for index, group in enumerate(unique)}
    predictions = np.empty(len(groups), dtype=np.float64)
    group_array = np.asarray(groups)
    for fold in range(folds):
        test = np.asarray([fold_by_group[group] == fold for group in groups])
        train = ~test
        coefficients = ridge_fit(X[train], target[train], alpha)
        predictions[test] = X[test] @ coefficients
    return predictions


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else 0.0


def feature_costs(path: Path, column: str) -> dict[str, float]:
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
    parser.add_argument("--output", default=str(
        REPO_ROOT / "results" / "router" / "runtime_cost_model.json"))
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--feature-times", required=True)
    parser.add_argument("--feature-column", default="production_full_cpu_input_ms")
    parser.add_argument("--validation-output", required=True)
    args = parser.parse_args()

    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    cusparse_actual: dict[tuple[str, int], tuple[float, float]] = {}
    with Path(args.sweep).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kernel") in KERNELS and truth(row.get("correct", False)):
                if row.get("ms_warm") and row.get("preprocess_ms"):
                    rows[row["kernel"]].append(row)
            elif row.get("kernel") == "CUSPARSE" and truth(row.get("correct", False)):
                cusparse_actual[(row["dataset"], int(row["N"]))] = (
                    float(row["ms_cold"]), float(row["ms_warm"]))
    measured_features = feature_costs(Path(args.feature_times), args.feature_column)

    model: dict[str, object] = {
        "schema_version": 1,
        "training_csv": str(Path(args.sweep)),
        "features": FEATURE_NAMES,
        "target_transform": "log(milliseconds)",
        "ridge_alpha": args.ridge,
        "validation": "graph-grouped out-of-fold predictions",
        "cv_folds": args.cv_folds,
        "kernels": {},
    }
    oof_by_key: dict[tuple[str, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    actual_by_key: dict[tuple[str, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    for kernel in KERNELS:
        samples = rows.get(kernel, [])
        if not samples:
            continue
        X = np.asarray([feature_vector(row) for row in samples], dtype=np.float64)
        warm = np.log(np.maximum(
            np.asarray([float(row["ms_warm"]) for row in samples]), 1e-9))
        setup = np.log(np.maximum(
            np.asarray([float(row["preprocess_ms"]) for row in samples]), 1e-9))
        warm_coef = ridge_fit(X, warm, args.ridge)
        setup_coef = ridge_fit(X, setup, args.ridge)
        warm_pred = np.exp(X @ warm_coef)
        setup_pred = np.exp(X @ setup_coef)
        groups = [row["dataset"] for row in samples]
        warm_oof = np.exp(grouped_oof_predictions(
            X, warm, groups, args.cv_folds, args.ridge))
        setup_oof = np.exp(grouped_oof_predictions(
            X, setup, groups, args.cv_folds, args.ridge))
        for row, predicted_warm, predicted_setup in zip(samples, warm_oof, setup_oof):
            key = (row["dataset"], int(row["N"]))
            oof_by_key[key][kernel] = (float(predicted_setup), float(predicted_warm))
            actual_by_key[key][kernel] = (float(row["ms_cold"]), float(row["ms_warm"]))
        model["kernels"][kernel] = {
            "samples": len(samples),
            "warm_coefficients": warm_coef.tolist(),
            "setup_coefficients": setup_coef.tolist(),
            "warm_geomean_relative_error": float(math.exp(np.mean(
                np.abs(np.log(warm_pred / np.exp(warm)))))) - 1.0,
            "setup_geomean_relative_error": float(math.exp(np.mean(
                np.abs(np.log(setup_pred / np.exp(setup)))))) - 1.0,
            "warm_grouped_cv_geomean_relative_error": float(math.exp(np.mean(
                np.abs(np.log(warm_oof / np.exp(warm)))))) - 1.0,
            "setup_grouped_cv_geomean_relative_error": float(math.exp(np.mean(
                np.abs(np.log(setup_oof / np.exp(setup)))))) - 1.0,
        }

    policy_validation = {}
    validation_rows: list[dict[str, object]] = []
    for calls in (1, 1000):
        ratios = []
        speedups = []
        hits = 0
        complete = 0
        for key in sorted(oof_by_key):
            if set(oof_by_key[key]) != set(KERNELS) or set(actual_by_key[key]) != set(KERNELS):
                continue
            dataset, N = key
            if dataset not in measured_features or key not in cusparse_actual:
                raise SystemExit(f"Missing feature or cuSPARSE lifecycle for {key}")
            predicted = min(
                KERNELS,
                key=lambda kernel: oof_by_key[key][kernel][0] +
                    calls * oof_by_key[key][kernel][1])
            actual_total = {
                kernel: actual_by_key[key][kernel][0] +
                    max(0, calls - 1) * actual_by_key[key][kernel][1]
                for kernel in KERNELS
            }
            oracle = min(actual_total, key=actual_total.get)
            ratio = actual_total[oracle] / actual_total[predicted]
            feature_ms = measured_features[dataset]
            selected_lifecycle = feature_ms + actual_total[predicted]
            oracle_lifecycle = feature_ms + actual_total[oracle]
            cusp_cold, cusp_warm = cusparse_actual[key]
            cusp_lifecycle = cusp_cold + max(0, calls - 1) * cusp_warm
            speedup = cusp_lifecycle / selected_lifecycle
            ratios.append(ratio)
            speedups.append(speedup)
            hits += predicted == oracle
            complete += 1
            validation_rows.append({
                "dataset": dataset, "N": N, "calls": calls,
                "policy": "graph-grouped out-of-fold predicted lifecycle cost",
                "feature_column": args.feature_column, "feature_ms": feature_ms,
                "chosen_kernel": predicted, "oracle_kernel": oracle,
                "chosen_measured_lifecycle_ms": selected_lifecycle,
                "oracle_measured_lifecycle_ms": oracle_lifecycle,
                "cusparse_measured_lifecycle_ms": cusp_lifecycle,
                "router_oracle_ratio": ratio,
                "speedup_vs_matching_cusparse": speedup,
                "oracle_hit": predicted == oracle,
            })
        policy_validation[f"K_{calls}"] = {
            "complete_configurations": complete,
            "router_oracle_geomean": geomean(ratios),
            "oracle_hits": hits,
            "hit_rate": hits / complete if complete else 0.0,
            "ratio_ge_0_85": sum(value >= 0.85 for value in ratios),
            "ratio_ge_0_99": sum(value >= 0.99 for value in ratios),
            "geomean_vs_matching_cusparse_including_feature_cost": geomean(speedups),
        }
    model["grouped_cv_policy_validation"] = policy_validation

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2) + "\n")
    validation_output = Path(args.validation_output)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with validation_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    print(f"Wrote {output} with {len(model['kernels'])} kernel models")
    print(f"Wrote {len(validation_rows)} lifecycle validation rows to {validation_output}")


if __name__ == "__main__":
    main()
