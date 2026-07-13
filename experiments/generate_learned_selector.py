#!/usr/bin/env python3
"""Compare rule routing with lightweight learned kernel selectors.

Splits are grouped by graph: all feature widths for a graph stay together. This
prevents train/test leakage across N. Models are trained only on strictly correct
warm measurements and are evaluated by both exact oracle hit rate and runtime
relative to the measured oracle.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_router_eval import KERNELS, simple_router


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else 0.0


def load_samples(path: Path, expected: int, extended: dict[str, list[float]]):
    grouped = defaultdict(dict)
    for row in csv.DictReader(path.open(newline="")):
        if row["kernel"] in KERNELS and row.get("correct", "").lower() in {"1", "true"}:
            grouped[(row["dataset"], int(row["N"]))][row["kernel"]] = row
    samples = []
    for (dataset, N), rows in sorted(grouped.items()):
        if set(rows) != set(KERNELS):
            continue
        sample = rows[KERNELS[0]]
        M, nnz = int(sample["M"]), int(sample["nnz"])
        times = {kernel: float(rows[kernel]["ms_warm"]) for kernel in KERNELS}
        oracle = min(times, key=times.get)
        samples.append({
            "dataset": dataset, "features": [math.log1p(M), math.log1p(nnz),
                math.log1p(nnz / max(M, 1)), math.log1p(float(sample["cv_d"])),
                math.log1p(N)],
            "extended_features": [math.log1p(M), math.log1p(nnz), math.log1p(N)] +
                extended.get(dataset, []),
            "raw": (nnz / max(M, 1), float(sample["cv_d"]), M, N, nnz),
            "oracle": oracle, "times": times,
        })
    if len(samples) != expected:
        raise SystemExit(f"Expected {expected} complete configurations, got {len(samples)}")
    return samples


def score(name, predictions, samples):
    ratios = []
    hits = 0
    for predicted, sample in zip(predictions, samples):
        if predicted not in sample["times"]:
            raise RuntimeError(f"{name} predicted unknown kernel {predicted}")
        ratios.append(sample["times"][sample["oracle"]] / sample["times"][predicted])
        hits += predicted == sample["oracle"]
    return {"model": name, "configs": len(samples), "oracle_hits": hits,
            "hit_rate": hits / len(samples), "router_oracle_geomean": geomean(ratios),
            "ratio_ge_0_85": sum(v >= 0.85 for v in ratios),
            "ratio_ge_0_99": sum(v >= 0.99 for v in ratios)}


def main() -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=192)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--features", default="",
                        help="Optional CSV from extract_production_features.py")
    args = parser.parse_args()
    extended = {}
    if args.features:
        feature_rows = list(csv.DictReader(Path(args.features).open(newline="")))
        feature_columns = sorted(
            key for key in feature_rows[0]
            if key.startswith("feature_") and key != "feature_N")
        extended = {row["dataset"]: [float(row[key]) for key in feature_columns]
                    for row in feature_rows}
    samples = load_samples(Path(args.sweep), args.expected, extended)
    groups = np.asarray([sample["dataset"] for sample in samples])
    unique_groups = np.unique(groups)
    if args.folds < 2 or args.folds > len(unique_groups):
        raise SystemExit(f"--folds must be in [2, {len(unique_groups)}]")
    X = np.asarray([sample["features"] for sample in samples], dtype=np.float64)
    y = np.asarray([sample["oracle"] for sample in samples])
    cv = GroupKFold(n_splits=args.folds)
    model_factories = {
        "decision_tree_depth4": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=3, random_state=args.seed),
        "random_forest_100": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=args.seed, n_jobs=-1),
        "multinomial_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
    }
    rows = [score("production_rules", [simple_router(*s["raw"]) for s in samples], samples)]
    for name, model in model_factories.items():
        predicted = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict")
        rows.append(score(f"{name}_base_features", predicted, samples))
    if extended:
        missing = sorted({sample["dataset"] for sample in samples} - set(extended))
        if missing:
            raise SystemExit(f"Missing production feature rows: {missing}")
        X_extended = np.asarray([sample["extended_features"] for sample in samples], dtype=np.float64)
        for name, model in model_factories.items():
            predicted = cross_val_predict(
                model, X_extended, y, groups=groups, cv=cv, method="predict")
            rows.append(score(f"{name}_production_features", predicted, samples))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}; {args.folds}-fold graph-grouped evaluation over {len(samples)} configs")


if __name__ == "__main__":
    main()
