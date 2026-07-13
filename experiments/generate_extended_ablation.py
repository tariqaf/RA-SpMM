#!/usr/bin/env python3
"""Generate feature-mask and leave-one-rule-out router ablations.

Feature ablations replace one production input with its corpus median. This is a
fixed-router sensitivity test: thresholds are not retuned after seeing outcomes.
Rule ablations skip exactly one first-match rule while preserving rule order.
Only strictly correct timing rows participate.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_router_eval import KERNELS, route_with_rules


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else 0.0


def load_configs(path: Path, regime: str, expected: int):
    grouped = defaultdict(dict)
    for row in csv.DictReader(path.open(newline="")):
        if row["kernel"] in KERNELS and row.get("correct", "").lower() in {"1", "true"}:
            grouped[(row["dataset"], int(row["N"]))][row["kernel"]] = row
    complete = {key: rows for key, rows in grouped.items() if set(rows) == set(KERNELS)}
    if len(complete) != expected:
        raise SystemExit(f"Expected {expected} complete configurations, got {len(complete)}")
    time_col = f"ms_{regime}"
    configs = []
    for (dataset, N), rows in sorted(complete.items()):
        sample = rows[KERNELS[0]]
        M, nnz = int(sample["M"]), int(sample["nnz"])
        configs.append({
            "dataset": dataset, "N": N, "M": M, "nnz": nnz,
            "d": nnz / max(M, 1), "cv": float(sample["cv_d"]),
            "times": {kernel: float(rows[kernel][time_col]) for kernel in KERNELS},
        })
    return configs


def evaluate(configs, *, mask=None, keep_only=None, median=None, disabled_rules=()):
    ratios, hits = [], 0
    for item in configs:
        features = {key: item[key] for key in ("M", "N", "d", "cv")}
        if mask:
            features[mask] = median[mask]
        if keep_only:
            for feature in features:
                if feature != keep_only:
                    features[feature] = median[feature]
        selected = route_with_rules(
            features["d"], features["cv"], features["M"], features["N"],
            item["nnz"], disabled_rules=disabled_rules)
        oracle = min(item["times"], key=item["times"].get)
        ratios.append(item["times"][oracle] / item["times"][selected])
        hits += selected == oracle
    return {
        "router_oracle_geomean": geomean(ratios),
        "oracle_hits": hits,
        "hit_rate": hits / len(configs),
        "ratio_ge_0_85": sum(v >= 0.85 for v in ratios),
        "ratio_ge_0_99": sum(v >= 0.99 for v in ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regime", choices=("warm", "cold"), default="warm")
    parser.add_argument("--expected", type=int, default=192)
    args = parser.parse_args()

    configs = load_configs(Path(args.sweep), args.regime, args.expected)
    medians = {name: statistics.median(item[name] for item in configs)
               for name in ("M", "N", "d", "cv")}
    rows = []

    def add(kind, removed, result):
        rows.append({"regime": args.regime, "ablation_kind": kind,
                     "removed": removed, "configs": len(configs), **result})

    add("none", "none", evaluate(configs))
    for feature in ("M", "N", "d", "cv"):
        add("feature_median_mask", feature,
            evaluate(configs, mask=feature, median=medians))
        add("only_feature_variable", feature,
            evaluate(configs, keep_only=feature, median=medians))
    for rule in range(1, 9):
        add("leave_one_rule_out", f"rule_{rule}",
            evaluate(configs, disabled_rules=(rule,)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} ablations over {len(configs)} configurations)")


if __name__ == "__main__":
    main()
