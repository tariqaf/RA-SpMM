#!/usr/bin/env python3
"""Produce plain-number summaries from a complete corrected sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_router_eval import KERNELS, simple_router


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--expected", type=int, default=192)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.sweep).open(newline="")))
    by_config: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["correct"].lower() in {"1", "true"}:
            by_config[(row["dataset"], int(row["N"]))][row["kernel"]] = row
    if len(by_config) != args.expected:
        raise SystemExit(f"Expected {args.expected} configurations, got {len(by_config)}")

    summary: dict[str, object] = {"configurations": len(by_config), "regimes": {}}
    markdown = ["# Corrected Fair Evaluation\n\n"]
    for regime in ("warm", "cold"):
        time_col = f"ms_{regime}"
        per_kernel: dict[str, list[float]] = defaultdict(list)
        router_speedups: list[float] = []
        oracle_speedups: list[float] = []
        router_oracle: list[float] = []
        hits = 0
        oracle_counts: Counter[str] = Counter()
        oracle_counts_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        for (dataset, N), config in sorted(by_config.items()):
            if "CUSPARSE" not in config or any(kernel not in config for kernel in KERNELS):
                raise SystemExit(f"Incomplete configuration: {dataset}, N={N}")
            cusp = float(config["CUSPARSE"][time_col])
            times = {kernel: float(config[kernel][time_col]) for kernel in KERNELS}
            sample = config[KERNELS[0]]
            for kernel, elapsed in times.items():
                per_kernel[kernel].append(cusp / elapsed)
            oracle = min(times, key=times.get)
            oracle_counts[oracle] += 1
            oracle_counts_by_category[sample["category"]][oracle] += 1
            oracle_speedups.append(cusp / times[oracle])
            M = int(sample["M"])
            nnz = int(sample["nnz"])
            router = simple_router(
                nnz / max(1, M), float(sample["cv_d"]), M, N, nnz)
            hits += router == oracle
            router_speedups.append(cusp / times[router])
            router_oracle.append(times[oracle] / times[router])

        regime_summary = {
            "router_policy": "production_rule_tree_no_lifecycle_cost",
            "per_kernel_geomean_vs_cusparse": {
                kernel: geomean(values) for kernel, values in per_kernel.items()},
            "oracle_geomean_vs_cusparse": geomean(oracle_speedups),
            "router_geomean_vs_cusparse": geomean(router_speedups),
            "router_oracle_geomean": geomean(router_oracle),
            "router_hits": hits,
            "router_hit_rate": hits / len(by_config),
            "empirical_ratio_ge_0_85": sum(value >= 0.85 for value in router_oracle),
            "empirical_ratio_ge_0_99": sum(value >= 0.99 for value in router_oracle),
            "oracle_counts": dict(oracle_counts),
            "oracle_counts_by_category": {
                category: dict(counts)
                for category, counts in sorted(oracle_counts_by_category.items())
            },
        }
        summary["regimes"][regime] = regime_summary
        markdown.extend([
            f"## {regime.title()}\n\n",
            "- Router policy: production rule tree (no measured-oracle or lifecycle look-ahead)\n",
            f"- Router geomean vs cuSPARSE: {regime_summary['router_geomean_vs_cusparse']:.6f}x\n",
            f"- Oracle geomean vs cuSPARSE: {regime_summary['oracle_geomean_vs_cusparse']:.6f}x\n",
            f"- Router/Oracle geomean: {regime_summary['router_oracle_geomean']:.6f}x\n",
            f"- Exact oracle hits: {hits}/{len(by_config)} ({100.0*hits/len(by_config):.2f}%)\n",
            f"- Empirical Router/Oracle >=0.85: {regime_summary['empirical_ratio_ge_0_85']}/{len(by_config)}\n",
            f"- Empirical Router/Oracle >=0.99: {regime_summary['empirical_ratio_ge_0_99']}/{len(by_config)}\n\n",
        ])
        for kernel in KERNELS:
            markdown.append(
                f"- {kernel}: {regime_summary['per_kernel_geomean_vs_cusparse'][kernel]:.6f}x\n")
        markdown.append("\n### Oracle Winners By Structural Regime\n\n")
        for category, counts in regime_summary["oracle_counts_by_category"].items():
            rendered = ", ".join(
                f"{kernel}={count}" for kernel, count in
                sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            markdown.append(f"- {category}: {rendered}\n")
        markdown.append("\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rendered = "".join(markdown).rstrip() + "\n"
    (outdir / "fair_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "FAIR_SUMMARY.md").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
