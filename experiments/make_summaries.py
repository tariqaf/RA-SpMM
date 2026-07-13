#!/usr/bin/env python3
"""Build the revision report exclusively from corrected fair-result files."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def mean(rows: list[dict[str, str]], column: str) -> float:
    values = [float(row[column]) for row in rows if row.get(column, "") not in {"", "nan"}]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-dir", default=str(ROOT / "fgcs_results/revision/fair"))
    parser.add_argument("--sweep", default="all_systems.csv")
    parser.add_argument("--expected", type=int, default=192)
    args = parser.parse_args()

    fair = Path(args.fair_dir)
    sweep = fair / args.sweep
    if not sweep.exists():
        raise SystemExit(f"Corrected sweep is missing: {sweep}")
    summary_dir = fair / "summary"
    run("experiments/summarize_fair_results.py", "--sweep", str(sweep),
        "--outdir", str(summary_dir), "--expected", str(args.expected))

    profile_dir = fair / "profile"
    if any(profile_dir.glob("*.meta.json")):
        run("experiments/profile_parse.py", "--profile-dir", str(profile_dir))
        run("experiments/summarize_profile_bottlenecks.py",
            "--summary", str(profile_dir / "profile_summary.csv"),
            "--output-csv", str(profile_dir / "profile_bottlenecks.csv"),
            "--output-md", str(profile_dir / "PROFILE_BOTTLENECKS.md"))

    sections = [
        "# Corrected Fair Evaluation Results\n\n",
        "All performance values in this report are generated from strictly correct rows ",
        "with matching warm/warm or cold/cold lifecycle comparisons.\n\n",
        (summary_dir / "FAIR_SUMMARY.md").read_text(),
    ]
    feature_path = fair / "feature_extraction.csv"
    if feature_path.exists():
        feature_rows = list(csv.DictReader(feature_path.open(newline="")))
        sections.extend([
            "\n## Production Feature Extraction\n\n",
            f"- Graphs measured: {len(feature_rows)}\n",
            f"- Mean full path, CPU-resident input: "
            f"{mean(feature_rows, 'production_full_cpu_input_ms'):.6f} ms\n",
            f"- Mean full path, GPU-resident input including required transfer: "
            f"{mean(feature_rows, 'production_full_gpu_input_ms'):.6f} ms\n",
            "- The lightweight degree-moment kernel is reported separately and is not used "
            "as the production router cost.\n",
        ])
    if (profile_dir / "PROFILE_BOTTLENECKS.md").exists():
        sections.extend([
            "\n## Profiling\n\n",
            "See `profile/PROFILE_SUMMARY.md` for per-pair metrics and "
            "`profile/PROFILE_BOTTLENECKS.md` for grouped bottlenecks.\n",
        ])
    output = fair / "REVISION_RESULTS.md"
    output.write_text("".join(sections))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
