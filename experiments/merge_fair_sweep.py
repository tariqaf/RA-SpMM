#!/usr/bin/env python3
"""Merge two-GPU fair sweep shards and enforce release completeness."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

EXPECTED_KERNELS = {
    "CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
    "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID", "CUSPARSE",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-configs", type=int, default=192)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for source in args.inputs:
        with Path(source).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise SystemExit(f"CSV schema mismatch: {source}")
            rows.extend(reader)

    keys = [(row["dataset"], int(row["N"]), row["kernel"]) for row in rows]
    duplicate_keys = [key for key, count in Counter(keys).items() if count != 1]
    if duplicate_keys:
        raise SystemExit(f"Duplicate rows: {duplicate_keys[:10]}")
    configs: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = (row["dataset"], int(row["N"]))
        configs.setdefault(key, set()).add(row["kernel"])
    incomplete = {key: sorted(EXPECTED_KERNELS - kernels)
                  for key, kernels in configs.items() if kernels != EXPECTED_KERNELS}
    if len(configs) != args.expected_configs or incomplete:
        raise SystemExit(
            f"Expected {args.expected_configs} complete configs; got {len(configs)}; "
            f"incomplete={list(incomplete.items())[:10]}")
    failures = [row for row in rows if row["correct"].lower() not in {"1", "true"}]
    if failures:
        raise SystemExit(f"Strict correctness failures: {len(failures)}")

    rows.sort(key=lambda row: (row["dataset"], int(row["N"]), row["kernel"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows / {len(configs)} complete configurations to {output}")


if __name__ == "__main__":
    main()
