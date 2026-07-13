#!/usr/bin/env python3
"""Merge external-baseline shards while preserving unioned failure columns."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=192)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    columns: set[str] = set()
    for source in args.inputs:
        with Path(source).open(newline="") as handle:
            shard = list(csv.DictReader(handle))
        rows.extend(shard)
        for row in shard:
            columns.update(row)

    keys = [(row["dataset"], int(row["N"])) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise SystemExit(f"Duplicate external rows: {duplicates[:10]}")
    if len(keys) != args.expected:
        raise SystemExit(f"Expected {args.expected} external rows, got {len(keys)}")

    preferred = ["dataset", "category", "M", "nnz", "N"]
    fieldnames = preferred + sorted(columns - set(preferred))
    rows.sort(key=lambda row: (row["dataset"], int(row["N"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} complete external configurations to {output}")


if __name__ == "__main__":
    main()
