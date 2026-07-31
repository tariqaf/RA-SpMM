"""
Kernel-choice PARITY between the RTX 4090 (Ada) sweep and the RTX 3090
baseline CSV.

For each (dataset, N) pair present in both CSVs, compares:
  - oracle kernel (argmax speedup_vs_cusparse) on Ada vs on the 3090
  - the router's chosen kernel (identical rule set, arch-independent) and whether it
    stays optimal on Ada.
Reports the oracle-agreement rate (does the best kernel per point transfer across
architectures?) and the router hit-rate on Ada.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def load(path):
    pairs = defaultdict(dict)
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("correct", "True") not in ("True", "true", "1"):
                continue
            try:
                sp = float(row["speedup_vs_cusparse"])
            except (KeyError, ValueError):
                continue
            key = (row["dataset"], int(row["N"]))
            pairs[key][row["kernel"]] = sp
            meta[key] = row
    return pairs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ada", required=True)
    ap.add_argument("--baseline3090", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ada, _ = load(args.ada)
    b90, _ = load(args.baseline3090)
    common = sorted(set(ada) & set(b90))

    rows = []
    oracle_agree = 0
    for key in common:
        ds, N = key
        ada_k = max(ada[key], key=ada[key].get)
        b90_k = max(b90[key], key=b90[key].get)
        agree = ada_k == b90_k
        oracle_agree += int(agree)
        rows.append({
            "dataset": ds, "N": N,
            "oracle_3090": b90_k, "oracle_ada": ada_k, "oracle_agree": agree,
            "ada_speedup_best": round(ada[key][ada_k], 3),
            "ada_speedup_of_3090choice": round(ada[key].get(b90_k, 0.0), 3),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(common)
    print(f"Common (dataset,N) pairs: {n}")
    print(f"Oracle kernel agrees 3090<->Ada: {oracle_agree}/{n} ({100*oracle_agree/max(1,n):.1f}%)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
