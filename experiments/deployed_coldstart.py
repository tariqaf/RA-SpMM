#!/usr/bin/env python3
"""Deployed cold-start measurement: the cold-aware policy, all overheads in.

Policy under test: a caller in a declared single-call / streaming context is
routed to the preprocessing-free CSR path. The measured first-call latency
includes the deployed 4-feature extraction (degree pass on GPU: d-bar, CV_d),
rule evaluation, and the CSR_DIRECT execution. Baseline: cuSPARSE cold
(plan + exec) via the same harness used everywhere else.

This replaces retrospective cold-oracle numbers with a deployed number.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset  # noqa: E402
from ra_router_eval import route_with_rules  # noqa: E402


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO / "fgcs_results" / "paper_combined_datasets.json"))
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "fgcs_results" / "revision" / "tf32" / "deployed_coldstart.csv"))
    args = ap.parse_args()

    manifest = json.loads(Path(args.datasets_file).read_text())
    if isinstance(manifest, dict):
        manifest = manifest["datasets"]

    # Warm up the tiny torch kernels used by the feature pass so the first
    # graph is not charged for one-time CUDA/JIT setup.
    warm = torch.arange(1024, device="cuda", dtype=torch.int32)
    (warm[1:] - warm[:-1]).float().std(correction=0)
    torch.cuda.synchronize()

    rows = []
    cats = defaultdict(list)
    for e in manifest:
        mat = load_dataset(e)
        if mat is None:
            continue
        M = mat["M"]; K = mat.get("K", M)
        rp = mat["rowptr"].int().cuda()
        ci = mat["colind"].int().cuda()
        vl = mat["vals"].float().cuda()
        nnz = int(mat["rowptr"][-1])
        B = torch.randn(K, args.N, device="cuda")

        # Deployed feature pass + routing (timed, averaged).
        fts = []
        for _ in range(args.iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            deg = (rp[1:] - rp[:-1]).float()
            dbar = nnz / max(1, M)
            cv = float((deg.std(correction=0) / deg.mean()).item()) if M > 0 else 0.0
            route_with_rules(dbar, cv, M, args.N, nnz)
            torch.cuda.synchronize()
            fts.append((time.perf_counter() - t0) * 1000)
        feat_ms = sum(fts) / len(fts)

        # Cold CSR execution (fresh single-shot calls).
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            C = ra_spmm.spmm_csr_direct(rp, ci, vl, B)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
            del C
        csr_ms = sum(ts) / len(ts)

        cus = ra_spmm.benchmark_cusparse_cold(rp, ci, vl, B, args.iters)
        cus_ms = float(cus["total_ms"])
        deployed = cus_ms / (feat_ms + csr_ms)
        cat = e.get("category", "?")
        cats[cat].append(deployed)
        cats["ALL"].append(deployed)
        rows.append({
            "dataset": e["name"], "category": cat, "M": M, "nnz": nnz,
            "N": args.N, "feature_route_ms": round(feat_ms, 4),
            "csr_cold_exec_ms": round(csr_ms, 4),
            "cusparse_cold_ms": round(cus_ms, 4),
            "deployed_cold_speedup": round(deployed, 4),
        })
        print(f"{e['name']:28s} feat={feat_ms:7.3f} csr={csr_ms:8.3f} "
              f"cusp={cus_ms:8.3f} deployed={deployed:6.3f}x", flush=True)
        del B
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\nDeployed cold-start geomeans (all overheads included):")
    for cat in sorted(cats):
        print(f"  {cat:20s} n={len(cats[cat]):3d} {gm(cats[cat]):.3f}x")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
