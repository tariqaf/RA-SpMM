#!/usr/bin/env python3
"""cuBLAS dense-GEMM baseline on small graphs, current fair protocol.

For each small graph (M <= 25k) and N: densify A once (cuBLAS's 'plan'),
then time steady-state dense GEMM (50 warmup / 200 timed) against
plan-reused cuSPARSE warm, plus cold comparisons (densify+first GEMM vs
cuSPARSE plan+exec). Router times are joined from router_quality_v3.csv
so cuBLAS is also compared against the deployed router pick.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset  # noqa: E402

GRAPHS = ["ca-GrQc", "ca-HepTh", "ca-CondMat", "amazon-photo", "amazon-computers",
          "synth_dense_small_d30", "synth_dense_small_d50", "synth_dense_small_d70",
          "synth_dense_small_d90", "synth_dense_small_d120",
          "Cora", "CiteSeer", "PPI"]


def bench(fn, warm=50, iters=200):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO / "fgcs_results" / "paper_combined_datasets.json"))
    ap.add_argument("--router-csv", default=str(REPO / "fgcs_results" / "revision" / "tf32" / "router_quality_v3.csv"))
    ap.add_argument("--out", default=str(REPO / "fgcs_results" / "revision" / "tf32" / "cublas_dense_v3.csv"))
    args = ap.parse_args()

    manifest = json.loads(Path(args.datasets_file).read_text())
    if isinstance(manifest, dict):
        manifest = manifest["datasets"]
    entries = {e["name"]: e for e in manifest}

    router = {}
    for r in csv.DictReader(open(args.router_csv)):
        router[(r["dataset"], int(r["N"]))] = (r["router_kernel"], float(r["router_ms"]))

    rows = []
    for name in GRAPHS:
        e = entries.get(name)
        mat = load_dataset(e) if e else None
        if mat is None:
            print(f"skip {name}")
            continue
        M = mat["M"]; K = mat.get("K", M)
        rp = mat["rowptr"].int().cuda()
        ci = mat["colind"].int().cuda()
        vl = mat["vals"].float().cuda()
        nnz = int(mat["rowptr"][-1])

        # Densify A once (cuBLAS's plan phase), timed for the cold number.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        A = torch.zeros(M, K, device="cuda")
        row_ids = torch.repeat_interleave(
            torch.arange(M, device="cuda"), (rp[1:] - rp[:-1]).long())
        A[row_ids, ci.long()] = vl
        torch.cuda.synchronize()
        densify_ms = (time.perf_counter() - t0) * 1000

        for N in (64, 128, 256, 512):
            B = torch.randn(K, N, device="cuda")
            cublas_warm = bench(lambda: torch.matmul(A, B))
            cus_warm = float(ra_spmm.benchmark_cusparse(rp, ci, vl, B, warmup=50, iters=200)["exec_ms"])
            cus_cold = float(ra_spmm.benchmark_cusparse_cold(rp, ci, vl, B, 5)["total_ms"])
            cublas_cold = densify_ms + cublas_warm  # densify + one GEMM
            rk, rms = router.get((name, N), ("?", float("nan")))
            rows.append({
                "dataset": name, "category": e.get("category", "?"),
                "M": M, "nnz": nnz, "N": N,
                "cublas_warm_ms": round(cublas_warm, 5),
                "cusparse_warm_ms": round(cus_warm, 5),
                "router_kernel": rk, "router_warm_ms": round(rms, 5),
                "densify_ms": round(densify_ms, 3),
                "cublas_cold_ms": round(cublas_cold, 4),
                "cusparse_cold_ms": round(cus_cold, 4),
                "cublas_vs_cusparse_warm": round(cus_warm / cublas_warm, 4),
                "cublas_vs_router_warm": round(rms / cublas_warm, 4),
                "cublas_vs_cusparse_cold": round(cus_cold / cublas_cold, 4),
            })
            print(f"{name:24s} N={N:3d} cublas={cublas_warm:8.4f} cusp={cus_warm:8.4f} "
                  f"router[{rk:>16s}]={rms:8.4f} | vs cusp {cus_warm/cublas_warm:6.3f}x "
                  f"vs router {rms/cublas_warm:6.3f}x cold {cus_cold/cublas_cold:6.3f}x", flush=True)
            del B
        del A, row_ids
        torch.cuda.empty_cache()

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
