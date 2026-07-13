#!/usr/bin/env python3
"""Precision-matched cuSPARSE (FP16 A/B, FP32 C/compute) baseline + accuracy.

For every (graph, N) config: warm/cold FP16-cuSPARSE timing with the exact
protocol of the fair sweep, plus max abs/rel error vs the FP32 cuSPARSE
reference for (a) each ME-BCRS tile kernel and (b) FP16 cuSPARSE itself.
Joins with fair_v2_*.csv on (dataset, N).
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import ra_spmm  # noqa: E402
import ra_real_graph_eval as ev  # noqa: E402

def max_abs_diff(A: "torch.Tensor", B: "torch.Tensor", chunk_rows: int = 262144) -> float:
    """Row-chunked max |A-B| to avoid materializing a full-size diff tensor."""
    m = 0.0
    for i in range(0, A.size(0), chunk_rows):
        m = max(m, (A[i:i + chunk_rows] - B[i:i + chunk_rows]).abs().max().item())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default="fgcs_results/paper_combined_datasets.json")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--Ns", default="64,128,256,512")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--cold-iters", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ns = [int(x) for x in args.Ns.split(",")]
    manifest = json.load(open(args.datasets_file))
    datasets = [d for d in manifest["datasets"] if d.get("enabled", True)]
    if args.datasets:
        keep = {x.strip() for x in args.datasets.split(",")}
        datasets = [d for d in datasets if d["name"] in keep]

    rows = []
    for entry in datasets:
        name = entry["name"]
        data = ev.load_dataset(entry)
        if data is None:
            print(f"[skip] {name}: load failed")
            continue
        rowptr_cpu = data["rowptr"].int()
        colind_cpu = data["colind"].int()
        vals_cpu = data["vals"].float()
        M, K = data["M"], data["K"]
        rowptr = rowptr_cpu.cuda()
        colind = colind_cpu.cuda()
        vals = vals_cpu.cuda()
        lens = (rowptr_cpu[1:] - rowptr_cpu[:-1])
        max_nnz_row = int(lens.max().item()) if M > 0 else 0

        tol = 1e-3 * max(1.0, math.sqrt(max_nnz_row)) * 10.0
        recs = {}
        # Phase 1: fp16 cuSPARSE timing + error per N (no tile plans resident).
        for N in ns:
            torch.manual_seed(123)
            B = torch.randn(K, N, device="cuda")
            C_ref = ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
            ref_max = C_ref.abs().max().item()
            w16 = ra_spmm.benchmark_cusparse_fp16(
                rowptr, colind, vals, B, warmup=args.warmup, iters=args.timed)
            c16 = ra_spmm.benchmark_cusparse_fp16_cold(
                rowptr, colind, vals, B, max(1, args.cold_iters))
            C16 = ra_spmm.spmm_cusparse_fp16(rowptr, colind, vals, B)
            err16 = max_abs_diff(C16, C_ref)
            recs[N] = {
                "dataset": name,
                "category": entry.get("category", ""),
                "N": N, "M": M, "nnz": int(colind_cpu.numel()),
                "max_nnz_per_row": max_nnz_row,
                "tolerance_tc": round(tol, 6),
                "ref_max": ref_max,
                "ms_cusparse_fp16_warm": round(float(w16["exec_ms"]), 6),
                "ms_cusparse_fp16_cold": round(float(c16["total_ms"]), 6),
                "cusparse_fp16_max_abs_err": err16,
                "cusparse_fp16_max_rel_err": err16 / max(ref_max, 1e-30),
            }
            del B, C_ref, C16
            torch.cuda.empty_cache()

        # Phase 2: one tile plan resident at a time; identical seeded B.
        for kname in ["TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"]:
            maker = {
                "TC_DIRECT": ra_spmm.make_tc_direct_plan,
                "COMMUNITY_TC": ra_spmm.make_community_tc_plan,
                "SEGMENT_HYBRID": ra_spmm.make_segment_hybrid_plan,
            }[kname]
            plan = maker(rowptr_cpu, colind_cpu, vals_cpu, M, K, 64)
            for N in ns:
                torch.manual_seed(123)
                B = torch.randn(K, N, device="cuda")
                C_ref = ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
                if kname == "SEGMENT_HYBRID":
                    C = ra_spmm.run_segment_hybrid_plan(plan, colind, vals, B)
                elif kname == "TC_DIRECT":
                    C = ra_spmm.run_tc_direct_plan(plan, B)
                else:
                    C = ra_spmm.run_community_tc_plan(plan, B)
                err = max_abs_diff(C, C_ref)
                rec = recs[N]
                rec[f"{kname}_max_abs_err"] = err
                rec[f"{kname}_max_rel_err"] = err / max(rec["ref_max"], 1e-30)
                rec[f"{kname}_passes_gate"] = bool(err <= tol and err < 1.0)
                del B, C_ref, C
                torch.cuda.empty_cache()
            del plan
            torch.cuda.empty_cache()

        for N in ns:
            rec = recs[N]
            rec.pop("ref_max", None)
            rows.append(rec)
            print(f"{name} N={N}: fp16cusp warm {rec['ms_cusparse_fp16_warm']:.4f} ms, "
                  f"err fp16cusp {rec['cusparse_fp16_max_abs_err']:.5f} "
                  f"TC {rec.get('TC_DIRECT_max_abs_err', float('nan')):.5f} (tol {tol:.4f})")

    with open(args.output, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
