#!/usr/bin/env python3
"""Phase-1 CHECKPOINT: routing-only pass (no timing) over the eight e2e datasets
and all three models, on the corrected backend. Reports resolved_path per SpMM
site + per-model post-normalization descriptors. NO measurement."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import torch  # noqa: load libtorch before ra_spmm
import ra_spmm as spmm_next  # noqa
from gnn_bench.router_vs_baselines_gcn import (DATASETS, load_csr,
    build_aggregation_operator, operator_descriptors, resolve_router_label)

DS8 = ["Reddit", "ogbn-proteins", "ogbn-arxiv", "PPI",
       "amazon-photo", "amazon-computers", "Cora", "CiteSeer"]
# site -> (which width, transpose) per model
SITES = {
    "gcn": [("forward_hidden", "hidden", False), ("forward_out", "out", False),
            ("backward_hidden", "hidden", True), ("backward_out", "out", True)],
    "graphsage": [("forward_in", "in", False), ("forward_hidden", "hidden", False),
                  ("backward_in", "in", True), ("backward_hidden", "hidden", True)],
    "gin": [("forward_in", "in", False), ("forward_hidden", "hidden", False),
            ("backward_in", "in", True), ("backward_hidden", "hidden", True)],
}


def route(rp, ci, vl, M, N):
    import torch
    plan = spmm_next.make_router_plan(
        torch.from_numpy(rp.astype(np.int32)), torch.from_numpy(ci.astype(np.int32)),
        torch.from_numpy(vl.astype(np.float32)), M, M, int(N), "MAIN")
    return resolve_router_label(plan)


def main():
    raw_desc = {}
    print(f"{'dataset':16s} {'model':10s} {'site':16s} {'N':>5s}  resolved_path")
    print("-" * 70)
    changed = []
    for name in DS8:
        spec = DATASETS[name]
        raw = load_csr(f"datasets/gnn/exports/{spec.npz_name}")
        raw_desc[name] = operator_descriptors(raw)
        widthmap = {"in": spec.in_dim, "hidden": spec.hidden_dim, "out": spec.out_dim}
        for model in ("gcn", "graphsage", "gin"):
            op = build_aggregation_operator(raw, model)
            opT = op.T.tocsr()
            d = operator_descriptors(op)
            op._checkpoint_desc = d
            for site, wkey, transpose in SITES[model]:
                N = widthmap[wkey]
                mat = opT if transpose else op
                lbl = route(mat.indptr, mat.indices, mat.data, op.shape[0], N)
                print(f"{name:16s} {model:10s} {site:16s} {N:>5d}  {lbl}")
                # flag GCN decisions that changed vs the raw (pre-self-loop) route
                if model == "gcn":
                    rmat = raw.T.tocsr() if transpose else raw
                    raw_lbl = route(rmat.indptr, rmat.indices, rmat.data, raw.shape[0], N)
                    if raw_lbl != lbl:
                        changed.append((name, site, N, raw_lbl, lbl))
    # per-model descriptors
    print("\n=== per-model post-normalization descriptors (M, nnz, d_bar, CV_d) ===")
    print(f"{'dataset':16s} {'RAW nnz':>10s} {'GCN nnz':>10s} {'GCN d_bar':>10s} {'GCN CV_d':>9s} {'SAGE d_bar':>11s} {'GIN d_bar':>10s}")
    for name in DS8:
        spec = DATASETS[name]
        raw = load_csr(f"datasets/gnn/exports/{spec.npz_name}")
        g = operator_descriptors(build_aggregation_operator(raw, "gcn"))
        s = operator_descriptors(build_aggregation_operator(raw, "graphsage"))
        i = operator_descriptors(build_aggregation_operator(raw, "gin"))
        print(f"{name:16s} {raw_desc[name]['nnz']:>10d} {g['nnz']:>10d} {g['d_bar']:>10.3f} "
              f"{g['CV_d']:>9.3f} {s['d_bar']:>11.3f} {i['d_bar']:>10.3f}")
    print("\n=== GCN router decisions changed by self-loops (raw -> A_hat) ===")
    if changed:
        for c in changed:
            print(f"  {c[0]} {c[1]} N={c[2]}: {c[3]} -> {c[4]}")
    else:
        print("  none")
    # the key checkpoint assertion
    print("\n=== KEY CHECK: Cora/CiteSeer forward_hidden @ N=128 ===")
    for name in ("Cora", "CiteSeer"):
        spec = DATASETS[name]
        op = build_aggregation_operator(load_csr(f"datasets/gnn/exports/{spec.npz_name}"), "gcn")
        lbl = route(op.indptr, op.indices, op.data, op.shape[0], 128)
        print(f"  {name} GCN forward_hidden N=128 -> {lbl}  ({'OK' if lbl=='TC_DIRECT_TF32' else 'UNEXPECTED'})")


if __name__ == "__main__":
    main()
