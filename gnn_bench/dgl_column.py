#!/usr/bin/env python3
"""Standalone DGL GCN column, protocol-identical to router_vs_baselines_gcn.py.

Runs in a dedicated venv (torch 2.6.x+cu118 + dgl 2.5.0+cu118 — DGL's newest
supported torch) because no DGL build exists for torch 2.7. Everything else
matches the main harness exactly: 2-layer GCN (bias-free linears), Adam,
seed 123, cross-entropy on random labels, 50 warmup + 200 timed full
training steps, per-direction sparse matrices cached and reused (warm).
Correctness gate: forward/backward SpMM vs torch.sparse.mm (cuSPARSE-backed)
under the harness tolerance BASE_ATOL * sqrt(max_row_nnz).
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp

BASE_ATOL = 1e-3

DATASETS = {
    "Reddit":        dict(npz="Reddit.npz",        in_dim=602, out_dim=41,  hidden_dim=128),
    "ogbn-proteins": dict(npz="ogbn-proteins.npz", in_dim=8,   out_dim=112, hidden_dim=128),
    "ogbn-arxiv":    dict(npz="ogbn-arxiv.npz",    in_dim=128, out_dim=40,  hidden_dim=128),
}


class DGLGraphPair:
    """Forward and transpose dgl.sparse matrices, cached (warm protocol)."""

    def __init__(self, csr: sp.csr_matrix, device: torch.device) -> None:
        import dgl.sparse as dglsp
        self.mats = {}
        for transpose in (False, True):
            m = csr.T.tocsr() if transpose else csr
            M = int(m.shape[0])
            indptr = torch.from_numpy(np.asarray(m.indptr, dtype=np.int64)).to(device)
            col = torch.from_numpy(np.asarray(m.indices, dtype=np.int64)).to(device)
            val = torch.from_numpy(np.asarray(m.data, dtype=np.float32)).to(device)
            # from_csr dispatches DGL's cuSPARSE-backed CSR SpMM — its fast
            # path (a COO-constructed matrix runs ~1.6-3.4x slower per op).
            self.mats[transpose] = dglsp.from_csr(indptr, col, val, shape=(M, M))

    def run(self, B: torch.Tensor, transpose: bool) -> torch.Tensor:
        return self.mats[transpose] @ B


class SparseMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, graph):
        ctx.graph = graph
        return graph.run(B, transpose=False)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.graph.run(grad_output.contiguous(), transpose=True), None


class GCNBench(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, X, graph):
        h = self.lin1(X)
        h = SparseMMFunction.apply(h, graph)
        h = F.relu(h)
        out = self.lin2(h)
        return SparseMMFunction.apply(out, graph)


def correctness(csr, graph, device, dims):
    indptr = np.asarray(csr.indptr)
    max_row = max(1, int(np.diff(indptr).max()))
    tol = BASE_ATOL * max(1.0, math.sqrt(max_row))
    coo = csr.tocoo()
    ref_mat = torch.sparse_coo_tensor(
        np.vstack([coo.row, coo.col]), coo.data.astype(np.float32),
        size=csr.shape, device=device).coalesce()
    worst = 0.0
    for n in dims:
        torch.manual_seed(123 + n)
        B = torch.randn(csr.shape[0], n, device=device)
        out = graph.run(B, transpose=False)
        ref = torch.sparse.mm(ref_mat, B)
        worst = max(worst, float((out - ref).abs().max().item()))
    return worst <= tol and worst < 1.0, worst, tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default=str(Path(__file__).resolve().parents[1] / "datasets/gnn/exports"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "fgcs_results/revision/tf32/gnn_v5_dgl/dgl_column.csv"))
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--timed_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    device = torch.device("cuda")
    rows = []
    for name, spec in DATASETS.items():
        csr = sp.load_npz(Path(args.datasets_dir) / spec["npz"]).tocsr()
        csr.sum_duplicates(); csr.sort_indices()
        M = int(csr.shape[0])
        torch.manual_seed(args.seed)
        X = torch.randn((M, spec["in_dim"]), device=device)
        y = torch.randint(0, spec["out_dim"], (M,), device=device)

        torch.manual_seed(args.seed)
        graph = DGLGraphPair(csr, device)
        ok, err, tol = correctness(csr, graph, device, (spec["hidden_dim"], spec["out_dim"]))
        model = GCNBench(spec["in_dim"], spec["hidden_dim"], spec["out_dim"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        def step():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X, graph), y)
            loss.backward(); optimizer.step()
            torch.cuda.synchronize(); return time.perf_counter() - t0

        for _ in range(args.warmup_steps):
            step()
        times = [step() for _ in range(args.timed_steps)]
        mean_s = sum(times) / len(times)
        rows.append({
            "dataset": name, "backend": "dgl",
            "torch_version": torch.__version__,
            "mean_step_sec": mean_s,
            "std_step_sec": float(np.std(times)),
            "correct": ok, "max_error": err, "tolerance": tol,
            "warmup_steps": args.warmup_steps, "timed_steps": args.timed_steps,
        })
        print(f"{name:14s} dgl step={mean_s*1000:.2f} ms correct={ok} "
              f"(err {err:.3g} tol {tol:.3g})", flush=True)
        del graph, model, X, y
        torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
