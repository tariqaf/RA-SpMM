#!/usr/bin/env python3
"""
GCN end-to-end benchmark: Extended 6-kernel router vs cuSPARSE / TC_DIRECT-only / PyG torch_sparse.

Adapted from the original RA-SpMM harness (router_vs_cusparse_gcn.py) to:
  - Import `ra_spmm` (Extended project binding) instead of `spmm_next`
  - Dispatch the 6 new Extended kernel paths in addition to the 4 original paths
  - Add TC_DIRECT-only backend (best single kernel, for "why not just one kernel?" answer)
  - Add PyG torch_sparse backend as external framework baseline

Usage:
  python gnn_bench/router_vs_baselines_gcn.py --datasets Reddit,ogbn-proteins,ogbn-arxiv \
    --datasets_dir /path/to/gnn/exports --results_dir results_gnn

The --datasets_dir points at a directory containing <Name>.npz CSR files (e.g. Reddit.npz).
"""
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ra_spmm as spmm_next  # alias keeps dispatcher body identical to the Original


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    npz_name: str       # file name inside --datasets_dir
    in_dim: int
    out_dim: int
    hidden_dim: int


# Mirrors the Original RA-SpMM dataset set; dimensions match the original paper's GCN setup.
DATASETS: Dict[str, DatasetSpec] = {
    "Reddit":           DatasetSpec("Reddit",           "Reddit.npz",           in_dim=602, out_dim=41,  hidden_dim=128),
    "ogbn-products":    DatasetSpec("ogbn-products",    "ogbn-products.npz",    in_dim=100, out_dim=47,  hidden_dim=128),
    "ogbn-arxiv":       DatasetSpec("ogbn-arxiv",       "ogbn-arxiv.npz",       in_dim=128, out_dim=40,  hidden_dim=128),
    "ogbn-proteins":    DatasetSpec("ogbn-proteins",    "ogbn-proteins.npz",    in_dim=8,   out_dim=112, hidden_dim=128),
    # FGCS additions (Mixed/Irregular + Dense Small) ----------------------
    "Flickr":           DatasetSpec("Flickr",           "Flickr.npz",           in_dim=500, out_dim=7,   hidden_dim=128),
    "PPI":              DatasetSpec("PPI",              "PPI.npz",              in_dim=50,  out_dim=121, hidden_dim=128),
    "amazon-photo":     DatasetSpec("amazon-photo",     "amazon-photo.npz",     in_dim=745, out_dim=8,   hidden_dim=128),
    "amazon-computers": DatasetSpec("amazon-computers", "amazon-computers.npz", in_dim=767, out_dim=10,  hidden_dim=128),
    # FGCS round-2 additions (Cora + CiteSeer Planetoid graphs) ----------
    "Cora":             DatasetSpec("Cora",             "Cora.npz",             in_dim=1433,out_dim=7,   hidden_dim=128),
    "CiteSeer":         DatasetSpec("CiteSeer",         "CiteSeer.npz",         in_dim=3703,out_dim=6,   hidden_dim=128),
}


# ---------------------------------------------------------------------------
# Backends (Extended adds TC_DIRECT-only and PyG torch_sparse)
# ---------------------------------------------------------------------------
ALL_BACKENDS = ("router", "cusparse", "tc_direct", "pyg")


# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------
def load_csr(path: Path) -> sp.csr_matrix:
    csr = sp.load_npz(path).tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def csr_to_tensors(csr: sp.csr_matrix, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "rowptr_cpu": torch.from_numpy(np.asarray(csr.indptr, dtype=np.int32)),
        "colind_cpu": torch.from_numpy(np.asarray(csr.indices, dtype=np.int32)),
        "vals_cpu":   torch.from_numpy(np.asarray(csr.data,    dtype=np.float32)),
        "rowptr":     torch.from_numpy(np.asarray(csr.indptr,  dtype=np.int32)).to(device=device),
        "colind":     torch.from_numpy(np.asarray(csr.indices, dtype=np.int32)).to(device=device),
        "vals":       torch.from_numpy(np.asarray(csr.data,    dtype=np.float32)).to(device=device),
    }


# ---------------------------------------------------------------------------
# GraphBackend — handles forward/backward CSRs and kernel dispatch
# ---------------------------------------------------------------------------
class GraphBackend:
    def __init__(self, csr: sp.csr_matrix, device: torch.device) -> None:
        self.device = device
        self.csr = csr
        self.csr_t = csr.transpose().tocsr()
        self.forward = csr_to_tensors(self.csr, device)
        self.backward = csr_to_tensors(self.csr_t, device)
        self.plan_cache: Dict[Tuple[bool, int], Dict[str, object]] = {}
        self.exec_cache: Dict[Tuple[str, bool, int], object] = {}

        # Optional: cache a torch_sparse SparseTensor for PyG backend (built lazily)
        self._pyg_sparse_fwd: Optional[object] = None
        self._pyg_sparse_bwd: Optional[object] = None

    # -- shared helpers ------------------------------------------------------
    def _tensors(self, transpose: bool) -> Dict[str, torch.Tensor]:
        return self.backward if transpose else self.forward

    def get_router_plan(self, transpose: bool, ncols: int) -> Dict[str, object]:
        key = (transpose, int(ncols))
        if key not in self.plan_cache:
            tensors = self._tensors(transpose)
            M = int(tensors["rowptr_cpu"].numel()) - 1
            K = M
            self.plan_cache[key] = spmm_next.make_router_plan(
                tensors["rowptr_cpu"],
                tensors["colind_cpu"],
                tensors["vals_cpu"],
                M,
                K,
                int(ncols),
                "MAIN",
            )
        return self.plan_cache[key]

    # -- PyG torch_sparse SparseTensor (lazy) --------------------------------
    def _pyg_get_sparse(self, transpose: bool):
        from torch_sparse import SparseTensor
        cache_attr = "_pyg_sparse_bwd" if transpose else "_pyg_sparse_fwd"
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached
        tensors = self._tensors(transpose)
        M = int(tensors["rowptr_cpu"].numel()) - 1
        st = SparseTensor(
            rowptr=tensors["rowptr"].to(torch.long),
            col=tensors["colind"].to(torch.long),
            value=tensors["vals"],
            sparse_sizes=(M, M),
        )
        setattr(self, cache_attr, st)
        return st

    # -- dispatcher ----------------------------------------------------------
    def _get_executor(self, backend: str, transpose: bool, ncols: int):
        key = (backend, transpose, int(ncols))
        if key in self.exec_cache:
            return self.exec_cache[key]

        tensors = self._tensors(transpose)
        M = int(tensors["rowptr_cpu"].numel()) - 1
        K = M

        # -- cuSPARSE baseline --
        if backend == "cusparse":
            executor = ("cusparse", None)

        # -- PyG torch_sparse external baseline --
        elif backend == "pyg":
            executor = ("pyg", self._pyg_get_sparse(transpose))

        # -- TC_DIRECT-only (best single kernel) --
        elif backend == "tc_direct":
            wrapper = spmm_next.make_tc_direct_plan(
                tensors["rowptr_cpu"],
                tensors["colind_cpu"],
                tensors["vals_cpu"],
                M,
                K,
                int(ncols),
            )
            executor = ("TC_DIRECT", wrapper)

        # -- Router: ask the C++ router which path to use --
        elif backend == "router":
            plan = self.get_router_plan(transpose, ncols)
            path = str(plan["chosen_path"])

            # Original 4-kernel portfolio paths
            if path == "CSR_DIRECT":
                executor = (path, None)
            elif path == "CUSPARSE":
                executor = (path, None)
            elif path == "ROW_SPLIT_CUDA":
                wrapper = spmm_next.make_row_split_plan(
                    tensors["rowptr_cpu"], M, K,
                )
                executor = (path, wrapper)
            elif path == "TC_REORDERED":
                wrapper = spmm_next.make_tc_reordered_plan(
                    tensors["rowptr_cpu"], tensors["colind_cpu"], tensors["vals_cpu"],
                    M, K, int(ncols),
                )
                executor = (path, wrapper)
            elif path == "HYBRID_TC_CUDA":
                wrapper = spmm_next.make_hybrid_tc_cuda_plan(
                    tensors["rowptr_cpu"], tensors["colind_cpu"], tensors["vals_cpu"],
                    M, K, int(ncols), 0.45,
                )
                executor = (path, wrapper)

            # Extended 6-kernel portfolio paths (new)
            elif path == "TC_DIRECT":
                wrapper = spmm_next.make_tc_direct_plan(
                    tensors["rowptr_cpu"], tensors["colind_cpu"], tensors["vals_cpu"],
                    M, K, int(ncols),
                )
                executor = (path, wrapper)
            elif path == "COMMUNITY_TC":
                wrapper = spmm_next.make_community_tc_plan(
                    tensors["rowptr_cpu"], tensors["colind_cpu"], tensors["vals_cpu"],
                    M, K, int(ncols),
                )
                executor = (path, wrapper)
            elif path == "RODE_ENHANCED":
                wrapper = spmm_next.make_rode_enhanced_plan(
                    tensors["rowptr_cpu"], M, K,
                )
                executor = (path, wrapper)
            elif path == "ZERO_OVERHEAD_CSR":
                wrapper = spmm_next.make_zero_overhead_plan(
                    tensors["rowptr_cpu"], M, K,
                )
                executor = (path, wrapper)
            elif path == "SEGMENT_HYBRID":
                wrapper = spmm_next.make_segment_hybrid_plan(
                    tensors["rowptr_cpu"], tensors["colind_cpu"], tensors["vals_cpu"],
                    M, K, int(ncols),
                )
                executor = (path, wrapper)
            else:
                raise RuntimeError(f"Unsupported routed path in benchmark: {path}")
        else:
            raise ValueError(f"Unknown backend {backend}")

        self.exec_cache[key] = executor
        return executor

    # -- runner --------------------------------------------------------------
    def run(self, B: torch.Tensor, backend: str, transpose: bool = False) -> torch.Tensor:
        tensors = self._tensors(transpose)
        path, wrapper = self._get_executor(backend, transpose, int(B.size(1)))

        # cuSPARSE / CUSPARSE router-path
        if path in ("cusparse", "CUSPARSE"):
            return spmm_next.spmm_cusparse(
                tensors["rowptr"], tensors["colind"], tensors["vals"], B,
            )

        # PyG torch_sparse
        if path == "pyg":
            import torch_sparse
            # torch_sparse.spmm expects (index, value, M, K, B) OR uses SparseTensor @ B
            # SparseTensor matmul is the idiomatic path; equivalent to cuSPARSE internally.
            return wrapper @ B

        # Original 4-kernel paths
        if path == "CSR_DIRECT":
            return spmm_next.spmm_csr_direct(
                tensors["rowptr"], tensors["colind"], tensors["vals"], B,
            )
        if path == "ROW_SPLIT_CUDA":
            return spmm_next.run_row_split_plan(
                wrapper, tensors["colind"], tensors["vals"], B,
            )
        if path == "TC_REORDERED":
            return spmm_next.run_tc_reordered_plan(wrapper, B)
        if path == "HYBRID_TC_CUDA":
            return spmm_next.run_hybrid_tc_cuda_plan(wrapper, B)

        # Extended 6-kernel paths (new)
        if path == "TC_DIRECT":
            return spmm_next.run_tc_direct_plan(wrapper, B)
        if path == "COMMUNITY_TC":
            return spmm_next.run_community_tc_plan(wrapper, B)
        if path == "RODE_ENHANCED":
            return spmm_next.run_rode_enhanced_plan(
                wrapper, tensors["colind"], tensors["vals"], B,
            )
        if path == "ZERO_OVERHEAD_CSR":
            return spmm_next.run_zero_overhead_plan(
                wrapper, tensors["rowptr"], tensors["colind"], tensors["vals"], B,
            )
        if path == "SEGMENT_HYBRID":
            return spmm_next.run_segment_hybrid_plan(
                wrapper, tensors["colind"], tensors["vals"], B,
            )

        raise RuntimeError(f"Unsupported executor path {path}")


# ---------------------------------------------------------------------------
# Autograd-friendly SpMM and GCN model
# ---------------------------------------------------------------------------
class SparseMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B: torch.Tensor, graph: GraphBackend, backend: str) -> torch.Tensor:
        ctx.graph = graph
        ctx.backend = backend
        return graph.run(B, backend, transpose=False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_B = ctx.graph.run(grad_output.contiguous(), ctx.backend, transpose=True)
        return grad_B, None, None


class GCNBench(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, graph: GraphBackend, backend: str) -> torch.Tensor:
        h = self.lin1(X)
        h = SparseMMFunction.apply(h, graph, backend)
        h = F.relu(h)
        out = self.lin2(h)
        out = SparseMMFunction.apply(out, graph, backend)
        return out


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def measure_step(
    model: GCNBench,
    graph: GraphBackend,
    backend: str,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(X, graph, backend)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark_dataset(
    spec: DatasetSpec,
    datasets_dir: Path,
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
    seed: int,
    lr: float,
    backends: Tuple[str, ...],
) -> List[Dict[str, object]]:
    npz_path = datasets_dir / spec.npz_name
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {npz_path}")
    csr = load_csr(npz_path)
    num_nodes = int(csr.shape[0])
    graph = GraphBackend(csr, device)

    torch.manual_seed(seed)
    X = torch.randn((num_nodes, spec.in_dim), device=device, dtype=torch.float32)
    y = torch.randint(0, spec.out_dim, (num_nodes,), device=device, dtype=torch.long)

    results: List[Dict[str, object]] = []
    for backend in backends:
        # Skip PyG if not importable — record as skipped rather than crashing the whole suite
        if backend == "pyg":
            try:
                import torch_sparse  # noqa: F401
            except ImportError:
                print(f"[gnn-bench] dataset={spec.name} backend=pyg SKIPPED (torch_sparse not installed)", flush=True)
                continue

        torch.manual_seed(seed)
        model = GCNBench(spec.in_dim, spec.hidden_dim, spec.out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(warmup_steps):
            _ = measure_step(model, graph, backend, X, y, optimizer)

        times: List[float] = []
        for _ in range(timed_steps):
            times.append(measure_step(model, graph, backend, X, y, optimizer))

        router_paths: Dict[str, str] = {}
        if backend == "router":
            for tag, ncols in (
                ("forward_hidden", spec.hidden_dim),
                ("backward_hidden", spec.hidden_dim),
                ("forward_out",   spec.out_dim),
                ("backward_out",  spec.out_dim),
            ):
                transpose = tag.startswith("backward")
                plan = graph.get_router_plan(transpose, ncols)
                router_paths[tag] = str(plan["chosen_path"])

        results.append(
            {
                "dataset":              spec.name,
                "num_nodes":            num_nodes,
                "nnz":                  int(csr.nnz),
                "in_dim":               spec.in_dim,
                "hidden_dim":           spec.hidden_dim,
                "out_dim":              spec.out_dim,
                "backend":              backend,
                "warmup_steps":         warmup_steps,
                "timed_steps":          timed_steps,
                "mean_step_sec":        float(np.mean(times)),
                "std_step_sec":         float(np.std(times)),
                "min_step_sec":         float(np.min(times)),
                "max_step_sec":         float(np.max(times)),
                "router_forward_hidden":  router_paths.get("forward_hidden", ""),
                "router_backward_hidden": router_paths.get("backward_hidden", ""),
                "router_forward_out":     router_paths.get("forward_out", ""),
                "router_backward_out":    router_paths.get("backward_out", ""),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_speedups(rows: List[Dict[str, object]]) -> None:
    """For each dataset, add speedup columns of every non-cuSPARSE backend vs cuSPARSE."""
    by_dataset: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["backend"]] = row

    for dataset, parts in by_dataset.items():
        if "cusparse" not in parts:
            continue
        cusparse = float(parts["cusparse"]["mean_step_sec"])
        for backend, row in parts.items():
            t = float(row["mean_step_sec"])
            row[f"speedup_vs_cusparse"] = cusparse / t if t > 0 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GCN end-to-end benchmark: router vs cuSPARSE vs TC_DIRECT-only vs PyG torch_sparse."
    )
    parser.add_argument("--datasets", default="Reddit,ogbn-proteins,ogbn-arxiv",
                        help="Comma-separated dataset names (subset of: Reddit, ogbn-products, ogbn-arxiv, ogbn-proteins)")
    parser.add_argument("--datasets_dir", default=str(REPO_ROOT / "datasets" / "gnn" / "exports"),
                        help="Directory containing <name>.npz files")
    parser.add_argument("--backends", default=",".join(ALL_BACKENDS),
                        help=f"Comma-separated backends. Available: {', '.join(ALL_BACKENDS)}")
    parser.add_argument("--results_dir", default="results_gnn_extended")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--timed_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    selected = [t.strip() for t in args.datasets.split(",") if t.strip()]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(DATASETS)}")

    backends = tuple(t.strip() for t in args.backends.split(",") if t.strip())
    unknown_bk = sorted(set(backends) - set(ALL_BACKENDS))
    if unknown_bk:
        raise ValueError(f"Unknown backend(s): {', '.join(unknown_bk)}. Available: {', '.join(ALL_BACKENDS)}")

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets dir not found: {datasets_dir}")

    device = torch.device("cuda")
    out_dir = REPO_ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for name in selected:
        spec = DATASETS[name]
        print(f"[gnn-bench] dataset={name} backends={','.join(backends)}", flush=True)
        rows.extend(benchmark_dataset(
            spec, datasets_dir, device,
            args.warmup_steps, args.timed_steps, args.seed, args.lr, backends,
        ))

    add_speedups(rows)
    csv_path = out_dir / "gcn_end_to_end.csv"
    write_csv(csv_path, rows)
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
