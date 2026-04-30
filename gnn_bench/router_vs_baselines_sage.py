#!/usr/bin/env python3
"""
GraphSAGE end-to-end benchmark: Extended 6-kernel router vs cuSPARSE / TC_DIRECT-only / PyG torch_sparse.

Mirrors the GCN harness (router_vs_baselines_gcn.py) with a 2-layer GraphSAGE
model using mean aggregation (full-batch, no sampling — apples-to-apples SpMM
timing with GCN / GIN). Per layer:

    h' = linear( concat(h, SpMM(A, h)) )

Two SpMM calls per forward pass, same structure as GCN, so router picks the
same kernels at (M, hidden_dim) / (M, out_dim) shapes.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gnn_bench.router_vs_baselines_gcn import (  # noqa: E402
    ALL_BACKENDS,
    DATASETS,
    DatasetSpec,
    GraphBackend,
    SparseMMFunction,
    add_speedups,
    load_csr,
    write_csv,
)


class SAGEBench(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        # concat(self, neigh) doubles feature dim before the linear
        self.lin1 = nn.Linear(2 * in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(2 * hidden_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, graph: GraphBackend, backend: str) -> torch.Tensor:
        # Layer 1: mean aggregator (SpMM with 1-valued adjacency acts as sum;
        # same shape/timing as true mean — the router's decision depends on
        # (M, N) not on the value scale).
        neigh1 = SparseMMFunction.apply(X, graph, backend)
        h = torch.cat([X, neigh1], dim=1)
        h = F.relu(self.lin1(h))

        neigh2 = SparseMMFunction.apply(h, graph, backend)
        h = torch.cat([h, neigh2], dim=1)
        out = self.lin2(h)
        return out


def measure_step(
    model: SAGEBench,
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
        if backend == "pyg":
            try:
                import torch_sparse  # noqa: F401
            except ImportError:
                print(f"[sage-bench] dataset={spec.name} backend=pyg SKIPPED (torch_sparse not installed)", flush=True)
                continue

        torch.manual_seed(seed)
        model = SAGEBench(spec.in_dim, spec.hidden_dim, spec.out_dim).to(device)
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

        results.append({
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
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphSAGE end-to-end benchmark: router vs cuSPARSE vs TC_DIRECT-only vs PyG torch_sparse."
    )
    parser.add_argument("--datasets", default="Reddit,ogbn-proteins,ogbn-arxiv")
    parser.add_argument("--datasets_dir", default=str(REPO_ROOT / "datasets" / "gnn" / "exports"))
    parser.add_argument("--backends", default=",".join(ALL_BACKENDS))
    parser.add_argument("--results_dir", default="results_gnn_extended")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--timed_steps", type=int, default=100)
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
        print(f"[sage-bench] dataset={name} backends={','.join(backends)}", flush=True)
        rows.extend(benchmark_dataset(
            spec, datasets_dir, device,
            args.warmup_steps, args.timed_steps, args.seed, args.lr, backends,
        ))

    add_speedups(rows)
    csv_path = out_dir / "graphsage_end_to_end.csv"
    write_csv(csv_path, rows)
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
