#!/usr/bin/env python3
"""True graph-cold end-to-end FIRST-training-step benchmark (GCN/GraphSAGE/GIN).

Timed unit is byte-for-byte the warm harness unit — zero_grad, forward,
cross-entropy loss, backward, optimizer.step, CUDA-synchronized at both ends —
with ONE addition that defines cold: graph-specific backend construction happens
INSIDE the timed region on both sides.
  router  (cold policy = deployed native CSR): 4-feature degree pass + rule
          route + CSR_DIRECT path prep + first execution. No tiled plan built
          inside the step, and none built outside any accounting.
  cusparse: graph-specific descriptor + workspace/preprocess build (once per
          direction and width) + first execution.
  pyg     : torch_sparse SparseTensor build (once per direction) + execution.

Process-global CUDA / PyTorch / extension state is preinitialized. Optimizer
(Adam) state is initialized once and an identical model+optimizer+RNG snapshot
is restored before EVERY trial, so no trial pays Adam-state initialization.
Between trials only graph-specific plans/descriptors are destroyed; the process,
CUDA context, loaded modules, and allocator pool are retained (no empty_cache).
Backend order is alternated per trial. >=25 independent trials; per-trial raw
latencies emitted (never ratios alone). Correctness on the cold step: per-SpMM
output vs cuSPARSE under the warm tolerance gate, plus loss agreement, gradient
agreement, and NaN/Inf absence. Claims are numerical agreement only.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, sys, time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import ra_spmm as spmm_next  # noqa: E402
from ra_router_eval import route_with_rules  # noqa: E402
import gnn_bench.router_vs_baselines_gcn as gcnmod  # noqa: E402
from gnn_bench.router_vs_baselines_gcn import DATASETS, GCNBench, BASE_ATOL, load_csr  # noqa: E402
import gnn_bench.router_vs_baselines_sage as sagemod  # noqa: E402
import gnn_bench.router_vs_baselines_gin as ginmod  # noqa: E402

MODELS = {"gcn": GCNBench, "graphsage": sagemod.SAGEBench, "gin": ginmod.GINBench}
DEFAULT_DATASETS = ["Reddit", "ogbn-proteins", "ogbn-arxiv", "PPI",
                    "amazon-photo", "amazon-computers", "Cora", "CiteSeer"]


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def _csr_tensors(csr, device):
    return dict(
        rowptr=torch.from_numpy(csr.indptr.astype(np.int32)).to(device),
        colind=torch.from_numpy(csr.indices.astype(np.int32)).to(device),
        vals=torch.from_numpy(csr.data.astype(np.float32)).to(device))


class ColdGraph:
    """Cold-mode SpMM backend: graph-specific state built lazily INSIDE the
    timed step, destroyed by reset() (no empty_cache). transpose selects A^T."""

    def __init__(self, csr, device, N_feature):
        self.device = device
        self.M = int(csr.shape[0]); self.nnz = int(csr.indptr[-1])
        self.fwd = _csr_tensors(csr, device)
        self.bwd = _csr_tensors(csr.T.tocsr(), device)
        deg = csr.indptr[1:] - csr.indptr[:-1]
        self.dbar = self.nnz / max(1, self.M); self.maxrow = max(1, int(deg.max()))
        self._N = int(N_feature)
        self.backend = "router"
        self._cus: Dict[tuple, object] = {}
        self._pyg: Dict[bool, object] = {}

    def _t(self, transpose):
        return self.bwd if transpose else self.fwd

    def reset(self):
        # Destroy only graph-specific plans/descriptors; keep context + allocator.
        self._cus.clear(); self._pyg.clear()
        torch.cuda.synchronize()

    def feature_route(self):
        t = self.fwd
        deg = (t["rowptr"][1:] - t["rowptr"][:-1]).float()
        cv = float((deg.std(correction=0) / deg.mean()).item()) if self.M else 0.0
        route_with_rules(self.dbar, cv, self.M, self._N, self.nnz)

    def run(self, B, transpose):
        t = self._t(transpose)
        if self.backend == "router":
            return spmm_next.spmm_csr_direct(t["rowptr"], t["colind"], t["vals"], B)
        if self.backend == "cusparse":
            k = (transpose, int(B.shape[1]))
            if k not in self._cus:
                tmpl = torch.empty((self.M, B.shape[1]), device=self.device, dtype=torch.float32)
                self._cus[k] = spmm_next.make_cusparse_plan(t["rowptr"], t["colind"], t["vals"], tmpl)
            return spmm_next.run_cusparse_plan(self._cus[k], B)
        if self.backend == "pyg":
            from torch_sparse import SparseTensor
            if transpose not in self._pyg:
                self._pyg[transpose] = SparseTensor(
                    rowptr=t["rowptr"].to(torch.long), col=t["colind"].to(torch.long),
                    value=t["vals"], sparse_sizes=(self.M, self.M))
            return self._pyg[transpose] @ B
        raise RuntimeError(self.backend)


def _install_dispatch():
    """Route the shared SparseMMFunction through ColdGraph when applicable."""
    class Dispatch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, graph, backend):
            ctx.graph = graph
            return graph.run(B, transpose=False)

        @staticmethod
        def backward(ctx, g):
            return ctx.graph.run(g.contiguous(), transpose=True), None, None
    gcnmod.SparseMMFunction = Dispatch
    sagemod.SparseMMFunction = Dispatch
    ginmod.SparseMMFunction = Dispatch


def timed_step(model, graph, backend, X, y, optimizer) -> float:
    """EXACT warm unit + cold graph-state build inside the timer."""
    graph.backend = backend
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if backend == "router":
        graph.feature_route()                       # RA feature+route, inside cold timer
    optimizer.zero_grad(set_to_none=True)
    logits = model(X, graph, backend)               # graph-specific descriptors built here (cold)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def correctness(graph, model, X, y, widths, backends, seed):
    """Per-SpMM output vs cuSPARSE (warm tol = BASE_ATOL*sqrt(max_row_nnz),
    CSR path no TC factor) + loss agreement + gradient agreement + NaN/Inf."""
    tol = BASE_ATOL * max(1.0, math.sqrt(graph.maxrow))
    res = {b: {"correct": True, "max_error": 0.0} for b in backends}
    with torch.no_grad():
        for transpose in (False, True):
            t = graph._t(transpose)
            for n in sorted(set(int(w) for w in widths)):
                torch.manual_seed(seed + n + int(transpose))
                B = torch.randn((graph.M, n), device=graph.device)
                ref = spmm_next.spmm_cusparse(t["rowptr"], t["colind"], t["vals"], B)
                for b in backends:
                    graph.backend = b; cur = graph.run(B, transpose); graph.reset()
                    err = float((cur.float() - ref.float()).abs().max().item())
                    res[b]["max_error"] = max(res[b]["max_error"], err)
                    if err > tol or err >= 1.0 or not math.isfinite(err):
                        res[b]["correct"] = False
    # loss + gradient agreement on one full step (router/pyg vs cusparse)
    def one_step_probe(backend):
        model.load_state_dict(snap_model); model.zero_grad(set_to_none=True)
        graph.backend = backend
        if backend == "router":
            graph.feature_route()
        logits = model(X, graph, backend); loss = F.cross_entropy(logits, y); loss.backward()
        g = next(p.grad.detach().clone() for p in model.parameters() if p.grad is not None)
        graph.reset()
        return float(loss.item()), logits.detach(), g
    snap_model = copy.deepcopy(model.state_dict())
    ref_loss, ref_logits, ref_grad = one_step_probe("cusparse")
    for b in backends:
        l, lo, gr = one_step_probe(b)
        # Per-SpMM aggregation output is the adopted (warm) gate above; the
        # accumulated 2-layer logits and gradients are checked RELATIVELY,
        # since their magnitude grows with depth and degree.
        denom_l = float(ref_logits.float().abs().max().item()) + 1e-9
        denom_g = float(ref_grad.float().abs().max().item()) + 1e-9
        logit_ok = bool((lo.float() - ref_logits.float()).abs().max().item() / denom_l <= 1e-2)
        loss_ok = bool(abs(l - ref_loss) <= max(1e-3, 1e-3 * abs(ref_loss)))
        grad_ok = bool((gr.float() - ref_grad.float()).abs().max().item() / denom_g <= 1e-2)
        finite = bool(math.isfinite(l))
        res[b]["loss_ok"] = loss_ok; res[b]["logit_ok"] = logit_ok
        res[b]["grad_ok"] = grad_ok; res[b]["finite"] = finite
        res[b]["correct"] = res[b]["correct"] and loss_ok and logit_ok and grad_ok and finite
    model.load_state_dict(snap_model)
    return res, tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(MODELS))
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--datasets_dir", default=str(REPO / "datasets/gnn/exports"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--include-pyg", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda")
    _install_dispatch()
    ModelCls = MODELS[args.arch]
    backends = ["router", "cusparse"] + (["pyg"] if args.include_pyg else [])

    # Preinitialize process-global CUDA/PyTorch/extension/torch_sparse state.
    w = torch.randn(256, 128, device=device); (w @ w.t()).sum().item()
    if args.include_pyg:
        from torch_sparse import SparseTensor  # noqa: F401
    torch.cuda.synchronize()

    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        spec = DATASETS[name]
        csr = load_csr(Path(args.datasets_dir) / spec.npz_name)
        graph = ColdGraph(csr, device, spec.hidden_dim)
        M = graph.M
        torch.manual_seed(args.seed)
        X = torch.randn((M, spec.in_dim), device=device)
        y = torch.randint(0, spec.out_dim, (M,), device=device)
        model = ModelCls(spec.in_dim, spec.hidden_dim, spec.out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        widths = [spec.in_dim, spec.hidden_dim, spec.out_dim]
        corr, tol = correctness(graph, model, X, y, widths, backends, args.seed)

        # Initialize Adam moment state with one throwaway warm step (not timed),
        # then snapshot identical model+optimizer state to restore before trials.
        graph.backend = "cusparse"
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(X, graph, "cusparse"), y).backward(); optimizer.step()
        graph.reset()
        snap_model = copy.deepcopy(model.state_dict())
        snap_opt = copy.deepcopy(optimizer.state_dict())
        rng_state = torch.cuda.get_rng_state(device)

        per_trial = {b: [] for b in backends}
        for trial in range(args.trials):
            order = backends if trial % 2 == 0 else list(reversed(backends))
            for b in order:
                model.load_state_dict(snap_model)
                optimizer.load_state_dict(snap_opt)
                torch.cuda.set_rng_state(rng_state, device)
                graph.reset()
                dt = timed_step(model, graph, b, X, y, optimizer) * 1000.0
                per_trial[b].append(dt)
                graph.reset()
        # per-trial rows (raw latencies, never ratios alone)
        for trial in range(args.trials):
            r = {"dataset": name, "architecture": args.arch, "mode": "cold",
                 "num_nodes": M, "nnz": graph.nnz,
                 "in_dim": spec.in_dim, "hidden_dim": spec.hidden_dim,
                 "out_dim": spec.out_dim, "feature_width": spec.hidden_dim,
                 "trial_id": trial, "seed": args.seed,
                 "router_ms": round(per_trial["router"][trial], 5),
                 "cusparse_ms": round(per_trial["cusparse"][trial], 5),
                 "pyg_ms": round(per_trial["pyg"][trial], 5) if "pyg" in backends else "",
                 "router_correct": corr["router"]["correct"],
                 "router_max_error": round(corr["router"]["max_error"], 6),
                 "pyg_correct": corr["pyg"]["correct"] if "pyg" in backends else "",
                 "tolerance": round(tol, 6)}
            r["router_vs_cusparse"] = round(r["cusparse_ms"] / r["router_ms"], 5)
            if "pyg" in backends:
                r["pyg_vs_cusparse"] = round(r["cusparse_ms"] / r["pyg_ms"], 5)
                r["router_vs_pyg"] = round(r["pyg_ms"] / r["router_ms"], 5)
            else:
                r["pyg_vs_cusparse"] = ""; r["router_vs_pyg"] = ""
            rows.append(r)
        med = {b: float(np.median(per_trial[b])) for b in backends}
        iqr = {b: float(np.percentile(per_trial[b], 75) - np.percentile(per_trial[b], 25)) for b in backends}
        pv = f" pyg={med.get('pyg', float('nan')):.3f}" if "pyg" in backends else ""
        print(f"{name:18s} router={med['router']:.3f}(IQR{iqr['router']:.3f}) "
              f"cus={med['cusparse']:.3f}{pv} R/cus={med['cusparse']/med['router']:.2f}x "
              f"correct={corr['router']['correct']}", flush=True)
        del X, y, model, optimizer, graph
        torch.cuda.empty_cache()  # only between DATASETS, never between trials

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    # dataset-median geomean for the log
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r["router_vs_cusparse"])
    print(f"\ncold {args.arch} router_vs_cusparse geomean(dataset medians): "
          f"{gm([float(np.median(v)) for v in by.values()]):.3f}x")
    print(f"wrote {out} ({len(rows)} per-trial rows)")


if __name__ == "__main__":
    main()
