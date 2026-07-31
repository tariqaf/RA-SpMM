#!/usr/bin/env python3
"""Emit the deliverable GNN CSVs + a strict key-value manifest.

Warm CSVs (gnn_warm_<arch>.csv): from the standing warm 8x3 run (100 timed
steps after 20 warmup), raw latencies with spread (mean/std/min/max) and the
router_vs_cusparse / pyg_vs_cusparse / router_vs_pyg ratios. The warm table is
NOT re-run (routing verified unchanged on the 7-rule binary; ALG2 frozen).
Cold CSVs (gnn_cold_<arch>.csv): per-trial passthrough from cold_first_step.py.
Both carry dataset, architecture, mode, feature_width, seed, correctness.
"""
from __future__ import annotations
import csv, json, math, os, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))  # so gnn_bench / ra_spmm import when run as a script
# Output dir is env-overridable so the corrected rerun can land beside (not on
# top of) the pre-correction gnn_v5_full baseline.
OUT = Path(os.environ.get("GNN_OUT", str(R / "fgcs_results/revision/tf32/gnn_v5_full")))
DS8 = ["Reddit", "ogbn-proteins", "ogbn-arxiv", "PPI",
       "amazon-photo", "amazon-computers", "Cora", "CiteSeer"]
WARM = {"gcn": OUT / "gcn/gcn_end_to_end.csv",
        "graphsage": OUT / "sage/graphsage_end_to_end.csv",
        "gin": OUT / "gin8/gin_end_to_end.csv"}
ARCH_KEY = {"gcn": "gcn", "graphsage": "graphsage", "gin": "gin"}


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def emit_warm(arch, path):
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r["dataset"], {})[r["backend"]] = r
    rows = []
    for d in DS8:
        if d not in by or "router" not in by[d] or "cusparse" not in by[d]:
            raise SystemExit(f"warm {arch}: missing {d}")
        b = by[d]

        def ms(bk, col="mean_step_sec"):
            return float(b[bk][col]) * 1000 if bk in b and b[bk].get(col) else ""
        rr, cc = ms("router"), ms("cusparse")
        pg = ms("pyg") if "pyg" in b else ""
        row = {"dataset": d, "architecture": arch, "mode": "warm",
               "num_nodes": b["router"]["num_nodes"], "nnz": b["router"]["nnz"],
               "feature_width": b["router"].get("hidden_dim", 128),
               "seed": b["router"].get("seed", 123), "n_timed_steps": b["router"].get("timed_steps", 100),
               "router_ms": round(rr, 5), "router_ms_std": round(ms("router", "std_step_sec"), 5),
               "router_ms_min": round(ms("router", "min_step_sec"), 5),
               "router_ms_max": round(ms("router", "max_step_sec"), 5),
               "cusparse_ms": round(cc, 5), "cusparse_ms_std": round(ms("cusparse", "std_step_sec"), 5),
               "pyg_ms": round(pg, 5) if pg != "" else "",
               "pyg_ms_std": round(ms("pyg", "std_step_sec"), 5) if "pyg" in b else "",
               "router_kernel": b["router"].get("router_forward_hidden", ""),
               "router_correct": b["router"]["correct"],
               "pyg_correct": b["pyg"]["correct"] if "pyg" in b else "",
               "router_vs_cusparse": round(cc / rr, 5),
               "pyg_vs_cusparse": round(cc / pg, 5) if pg != "" else "",
               "router_vs_pyg": round(pg / rr, 5) if pg != "" else ""}
        rows.append(row)
    outp = OUT / f"gnn_warm_{arch}.csv"
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"WARM {arch}: R/cus gm={gm([r['router_vs_cusparse'] for r in rows]):.3f}x "
          f"R/pyg gm={gm([r['router_vs_pyg'] for r in rows if r['router_vs_pyg']]):.3f}x -> {outp.name}")


def passthrough_cold(arch):
    src = OUT / f"cold_{arch}.csv"
    dst = OUT / f"gnn_cold_{arch}.csv"
    rows = list(csv.DictReader(open(src)))
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # dataset-median geomean
    import numpy as np
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(float(r["router_vs_cusparse"]))
    print(f"COLD {arch}: R/cus gm(medians)={gm([float(np.median(v)) for v in by.values()]):.3f}x "
          f"({len(rows)} per-trial rows) -> {dst.name}")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(R), "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# Per-model SpMM sites: (site name, feature-width key, transpose).
SITES = {
    "gcn": [("forward_hidden", "hidden", False), ("forward_out", "out", False),
            ("backward_hidden", "hidden", True), ("backward_out", "out", True)],
    "graphsage": [("forward_in", "in", False), ("forward_hidden", "hidden", False),
                  ("backward_in", "in", True), ("backward_hidden", "hidden", True)],
    "gin": [("forward_in", "in", False), ("forward_hidden", "hidden", False),
            ("backward_in", "in", True), ("backward_hidden", "hidden", True)],
}


def operator_manifest():
    """Per (dataset, model): forward AND transposed operator descriptors, plus the
    resolved routed path (kernel family + precision) at each SpMM site. The
    transposed descriptors make the forward/backward routing asymmetry (e.g.
    ogbn-arxiv, where A^T's degree CV crosses the ZERO_OVERHEAD_CSR threshold)
    verifiable rather than asserted. Everything is derived from the same faithful
    per-model operators the timed harness multiplies."""
    import numpy as np
    import torch
    import ra_spmm as spmm_next
    from gnn_bench.router_vs_baselines_gcn import (
        DATASETS, load_csr, build_aggregation_operator, operator_descriptors,
        resolve_router_label)

    def route(mat, M, N):
        plan = spmm_next.make_router_plan(
            torch.from_numpy(mat.indptr.astype(np.int32)),
            torch.from_numpy(mat.indices.astype(np.int32)),
            torch.from_numpy(mat.data.astype(np.float32)), M, M, int(N), "MAIN")
        return resolve_router_label(plan)

    ops = {}
    for name in DS8:
        spec = DATASETS[name]
        raw = load_csr(R / "datasets/gnn/exports" / spec.npz_name)
        widthmap = {"in": spec.in_dim, "hidden": spec.hidden_dim, "out": spec.out_dim}
        ops[name] = {}
        for model in ("gcn", "graphsage", "gin"):
            op = build_aggregation_operator(raw, model)
            opT = op.T.tocsr()
            M = int(op.shape[0])
            sites = {}
            for site, wkey, transpose in SITES[model]:
                N = widthmap[wkey]
                mat = opT if transpose else op
                sites[site] = {"N": N, "transpose": transpose,
                               "resolved_path": route(mat, M, N)}
            ops[name][model] = {
                "forward_descriptors": operator_descriptors(op),
                "transposed_descriptors": operator_descriptors(opT),
                "sites": sites,
            }
    return ops


def manifest():
    import torch
    m = {
        "code_commit": git_commit(),
        "router_rule_count": 7,
        "cusparse_algorithm": "CUSPARSE_SPMM_CSR_ALG2",
        "pyg_backend": "torch_sparse SparseTensor fused SpMM path (sparse-aggregation backend), NOT full torch_geometric message-passing layers",
        "pyg_torch_sparse_version": "0.6.18+pt27cu118",
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "per-site; FP32 baseline with router-gated TF32 tile paths "
                     "(see operators[dataset][model].sites[*].resolved_path — a "
                     "_TF32 suffix denotes the TF32 mma tile path, FP32 accumulate)",
        "trial_count_cold": 25,
        "warm_timed_steps": 100,
        "warm_warmup_steps": 20,
        "warm_reset_definition": "graph-specific plans/descriptors built ONCE outside the timed region; timed unit = zero_grad+forward+cross_entropy+backward+optimizer.step, mean of 100 timed steps",
        "cold_reset_definition": "graph-specific plans/descriptors built INSIDE the timed region (router: feature+route+native-CSR; cusparse: descriptor+workspace); optimizer(Adam) state preinitialized and identical model+optimizer+RNG snapshot restored before every trial; between trials only graph-specific state destroyed + synchronize, no empty_cache; per-trial median over 25 trials",
        "timed_unit": "zero_grad,forward,cross_entropy_loss,backward,optimizer_step,cuda_synchronized_both_ends",
        "tolerance": "per_spmm BASE_ATOL*sqrt(max_row_nnz); logits/gradients relative<=1e-2; loss relative<=1e-3",
        "datasets": DS8,
        "architectures": ["gcn", "graphsage", "gin"],
        "operator_definitions": {
            "gcn": "A_hat = D~^-1/2 (A + I) D~^-1/2, self-loop added only where absent",
            "graphsage": "A_mean = D^-1 A, no self-loop, zero-degree rows preserved",
            "gin": "raw adjacency A ((1+eps) self term supplied inside the model)",
        },
        "operators": operator_manifest(),
    }
    outp = OUT / "gnn_manifest.json"
    outp.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(f"manifest -> {outp.name}")


if __name__ == "__main__":
    for arch in ("gcn", "graphsage", "gin"):
        emit_warm(arch, WARM[arch])
        passthrough_cold(arch)
    manifest()
