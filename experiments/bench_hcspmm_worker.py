"""
Benchmark HC-SpMM on ONE graph, in an isolated process.

HC-SpMM's CUDA kernels can raise an illegal-memory-access on some graphs/dims,
which corrupts the CUDA context irrecoverably; running one graph per subprocess
contains the blast radius. Prints one JSON line per (graph, N) to stdout; exits 0
on success, nonzero if the kernel crashed (parent records a BUILD_NOTE).

HC-SpMM natively supports GNN embedding dims via fixed kernels: forward_fixed32
(N=32) and forward_fixed64 (N=64). N=64 is in the paper's N set, so it is the fair
native comparison point. The arbitrary-dim `forward` (N>=128) is unstable and is
not used.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "baselines" / "HC-SpMM" / "hybrid_kernel"))
import ra_spmm  # noqa
from ra_real_graph_eval import load_dataset, measure_ms, run_kernel  # noqa
from ra_router_eval import simple_router  # noqa
import HCSPMM  # noqa

FIXED = {32: "forward_fixed32", 64: "forward_fixed64"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--datasets-file", default=str(R / "paper_datasets.json"))
    ap.add_argument("--Ns", default="64")
    args = ap.parse_args()

    man = {d["name"]: d for d in json.loads(Path(args.datasets_file).read_text())["datasets"]}
    entry = man[args.dataset]
    mat = load_dataset(entry)
    if mat is None:
        print(json.dumps({"dataset": args.dataset, "error": "not_found"})); sys.exit(3)
    M = mat["M"]
    rp = mat["rowptr"].cuda().int(); ci = mat["colind"].cuda().int(); v = mat["vals"].cuda().float()
    nnz = int(rp[-1].item()); nrw = (M + 15) // 16
    d_bar = nnz / max(1, M)
    deg = (rp[1:] - rp[:-1]).float(); cv_d = float((deg.std() / deg.mean()).item()) if d_bar > 0 else 0.0

    t0 = time.perf_counter(); meta = HCSPMM.preprocess(ci, rp, M, nnz, nrw); torch.cuda.synchronize()
    preproc_ms = (time.perf_counter() - t0) * 1e3
    bp, e2c, e2r, ht, rnzr, cnzr = meta

    allowed = [int(x) for x in args.Ns.split(",")]
    entry_Ns = [int(n) for n in entry.get("Ns", [64, 128, 256, 512])]
    for N in allowed:
        if N not in FIXED or N not in entry_Ns:
            continue
        fn = getattr(HCSPMM, FIXED[N])
        X = torch.randn(M, N, device="cuda").contiguous()
        call = lambda: fn(X, rp, ci, bp, e2c, e2r, ht, rnzr, cnzr)
        out = call(); torch.cuda.synchronize()
        C = out[0] if isinstance(out, (list, tuple)) else out
        Cref = ra_spmm.spmm_cusparse(rp, ci, v, X)
        max_err = (C.float() - Cref).abs().max().item()
        ms_hc = measure_ms(call, 50, 200)
        ms_cus = measure_ms(lambda: ra_spmm.spmm_cusparse(rp, ci, v, X), 50, 200)
        rk = simple_router(d_bar, cv_d, M, N, nnz); pc = {}
        try:
            ms_router = measure_ms(lambda: run_kernel(rk, rp, ci, v, X, pc, f"{rk}_{N}"), 50, 200)
        except Exception:
            ms_router = float("nan")
        print(json.dumps({
            "dataset": args.dataset, "category": entry.get("category", "?"),
            "M": M, "nnz": nnz, "N": N, "kernel": "HC-SpMM",
            "ms": round(ms_hc, 4), "ms_cusparse": round(ms_cus, 4),
            "speedup_vs_cusparse": round(ms_cus / ms_hc, 3) if ms_hc > 0 else 0,
            "router_kernel": rk, "ms_router": round(ms_router, 4),
            "speedup_router_vs_hcspmm": round(ms_hc / ms_router, 3) if ms_router == ms_router and ms_router > 0 else 0,
            "correct": bool(max_err < 1e-1), "max_error": round(max_err, 5),
            "preproc_ms": round(preproc_ms, 4),
        }), flush=True)
        del X
    sys.exit(0)


if __name__ == "__main__":
    main()
