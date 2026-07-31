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
from ra_real_graph_eval import (BASE_ATOL, TC_EXTRA_FACTOR, load_dataset,
                                measure_ms, measure_one_ms, population_cv)  # noqa
import HCSPMM  # noqa

FIXED = {32: "forward_fixed32", 64: "forward_fixed64"}
WARMUP = 50
TIMED = 200
COLD_ITERS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--datasets-file", default=str(R / "paper_datasets.json"))
    ap.add_argument("--Ns", default="64")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--timed", type=int, default=TIMED)
    ap.add_argument("--cold-iters", type=int, default=COLD_ITERS)
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
    deg = (mat["rowptr"][1:] - mat["rowptr"][:-1]).float()
    cv_d = population_cv(deg)
    max_row_nnz = max(1, int(deg.max().item()))

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
        tolerance = BASE_ATOL * max(1.0, max_row_nnz ** 0.5) * TC_EXTRA_FACTOR
        correct = max_err <= tolerance and max_err < 1.0
        ms_hc_warm = measure_ms(call, args.warmup, args.timed) if correct else float("nan")
        setup_total = 0.0
        exec_total = 0.0
        if correct:
            for _ in range(max(1, args.cold_iters)):
                torch.cuda.synchronize()
                cold_start = time.perf_counter()
                cold_meta = HCSPMM.preprocess(ci, rp, M, nnz, nrw)
                torch.cuda.synchronize()
                setup_total += (time.perf_counter() - cold_start) * 1e3
                cbp, ce2c, ce2r, cht, crnzr, ccnzr = cold_meta
                cold_out, cold_exec = measure_one_ms(
                    lambda: fn(X, rp, ci, cbp, ce2c, ce2r, cht, crnzr, ccnzr))
                exec_total += cold_exec
                del cold_out, cold_meta
            preproc_ms = setup_total / max(1, args.cold_iters)
            cold_exec_ms = exec_total / max(1, args.cold_iters)
            ms_hc_cold = preproc_ms + cold_exec_ms
        else:
            cold_exec_ms = ms_hc_cold = float("nan")
        cus_warm = ra_spmm.benchmark_cusparse(rp, ci, v, X, args.warmup, args.timed)
        cus_cold = ra_spmm.benchmark_cusparse_cold(rp, ci, v, X, args.cold_iters)
        print(json.dumps({
            "dataset": args.dataset, "category": entry.get("category", "?"),
            "M": M, "nnz": nnz, "N": N, "kernel": "HC-SpMM",
            "status": "OK" if correct else "INCORRECT",
            "ms_warm": round(ms_hc_warm, 6),
            "preprocess_ms": round(preproc_ms, 6),
            "cold_exec_ms": round(cold_exec_ms, 6),
            "ms_cold": round(ms_hc_cold, 6),
            "ms_cusparse_warm": round(float(cus_warm["exec_ms"]), 6),
            "ms_cusparse_cold": round(float(cus_cold["total_ms"]), 6),
            "speedup_vs_cusparse_warm": round(float(cus_warm["exec_ms"]) / ms_hc_warm, 6) if correct else None,
            "speedup_vs_cusparse_cold": round(float(cus_cold["total_ms"]) / ms_hc_cold, 6) if correct else None,
            "correct": correct,
            "soft_fail": tolerance < max_err < 1.0,
            "hard_fail": max_err >= 1.0,
            "max_error": round(max_err, 8),
            "tolerance": tolerance,
            "warmup": args.warmup, "timed_iters": args.timed,
            "cold_iters": args.cold_iters,
            "error": "",
        }), flush=True)
        del X
    sys.exit(0)


if __name__ == "__main__":
    main()
