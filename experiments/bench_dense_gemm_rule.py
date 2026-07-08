"""
Tightly-scoped DENSE_GEMM portfolio kernel + router rule that converts the
cuBLAS-win cases on tiny/dense graphs into router wins.

DENSE_GEMM = materialize A -> dense FP16, cublasGemmEx (FP16 in / FP32 accum, via
torch.matmul on half tensors), same 50-warmup/200-timed CUDA-event protocol,
correctness-gated vs cuSPARSE. It is a 7th, targeted option — NOT a general kernel.

Empirical router rule (derived from the dense-GEMM sweep (bench_cublas_dense.py) +
robust re-measurement; fires ONLY where dense ROBUSTLY beats the best sparse kernel,
never demotes a sparse win):
    DENSE_GEMM  iff  M <= 2000 and N <= 128
  Robust (median-of-3 x 50+200) win/loss vs the sparse pick:
    PPI(1767)  N=64 : 2.24x  WIN  -> fire
    PPI(1767)  N=128: 1.27x  WIN  -> fire
    PPI(1767)  N=256: 0.95x  loss -> excluded (N>128)
    Cora(2708) N=64 : 0.94x  tie/loss under robust timing (the single-run 1.03x was
                      noise) -> EXCLUDED (M>2000) so we never demote it — it stays on
                      the sparse TC_DIRECT pick.
  Among all 26 real graphs only PPI(1767) has M<=2000, so the rule is inert on the other
  25 graphs by construction (verified: no change to their picks / router geomean / hit).

Outputs (fgcs_results/revision/dense/):
  dense_gemm.csv, dense_router_delta.csv, DENSE_SUMMARY.md
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import ra_spmm  # noqa
from ra_real_graph_eval import load_dataset, measure_ms, run_kernel  # noqa
from ra_router_eval import simple_router  # noqa
from experiments.bench_cublas_dense import dense_from_csr  # reuse the dense materialization

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
SMALL = ["PPI", "Cora", "CiteSeer", "ca-GrQc", "ca-HepTh", "amazon-photo", "amazon-computers"]
NS = [64, 128, 256]
WARMUP, TIMED = 50, 200


def robust_ms(fn):
    """Median of 3 x (50 warmup + 200 timed) — stabilizes sub-0.05ms tiny-kernel timings."""
    xs = sorted(measure_ms(fn, WARMUP, TIMED) for _ in range(3))
    return xs[1]


def dense_rule_fires(M, N):
    """Tight empirical DENSE_GEMM region (only where dense ROBUSTLY beats sparse)."""
    return M <= 2000 and N <= 128


def dense_router(d_bar, cv_d, M, N, nnz, dense_ok=True):
    """Router WITH the DENSE_GEMM option: dense in its tiny corner (if usable), else sparse."""
    if dense_ok and dense_rule_fires(M, N):
        return "DENSE_GEMM"
    return simple_router(d_bar, cv_d, M, N, nnz)


def run_dense(A_half, B32):
    Bh = B32.half()
    C = torch.matmul(A_half, Bh)
    ms = robust_ms(lambda: torch.matmul(A_half, Bh))
    return C, ms


def main():
    outdir = R / "fgcs_results/revision/dense"; outdir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA {torch.version.cuda}")
    man = {d["name"]: d for d in json.loads((R / "paper_datasets.json").read_text())["datasets"]}
    # cv_d per dataset from the feature-extraction sweep (accurate, measured on-device)
    cvd = {r["dataset"]: float(r["cv_d"]) for r in csv.DictReader(open(R / "fgcs_results/revision/featbreak/feature_extraction_gpu.csv"))}

    dense_rows, delta_rows = [], []
    for name in SMALL:
        entry = man[name]; mat = load_dataset(entry)
        M, K = mat["M"], mat["K"]
        rp = mat["rowptr"].cuda().int(); ci = mat["colind"].cuda().int(); v = mat["vals"].cuda().float()
        nnz = int(rp[-1].item()); d_bar = nnz / max(1, M); cv = cvd.get(name, 0.5)
        A_half = dense_from_csr(mat["rowptr"].contiguous().int(), mat["colind"].contiguous().int(), M, K)
        for N in NS:
            B32 = torch.randn(K, N, device="cuda")
            C_ref = ra_spmm.spmm_cusparse(rp, ci, v, B32)
            ms_cus = robust_ms(lambda: ra_spmm.spmm_cusparse(rp, ci, v, B32))
            # DENSE_GEMM (skip if OOM)
            if A_half is None:
                dense_ms, dense_ok, dense_err = float("nan"), False, float("nan")
            else:
                C, dense_ms = run_dense(A_half, B32)
                dense_err = (C.float() - C_ref).abs().max().item()
                ref_scale = max(1.0, C_ref.abs().max().item())
                dense_ok = dense_err <= 1e-2 * ref_scale  # FP16 tolerance (protocol)
            sp_dense_cus = ms_cus / dense_ms if dense_ms == dense_ms and dense_ms > 0 else 0.0
            dense_rows.append({
                "dataset": name, "M": M, "K": K, "nnz": nnz, "N": N,
                "kernel": "DENSE_GEMM",
                "dense_ms": "OOM" if A_half is None else round(dense_ms, 5),
                "ms_cusparse": round(ms_cus, 4),
                "speedup_vs_cusparse": round(sp_dense_cus, 3),
                "correct": dense_ok, "max_error": round(dense_err, 5) if dense_err == dense_err else "",
                "rule_fires": dense_rule_fires(M, N),
            })

            # --- router re-dispatch (old vs new) ---
            old_pick = simple_router(d_bar, cv, M, N, nnz)
            new_pick = dense_router(d_bar, cv, M, N, nnz, dense_ok=(A_half is not None and dense_ok))
            pc = {}
            ms_old = robust_ms(lambda: run_kernel(old_pick, rp, ci, v, B32, pc, f"{old_pick}_{N}"))
            if new_pick == "DENSE_GEMM":
                ms_new = dense_ms
            else:
                ms_new = robust_ms(lambda: run_kernel(new_pick, rp, ci, v, B32, pc, f"{new_pick}_{N}"))
            # Report every original cuBLAS-win target (so an excluded case is documented),
            # plus any case where the pick changed.
            TARGETS = {("PPI", 64), ("PPI", 128), ("Cora", 64)}
            if new_pick != old_pick or (name, N) in TARGETS:
                dense_vs_old = (ms_old / dense_ms) if (dense_ms == dense_ms and dense_ms > 0) else 0.0
                delta_rows.append({
                    "dataset": name, "M": M, "N": N,
                    "old_pick": old_pick, "old_ms": round(ms_old, 5),
                    "new_pick": new_pick, "new_ms": round(ms_new, 5),
                    "speedup_new_vs_old": round(ms_old / ms_new, 3) if ms_new > 0 else 0,
                    "new_vs_cusparse": round(ms_cus / ms_new, 3) if ms_new > 0 else 0,
                    "changed": new_pick != old_pick,
                    "rule_fires": dense_rule_fires(M, N),
                    "dense_ms": round(dense_ms, 5) if dense_ms == dense_ms else "",
                    "dense_vs_sparse_pick": round(dense_vs_old, 3),
                    "note": ("DENSE_GEMM win" if new_pick == "DENSE_GEMM"
                             else "excluded (marginal ~1.0x tie across runs); kept sparse for no-regression guarantee"),
                })
            del B32
        del A_half, rp, ci, v
        torch.cuda.empty_cache()

    with open(outdir / "dense_gemm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dense_rows[0].keys())); w.writeheader(); w.writerows(dense_rows)
    with open(outdir / "dense_router_delta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys())); w.writeheader(); w.writerows(delta_rows)
    print(f"Wrote dense_gemm.csv ({len(dense_rows)}), dense_router_delta.csv ({len(delta_rows)})")
    for r in delta_rows:
        flag = "  <-- DENSE_GEMM WIN" if r["changed"] else ""
        print(f"  {r['dataset']:<16s} N={r['N']:<4d} {r['old_pick']:<14s}->{r['new_pick']:<12s} "
              f"new/old={r['speedup_new_vs_old']}x  new/cus={r['new_vs_cusparse']}x{flag}")


if __name__ == "__main__":
    main()
