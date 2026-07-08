"""
MP-SpMM baseline (SC'25, 2:4 Sparse Tensor Cores).
Code: Zenodo 10.5281/zenodo.16933452 (CGCL-codes/MP-SpMM).

MP-SpMM accelerates unstructured SpMM with 2:4 Structured Sparse Tensor Cores. It
PADS (does not prune) so the result is exact. It is preprocessing-heavy: a
"match-and-pad" step converts CSR into 2:4 block metadata, which we time SEPARATELY
from the kernel. Its SpMM kernel supports N in {32,128}; N=128 is
the comparison point in the paper's N set.

Pipeline per graph:
  1. export our CSR to Matrix Market (.mtx) in a scratch dataset/ dir,
  2. run the match-and-pad preprocessing binary (wall-clock timed = preprocessing),
  3. run the MP-SpMM kernel binary at N=128 (prints kernel ms + GFLOP/s),
  4. compare kernel-only time vs cuSPARSE (measured here, same protocol) and vs our
     router's chosen kernel.
Binaries were built for SM 86 (impl_cu_sm86, spmm_sm86). Graphs whose .mtx/preproc
is too heavy are skipped with a BUILD_NOTE.

Outputs:
  fgcs_results/revision/baselines/mp_spmm.csv
  fgcs_results/revision/baselines/mp_spmm_preproc.csv
  fgcs_results/revision/baselines/mpspmm_BUILD_NOTE.txt
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import ra_spmm  # noqa
from ra_real_graph_eval import load_dataset, measure_ms, run_kernel  # noqa
from ra_router_eval import simple_router  # noqa

MP = R / "baselines/MP-SpMM_code/MP-SpMM_SC25/mpspmm"
PREPROC_BIN = MP / "preprocessing" / "impl_cu_sm86"
SPMM_BIN = MP / "SpMM" / "spmm_sm86"
N_MP = 128
WARMUP, TIMED = 50, 200


def write_mtx(path: Path, rowptr, colind, M, K):
    """Write CSR adjacency (unit values) as a Matrix Market coordinate-real general file."""
    nnz = int(rowptr[-1])
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{M} {K} {nnz}\n")
        rp = rowptr.tolist()
        ci = colind.tolist()
        # MTX is 1-indexed
        lines = []
        for i in range(M):
            for p in range(rp[i], rp[i + 1]):
                lines.append(f"{i+1} {ci[p]+1} 1\n")
        f.writelines(lines)


def setup_workdir(work: Path):
    (work / "dataset").mkdir(parents=True, exist_ok=True)
    (work / "result" / "time_and_tb_num").mkdir(parents=True, exist_ok=True)
    # path.txt is read relative to the preprocessing binary's CWD as ../../path.txt,
    # i.e. mpspmm/path.txt. We point both paths at the scratch workdir.
    (MP.parent / "path.txt").write_text(f"project_path={work}\ndataset_path={work}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(R / "paper_datasets.json"))
    ap.add_argument("--out", default=str(R / "fgcs_results/revision/baselines/mp_spmm.csv"))
    ap.add_argument("--preproc-out", default=str(R / "fgcs_results/revision/baselines/mp_spmm_preproc.csv"))
    ap.add_argument("--note-out", default=str(R / "fgcs_results/revision/baselines/mpspmm_BUILD_NOTE.txt"))
    ap.add_argument("--work", default=str(R / "fgcs_results/revision/baselines/_mpspmm_work"))
    ap.add_argument("--max-nnz", type=int, default=20_000_000, help="skip graphs with more nnz (mtx/preproc cost)")
    args = ap.parse_args()

    assert PREPROC_BIN.exists() and SPMM_BIN.exists(), "MP-SpMM binaries not built"
    work = Path(args.work); setup_workdir(work)
    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    rows, preproc_rows, notes = [], [], []
    notes.append(f"MP-SpMM (2:4 SpTC) built for SM 86 (CUDA {torch.version.cuda}); kernel supports N in {{32,128}}, "
                 f"compared at N=128. PADS not prunes (exact). Preprocessing = match-and-pad, timed separately.")

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        if N_MP not in [int(n) for n in entry.get("Ns", [64, 128, 256, 512])]:
            continue
        name = entry["name"]
        mat = load_dataset(entry)
        if mat is None:
            continue
        M, K = mat["M"], mat["K"]
        rp = mat["rowptr"].contiguous().int(); ci = mat["colind"].contiguous().int()
        nnz = int(rp[-1].item())
        if nnz > args.max_nnz:
            notes.append(f"{name}: SKIP (nnz={nnz} > {args.max_nnz}; .mtx/preproc too heavy)")
            print(f"  [skip] {name}: nnz={nnz}")
            continue
        mtx_path = work / "dataset" / f"{name}.mtx"
        write_mtx(mtx_path, rp.numpy(), ci.numpy(), M, K)

        # (2) preprocessing (match-and-pad), wall-clock timed
        try:
            t0 = time.perf_counter()
            pr = subprocess.run([str(PREPROC_BIN), name], cwd=str(PREPROC_BIN.parent),
                                capture_output=True, text=True, timeout=1800)
            preproc_ms = (time.perf_counter() - t0) * 1e3
        except subprocess.TimeoutExpired:
            notes.append(f"{name}: preprocessing TIMEOUT"); continue
        data_bin = work / "dataset_mp_processed" / "2-4-cu" / "0.Adjacent-matching" / f"{name}_data.bin"
        if pr.returncode != 0 or not data_bin.exists():
            notes.append(f"{name}: preprocessing FAILED rc={pr.returncode}: {(pr.stderr or pr.stdout).strip()[:160]}")
            print(f"  [preproc fail] {name}: {(pr.stderr or pr.stdout).strip()[:80]}")
            continue
        preproc_rows.append({"dataset": name, "M": M, "nnz": nnz, "preproc_ms": round(preproc_ms, 3)})

        # (3) MP-SpMM kernel at N=128
        try:
            sp = subprocess.run([str(SPMM_BIN), str(data_bin), str(N_MP)],
                                cwd=str(SPMM_BIN.parent), capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            notes.append(f"{name}: spmm TIMEOUT"); continue
        out = sp.stdout.strip().splitlines()[-1] if sp.stdout.strip() else ""
        # format: M,K,nnz,<ms>ms,<gflops>
        try:
            parts = out.split(",")
            ms_mp = float(parts[3].replace("ms", ""))
            gflops = float(parts[4])
        except (IndexError, ValueError):
            notes.append(f"{name}: spmm output parse fail: '{out}'  stderr={sp.stderr.strip()[:120]}")
            print(f"  [spmm parse fail] {name}: '{out}'")
            continue

        # (4) cuSPARSE + router reference at N=128 on our stack
        rp_g = rp.cuda(); ci_g = ci.cuda(); v_g = mat["vals"].cuda().float()
        B = torch.randn(M, N_MP, device="cuda")
        ms_cus = measure_ms(lambda: ra_spmm.spmm_cusparse(rp_g, ci_g, v_g, B), WARMUP, TIMED)
        d_bar = nnz / max(1, M); deg = (rp_g[1:] - rp_g[:-1]).float()
        cv_d = float((deg.std() / deg.mean()).item()) if d_bar > 0 else 0.0
        rk = simple_router(d_bar, cv_d, M, N_MP, nnz); pc = {}
        try:
            ms_router = measure_ms(lambda: run_kernel(rk, rp_g, ci_g, v_g, B, pc, f"{rk}_{N_MP}"), WARMUP, TIMED)
        except Exception:
            ms_router = float("nan")
        rows.append({
            "dataset": name, "category": entry.get("category", "?"),
            "M": M, "nnz": nnz, "N": N_MP, "kernel": "MP-SpMM",
            "ms": round(ms_mp, 5), "gflops": round(gflops, 2),
            "ms_cusparse": round(ms_cus, 4),
            "speedup_vs_cusparse": round(ms_cus / ms_mp, 3) if ms_mp > 0 else 0,
            "router_kernel": rk, "ms_router": round(ms_router, 4),
            "speedup_router_vs_mpspmm": round(ms_mp / ms_router, 3) if ms_router == ms_router and ms_router > 0 else 0,
            "preproc_ms": round(preproc_ms, 3),
            "preproc_over_kernel_ratio": round(preproc_ms / ms_mp, 1) if ms_mp > 0 else 0,
        })
        print(f"  {name:<20s} MP={ms_mp:.5f}ms ({gflops:.1f} GFLOP/s) cus={ms_cus:.4f}ms "
              f"preproc={preproc_ms:.1f}ms ({preproc_ms/ms_mp:.0f}x kernel)")
        del B, rp_g, ci_g, v_g
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out} ({len(rows)} rows)")
    if preproc_rows:
        with open(args.preproc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preproc_rows[0].keys())); w.writeheader(); w.writerows(preproc_rows)
        print(f"Wrote {args.preproc_out} ({len(preproc_rows)} rows)")
    Path(args.note_out).write_text("\n".join(notes) + "\n")
    print(f"Wrote {args.note_out} ({len(notes)} notes)")


if __name__ == "__main__":
    main()
