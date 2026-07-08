"""
Correctness check for MP-SpMM's 2:4 match-and-pad (must be EXACT).

MP-SpMM PADS (not prunes), so A_padded @ B == A @ B exactly. We verify by running the
kernel with B = all-ones: then C[i][0] = row-sum(A) = degree(i) for our unit-value
adjacency. If the 2:4 conversion dropped any nonzero (pruning), C[i] would be < degree(i).
Compare C[:,0] from spmm_verify against the true degrees from our CSR.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from experiments.bench_mpspmm import write_mtx, setup_workdir, MP, PREPROC_BIN  # reuse
from ra_real_graph_eval import load_dataset

SPMM_VERIFY = MP / "SpMM" / "spmm_verify"
N = 128
GRAPHS = ["amazon-photo", "ca-CondMat", "roadNet-PA", "Cora", "PPI"]


def main():
    work = R / "fgcs_results/revision/baselines/_mpspmm_verify"
    setup_workdir(work)
    man = {d["name"]: d for d in json.loads((R / "paper_datasets.json").read_text())["datasets"]}
    print(f"{'dataset':<16s} {'M':>8s} {'exact?':>7s} {'mismatch_rows':>13s}  {'maxΔ':>6s}")
    all_ok = True
    for name in GRAPHS:
        mat = load_dataset(man[name])
        M, K = mat["M"], mat["K"]
        rp = mat["rowptr"].numpy(); ci = mat["colind"].numpy()
        deg = (rp[1:] - rp[:-1]).astype(np.float64)
        write_mtx(work / "dataset" / f"{name}.mtx", rp, ci, M, K)
        pr = subprocess.run([str(PREPROC_BIN), name], cwd=str(PREPROC_BIN.parent),
                            capture_output=True, text=True, timeout=1800)
        data_bin = work / "dataset_mp_processed" / "2-4-cu" / "0.Adjacent-matching" / f"{name}_data.bin"
        if not data_bin.exists():
            print(f"{name:<16s}  preprocessing failed: {(pr.stderr or pr.stdout)[-100:]}"); all_ok = False; continue
        vout = work / f"{name}_Ccol0.txt"
        env = dict(os.environ); env["VERIFY_OUT"] = str(vout); env["CUDA_VISIBLE_DEVICES"] = "0"
        sp = subprocess.run([str(SPMM_VERIFY), str(data_bin), str(N)],
                            cwd=str(SPMM_VERIFY.parent), capture_output=True, text=True, timeout=600, env=env)
        if not vout.exists():
            print(f"{name:<16s}  spmm_verify produced no output: {sp.stderr[-120:]}"); all_ok = False; continue
        c0 = np.atleast_1d(np.loadtxt(vout))
        d = np.abs(c0[:M] - deg)                 # first M rows = real rows
        mism = int((d > 0.5).sum())
        pad_rows = c0[M:]                          # trailing rows are M padded up to mult. of 16
        pad_ok = (np.abs(pad_rows) < 0.5).all() if len(pad_rows) else True
        exact = mism == 0 and pad_ok
        all_ok = all_ok and exact
        print(f"{name:<16s} {M:>8d} {str(exact):>7s} {mism:>13d}  {d.max():>6.1f}   "
              f"(dumped {len(c0)} rows, {len(pad_rows)} zero-pad)")
    print("\nVERDICT: MP-SpMM 2:4 match-and-pad is", "EXACT (pad, not prune): C[:,0]=degree on all real rows,"
          " trailing rows are zero padding to a multiple of tile_M=16."
          if all_ok else "NOT exact on some graph — see above.")


if __name__ == "__main__":
    main()
