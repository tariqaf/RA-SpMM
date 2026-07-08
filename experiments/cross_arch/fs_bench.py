"""
Drives FlashSparse over the RA-SpMM 26-graph suite on the RTX 4090.

FlashSparse (https://github.com/ParCIS/FlashSparse) exposes a CUDA SpMM built for
SM 89/90 tensor cores. Its Python entry points differ across commits, so the call
into FlashSparse is isolated in `flashsparse_prepare()` below — adapt THAT ONE
FUNCTION to the installed FlashSparse API (the README of the cloned repo shows the
exact import + call). Everything else — graph loading, the 50+200 timing protocol,
correctness vs cuSPARSE, and the kernel-only / preprocessing split — is fixed and
matches the paper.

Emits:
  --out          kernel-only CSV: dataset,N,ms,speedup_vs_cusparse,correct,...
  --preproc-out  preprocessing CSV: dataset,N,preproc_ms   (format conversion only)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from ra_real_graph_eval import load_dataset, measure_ms  # noqa: E402
import ra_spmm  # noqa: E402  (for cuSPARSE reference)

WARMUP, TIMED = 50, 200


# --------------------------------------------------------------------------- #
#  ADAPTER: replace the body to match the installed FlashSparse API.          #
#  Must return (preproc_handle, run_fn) where run_fn() executes ONE SpMM and  #
#  returns the dense result C (M x N) as a torch.cuda tensor, and the         #
#  preprocessing (format build) has already happened when this returns.       #
# --------------------------------------------------------------------------- #
def flashsparse_prepare(rowptr, colind, vals, M, K, N, B):
    """
    Build FlashSparse's format from CSR (this is the PREPROCESSING step) and
    return a zero-arg closure that runs the kernel. Time spent HERE is recorded
    as preprocessing; time spent in the returned closure is kernel-only.

    Example skeleton (adapt to the cloned FlashSparse README):

        import FlashSparse as fs
        mat = fs.build(rowptr.cpu(), colind.cpu(), vals.cpu(), M, K)  # preprocessing
        def run():
            return fs.spmm(mat, B)                                    # kernel-only
        return mat, run
    """
    raise NotImplementedError(
        "Adapt flashsparse_prepare() to the installed FlashSparse API "
        "(see baselines/FlashSparse/README). Until then this graph is skipped "
        "and a BUILD_NOTE is emitted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default="paper_datasets.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preproc-out", required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    rows, preproc_rows = [], []
    build_note = []

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        mat = load_dataset(entry)
        if mat is None:
            continue
        M, K = mat["M"], mat["K"]
        rp = mat["rowptr"].cuda().int(); ci = mat["colind"].cuda().int(); v = mat["vals"].cuda().float()
        for N in [int(n) for n in entry.get("Ns", [64, 128, 256, 512])]:
            B = torch.randn(M, N, device="cuda")
            C_ref = ra_spmm.spmm_cusparse(rp, ci, v, B)
            ms_cus = measure_ms(lambda: ra_spmm.spmm_cusparse(rp, ci, v, B), WARMUP, TIMED)
            try:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                handle, run_fn = flashsparse_prepare(rp, ci, v, M, K, N, B)
                torch.cuda.synchronize(); preproc_ms = (time.perf_counter() - t0) * 1e3
                C = run_fn()
                max_err = (C.float() - C_ref).abs().max().item()
                ms_fs = measure_ms(run_fn, WARMUP, TIMED)
                rows.append({
                    "dataset": entry["name"], "category": entry.get("category", "?"),
                    "M": M, "nnz": int(rp[-1].item()), "N": N,
                    "kernel": "FlashSparse", "ms": round(ms_fs, 4),
                    "ms_cusparse": round(ms_cus, 4),
                    "speedup_vs_cusparse": round(ms_cus / ms_fs, 3) if ms_fs > 0 else 0,
                    "correct": max_err < 1e-1, "max_error": round(max_err, 5),
                })
                preproc_rows.append({
                    "dataset": entry["name"], "N": N, "preproc_ms": round(preproc_ms, 4),
                })
                print(f"  {entry['name']:<20s} N={N:<4d} FS={ms_fs:.4f}ms preproc={preproc_ms:.2f}ms "
                      f"({ms_cus/ms_fs:.2f}x vs cuSPARSE)")
            except NotImplementedError as e:
                build_note.append(f"{entry['name']} N={N}: {e}")
            except Exception as e:
                build_note.append(f"{entry['name']} N={N}: RUNTIME {type(e).__name__}: {e}")
            del B
            torch.cuda.empty_cache()

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if preproc_rows:
        with open(args.preproc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preproc_rows[0].keys())); w.writeheader(); w.writerows(preproc_rows)
    if build_note:
        note_path = Path(args.out).parent / "FLASHSPARSE_BUILD_NOTE.txt"
        note_path.write_text("\n".join(build_note))
        print(f"\n{len(build_note)} graphs not run — see {note_path}")


if __name__ == "__main__":
    main()
