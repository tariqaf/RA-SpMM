"""
Crash attribution for HC-SpMM.

Question: are the illegal-memory crashes intrinsic to HC-SpMM's kernel, or an
artifact of OUR CSR feeding? This test removes our loader entirely: it loads graphs
through HC-SpMM's OWN native loader (HCSPMM_dataset, scipy COO->CSR from a dst,src
.txt) and runs HC-SpMM's own preprocess + forward_fixed64 (N=64), each in isolation.

  (A) positive control: a bundled HC-SpMM example (PROTEINS_full_A) — expect OK.
  (B) a crashed graph (amazon-computers), converted to HC-SpMM's native .txt format —
      if HC-SpMM's own loader + kernel ALSO crash here, the crash is INTRINSIC.

Run one --case per process (a crash kills the CUDA context):
    python experiments/bench_hcspmm_crash_check.py --case control
    python experiments/bench_hcspmm_crash_check.py --case amazon-computers
"""
from __future__ import annotations
import argparse, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
HC = R / "baselines" / "HC-SpMM"
sys.path.insert(0, str(R))
sys.path.insert(0, str(HC))
sys.path.insert(0, str(HC / "hybrid_kernel"))
import HCSPMM  # noqa
from dataset import HCSPMM_dataset  # noqa: HC-SpMM's OWN loader
from ra_real_graph_eval import load_dataset  # only to export our graph to their .txt


def ensure_control_txt():
    """Extract the smallest bundled example (PROTEINS_full_A) from Dataset.zip."""
    dsdir = HC / "Dataset"; dsdir.mkdir(exist_ok=True)
    txt = dsdir / "PROTEINS_full_A.txt"
    if not txt.exists():
        with zipfile.ZipFile(HC / "Dataset.zip") as z:
            z.extract("Dataset/PROTEINS_full_A.txt", HC)
    return "PROTEINS_full_A"


def export_our_graph_to_hc_txt(name):
    """Write our graph as HC-SpMM's native 'dst,src' (1-indexed) .txt via their loader path."""
    man = {d["name"]: d for d in json.loads((R / "paper_datasets.json").read_text())["datasets"]}
    mat = load_dataset(man[name])
    rp = mat["rowptr"].numpy(); ci = mat["colind"].numpy(); M = mat["M"]
    dsdir = HC / "Dataset"; dsdir.mkdir(exist_ok=True)
    txt = dsdir / f"{name}.txt"
    with open(txt, "w") as f:
        lines = []
        for i in range(M):
            for p in range(rp[i], rp[i + 1]):
                j = ci[p]
                lines.append(f"{i+1},{j+1}\n")  # symmetric graphs: orientation irrelevant
        f.writelines(lines)
    return name


def run(name):
    dim = 64
    ds = HCSPMM_dataset(str(HC / "Dataset" / f"{name}.txt"), dim, num_class=2, load_from_txt=True)
    M = ds.num_nodes
    col = ds.column_index.cuda(); rp = ds.row_pointers.cuda()
    nrw = (M + 15) // 16
    print(f"[{name}] loaded via HC-SpMM native loader: M={M}, edges={ds.num_edges}")
    meta = HCSPMM.preprocess(col, rp, M, ds.num_edges, nrw)
    print(f"[{name}] HC preprocess OK")
    bp, e2c, e2r, ht, rnzr, cnzr = meta
    X = torch.randn(M, dim, device="cuda").contiguous()
    out = HCSPMM.forward_fixed64(X, rp, col, bp, e2c, e2r, ht, rnzr, cnzr)
    C = out[0] if isinstance(out, (list, tuple)) else out
    torch.cuda.synchronize()
    print(f"[{name}] HC forward_fixed64 OK  out_shape={tuple(C.shape)}")
    print(f"VERDICT[{name}]=RAN_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="'control' or a dataset name (e.g. amazon-computers)")
    args = ap.parse_args()
    if args.case == "control":
        name = ensure_control_txt()
    else:
        name = export_our_graph_to_hc_txt(args.case)
    run(name)


if __name__ == "__main__":
    main()
