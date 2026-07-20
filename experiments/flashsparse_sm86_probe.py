#!/usr/bin/env python3
"""Task 1 gate: does FlashSparse BUILD and RUN on SM 86 (RTX 3090)?

Builds were produced with TORCH_CUDA_ARCH_LIST=8.6. This script imports the
sm_86 FS_SpMM extension, runs its fp16 (8x8, mma.m16n8k8) and tf32 (8x4,
mma.m16n8k4) SpMM paths on real graphs at N=128, and checks correctness by
relative Frobenius norm against cuSPARSE (ra_spmm.spmm_cusparse) on the SAME
CSR. Emits a per-(graph,path) CSV. Any failure is captured and printed, not
swallowed.
"""
from __future__ import annotations
import sys, csv, math, traceback
from pathlib import Path

REPO = Path("/mnt/shared/development/tariq/RA-SpMM")
FS = REPO / "external/flashsparse/FlashSparse"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FS))

import torch
import numpy as np
import ra_spmm
from ra_real_graph_eval import load_dataset

import FS_Block
import FS_SpMM

N = 128
GRAPHS = ["Cora", "ca-HepTh", "PPI", "ogbn-arxiv"]  # real, span regimes/sizes


def load_manifest():
    import json
    raw = json.load(open(REPO / "paper_datasets.json"))
    ds = raw["datasets"] if isinstance(raw, dict) else raw
    return {d["name"]: d for d in ds}


def rel_frob(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm((a - b).float()) / (torch.linalg.norm(b.float()) + 1e-30))


def run_one(name, entry):
    out = []
    data = load_dataset(entry)
    rowptr = data["rowptr"].cpu().to(torch.int32)
    colind = data["colind"].cpu().to(torch.int32)
    M = int(data["M"])
    nnz = int(colind.numel())
    # reference B and cuSPARSE ground truth on original CSR (unit values)
    torch.manual_seed(123)
    vals = torch.ones(nnz, dtype=torch.float32, device="cuda")
    rp_g = rowptr.cuda(); ci_g = colind.cuda()
    B = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref = ra_spmm.spmm_cusparse(rp_g, ci_g, vals, B)  # M x N, FP32

    for path, dtype, blk in [("fp16_8x8", torch.float16, (8, 8)), ("tf32_8x4", torch.float32, (8, 4))]:
        rec = {"dataset": name, "M": M, "nnz": nnz, "N": N, "path": path,
               "built_arch": "sm_86", "ran": False, "rel_frob": "", "kernel_ms": "",
               "status": "", "detail": ""}
        try:
            dd = torch.ones(nnz, dtype=dtype)
            Mpad = M + (16 - M % 16) % 16  # FlashSparse pads node count to mult of 16
            if path.startswith("fp16"):
                rp, ci, deg = FS_Block.blockProcess_fp16(rowptr.clone(), colind.clone(), dd, blk[0], blk[1])
                rhs = B.to(torch.float16).cpu()
                rp = rp.cuda(); ci = ci.cuda(); deg = deg.cuda(); rhs = rhs.cuda()
                # (rp, ci, values, rhs, num_nodes_pad, dimN, num_nodes, epoches)
                result, ms = FS_SpMM.forward_fp16(rp, ci, deg, rhs, Mpad, N, M, 1)
            else:
                rp, ci, deg = FS_Block.blockProcess_tf32(rowptr.clone(), colind.clone(), dd, blk[0], blk[1])
                rhs = B.cpu().cuda()
                rp = rp.cuda(); ci = ci.cuda(); deg = deg.cuda()
                result, ms = FS_SpMM.forward_tf32(rp, ci, deg, rhs, Mpad, N, M, 1)
            result = result[:M].float().to(ref.device)
            err = rel_frob(result, ref)
            rec.update(ran=True, rel_frob=round(err, 6), kernel_ms=round(float(ms), 5),
                       status="OK" if err < 5e-2 else "RAN_HIGH_ERR")
        except Exception as e:
            rec.update(status="RUNTIME_FAIL", detail=f"{type(e).__name__}: {e}")
            print(f"[{name}/{path}] RUNTIME_FAIL:\n{traceback.format_exc()}")
        out.append(rec)
        print(f"  {name:12s} {path:9s} ran={rec['ran']} err={rec['rel_frob']} ms={rec['kernel_ms']} {rec['status']} {rec['detail'][:60]}")
    return out


def main():
    print(f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0)} "
          f"cc {torch.cuda.get_device_capability(0)}")
    print(f"FS_SpMM: {FS_SpMM.__file__}")
    man = load_manifest()
    rows = []
    for g in GRAPHS:
        if g not in man:
            print(f"  {g}: NOT IN MANIFEST, skip"); continue
        try:
            rows += run_one(g, man[g])
        except Exception as e:
            print(f"[{g}] LOAD/SETUP FAIL: {e}")
            rows.append({"dataset": g, "path": "all", "ran": False, "status": "SETUP_FAIL", "detail": str(e)})
    out = REPO / "fgcs_results/revision/tf32/baseline_audit/flashsparse_sm86.csv"
    fields = ["dataset", "M", "nnz", "N", "path", "built_arch", "ran", "rel_frob", "kernel_ms", "status", "detail"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    ran = [r for r in rows if r.get("ran")]
    print(f"\nVERDICT INPUT: {len(ran)}/{len(rows)} (graph,path) ran; "
          f"correct(<5%)={sum(1 for r in ran if r.get('status')=='OK')}")


if __name__ == "__main__":
    main()
