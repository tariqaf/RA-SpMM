#!/usr/bin/env python3
"""E3 Step 1 verification: ZC-BCRS vs baseline ME-BCRS.

Checks, per graph and N:
  1. EXACT equality of ZC output vs baseline TC_DIRECT output (same mma
     fragments -> same accumulation order -> bit-identical is the claim).
     Same for the TF32 pair.
  2. max_error vs FP32 cuSPARSE for all four paths (strict-gate context).
  3. Plan bytes baseline vs ZC (measured, not estimated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(REPO / "fgcs_results" / "paper_combined_datasets.json"))
    ap.add_argument("--datasets", default="roadNet-TX,com-Amazon,synth_community_nc1000,twitter-combined,ca-GrQc,ogbn-arxiv")
    ap.add_argument("--Ns", default="64,128")
    args = ap.parse_args()

    manifest = json.loads(Path(args.datasets_file).read_text())
    if isinstance(manifest, dict):
        manifest = manifest.get("datasets", [])
    wanted = [s.strip() for s in args.datasets.split(",")]
    entries = {e["name"]: e for e in manifest}
    Ns = [int(x) for x in args.Ns.split(",")]

    all_ok = True
    for name in wanted:
        mat = load_dataset(entries[name])
        if mat is None:
            print(f"skip {name}")
            continue
        M = mat["M"]; K = mat.get("K", M)
        rowptr = mat["rowptr"].contiguous().int()
        colind = mat["colind"].contiguous().int()
        vals = mat["vals"].contiguous().float()
        rp, ci, vl = rowptr.cuda(), colind.cuda(), vals.cuda()
        for N in Ns:
            torch.manual_seed(1234 + M + N)
            B = torch.randn(K, N, device="cuda")
            C_ref = ra_spmm.spmm_cusparse(rp, ci, vl, B)

            base = ra_spmm.make_tc_direct_plan(rowptr, colind, vals, M, K, N)
            zc = ra_spmm.make_tc_direct_zc_plan(rowptr, colind, vals, M, K, N)

            C_b16 = ra_spmm.run_tc_direct_plan(base, B)
            C_z16 = ra_spmm.run_tc_direct_plan(zc, B)
            C_b32 = ra_spmm.run_tc_direct_plan_tf32(base, B)
            C_z32 = ra_spmm.run_tc_direct_plan_tf32(zc, B)

            bit16 = bool((C_z16 == C_b16).all().item())
            bit32 = bool((C_z32 == C_b32).all().item())
            e_b16 = (C_b16 - C_ref).abs().max().item()
            e_z16 = (C_z16 - C_ref).abs().max().item()
            e_z32 = (C_z32 - C_ref).abs().max().item()
            shrink = base.plan_bytes / max(1, zc.plan_bytes)
            ok = bit16 and bit32
            all_ok &= ok
            print(f"{name:26s} N={N:3d} bit16={'PASS' if bit16 else 'FAIL'} "
                  f"bit32={'PASS' if bit32 else 'FAIL'} "
                  f"err base16={e_b16:.3e} zc16={e_z16:.3e} zc32={e_z32:.3e} "
                  f"plan {base.plan_bytes/1e6:.1f}->{zc.plan_bytes/1e6:.1f}MB ({shrink:.2f}x)")
            del base, zc, C_b16, C_z16, C_b32, C_z32
            torch.cuda.empty_cache()
    print("\nALL BIT-IDENTICAL" if all_ok else "\nBIT-IDENTITY FAILURES — investigate before benchmarking")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
