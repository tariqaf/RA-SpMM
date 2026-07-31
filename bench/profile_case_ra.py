"""
bench/profile_case_ra.py — single-launch SpMM harness for Nsight Compute.

Loads ONE graph + ONE kernel via the ra_spmm bindings, builds the kernel plan
(outside the profiled region), warms up, then issues exactly ONE SpMM inside an
NVTX range named "SPMM_PROFILE". Run under:

    ncu --nvtx --nvtx-include "SPMM_PROFILE/" --set full ... \
        python bench/profile_case_ra.py --dataset amazon-photo --kernel TC_DIRECT --N 128

so ncu captures only the compute launch(es) of interest, not the plan build or
warmup iterations.

Supported kernels: CSR_DIRECT, RODE_ENHANCED, ZERO_OVERHEAD_CSR, TC_DIRECT,
COMMUNITY_TC, SEGMENT_HYBRID, CUSPARSE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import ra_spmm  # noqa: E402
from ra_real_graph_eval import load_dataset, run_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name in the manifest")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "paper_datasets.json"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    manifest = {d["name"]: d for d in json.loads(Path(args.datasets_file).read_text())["datasets"]}
    entry = manifest[args.dataset]
    mat = load_dataset(entry)
    if mat is None:
        print(f"ERROR: dataset {args.dataset} not found", file=sys.stderr)
        sys.exit(2)

    M = mat["M"]
    rp = mat["rowptr"].cuda().int()
    ci = mat["colind"].cuda().int()
    v = mat["vals"].cuda().float()
    torch.manual_seed(args.seed + M + args.N)
    B = torch.randn(M, args.N, device="cuda", dtype=torch.float32)

    plan_cache = {}
    cache_key = f"{args.kernel}_{args.N}"
    # Build plan + warm up OUTSIDE the profiled NVTX range.
    for _ in range(args.warmup):
        run_kernel(args.kernel, rp, ci, v, B, plan_cache, cache_key)
    torch.cuda.synchronize()

    print(json.dumps({"dataset": args.dataset, "kernel": args.kernel, "N": args.N,
                      "M": M, "nnz": int(rp[-1].item())}))

    # Exactly one profiled launch.
    nvtx.range_push("SPMM_PROFILE")
    out = run_kernel(args.kernel, rp, ci, v, B, plan_cache, cache_key)
    torch.cuda.synchronize()
    nvtx.range_pop()

    if torch.is_tensor(out):
        print(json.dumps({"output_shape": list(out.shape)}))


if __name__ == "__main__":
    main()
