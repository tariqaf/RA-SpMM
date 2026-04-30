#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import paper_eval_utils as pe
import test_next as harness
import ra_spmm


def _load_case(name: str, size: str | None, manifest: str | None) -> Tuple[str, int, int, Dict[str, object]]:
    real_cases = {c.name: c for c in pe.load_real_cases(manifest, (64, 128, 256, 512))}
    if name in real_cases:
        case = real_cases[name]
        mat = case.loader()
        return "real", case.M, case.K, mat

    synth_cases = {c.name: c for c in harness.build_graph_cases()}
    if name not in synth_cases:
        raise KeyError(f"Unknown case {name}")
    case = synth_cases[name]
    if size is None:
        M, K = case.sizes[0]
    else:
        parts = size.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid --size {size}; expected MxK")
        M, K = int(parts[0]), int(parts[1])
    mat = case.builder(M, K, 42)
    return "synthetic", M, K, mat


def _prepare_tensors(mat: Dict[str, object], K: int, N: int, seed: int):
    torch.manual_seed(seed)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    rp_cpu = mat["rowptr"].cpu().contiguous().int()
    ci_cpu = mat["colind"].cpu().contiguous().int()
    v_cpu = mat["vals"].cpu().contiguous().float()
    rp = rp_cpu.cuda().int()
    ci = ci_cpu.cuda().int()
    v = v_cpu.cuda().float()
    return rp_cpu, ci_cpu, v_cpu, rp, ci, v, B


def _warm_run(path: str, rp_cpu, ci_cpu, v_cpu, rp, ci, v, B, warmup: int):
    if path == "CSR_DIRECT":
        for _ in range(warmup):
            ra_spmm.spmm_csr_direct(rp, ci, v, B)
        torch.cuda.synchronize()
        return lambda: ra_spmm.spmm_csr_direct(rp, ci, v, B)

    if path == "CSR_ADAPTIVE":
        plan = ra_spmm.make_csr_adaptive_plan(rp_cpu, ci_cpu, int(rp_cpu.numel()) - 1, B.size(0))
        for _ in range(warmup):
            ra_spmm.run_csr_adaptive_plan(plan, rp, ci, v, B)
        torch.cuda.synchronize()
        return lambda: ra_spmm.run_csr_adaptive_plan(plan, rp, ci, v, B)


    if path == "ROW_SPLIT_CUDA":
        plan = ra_spmm.make_row_split_plan(rp_cpu, int(rp_cpu.numel()) - 1, B.size(0))
        for _ in range(warmup):
            ra_spmm.run_row_split_plan(plan, ci, v, B)
        torch.cuda.synchronize()
        return lambda: ra_spmm.run_row_split_plan(plan, ci, v, B)

    if path == "TC_REORDERED":
        plan = ra_spmm.make_tc_reordered_plan(rp_cpu, ci_cpu, v_cpu, int(rp_cpu.numel()) - 1, B.size(0), B.size(1))
        for _ in range(warmup):
            ra_spmm.run_tc_reordered_plan(plan, B)
        torch.cuda.synchronize()
        return lambda: ra_spmm.run_tc_reordered_plan(plan, B)

    if path == "HYBRID_TC_CUDA":
        plan = ra_spmm.make_hybrid_tc_cuda_plan(rp_cpu, ci_cpu, v_cpu, int(rp_cpu.numel()) - 1, B.size(0), B.size(1), 0.45)
        for _ in range(warmup):
            ra_spmm.run_hybrid_tc_cuda_plan(plan, B)
        torch.cuda.synchronize()
        return lambda: ra_spmm.run_hybrid_tc_cuda_plan(plan, B)

    if path == "CUSPARSE":
        torch.cuda.synchronize()
        return lambda: ra_spmm.benchmark_cusparse(rp, ci, v, B, warmup=warmup, iters=1)

    raise ValueError(f"Unsupported path {path}")


def main():
    parser = argparse.ArgumentParser(description="Profile one case/path with warm semantics")
    parser.add_argument("--case", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--size", type=str, default=None, help="Synthetic size as MxK")
    parser.add_argument("--dataset_manifest", type=str, default="paper_datasets.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    source, M, K, mat = _load_case(args.case, args.size, args.dataset_manifest)
    rp_cpu, ci_cpu, v_cpu, rp, ci, v, B = _prepare_tensors(mat, K, args.N, args.seed + M + args.N)
    runner = _warm_run(args.path, rp_cpu, ci_cpu, v_cpu, rp, ci, v, B, args.warmup)

    print(json.dumps({
        "case": args.case,
        "source": source,
        "path": args.path,
        "M": M,
        "K": K,
        "N": args.N,
        "warmup": args.warmup,
        "nnz": int(mat["nnz"]),
    }))
    out = runner()
    torch.cuda.synchronize()
    if torch.is_tensor(out):
        print(json.dumps({"output_shape": list(out.shape)}))
    elif isinstance(out, dict):
        print(json.dumps({"benchmark_keys": sorted(out.keys())}))


if __name__ == "__main__":
    main()
