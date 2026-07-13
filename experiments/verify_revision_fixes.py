#!/usr/bin/env python3
"""GPU regression checks for repaired correctness defects."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ra_spmm  # noqa: E402


def check_zero_overhead_long_row() -> None:
    nnz = 70_000
    N = 64
    rowptr_cpu = torch.tensor([0, nnz], dtype=torch.int32)
    colind_cpu = torch.arange(nnz, dtype=torch.int32)
    vals_cpu = torch.ones(nnz, dtype=torch.float32)
    rowptr = rowptr_cpu.cuda()
    colind = colind_cpu.cuda()
    vals = vals_cpu.cuda()
    B = torch.ones((nnz, N), device="cuda", dtype=torch.float32)
    plan = ra_spmm.make_zero_overhead_plan(rowptr_cpu, 1, nnz)
    output = ra_spmm.run_zero_overhead_plan(plan, rowptr, colind, vals, B)
    expected = torch.full_like(output, float(nnz))
    max_error = float((output - expected).abs().max().item())
    if max_error != 0.0:
        raise AssertionError(f"70,000-nnz row max_error={max_error}")
    print("PASS ZERO_OVERHEAD_CSR 70,000-nnz row: max_error=0")


def check_narrow_router_fallback() -> None:
    M = 150_000
    N = 8
    rowptr_cpu = torch.arange(M + 1, dtype=torch.int32)
    colind_cpu = torch.arange(M, dtype=torch.int32)
    vals_cpu = torch.ones(M, dtype=torch.float32)
    rowptr = rowptr_cpu.cuda()
    colind = colind_cpu.cuda()
    vals = vals_cpu.cuda()
    B = torch.ones((M, N), device="cuda", dtype=torch.float32)
    plan = ra_spmm.make_router_plan(
        rowptr, colind, vals, M, M, N, "MAIN")
    chosen = str(plan["chosen_path"])
    if chosen != "CSR_DIRECT":
        raise AssertionError(f"N=8 router chose {chosen}, expected CSR_DIRECT")
    output = ra_spmm.run_router_plan(plan, rowptr, colind, vals, B)
    max_error = float((output - 1.0).abs().max().item())
    if max_error != 0.0:
        raise AssertionError(f"N=8 routed output max_error={max_error}")
    print("PASS M=150000,N=8 router fallback: CSR_DIRECT, max_error=0")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    check_zero_overhead_long_row()
    check_narrow_router_fallback()


if __name__ == "__main__":
    main()
