"""
pyg_baseline.py — Wrapper for PyG torch_sparse as an external GNN-framework baseline.

Why: reviewers of a GNN-focused SpMM paper expect a comparison against what GNN
practitioners actually use in PyG. torch_sparse's SparseTensor.matmul() is the
idiomatic path; internally it dispatches to cuSPARSE for most cases but the
end-to-end invocation path (Python overhead, tensor formatting) is what we
actually time here — this is what a PyG user sees.

Usage (from a driver script):
    from pyg_baseline import build_pyg_sparse, run_pyg_spmm, is_pyg_available

    if is_pyg_available():
        pyg = build_pyg_sparse(rowptr_gpu, colind_gpu, vals_gpu, M)
        C = run_pyg_spmm(pyg, B)
"""
from typing import Optional
import torch


_AVAILABLE = None


def is_pyg_available() -> bool:
    """Return True if torch_sparse is importable in this environment."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import torch_sparse  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


def build_pyg_sparse(
    rowptr_gpu: torch.Tensor,
    colind_gpu: torch.Tensor,
    vals_gpu: torch.Tensor,
    M: int,
    K: Optional[int] = None,
):
    """
    Build a torch_sparse.SparseTensor from pre-uploaded GPU CSR tensors.
    rowptr/colind must be int (torch_sparse coerces to int64 internally).
    """
    from torch_sparse import SparseTensor
    K = K if K is not None else M
    return SparseTensor(
        rowptr=rowptr_gpu.to(torch.long),
        col=colind_gpu.to(torch.long),
        value=vals_gpu,
        sparse_sizes=(M, K),
    )


def run_pyg_spmm(sparse_tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Run SpMM via torch_sparse.SparseTensor.matmul(). This is the idiomatic
    PyG path — `adj @ x` inside PyG convolutions eventually calls this.
    """
    return sparse_tensor @ B


def time_pyg_spmm(
    sparse_tensor,
    B: torch.Tensor,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> float:
    """
    Time torch_sparse SpMM using CUDA events.
    Returns mean milliseconds per call.
    """
    for _ in range(warmup_iters):
        _ = run_pyg_spmm(sparse_tensor, B)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(timed_iters):
        start.record()
        _ = run_pyg_spmm(sparse_tensor, B)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(sum(times_ms) / len(times_ms))
