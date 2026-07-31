"""Deployable model-based lifecycle router; it never benchmarks candidates."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import ra_spmm
from ra_real_graph_eval import build_kernel_plan, run_planned_kernel

KERNELS = [
    "CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
    "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID",
]
DEFAULT_MODEL = Path(__file__).resolve().parent / "results" / "router" / "runtime_cost_model.json"


def feature_vector(M: int, nnz: int, N: int, avg_nnz: float, cv_d: float) -> list[float]:
    return [
        1.0, math.log1p(M), math.log1p(nnz), math.log1p(N),
        math.log1p(nnz * N), math.log1p(avg_nnz), math.log1p(cv_d),
    ]


def prediction(coefficients: list[float], features: list[float]) -> float:
    return math.exp(sum(coef * value for coef, value in zip(coefficients, features)))


@dataclass
class RuntimeRouterPlan:
    chosen_path: str
    expected_calls: int
    kernel_plan: Any
    diagnostics: dict[str, Any]

    def run(self, rowptr: torch.Tensor, colind: torch.Tensor,
            vals: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return run_planned_kernel(
            self.chosen_path, self.kernel_plan, rowptr, colind, vals, B)


def make_runtime_router_plan(
    rowptr: torch.Tensor,
    colind: torch.Tensor,
    vals: torch.Tensor,
    M: int,
    K: int,
    N: int,
    *,
    expected_calls: int = 1,
    assume_cold: bool = False,
    model_path: str | Path = DEFAULT_MODEL,
) -> RuntimeRouterPlan:
    calls = 1 if assume_cold else max(1, int(expected_calls))
    model = json.loads(Path(model_path).read_text())
    start = time.perf_counter()
    rule_plan = ra_spmm.make_router_plan(rowptr, colind, vals, M, K, N, "MAIN")
    routing_ms = (time.perf_counter() - start) * 1e3
    features_dict = dict(rule_plan["feature_values"])
    nnz = int(features_dict.get("nnz", int(colind.numel())))
    avg = float(features_dict["avg_nnz_per_row"])
    cv = float(features_dict["degree_cv"])
    x = feature_vector(M, nnz, N, avg, cv)
    feasible = dict(rule_plan["feasible_by_path"])

    estimates: dict[str, dict[str, float]] = {}
    for kernel in KERNELS:
        kernel_model = model.get("kernels", {}).get(kernel)
        if not kernel_model or not bool(feasible.get(kernel, False)):
            continue
        setup_ms = prediction(kernel_model["setup_coefficients"], x)
        warm_ms = prediction(kernel_model["warm_coefficients"], x)
        estimates[kernel] = {
            "predicted_setup_ms": setup_ms,
            "predicted_warm_ms": warm_ms,
            "predicted_total_ms": setup_ms + calls * warm_ms,
        }
    if not estimates:
        raise RuntimeError("No feasible kernel has a runtime cost model")
    chosen = min(estimates, key=lambda kernel: estimates[kernel]["predicted_total_ms"])

    rowptr_cpu = rowptr.cpu().contiguous().int()
    colind_cpu = colind.cpu().contiguous().int()
    vals_cpu = vals.cpu().contiguous().float()
    kernel_plan = build_kernel_plan(chosen, rowptr_cpu, colind_cpu, vals_cpu, M, K, N)
    return RuntimeRouterPlan(
        chosen_path=chosen,
        expected_calls=calls,
        kernel_plan=kernel_plan,
        diagnostics={
            "policy": "model_predicted_lifecycle_cost",
            "assume_cold": bool(assume_cold),
            "expected_calls": calls,
            "rule_router_path": str(rule_plan["chosen_path"]),
            "routing_feature_ms": routing_ms,
            "candidate_estimates": estimates,
            "model_path": str(model_path),
        },
    )
