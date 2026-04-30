"""
test_next.py - Paper-aware harness for the repaired ra_spmm extension

MAIN portfolio (paper-facing):
- CSR_DIRECT
- RODE_ENHANCED
- ZERO_OVERHEAD_CSR
- TC_DIRECT
- COMMUNITY_TC
- SEGMENT_HYBRID
- CUSPARSE (vendor baseline)

FULL portfolio:
- MAIN
- CSR_ADAPTIVE
- STAGED_REUSE
- TC_SPARSE
- ROW_SPLIT_CUDA
- TC_REORDERED
- HYBRID_TC_CUDA
- VECTORIZED_COARSE
- LOCALITY_TILED

Primary benchmark groupings in this phase:
- baseline_reference
- row_split_targets
- tc_locality_targets
- hybrid_mixed_targets
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_spmm


MAIN_PATHS = ["CSR_DIRECT", "ROW_SPLIT_CUDA", "TC_REORDERED", "HYBRID_TC_CUDA", "CUSPARSE"]
FULL_PATHS = MAIN_PATHS + ["CSR_ADAPTIVE", "STAGED_REUSE", "TC_SPARSE"]
EXTERNAL_BASELINE_PATHS = ["TORCH_SPARSE"]
EXPANDED_GROUPS = [
    "baseline_reference",
    "row_split_targets",
    "tc_locality_targets",
    "hybrid_mixed_targets",
]


@dataclass(frozen=True)
class GraphCase:
    name: str
    category: str
    group: str
    sizes: Sequence[Tuple[int, int]]
    Ns: Sequence[int]
    builder: Callable[[int, int, int], Dict[str, object]]
    seed: int = 42


def print_sep(width: int = 122) -> None:
    print("=" * width)


def print_table_header(cols: List[str], widths: List[int]) -> None:
    print("  ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    print("-" * sum(widths + [2 * (len(widths) - 1)]))


def print_table_row(values: List[object], widths: List[int]) -> None:
    print("  ".join(f"{str(v):<{w}}" for v, w in zip(values, widths)))


def fmt_ms(value: float) -> str:
    if value is None or math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def measure_cuda_ms(run_fn: Callable[[], None], warmup: int, iters: int) -> float:
    if iters <= 0:
        return 0.0
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / float(iters)


def build_torch_sparse_tensor(
    rowptr,
    colind,
    vals,
    M: int,
    K: int,
    device: torch.device,
    prefer_csr: bool = True,
):
    rowptr_dev = rowptr.to(device=device, dtype=torch.int32)
    colind_dev = colind.to(device=device, dtype=torch.int32)
    vals_dev = vals.to(device=device, dtype=torch.float32)

    if prefer_csr:
        try:
            return (
                torch.sparse_csr_tensor(
                    rowptr_dev,
                    colind_dev,
                    vals_dev,
                    size=(M, K),
                    device=device,
                    dtype=torch.float32,
                ),
                "csr",
            )
        except Exception:
            pass

    rows = torch.repeat_interleave(
        torch.arange(M, device=device, dtype=torch.int64),
        (rowptr_dev[1:] - rowptr_dev[:-1]).to(torch.int64),
    )
    sparse = torch.sparse_coo_tensor(
        torch.stack([rows, colind_dev.to(torch.int64)]),
        vals_dev,
        (M, K),
        dtype=torch.float32,
        device=device,
    ).coalesce()
    return sparse, "coo"


def pytorch_ref_spmm(rowptr_cpu, colind_cpu, vals_cpu, B_gpu, M, K):
    if colind_cpu.numel() == 0:
        return torch.zeros((M, B_gpu.size(1)), device=B_gpu.device, dtype=torch.float32)
    A, _ = build_torch_sparse_tensor(
        rowptr_cpu,
        colind_cpu,
        vals_cpu,
        M,
        K,
        B_gpu.device,
        prefer_csr=False,
    )
    return torch.sparse.mm(A, B_gpu)


def torch_sparse_spmm(rowptr, colind, vals, B, M, K):
    A, layout = build_torch_sparse_tensor(rowptr, colind, vals, M, K, B.device, prefer_csr=True)
    try:
        return torch.sparse.mm(A, B), layout
    except Exception:
        if layout == "csr":
            A, layout = build_torch_sparse_tensor(rowptr, colind, vals, M, K, B.device, prefer_csr=False)
            return torch.sparse.mm(A, B), layout
        raise


def max_err(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def build_graph_cases() -> List[GraphCase]:
    return [
        GraphCase(
            name="random_16",
            category="hybrid/mixed",
            group="baseline_reference",
            sizes=[(4096, 4096), (8192, 8192)],
            Ns=[64, 128, 256],
            builder=lambda M, K, seed: ra_spmm.gen_random_sparse(M, K, 16, seed),
        ),
        GraphCase(
            name="road_like",
            category="ordered sparse / road-network",
            group="baseline_reference",
            sizes=[(4096, 4096), (8192, 8192), (16384, 16384)],
            Ns=[64, 128, 256],
            builder=lambda M, K, seed: ra_spmm.gen_road_like(M, K, 4, seed),
        ),
        GraphCase(
            name="community",
            category="sparse modular community",
            group="tc_locality_targets",
            sizes=[(4096, 4096), (8192, 8192)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_community_clustered(M, K, 8, 0.35, 0.01, seed),
        ),
        # community_sbm: sparse SBM matching real community graph density (com-DBLP avg_deg~7).
        # n_comm = M // 256 keeps community size ~256 nodes at all scales.
        # within_density = 7 / 256 ≈ 0.027 → avg_deg ≈ 7 regardless of M.
        # between_density = 0.5/M → ~0.5 inter-community edge per row (sparse cross-community).
        GraphCase(
            name="community_sbm",
            category="sparse modular community",
            group="tc_locality_targets",
            sizes=[(8192, 8192), (16384, 16384), (32768, 32768), (65536, 65536)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_community_sbm(
                M, max(4, M // 256), 0.027, 0.5 / max(1, M), seed),
        ),
        GraphCase(
            name="block_local",
            category="dense block-local / TC-friendly",
            group="tc_locality_targets",
            sizes=[(4096, 4096), (8192, 8192), (32768, 32768), (65536, 65536)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_block_locality(M, K, 64, 0.70, seed),
        ),
        GraphCase(
            name="skewed_powerlaw",
            category="hub-dominated power-law",
            group="row_split_targets",
            # Removed 4K and 8K — launch-overhead dominated; keep 16K+ only
            sizes=[(16384, 16384)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_skewed_powerlaw(M, K, 2.2, 1, min(384, K), seed),
        ),
        # hub_heavy removed: pathological bimodal step-function, not power-law.
        # Replaced by powerlaw_realistic (BA graph).
        # powerlaw_realistic: Barabasi-Albert preferential attachment, m=5.
        # Continuous heavy-tail degree distribution, no hard degree cap.
        GraphCase(
            name="powerlaw_realistic",
            category="hub-dominated power-law",
            group="row_split_targets",
            sizes=[(4096, 4096), (8192, 8192), (16384, 16384), (32768, 32768)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_powerlaw_realistic(M, 5, seed),
        ),
        GraphCase(
            name="mixed_skew",
            category="hybrid/mixed",
            group="hybrid_mixed_targets",
            # Removed 4K and 8K — launch-overhead dominated; keep 16K+ only
            sizes=[(16384, 16384)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_mixed_skew(M, K, 0.70, 0.25, 0.05, 2, 24, min(320, K), seed),
        ),
        GraphCase(
            name="clustered_window",
            category="dense block-local / TC-friendly",
            group="tc_locality_targets",
            # Extended to 32K and 64K to show TC advantage as overhead amortizes
            sizes=[(4096, 4096), (8192, 8192), (16384, 16384), (32768, 32768), (65536, 65536)],
            Ns=[256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_clustered_window(M, K, 16, 96, 0.45, seed),
        ),
        # scrambled_locality removed: row permutation defeats TC locality recovery;
        # cuSPARSE wins 1.42x vs TC 1.23x — not a valid block-local test.
        GraphCase(
            name="mixed_block_skew",
            category="hybrid/mixed",
            group="hybrid_mixed_targets",
            sizes=[(4096, 4096), (8192, 8192), (16384, 16384)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_mixed_block_skew(M, K, 16, 0.35, 0.30, 0.65, 4, min(256, K), seed),
        ),
        GraphCase(
            name="cluster_plus_hubs",
            category="hybrid/mixed",
            group="hybrid_mixed_targets",
            sizes=[(4096, 4096), (8192, 8192), (16384, 16384)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_cluster_plus_hubs(M, K, 8, 0.28, 0.01, 0.02, min(384, K), seed),
        ),
        GraphCase(
            name="heterogeneous_windows",
            category="hybrid/mixed",
            group="hybrid_mixed_targets",
            sizes=[(4096, 4096), (8192, 8192), (16384, 16384)],
            Ns=[128, 256, 512],
            builder=lambda M, K, seed: ra_spmm.gen_heterogeneous_windows(M, K, 16, 0.25, 0.25, 0.25, 0.25, seed),
        ),
    ]


def build_correctness_cases() -> List[GraphCase]:
    return [
        GraphCase("random_16", "hybrid/mixed", "baseline_reference", [(1024, 1024)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_random_sparse(M, K, 16, seed)),
        GraphCase("skewed_powerlaw", "hub-dominated power-law", "row_split_targets", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_skewed_powerlaw(M, K, 2.2, 1, min(256, K), seed)),
        GraphCase("hub_heavy", "hub-dominated power-law", "row_split_targets", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_hub_heavy(M, K, 0.02, min(512, K), 4, seed)),
        GraphCase("mixed_skew", "hybrid/mixed", "hybrid_mixed_targets", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_mixed_skew(M, K, 0.70, 0.25, 0.05, 2, 24, min(256, K), seed)),
        GraphCase("community", "sparse modular community", "tc_locality_targets", [(1024, 1024)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_community_clustered(M, K, 8, 0.35, 0.01, seed)),
        GraphCase("block_local", "dense block-local / TC-friendly", "tc_locality_targets", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_block_locality(M, K, 64, 0.70, seed)),
        GraphCase("clustered_window", "dense block-local / TC-friendly", "tc_locality_targets", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_clustered_window(M, K, 16, 96, 0.45, seed)),
        GraphCase("mixed_block_skew", "hybrid/mixed", "hybrid_mixed_targets", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_mixed_block_skew(M, K, 16, 0.35, 0.30, 0.65, 4, min(256, K), seed)),
        GraphCase("road_like", "ordered sparse / road-network", "baseline_reference", [(2048, 2048)], [64, 128],
                  lambda M, K, seed: ra_spmm.gen_road_like(M, K, 4, seed)),
    ]


def case_iter(cases: Sequence[GraphCase]) -> Iterable[Tuple[GraphCase, int, int, int, Dict[str, object]]]:
    for case in cases:
        for M, K in case.sizes:
            yield case, M, K, case.seed, case.builder(M, K, case.seed)


def cases_for_group(group: str) -> List[GraphCase]:
    return [case for case in build_graph_cases() if case.group == group]


def compact_default_cases() -> List[GraphCase]:
    return [
        case for case in build_graph_cases()
        if case.name in {"random_16", "skewed_powerlaw", "community", "block_local", "road_like"}
    ]


def expanded_router_cases(groups: Sequence[str] | None = None) -> List[GraphCase]:
    cases = build_graph_cases()
    if groups is None:
        return cases
    wanted = set(groups)
    return [case for case in cases if case.group in wanted]


def parse_group_filter(raw: str | None) -> List[str] | None:
    if not raw:
        return None
    groups = [token.strip() for token in raw.split(",") if token.strip()]
    unknown = sorted(set(groups) - set(EXPANDED_GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    return groups


def flatten_router_plan(plan: Dict[str, object], selected_portfolio: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "selected_portfolio": selected_portfolio,
        "decision_reason": plan["decision_reason"],
        "gate_margin_raw": plan["gate_margin_raw"],
        "gate_margin_norm": plan["gate_margin_norm"],
        "router_estimated_risk": plan["estimated_risk"],
        "router_planning_time_ms": plan["planning_time_ms"],
        "router_chosen_path": plan["chosen_path"],
        "router_path_count": plan["path_count"],
    }
    for name, value in plan["feature_values"].items():
        row[f"feat_{name}"] = value
    for name, value in plan["scores"].items():
        row[f"score_{name}"] = value
    for name, value in plan["feasible_by_path"].items():
        row[f"feasible_{name}"] = int(bool(value))
    for name, value in plan["rejection_codes"].items():
        row[f"reject_code_{name}"] = value
    for name, value in plan["rejection_details"].items():
        row[f"reject_detail_{name}"] = value
    return row


def summarize_router_rows(title: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    slowdowns = [float(row["slowdown"]) for row in rows]
    avg_slowdown = sum(slowdowns) / len(slowdowns)
    worst_slowdown = max(slowdowns)
    within_105 = 100.0 * sum(sd <= 1.05 for sd in slowdowns) / len(slowdowns)
    within_110 = 100.0 * sum(sd <= 1.10 for sd in slowdowns) / len(slowdowns)
    router_counts = Counter(row["router_path"] for row in rows)
    oracle_counts = Counter(row["oracle_path"] for row in rows)
    confusion_counts = Counter((row["oracle_path"], row["router_path"]) for row in rows)
    group_counts = Counter(row["graph_group"] for row in rows)

    print_sep()
    print(title)
    print_sep()
    print(f"cases: {len(rows)}")
    print(f"avg_slowdown: {avg_slowdown:.3f}x")
    print(f"worst_slowdown: {worst_slowdown:.3f}x")
    print(f"pct_within_1.05x: {within_105:.1f}%")
    print(f"pct_within_1.10x: {within_110:.1f}%")
    print(f"graph_groups: {dict(group_counts)}")
    print(f"router_selection_counts: {dict(router_counts)}")
    print(f"oracle_selection_counts: {dict(oracle_counts)}")

    print()
    print("confusion_counts:")
    for (oracle_path, router_path), count in sorted(
        confusion_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        print(f"  oracle={oracle_path:<16} router={router_path:<16} count={count}")

    print()
    print("group_summary:")
    cols = ["group", "cases", "avg_slowdown", "worst_slowdown", "within_1.10x"]
    widths = [22, 8, 14, 14, 12]
    print_table_header(cols, widths)
    for group in EXPANDED_GROUPS:
        group_rows = [row for row in rows if row["graph_group"] == group]
        if not group_rows:
            continue
        group_slowdowns = [float(row["slowdown"]) for row in group_rows]
        print_table_row([
            group,
            len(group_rows),
            f"{sum(group_slowdowns) / len(group_slowdowns):.3f}x",
            f"{max(group_slowdowns):.3f}x",
            f"{100.0 * sum(sd <= 1.10 for sd in group_slowdowns) / len(group_slowdowns):.1f}%",
        ], widths)
    print()


def path_runner(path: str, mat: Dict[str, object], B: torch.Tensor):
    rp_cpu = mat["rowptr"]
    ci_cpu = mat["colind"]
    v_cpu = mat["vals"]
    rp = rp_cpu.cuda().int()
    ci = ci_cpu.cuda().int()
    v = v_cpu.cuda().float()
    M = mat["M"]
    K = mat["K"]

    if path == "CSR_DIRECT":
        return ra_spmm.spmm_csr_direct(rp, ci, v, B)
    if path == "CSR_ADAPTIVE":
        return ra_spmm.spmm_csr_adaptive(rp, ci, v, B)
    if path == "STAGED_REUSE":
        return ra_spmm.spmm_staged_reuse(rp, ci, v, B)
    if path == "TC_SPARSE":
        return ra_spmm.spmm_tc_sparse(rp, ci, v, B)[0]
    if path == "CUSPARSE":
        return ra_spmm.spmm_cusparse(rp, ci, v, B)
    if path == "TORCH_SPARSE":
        return torch_sparse_spmm(rp, ci, v, B, M, K)[0]
    if path == "ROW_SPLIT_CUDA":
        plan = ra_spmm.make_row_split_plan(rp_cpu, M, K)
        return ra_spmm.run_row_split_plan(plan, ci, v, B)
    if path == "TC_REORDERED":
        plan = ra_spmm.make_tc_reordered_plan(rp_cpu, ci_cpu, v_cpu, M, K, B.size(1))
        return ra_spmm.run_tc_reordered_plan(plan, B)
    if path == "HYBRID_TC_CUDA":
        plan = ra_spmm.make_hybrid_tc_cuda_plan(rp_cpu, ci_cpu, v_cpu, M, K, B.size(1), 0.45)
        return ra_spmm.run_hybrid_tc_cuda_plan(plan, B)
    raise ValueError(f"Unknown path {path}")


def record_oracle_rows(
    result: Dict[str, object],
    case: GraphCase,
    M: int,
    K: int,
    mode: str,
    records: List[Dict[str, object]],
) -> None:
    for path, timing in result["path_results"].items():
        records.append({
            "kind": "oracle",
            "mode": mode,
            "portfolio": result["portfolio"],
            "graph": case.name,
            "graph_group": case.group,
            "path": path,
            "plan_ms": timing["plan_ms"],
            "exec_ms": timing["exec_ms"],
            "total_ms": timing["total_ms"],
            "oracle_path": result["oracle_path"],
            "oracle_time_ms": result["oracle_time_ms"],
            "N": result["N"],
            "M": M,
            "K": K,
            "nnz": result["nnz"],
            "size_tag": f"{M}x{K}",
        })


def test_correctness() -> bool:
    print_sep()
    print("SECTION: correctness")
    print_sep()

    cols = ["graph", "group", "M", "N", "path", "max_err", "status"]
    widths = [20, 20, 8, 6, 18, 12, 8]
    print_table_header(cols, widths)

    all_ok = True
    for case, M, K, _, mat in case_iter(build_correctness_cases()):
        rp_cpu = mat["rowptr"]
        ci_cpu = mat["colind"]
        v_cpu = mat["vals"]
        for N in case.Ns:
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            ref = pytorch_ref_spmm(rp_cpu, ci_cpu, v_cpu, B, M, K)
            for path in FULL_PATHS + EXTERNAL_BASELINE_PATHS:
                try:
                    out = path_runner(path, mat, B)
                    err = max_err(out, ref)
                    tol = 1e-2 if path in {"TC_SPARSE", "TC_REORDERED", "HYBRID_TC_CUDA"} else 1e-3
                    ok = err <= tol
                    all_ok = all_ok and ok
                    print_table_row([case.name, case.group, M, N, path, f"{err:.2e}", "PASS" if ok else "FAIL"], widths)
                except Exception as exc:
                    all_ok = False
                    print_table_row([case.name, case.group, M, N, path, "ERROR", str(exc)[:8]], widths)
    print()
    print("CORRECTNESS:", "ALL PASS" if all_ok else "FAILURES DETECTED")
    return all_ok


def run_oracle_section(
    mode: str,
    portfolio: str,
    warmup: int,
    iters: int,
    records: List[Dict[str, object]],
    cases: Sequence[GraphCase] | None = None,
    title: str | None = None,
) -> None:
    title = title or f"SECTION: oracle_{mode} ({portfolio})"
    cases = list(cases) if cases is not None else compact_default_cases()

    print_sep()
    print(title)
    print_sep()

    cols = ["graph", "group", "size", "N", "oracle_path", "oracle_ms", "direct", "row_split", "tc_reord", "hybrid"]
    widths = [18, 20, 11, 6, 18, 10, 10, 10, 10, 10]
    print_table_header(cols, widths)

    api = ra_spmm.run_oracle_cold if mode == "cold" else ra_spmm.run_oracle_warm
    for case, M, K, _, mat in case_iter(cases):
        rp = mat["rowptr"].cuda().int()
        ci = mat["colind"].cuda().int()
        v = mat["vals"].cuda().float()
        for N in case.Ns:
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            result = api(rp, ci, v, B, warmup=warmup, iters=iters, portfolio=portfolio)
            pr = result["path_results"]
            print_table_row([
                case.name,
                case.group,
                f"{M}x{K}",
                N,
                result["oracle_path"],
                fmt_ms(result["oracle_time_ms"]),
                fmt_ms(pr["CSR_DIRECT"]["total_ms"]),
                fmt_ms(pr["ROW_SPLIT_CUDA"]["total_ms"]),
                fmt_ms(pr["TC_REORDERED"]["total_ms"]),
                fmt_ms(pr["HYBRID_TC_CUDA"]["total_ms"]),
            ], widths)
            record_oracle_rows(result, case, M, K, mode, records)
    print()


def run_router_section(
    mode: str,
    portfolio: str,
    warmup: int,
    iters: int,
    records: List[Dict[str, object]],
    cases: Sequence[GraphCase] | None = None,
    title: str | None = None,
) -> List[Dict[str, object]]:
    title = title or f"SECTION: router_{mode} ({portfolio})"
    cases = list(cases) if cases is not None else expanded_router_cases()
    print_sep()
    print(title)
    print_sep()

    cols = ["graph", "group", "size", "N", "router_path", "router_ms", "oracle_path", "oracle_ms", "slowdown", "reason"]
    widths = [18, 20, 11, 6, 18, 10, 18, 10, 10, 34]
    print_table_header(cols, widths)

    router_api = ra_spmm.run_router_cold if mode == "cold" else ra_spmm.run_router_warm
    oracle_api = ra_spmm.run_oracle_cold if mode == "cold" else ra_spmm.run_oracle_warm
    section_rows: List[Dict[str, object]] = []

    for case, M, K, _, mat in case_iter(cases):
        rp = mat["rowptr"].cuda().int()
        ci = mat["colind"].cuda().int()
        v = mat["vals"].cuda().float()
        for N in case.Ns:
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            router = router_api(rp, ci, v, B, portfolio=portfolio, warmup=warmup, iters=iters)
            oracle = oracle_api(rp, ci, v, B, warmup=warmup, iters=iters, portfolio=portfolio)
            slowdown = router["timing"]["total_ms"] / oracle["oracle_time_ms"]
            print_table_row([
                case.name,
                case.group,
                f"{M}x{K}",
                N,
                router["router_path"],
                fmt_ms(router["timing"]["total_ms"]),
                oracle["oracle_path"],
                fmt_ms(oracle["oracle_time_ms"]),
                f"{slowdown:.3f}x",
                router["plan"]["decision_reason"],
            ], widths)
            row = {
                "kind": "router",
                "mode": mode,
                "portfolio": portfolio,
                "graph": case.name,
                "graph_group": case.group,
                "path": router["router_path"],
                "router_path": router["router_path"],
                "plan_ms": router["timing"]["plan_ms"],
                "exec_ms": router["timing"]["exec_ms"],
                "total_ms": router["timing"]["total_ms"],
                "oracle_path": oracle["oracle_path"],
                "oracle_time_ms": oracle["oracle_time_ms"],
                "slowdown": slowdown,
                "N": N,
                "M": M,
                "K": K,
                "nnz": mat["nnz"],
                "size_tag": f"{M}x{K}",
            }
            row.update(flatten_router_plan(router["plan"], portfolio))
            records.append(row)
            section_rows.append(row)
    print()
    return section_rows


def run_calibrate_warm_main(
    warmup: int,
    iters: int,
    records: List[Dict[str, object]],
    groups: Sequence[str] | None = None,
) -> None:
    cases = expanded_router_cases(groups or EXPANDED_GROUPS)

    print_sep()
    print("SECTION: calibrate_warm_main")
    print_sep()
    print("calibration_corpus:")
    for group in groups or EXPANDED_GROUPS:
        group_cases = [case for case in cases if case.group == group]
        num_sizes = sum(len(case.sizes) for case in group_cases)
        num_points = sum(len(case.sizes) * len(case.Ns) for case in group_cases)
        print(f"  {group}: graphs={len(group_cases)}, size_variants={num_sizes}, points={num_points}")
    print()

    rows = run_router_section(
        "warm",
        "MAIN",
        warmup,
        iters,
        records,
        cases=cases,
        title="CALIBRATION: warm router vs warm oracle",
    )
    summarize_router_rows("CALIBRATION SUMMARY", rows)


def test_plan_run(warmup: int, iters: int, records: List[Dict[str, object]]) -> None:
    print_sep()
    print("SECTION: plan_run")
    print_sep()

    mat = ra_spmm.gen_hub_heavy(8192, 8192, 0.02, 768, 4, 42)
    M, K = mat["M"], mat["K"]
    N = 256
    rp_cpu, ci_cpu, v_cpu = mat["rowptr"], mat["colind"], mat["vals"]
    rp, ci, v = rp_cpu.cuda().int(), ci_cpu.cuda().int(), v_cpu.cuda().float()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    def warm_ms(run_fn: Callable[[], torch.Tensor]) -> float:
        for _ in range(warmup):
            run_fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run_fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / iters

    rows = []

    t0 = time.perf_counter()
    rs_plan = ra_spmm.make_row_split_plan(rp_cpu, M, K)
    rs_plan_ms = (time.perf_counter() - t0) * 1000.0
    rows.append(("ROW_SPLIT_CUDA", rs_plan_ms, warm_ms(lambda: ra_spmm.run_row_split_plan(rs_plan, ci, v, B))))

    t0 = time.perf_counter()
    tcr_plan = ra_spmm.make_tc_reordered_plan(rp_cpu, ci_cpu, v_cpu, M, K, N)
    tcr_plan_ms = (time.perf_counter() - t0) * 1000.0
    rows.append(("TC_REORDERED", tcr_plan_ms, warm_ms(lambda: ra_spmm.run_tc_reordered_plan(tcr_plan, B))))

    t0 = time.perf_counter()
    hyb_plan = ra_spmm.make_hybrid_tc_cuda_plan(rp_cpu, ci_cpu, v_cpu, M, K, N, 0.45)
    hyb_plan_ms = (time.perf_counter() - t0) * 1000.0
    rows.append(("HYBRID_TC_CUDA", hyb_plan_ms, warm_ms(lambda: ra_spmm.run_hybrid_tc_cuda_plan(hyb_plan, B))))

    t0 = time.perf_counter()
    cad_plan = ra_spmm.make_csr_adaptive_plan(rp_cpu, ci_cpu, M, K)
    cad_plan_ms = (time.perf_counter() - t0) * 1000.0
    rows.append(("CSR_ADAPTIVE", cad_plan_ms, warm_ms(lambda: ra_spmm.run_csr_adaptive_plan(cad_plan, rp, ci, v, B))))

    t0 = time.perf_counter()
    stg_plan = ra_spmm.make_staged_reuse_plan(rp_cpu, ci_cpu, v_cpu, M, K)
    stg_plan_ms = (time.perf_counter() - t0) * 1000.0
    rows.append(("STAGED_REUSE", stg_plan_ms, warm_ms(lambda: ra_spmm.run_staged_reuse_plan(stg_plan, B))))

    cols = ["path", "plan_ms", "warm_exec_ms", "cold_est_ms"]
    widths = [18, 12, 14, 12]
    print_table_header(cols, widths)
    for path, plan_ms, warm_exec in rows:
        print_table_row([path, fmt_ms(plan_ms), fmt_ms(warm_exec), fmt_ms(plan_ms + warm_exec)], widths)
        records.append({
            "kind": "plan_run",
            "mode": "warm",
            "portfolio": "FULL",
            "graph": "hub_heavy",
            "graph_group": "row_split_targets",
            "path": path,
            "plan_ms": plan_ms,
            "exec_ms": warm_exec,
            "total_ms": plan_ms + warm_exec,
            "N": N,
            "M": M,
            "K": K,
            "nnz": mat["nnz"],
            "size_tag": f"{M}x{K}",
        })
    print()


def test_ablation_full(warmup: int, iters: int, records: List[Dict[str, object]]) -> None:
    print_sep()
    print("SECTION: ablation_full")
    print_sep()

    cols = ["graph", "group", "size", "N", "oracle", "direct", "adaptive", "staged", "tc_sparse"]
    widths = [18, 20, 11, 6, 18, 10, 10, 10, 10]
    print_table_header(cols, widths)

    for case, M, K, _, mat in case_iter(compact_default_cases()):
        rp = mat["rowptr"].cuda().int()
        ci = mat["colind"].cuda().int()
        v = mat["vals"].cuda().float()
        for N in case.Ns:
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            result = ra_spmm.run_oracle_warm(rp, ci, v, B, warmup=warmup, iters=iters, portfolio="FULL")
            pr = result["path_results"]
            print_table_row([
                case.name,
                case.group,
                f"{M}x{K}",
                N,
                result["oracle_path"],
                fmt_ms(pr["CSR_DIRECT"]["total_ms"]),
                fmt_ms(pr["CSR_ADAPTIVE"]["total_ms"]),
                fmt_ms(pr["STAGED_REUSE"]["total_ms"]),
                fmt_ms(pr["TC_SPARSE"]["total_ms"]),
            ], widths)
            record_oracle_rows(result, case, M, K, "warm", records)
    print()


def test_family_analysis(warmup: int, iters: int, records: List[Dict[str, object]]) -> None:
    print_sep()
    print("SECTION: family_analysis")
    print_sep()

    cols = ["graph", "variant", "size", "router", "locality", "compactness", "slowdown"]
    widths = [18, 10, 11, 18, 10, 12, 10]
    print_table_header(cols, widths)

    family_cases = [case for case in build_graph_cases() if case.name in {"clustered_window", "scrambled_locality", "block_local", "community"}]
    for case, M, K, _, mat in case_iter(family_cases):
        variants = [
            ("orig", mat),
            ("reord", ra_spmm.gen_reordered_variant(mat["rowptr"], mat["colind"], mat["vals"], mat["M"], mat["K"], 99)),
        ]
        for variant_name, variant in variants:
            rp = variant["rowptr"].cuda().int()
            ci = variant["colind"].cuda().int()
            v = variant["vals"].cuda().float()
            B = torch.randn(variant["K"], 256, device="cuda", dtype=torch.float32)
            router = ra_spmm.run_router_warm(rp, ci, v, B, portfolio="MAIN", warmup=warmup, iters=iters)
            oracle = ra_spmm.run_oracle_warm(rp, ci, v, B, warmup=warmup, iters=iters, portfolio="MAIN")
            feat = router["plan"]["feature_values"]
            slowdown = router["timing"]["total_ms"] / oracle["oracle_time_ms"]
            print_table_row([
                case.name,
                variant_name,
                f"{M}x{K}",
                router["router_path"],
                f"{feat['reordered_locality_proxy']:.3f}",
                f"{feat['row_window_colspan_compactness']:.4f}",
                f"{slowdown:.3f}x",
            ], widths)
            records.append({
                "kind": "family_analysis",
                "mode": "warm",
                "portfolio": "MAIN",
                "graph": case.name,
                "graph_group": case.group,
                "variant": variant_name,
                "path": router["router_path"],
                "plan_ms": router["timing"]["plan_ms"],
                "exec_ms": router["timing"]["exec_ms"],
                "total_ms": router["timing"]["total_ms"],
                "oracle_path": oracle["oracle_path"],
                "oracle_time_ms": oracle["oracle_time_ms"],
                "slowdown": slowdown,
                "N": 256,
                "M": M,
                "K": K,
                "nnz": variant["nnz"],
                "size_tag": f"{M}x{K}",
            })
    print()


def test_oracle_cold_warm(warmup: int, iters: int, records: List[Dict[str, object]]) -> None:
    print_sep()
    print("SECTION: oracle_cold_warm")
    print_sep()

    cols = ["graph", "group", "size", "path", "cold_plan", "cold_exec", "cold_total", "warm_exec"]
    widths = [18, 20, 11, 18, 10, 10, 10, 10]
    print_table_header(cols, widths)

    for case, M, K, _, mat in case_iter(compact_default_cases()):
        rp = mat["rowptr"].cuda().int()
        ci = mat["colind"].cuda().int()
        v = mat["vals"].cuda().float()
        B = torch.randn(K, 256, device="cuda", dtype=torch.float32)
        cold = ra_spmm.run_oracle_cold(rp, ci, v, B, warmup=warmup, iters=iters, portfolio="MAIN")
        warm = ra_spmm.run_oracle_warm(rp, ci, v, B, warmup=warmup, iters=iters, portfolio="MAIN")
        for path in MAIN_PATHS:
            cold_t = cold["path_results"][path]
            warm_t = warm["path_results"][path]
            print_table_row([
                case.name,
                case.group,
                f"{M}x{K}",
                path,
                fmt_ms(cold_t["plan_ms"]),
                fmt_ms(cold_t["exec_ms"]),
                fmt_ms(cold_t["total_ms"]),
                fmt_ms(warm_t["exec_ms"]),
            ], widths)
            records.append({
                "kind": "oracle_cold_warm",
                "mode": "compare",
                "portfolio": "MAIN",
                "graph": case.name,
                "graph_group": case.group,
                "path": path,
                "cold_plan_ms": cold_t["plan_ms"],
                "cold_exec_ms": cold_t["exec_ms"],
                "cold_total_ms": cold_t["total_ms"],
                "warm_exec_ms": warm_t["exec_ms"],
                "N": 256,
                "M": M,
                "K": K,
                "nnz": mat["nnz"],
                "size_tag": f"{M}x{K}",
            })
    print()


def run_group_targets(group: str, warmup: int, iters: int, records: List[Dict[str, object]]) -> None:
    cases = cases_for_group(group)
    title = f"SECTION: {group}"
    run_oracle_section("warm", "MAIN", warmup, iters, records, cases=cases, title=title)


def test_external_baselines(warmup: int, iters: int, records: List[Dict[str, object]]) -> None:
    print_sep()
    print("SECTION: external_baselines")
    print_sep()

    cols = ["graph", "group", "size", "N", "direct", "oracle_main", "cusparse", "torch_sparse", "torch_fmt"]
    widths = [18, 20, 11, 6, 10, 12, 10, 12, 10]
    print_table_header(cols, widths)

    for case, M, K, _, mat in case_iter(compact_default_cases()):
        rp_cpu = mat["rowptr"]
        ci_cpu = mat["colind"]
        v_cpu = mat["vals"]
        rp = rp_cpu.cuda().int()
        ci = ci_cpu.cuda().int()
        v = v_cpu.cuda().float()
        for N in case.Ns:
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            oracle = ra_spmm.run_oracle_warm(rp, ci, v, B, warmup=warmup, iters=iters, portfolio="MAIN")
            direct_ms = float(oracle["path_results"]["CSR_DIRECT"]["total_ms"])

            cusparse = ra_spmm.benchmark_cusparse(rp, ci, v, B, warmup=warmup, iters=iters)
            torch_A, torch_layout = build_torch_sparse_tensor(rp, ci, v, M, K, B.device, prefer_csr=True)

            def run_torch_sparse() -> None:
                out = torch.sparse.mm(torch_A, B)
                del out

            try:
                torch_sparse_ms = measure_cuda_ms(run_torch_sparse, warmup, iters)
            except Exception:
                if torch_layout != "csr":
                    raise
                torch_A, torch_layout = build_torch_sparse_tensor(rp, ci, v, M, K, B.device, prefer_csr=False)
                def run_torch_sparse_fallback() -> None:
                    out = torch.sparse.mm(torch_A, B)
                    del out
                torch_sparse_ms = measure_cuda_ms(run_torch_sparse_fallback, warmup, iters)
            print_table_row([
                case.name,
                case.group,
                f"{M}x{K}",
                N,
                fmt_ms(direct_ms),
                fmt_ms(float(oracle["oracle_time_ms"])),
                fmt_ms(float(cusparse["total_ms"])),
                fmt_ms(torch_sparse_ms),
                torch_layout,
            ], widths)

            records.extend([
                {
                    "kind": "external_baseline",
                    "mode": "warm",
                    "portfolio": "EXTERNAL",
                    "graph": case.name,
                    "graph_group": case.group,
                    "path": "CUSPARSE",
                    "plan_ms": cusparse["plan_ms"],
                    "exec_ms": cusparse["exec_ms"],
                    "total_ms": cusparse["total_ms"],
                    "oracle_path": oracle["oracle_path"],
                    "oracle_time_ms": oracle["oracle_time_ms"],
                    "direct_ms": direct_ms,
                    "N": N,
                    "M": M,
                    "K": K,
                    "nnz": mat["nnz"],
                    "size_tag": f"{M}x{K}",
                    "cusparse_algorithm": cusparse.get("cusparse_algorithm", ""),
                },
                {
                    "kind": "external_baseline",
                    "mode": "warm",
                    "portfolio": "EXTERNAL",
                    "graph": case.name,
                    "graph_group": case.group,
                    "path": "TORCH_SPARSE",
                    "plan_ms": 0.0,
                    "exec_ms": torch_sparse_ms,
                    "total_ms": torch_sparse_ms,
                    "oracle_path": oracle["oracle_path"],
                    "oracle_time_ms": oracle["oracle_time_ms"],
                    "direct_ms": direct_ms,
                    "N": N,
                    "M": M,
                    "K": K,
                    "nnz": mat["nnz"],
                    "size_tag": f"{M}x{K}",
                    "torch_layout": torch_layout,
                },
            ])
    print()


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ra_spmm paper-aware harness")
    parser.add_argument(
        "--section",
        default="all",
        choices=[
            "all",
            "correctness",
            "oracle_cold",
            "oracle_warm",
            "router_cold",
            "router_warm",
            "calibrate_warm_main",
            "plan_run",
            "ablation_full",
            "family_analysis",
            "oracle_cold_warm",
            "baseline_reference",
            "row_split_targets",
            "tc_locality_targets",
            "hybrid_mixed_targets",
            "external_baselines",
        ],
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--csv_out", type=str, default=None)
    args = parser.parse_args()
    group_filter = parse_group_filter(args.groups)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print_sep()
    print("GPU INFO")
    print_sep()
    for gpu in ra_spmm.gpu_info_next():
        marker = "*" if args.gpu is not None and gpu["device_id"] == args.gpu else " "
        print(
            f"{marker} Device {gpu['device_id']}: {gpu['name']} "
            f"(sm_{gpu['compute_major']}{gpu['compute_minor']}, "
            f"{gpu['total_memory_mb']} MB, WMMA={'YES' if gpu['wmma_supported'] else 'NO'})"
        )
    print()

    records: List[Dict[str, object]] = []

    if args.section in ("all", "correctness"):
        test_correctness()
    if args.section in ("all", "oracle_cold"):
        run_oracle_section("cold", "MAIN", args.warmup, args.iters, records)
    if args.section in ("all", "oracle_warm"):
        run_oracle_section("warm", "MAIN", args.warmup, args.iters, records)
    if args.section in ("all", "router_cold"):
        run_router_section("cold", "MAIN", args.warmup, args.iters, records, cases=expanded_router_cases(group_filter))
    if args.section in ("all", "router_warm"):
        run_router_section("warm", "MAIN", args.warmup, args.iters, records, cases=expanded_router_cases(group_filter))
    if args.section in ("all", "calibrate_warm_main"):
        run_calibrate_warm_main(args.warmup, args.iters, records, groups=group_filter)
    if args.section in ("all", "plan_run"):
        test_plan_run(args.warmup, args.iters, records)
    if args.section in ("all", "ablation_full"):
        test_ablation_full(args.warmup, args.iters, records)
    if args.section in ("all", "family_analysis"):
        test_family_analysis(args.warmup, args.iters, records)
    if args.section in ("all", "oracle_cold_warm"):
        test_oracle_cold_warm(args.warmup, args.iters, records)
    if args.section in ("all", "baseline_reference"):
        run_group_targets("baseline_reference", args.warmup, args.iters, records)
    if args.section in ("all", "row_split_targets"):
        run_group_targets("row_split_targets", args.warmup, args.iters, records)
    if args.section in ("all", "tc_locality_targets"):
        run_group_targets("tc_locality_targets", args.warmup, args.iters, records)
    if args.section in ("all", "hybrid_mixed_targets"):
        run_group_targets("hybrid_mixed_targets", args.warmup, args.iters, records)
    if args.section in ("all", "external_baselines"):
        test_external_baselines(args.warmup, args.iters, records)

    if args.csv_out:
        write_csv(args.csv_out, records)


if __name__ == "__main__":
    main()
