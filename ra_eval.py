import argparse
import atexit
import math
import os
import signal
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

import test_next as harness
from paper_eval_utils import (
    EXTERNAL_BASELINE_PATHS,
    MAIN_PATHS,
    MEMORY_POLICIES,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_OOM,
    STATUS_SKIPPED_BY_MANIFEST,
    STATUS_SKIPPED_MEMORY,
    EvalMatrixCase,
    ExperimentRunner,
    bandwidth_gbps,
    choose_policy_path,
    collect_cases,
    dataset_inventory_rows,
    ensure_results_tree,
    first_non_ok_status,
    geomean,
    is_ok_status,
    measure_row_split_ablation,
    percentile,
    CATEGORY_ORDER,
    restricted_choice_from_path_results,
    safe_div,
    status_counts,
    write_csv_rows,
    write_json,
    write_latex_table,
)


DEFAULT_NS = (64, 128, 256, 512)
ROUTER_ABLATIONS = [
    "always_direct",
    "no_row_split",
    "no_tc_paths",
    "direct_vs_row_split_only",
    "current_router",
]
FEATURE_ABLATIONS = ["none", "no_skew", "no_locality", "no_mixedness"]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_results_lock(results_root: str) -> None:
    lock_path = os.path.join(results_root, ".paper_eval.lock")
    os.makedirs(results_root, exist_ok=True)
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                existing_pid = int(handle.read().strip() or "0")
        except (OSError, ValueError):
            existing_pid = 0
        if existing_pid > 0 and _pid_alive(existing_pid):
            raise RuntimeError(
                f"results_dir is already locked by running paper_eval process pid={existing_pid}: {results_root}"
            )
        try:
            os.remove(lock_path)
        except OSError:
            pass

    with open(lock_path, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    def _release_lock(*_args: object) -> None:
        try:
            if os.path.exists(lock_path):
                with open(lock_path, "r", encoding="utf-8") as handle:
                    owner = int(handle.read().strip() or "0")
                if owner == os.getpid():
                    os.remove(lock_path)
        except (OSError, ValueError):
            pass

    atexit.register(_release_lock)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(signum)

        def _handler(sig: int, frame: object, prev=previous) -> None:
            _release_lock()
            if callable(prev):
                prev(sig, frame)
            raise SystemExit(130 if sig == signal.SIGINT else 143)

        signal.signal(signum, _handler)


def print_sep(width: int = 120) -> None:
    print("=" * width)


def _ok_metric_values(
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    status_key: str = "status",
) -> List[float]:
    values: List[float] = []
    for row in rows:
        if not is_ok_status(row.get(status_key)):
            continue
        value = row.get(value_key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def _attach_status_counts(
    entry: Dict[str, object],
    rows: Sequence[Mapping[str, object]],
    prefix: str,
    status_key: str = "status",
) -> None:
    counts = status_counts(rows, status_key=status_key)
    for status, count in counts.items():
        entry[f"{prefix}_{status.lower()}"] = count
    entry[f"{prefix}_cases"] = len(rows)


def _format_metric_or_status(value: float, counts: Mapping[str, int], fmt: str = "{:.2f}") -> str:
    if int(counts.get(STATUS_OK, 0)) > 0:
        return fmt.format(value)
    label = first_non_ok_status(counts)
    if label in {STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST}:
        return "SKIP"
    if label == STATUS_OOM:
        return "OOM"
    if label == STATUS_ERROR:
        return "ERR"
    return "--"


def parse_sections(raw: str) -> List[str]:
    sections = [token.strip() for token in raw.split(",") if token.strip()]
    return sections or ["all"]


def parse_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def filter_cases(
    cases: Sequence[EvalMatrixCase],
    include_groups: Sequence[str] | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    include_sources: Sequence[str] | None = None,
) -> List[EvalMatrixCase]:
    out = list(cases)
    if include_groups:
        wanted_groups = set(include_groups)
        out = [case for case in out if case.group in wanted_groups]
    if include_names:
        wanted_names = set(include_names)
        out = [case for case in out if case.name in wanted_names]
    if exclude_names:
        blocked_names = set(exclude_names)
        out = [case for case in out if case.name not in blocked_names]
    if include_sources:
        wanted_sources = set(include_sources)
        out = [case for case in out if case.source in wanted_sources]
    return out


def write_dataset_inventory(cases: Sequence[EvalMatrixCase], results_dirs: Mapping[str, str]) -> None:
    rows = dataset_inventory_rows(cases)
    write_csv_rows(os.path.join(results_dirs["csv"], "dataset_inventory.csv"), rows)
    write_json(os.path.join(results_dirs["json"], "dataset_inventory.json"), rows)


def _summary_rows_from_group_path_metric(
    rows: Sequence[Mapping[str, object]],
    metric_key: str,
    value_key: str,
    groups: Sequence[str],
    paths: Sequence[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for group in list(groups) + ["overall"]:
        group_rows = list(rows) if group == "overall" else [row for row in rows if row[metric_key] == group]
        if not group_rows:
            continue
        entry: Dict[str, object] = {"group": group}
        for path in paths:
            path_rows = [row for row in group_rows if row["path"] == path]
            entry[path] = geomean(_ok_metric_values(path_rows, value_key)) if path_rows else 0.0
            _attach_status_counts(entry, path_rows, path)
        out.append(entry)
    return out


def run_main_kernel_comparison(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for N in case.Ns:
            warm = runner.warm_oracle(case, N)
            path_results = warm["path_results"]
            direct_ok = is_ok_status(path_results["CSR_DIRECT"]["status"])
            direct_ms = float(path_results["CSR_DIRECT"]["total_ms"]) if direct_ok and path_results["CSR_DIRECT"]["total_ms"] is not None else math.inf
            oracle_ms = float(warm["oracle_time_ms"]) if warm["oracle_path"] != "NONE" else math.inf
            for path in MAIN_PATHS:
                timing = path_results[path]
                total_ms = float(timing["total_ms"]) if timing["total_ms"] is not None else None
                exec_ms = float(timing["exec_ms"]) if timing["exec_ms"] is not None else None
                row = {
                    "graph": case.name,
                    "source": case.source,
                    "category": case.category,
                    "graph_group": case.group,
                    "tags": ",".join(case.tags),
                    "size_tag": case.size_tag,
                    "M": case.M,
                    "K": case.K,
                    "N": N,
                    "nnz": warm["nnz"],
                    "oracle_path": warm["oracle_path"],
                    "oracle_time_ms": oracle_ms,
                    "oracle_dataset_status": warm["dataset_status"],
                }
                row.update(timing)
                row["speedup_vs_direct"] = (
                    safe_div(direct_ms, total_ms)
                    if is_ok_status(timing["status"]) and math.isfinite(direct_ms) and total_ms is not None
                    else None
                )
                row["speedup_vs_oracle"] = (
                    safe_div(oracle_ms, total_ms)
                    if is_ok_status(timing["status"]) and math.isfinite(oracle_ms) and total_ms is not None
                    else None
                )
                row["estimated_bandwidth_gbps"] = (
                    bandwidth_gbps(int(warm["nnz"]), case.M, N, exec_ms or total_ms)
                    if is_ok_status(timing["status"]) and (exec_ms or total_ms)
                    else None
                )
                rows.append(row)

    summary_rows = _summary_rows_from_group_path_metric(
        rows, "category", "speedup_vs_direct",
        CATEGORY_ORDER,
        MAIN_PATHS,
    )

    write_csv_rows(os.path.join(results_dirs["csv"], "main_kernel_points.csv"), rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "main_kernel_summary.csv"), summary_rows)

    latex_rows = []
    for row in summary_rows:
        table_row = [row["group"]]
        for path in MAIN_PATHS:
            counts = {status: int(row.get(f"{path}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}
            table_row.append(_format_metric_or_status(row.get(path, ""), counts))
        latex_rows.append(table_row)
    latex_headers = ["Category"] + [p.replace("_CUDA", "").replace("HYBRID_TC", "HYBRID") for p in MAIN_PATHS]
    write_latex_table(
        os.path.join(results_dirs["tables"], "kernel_comparison_main.tex"),
        headers=latex_headers,
        rows=latex_rows,
        caption="Warm-mode geomean speedup versus CSR_DIRECT for the MAIN kernel portfolio.",
        label="tab:kernel-main",
    )
    return rows, summary_rows


def run_external_baselines(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for N in case.Ns:
            warm = runner.warm_oracle(case, N)
            external_results = runner.warm_external_baselines(case, N)
            direct = warm["path_results"]["CSR_DIRECT"]
            direct_ok = is_ok_status(direct["status"])
            direct_ms = float(direct["total_ms"]) if direct_ok and direct["total_ms"] is not None else math.inf
            oracle_ms = float(warm["oracle_time_ms"]) if warm["oracle_path"] != "NONE" else math.inf
            for path in EXTERNAL_BASELINE_PATHS:
                timing = external_results[path]
                total_ms = float(timing["total_ms"]) if timing["total_ms"] is not None else None
                exec_ms = float(timing["exec_ms"]) if timing["exec_ms"] is not None else None
                row = {
                    "graph": case.name,
                    "source": case.source,
                    "category": case.category,
                    "graph_group": case.group,
                    "tags": ",".join(case.tags),
                    "size_tag": case.size_tag,
                    "M": case.M,
                    "K": case.K,
                    "N": N,
                    "nnz": warm["nnz"],
                    "oracle_path": warm["oracle_path"],
                    "oracle_time_ms": oracle_ms,
                    "oracle_dataset_status": warm["dataset_status"],
                }
                row.update(timing)
                row["speedup_vs_direct"] = (
                    safe_div(direct_ms, total_ms)
                    if is_ok_status(timing["status"]) and math.isfinite(direct_ms) and total_ms is not None
                    else None
                )
                row["speedup_vs_oracle"] = (
                    safe_div(oracle_ms, total_ms)
                    if is_ok_status(timing["status"]) and math.isfinite(oracle_ms) and total_ms is not None
                    else None
                )
                row["estimated_bandwidth_gbps"] = (
                    bandwidth_gbps(int(warm["nnz"]), case.M, N, exec_ms or total_ms)
                    if is_ok_status(timing["status"]) and (exec_ms or total_ms)
                    else None
                )
                rows.append(row)

    summary_rows = _summary_rows_from_group_path_metric(
        rows, "category", "speedup_vs_direct",
        CATEGORY_ORDER,
        EXTERNAL_BASELINE_PATHS,
    )

    write_csv_rows(os.path.join(results_dirs["csv"], "external_baseline_points.csv"), rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "external_baseline_summary.csv"), summary_rows)
    ext_headers = ["Category"] + EXTERNAL_BASELINE_PATHS
    ext_table_rows = []
    for row in summary_rows:
        table_row = [row["group"]]
        for path in EXTERNAL_BASELINE_PATHS:
            counts = {status: int(row.get(f"{path}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}
            table_row.append(_format_metric_or_status(row.get(path, ""), counts))
        ext_table_rows.append(table_row)
    write_latex_table(
        os.path.join(results_dirs["tables"], "external_baselines.tex"),
        headers=ext_headers,
        rows=ext_table_rows,
        caption="Warm-mode geomean speedup versus CSR_DIRECT for external GPU baselines.",
        label="tab:external-baselines",
    )
    return rows, summary_rows


def build_n_scaling_summary(
    main_rows: Sequence[Mapping[str, object]],
    results_dirs: Mapping[str, str],
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    by_key: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in main_rows:
        if is_ok_status(row.get("status")) and row.get("speedup_vs_direct") not in (None, ""):
            by_key[(int(row["N"]), str(row["path"]))].append(float(row["speedup_vs_direct"]))
    for (N, path), values in sorted(by_key.items()):
        summary.append({"N": N, "path": path, "geomean_speedup_vs_direct": geomean(values)})
    write_csv_rows(os.path.join(results_dirs["csv"], "n_scaling_summary.csv"), summary)
    return summary


def run_router_vs_oracle(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for N in case.Ns:
            oracle = runner.warm_oracle(case, N)
            plan = runner.router_plan(case, N)
            chosen_path = str(plan["chosen_path"])
            # Use the warm oracle's per-path timing table for router-quality
            # accounting so the live router section is directly comparable to
            # the offline current_router ablation. We intentionally avoid
            # re-executing the chosen path here because a single illegal-access
            # failure would poison the CUDA context for all subsequent router
            # cases in the same process. The router section is a policy-quality
            # study, so the warm oracle timing table is the correct basis.
            router_timing = oracle["path_results"][chosen_path]
            slowdown = (
                safe_div(float(router_timing["total_ms"]), float(oracle["oracle_time_ms"]))
                if is_ok_status(router_timing["status"]) and oracle["oracle_path"] != "NONE" and router_timing["total_ms"] is not None
                else None
            )
            row = {
                "graph": case.name,
                "source": case.source,
                "category": case.category,
                "graph_group": case.group,
                "size_tag": case.size_tag,
                "M": case.M,
                "K": case.K,
                "N": N,
                "nnz": oracle["nnz"],
                "router_path": chosen_path,
                "oracle_path": oracle["oracle_path"],
                "router_status": router_timing["status"],
                "router_status_reason": router_timing["status_reason"],
                "router_total_ms": float(router_timing["total_ms"]) if router_timing["total_ms"] is not None else None,
                "router_exec_ms": float(router_timing["exec_ms"]) if router_timing["exec_ms"] is not None else None,
                "router_plan_ms": float(plan["planning_time_ms"]) if plan.get("planning_time_ms") is not None else None,
                "oracle_time_ms": float(oracle["oracle_time_ms"]),
                "oracle_dataset_status": oracle["dataset_status"],
                "oracle_ok_paths": int(oracle["status_counts"][STATUS_OK]),
                "slowdown": slowdown,
            }
            row.update({
                "status": router_timing["status"],
                "status_reason": router_timing["status_reason"],
                "attempted": router_timing["attempted"],
                "timed": router_timing["timed"],
                "memory_policy": router_timing["memory_policy"],
                "oom_retry_attempted": router_timing["oom_retry_attempted"],
                "oom_retry_succeeded": router_timing["oom_retry_succeeded"],
                "memory_estimate_bytes": router_timing["memory_estimate_bytes"],
                "memory_estimate_gb": router_timing["memory_estimate_gb"],
                "memory_limit_gb": router_timing["memory_limit_gb"],
            })
            row.update(harness.flatten_router_plan(plan, "MAIN"))
            rows.append(row)

    summary_rows: List[Dict[str, object]] = []
    for group in ["baseline_reference", "row_split_targets", "tc_locality_targets", "hybrid_mixed_targets", "overall"]:
        group_rows = rows if group == "overall" else [row for row in rows if row["graph_group"] == group]
        if not group_rows:
            continue
        slowdowns = [float(row["slowdown"]) for row in group_rows if row["slowdown"] not in (None, "")]
        entry = {
            "group": group,
            "cases": len(group_rows),
            "avg_slowdown": (sum(slowdowns) / len(slowdowns)) if slowdowns else math.inf,
            "worst_slowdown": max(slowdowns) if slowdowns else math.inf,
            "pct_within_1.05x": (100.0 * sum(sd <= 1.05 for sd in slowdowns) / len(slowdowns)) if slowdowns else 0.0,
            "pct_within_1.10x": (100.0 * sum(sd <= 1.10 for sd in slowdowns) / len(slowdowns)) if slowdowns else 0.0,
            "router_selection_counts": dict(Counter(row["router_path"] for row in group_rows)),
            "oracle_selection_counts": dict(Counter(row["oracle_path"] for row in group_rows)),
        }
        _attach_status_counts(entry, group_rows, "router", status_key="router_status")
        entry["no_oracle_cases"] = sum(row["oracle_path"] == "NONE" for row in group_rows)
        summary_rows.append(entry)

    write_csv_rows(os.path.join(results_dirs["csv"], "router_vs_oracle_points.csv"), rows)
    write_json(os.path.join(results_dirs["json"], "router_vs_oracle_summary.json"), summary_rows)
    write_csv_rows(
        os.path.join(results_dirs["csv"], "router_vs_oracle_summary.csv"),
        [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in summary_rows],
    )
    write_latex_table(
        os.path.join(results_dirs["tables"], "router_vs_oracle.tex"),
        headers=["Group", "Cases", "Avg Slowdown", "Worst", "Within 1.05x", "Within 1.10x"],
        rows=[[
            row["group"],
            row["cases"],
            _format_metric_or_status(row["avg_slowdown"], {status: int(row.get(f"router_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}x"),
            _format_metric_or_status(row["worst_slowdown"], {status: int(row.get(f"router_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}x"),
            f"{row['pct_within_1.05x']:.1f}\\%" if int(row.get("router_ok", 0)) > 0 else "--",
            f"{row['pct_within_1.10x']:.1f}\\%" if int(row.get("router_ok", 0)) > 0 else "--",
        ] for row in summary_rows],
        caption="Warm router quality relative to the warm oracle on the paper evaluation suite.",
        label="tab:router-vs-oracle",
    )
    return rows, summary_rows


def run_router_ablation_study(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for N in case.Ns:
            warm = runner.warm_oracle(case, N)
            features = runner.features(case, N)
            for policy_mode in ROUTER_ABLATIONS:
                chosen_path, reason = restricted_choice_from_path_results(
                    warm["path_results"], features, N, policy_mode)
                chosen_timing = warm["path_results"][chosen_path]
                slowdown = (
                    safe_div(float(chosen_timing["total_ms"]), float(warm["oracle_time_ms"]))
                    if is_ok_status(chosen_timing["status"]) and warm["oracle_path"] != "NONE" and chosen_timing["total_ms"] is not None
                    else None
                )
                rows.append({
                    "ablation": policy_mode,
                    "graph": case.name,
                    "source": case.source,
                    "category": case.category,
                    "graph_group": case.group,
                    "size_tag": case.size_tag,
                    "N": N,
                    "path": chosen_path,
                    "status": chosen_timing["status"],
                    "status_reason": chosen_timing["status_reason"],
                    "decision_reason": reason,
                    "total_ms": float(chosen_timing["total_ms"]) if chosen_timing["total_ms"] is not None else None,
                    "oracle_path": warm["oracle_path"],
                    "oracle_time_ms": float(warm["oracle_time_ms"]),
                    "slowdown": slowdown,
                })

    summary_rows: List[Dict[str, object]] = []
    for ablation in ROUTER_ABLATIONS:
        ablation_rows = [row for row in rows if row["ablation"] == ablation]
        slowdowns = [float(row["slowdown"]) for row in ablation_rows if row["slowdown"] not in (None, "")]
        entry = {
            "ablation": ablation,
            "cases": len(ablation_rows),
            "avg_slowdown": (sum(slowdowns) / len(slowdowns)) if slowdowns else math.inf,
            "worst_slowdown": max(slowdowns) if slowdowns else math.inf,
            "pct_within_1.10x": (100.0 * sum(sd <= 1.10 for sd in slowdowns) / len(slowdowns)) if slowdowns else 0.0,
            "selection_counts": dict(Counter(row["path"] for row in ablation_rows)),
        }
        _attach_status_counts(entry, ablation_rows, "ablation")
        summary_rows.append(entry)

    write_csv_rows(os.path.join(results_dirs["csv"], "router_ablations_points.csv"), rows)
    write_json(os.path.join(results_dirs["json"], "router_ablations_summary.json"), summary_rows)
    write_latex_table(
        os.path.join(results_dirs["tables"], "router_ablations.tex"),
        headers=["Ablation", "Cases", "Avg Slowdown", "Worst", "Within 1.10x"],
        rows=[[
            row["ablation"],
            row["cases"],
            _format_metric_or_status(row["avg_slowdown"], {status: int(row.get(f"ablation_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}x"),
            _format_metric_or_status(row["worst_slowdown"], {status: int(row.get(f"ablation_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}x"),
            f"{row['pct_within_1.10x']:.1f}\\%" if int(row.get("ablation_ok", 0)) > 0 else "--",
        ] for row in summary_rows],
        caption="Router portfolio ablations using the current warm-calibrated policy replicated offline for evaluation.",
        label="tab:router-ablations",
    )
    return rows, summary_rows


def run_feature_ablation_study(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for N in case.Ns:
            warm = runner.warm_oracle(case, N)
            features = runner.features(case, N)
            for ablation in FEATURE_ABLATIONS:
                chosen_path, reason = choose_policy_path(features, N, feature_ablation=ablation if ablation != "none" else None)
                chosen_timing = warm["path_results"][chosen_path]
                slowdown = (
                    safe_div(float(chosen_timing["total_ms"]), float(warm["oracle_time_ms"]))
                    if is_ok_status(chosen_timing["status"]) and warm["oracle_path"] != "NONE" and chosen_timing["total_ms"] is not None
                    else None
                )
                rows.append({
                    "feature_ablation": ablation,
                    "graph": case.name,
                    "source": case.source,
                    "category": case.category,
                    "graph_group": case.group,
                    "size_tag": case.size_tag,
                    "N": N,
                    "path": chosen_path,
                    "status": chosen_timing["status"],
                    "status_reason": chosen_timing["status_reason"],
                    "decision_reason": reason,
                    "total_ms": float(chosen_timing["total_ms"]) if chosen_timing["total_ms"] is not None else None,
                    "oracle_path": warm["oracle_path"],
                    "oracle_time_ms": float(warm["oracle_time_ms"]),
                    "slowdown": slowdown,
                })

    summary_rows: List[Dict[str, object]] = []
    for ablation in FEATURE_ABLATIONS:
        ablation_rows = [row for row in rows if row["feature_ablation"] == ablation]
        slowdowns = [float(row["slowdown"]) for row in ablation_rows if row["slowdown"] not in (None, "")]
        entry = {
            "feature_ablation": ablation,
            "cases": len(ablation_rows),
            "avg_slowdown": (sum(slowdowns) / len(slowdowns)) if slowdowns else math.inf,
            "worst_slowdown": max(slowdowns) if slowdowns else math.inf,
            "pct_within_1.10x": (100.0 * sum(sd <= 1.10 for sd in slowdowns) / len(slowdowns)) if slowdowns else 0.0,
            "selection_counts": dict(Counter(row["path"] for row in ablation_rows)),
        }
        _attach_status_counts(entry, ablation_rows, "feature")
        summary_rows.append(entry)

    write_csv_rows(os.path.join(results_dirs["csv"], "feature_ablations_points.csv"), rows)
    write_json(os.path.join(results_dirs["json"], "feature_ablations_summary.json"), summary_rows)
    write_latex_table(
        os.path.join(results_dirs["tables"], "feature_ablations.tex"),
        headers=["Feature Ablation", "Cases", "Avg Slowdown", "Worst", "Within 1.10x"],
        rows=[[
            row["feature_ablation"],
            row["cases"],
            _format_metric_or_status(row["avg_slowdown"], {status: int(row.get(f"feature_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}x"),
            _format_metric_or_status(row["worst_slowdown"], {status: int(row.get(f"feature_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}x"),
            f"{row['pct_within_1.10x']:.1f}\\%" if int(row.get("feature_ok", 0)) > 0 else "--",
        ] for row in summary_rows],
        caption="Feature-group ablations using the current router policy replicated offline for analysis.",
        label="tab:feature-ablations",
    )
    return rows, summary_rows


def run_kernel_ablation_study(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    target_cases = [
        case for case in cases
        if case.group in {"row_split_targets", "hybrid_mixed_targets"}
    ]
    rows: List[Dict[str, object]] = []
    for case in target_cases:
        for N in [n for n in case.Ns if n >= 128]:
            rows.append(measure_row_split_ablation(runner, case, N))

    summary_rows: List[Dict[str, object]] = []
    for group in ["row_split_targets", "hybrid_mixed_targets", "overall"]:
        group_rows = rows if group == "overall" else [row for row in rows if row["graph_group"] == group]
        if not group_rows:
            continue
        ok_rows = [row for row in group_rows if is_ok_status(row.get("status"))]
        entry = {
            "group": group,
            "cases": len(group_rows),
            "median_row_split_long_row_speedup": percentile([row["row_split_long_row_speedup"] for row in ok_rows], 0.5),
            "median_row_split_vectorization_speedup": percentile([row["row_split_vectorization_speedup"] for row in ok_rows], 0.5),
            "median_csr_direct_vectorization_speedup": percentile([row["csr_direct_vectorization_speedup"] for row in ok_rows], 0.5),
        }
        _attach_status_counts(entry, group_rows, "ablation")
        summary_rows.append(entry)

    write_csv_rows(os.path.join(results_dirs["csv"], "kernel_ablations_points.csv"), rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "kernel_ablations_summary.csv"), summary_rows)
    write_latex_table(
        os.path.join(results_dirs["tables"], "kernel_ablations.tex"),
        headers=["Group", "Cases", "Long-Row Ablation", "ROW_SPLIT Vec4 Ablation", "CSR_DIRECT Vec4 Ablation"],
        rows=[[
            row["group"],
            row["cases"],
            _format_metric_or_status(row["median_row_split_long_row_speedup"], {status: int(row.get(f"ablation_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
            _format_metric_or_status(row["median_row_split_vectorization_speedup"], {status: int(row.get(f"ablation_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
            _format_metric_or_status(row["median_csr_direct_vectorization_speedup"], {status: int(row.get(f"ablation_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
        ] for row in summary_rows],
        caption="Kernel ablations for long-row handling and vectorization. Values above 1.0x indicate slowdown when the optimization is disabled.",
        label="tab:kernel-ablations",
    )
    return rows, summary_rows


def run_reuse_analysis(
    runner: ExperimentRunner,
    cases: Sequence[EvalMatrixCase],
    results_dirs: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        for N in case.Ns:
            warm = runner.warm_oracle(case, N)
            cold = runner.cold_oracle(case, N)
            direct_result = warm["path_results"]["CSR_DIRECT"]
            direct_ms = float(direct_result["total_ms"]) if is_ok_status(direct_result["status"]) and direct_result["total_ms"] is not None else math.inf
            for path in ["ROW_SPLIT_CUDA", "TC_REORDERED", "HYBRID_TC_CUDA"]:
                cold_t = cold["path_results"][path]
                warm_t = warm["path_results"][path]
                ok = (
                    is_ok_status(cold_t["status"]) and
                    is_ok_status(warm_t["status"]) and
                    math.isfinite(direct_ms) and
                    cold_t["plan_ms"] is not None and
                    warm_t["exec_ms"] is not None
                )
                plan_ms = float(cold_t["plan_ms"]) if ok else None
                warm_exec_ms = float(warm_t["exec_ms"]) if ok else None
                benefit = (direct_ms - warm_exec_ms) if ok and warm_exec_ms is not None else None
                row = {
                    "graph": case.name,
                    "source": case.source,
                    "category": case.category,
                    "graph_group": case.group,
                    "size_tag": case.size_tag,
                    "M": case.M,
                    "K": case.K,
                    "N": N,
                    "path": path,
                    "direct_ms": direct_ms,
                    "plan_ms": plan_ms,
                    "warm_exec_ms": warm_exec_ms,
                    "benefit_ms": benefit,
                    "status": STATUS_OK if ok else (warm_t["status"] if not is_ok_status(warm_t["status"]) else cold_t["status"]),
                    "status_reason": "" if ok else f"warm={warm_t['status_reason']} cold={cold_t['status_reason']}",
                    "break_even_runs": (math.inf if benefit is not None and benefit <= 0.0 else (plan_ms / benefit if ok and benefit is not None else None)),
                    "plan_exec_ratio": (safe_div(plan_ms, warm_exec_ms) if ok and warm_exec_ms is not None else None),
                    "warm_status": warm_t["status"],
                    "cold_status": cold_t["status"],
                }
                rows.append(row)

    summary_rows: List[Dict[str, object]] = []
    for path in ["ROW_SPLIT_CUDA", "TC_REORDERED", "HYBRID_TC_CUDA"]:
        for N in DEFAULT_NS:
            path_rows = [row for row in rows if row["path"] == path and int(row["N"]) == N]
            if not path_rows:
                continue
            ok_rows = [row for row in path_rows if is_ok_status(row.get("status"))]
            finite_break_even = [float(row["break_even_runs"]) for row in ok_rows if row["break_even_runs"] is not None and math.isfinite(float(row["break_even_runs"]))]
            entry = {
                "path": path,
                "N": N,
                "median_break_even_runs": percentile(finite_break_even, 0.5) if finite_break_even else math.inf,
                "median_plan_exec_ratio": percentile([float(row["plan_exec_ratio"]) for row in ok_rows if row["plan_exec_ratio"] is not None], 0.5),
            }
            _attach_status_counts(entry, path_rows, "reuse")
            summary_rows.append(entry)

    write_csv_rows(os.path.join(results_dirs["csv"], "reuse_analysis_points.csv"), rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "reuse_analysis_summary.csv"), summary_rows)
    write_latex_table(
        os.path.join(results_dirs["tables"], "reuse_break_even.tex"),
        headers=["Path", "N", "Median Break-Even Runs", "Median Plan/Exec"],
        rows=[[
            row["path"],
            row["N"],
            _format_metric_or_status(row["median_break_even_runs"], {status: int(row.get(f"reuse_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.1f}"),
            _format_metric_or_status(row["median_plan_exec_ratio"], {status: int(row.get(f"reuse_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
        ] for row in summary_rows],
        caption="Reuse break-even analysis aligned with the preprocessing-versus-reuse style used in RoDe-like evaluations.",
        label="tab:reuse-break-even",
    )
    return rows, summary_rows


def build_external_alignment_artifacts(
    main_rows: Sequence[Mapping[str, object]],
    reuse_rows: Sequence[Mapping[str, object]],
    results_dirs: Mapping[str, str],
) -> None:
    rode_rows = [
        row for row in main_rows
        if row["graph_group"] == "row_split_targets" and row["path"] in {"CSR_DIRECT", "ROW_SPLIT_CUDA"}
    ]
    tc_rows = [
        row for row in main_rows
        if row["graph_group"] == "tc_locality_targets"
    ]
    hybrid_rows = [
        row for row in main_rows
        if row["graph_group"] == "hybrid_mixed_targets"
    ]

    write_csv_rows(os.path.join(results_dirs["csv"], "rode_aligned_points.csv"), rode_rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "tc_aligned_points.csv"), tc_rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "hybrid_aligned_points.csv"), hybrid_rows)

    rode_summary_rows: List[Dict[str, object]] = []
    rode_break_even = {
        (str(row["graph"]), int(row["N"])): []
        for row in reuse_rows
        if row["path"] == "ROW_SPLIT_CUDA" and row["graph_group"] == "row_split_targets"
    }
    for row in reuse_rows:
        if row["path"] == "ROW_SPLIT_CUDA" and row["graph_group"] == "row_split_targets":
            if row.get("break_even_runs") not in (None, ""):
                rode_break_even[(str(row["graph"]), int(row["N"]))].append(float(row["break_even_runs"]))

    for graph in sorted({str(row["graph"]) for row in rode_rows}):
        for N in DEFAULT_NS:
            graph_rows = [row for row in rode_rows if row["graph"] == graph and int(row["N"]) == N]
            if not graph_rows:
                continue
            row_split_speedups = [
                float(row["speedup_vs_direct"]) for row in graph_rows
                if row["path"] == "ROW_SPLIT_CUDA" and row["speedup_vs_direct"] not in (None, "")
            ]
            direct_rows = [row for row in graph_rows if row["path"] == "CSR_DIRECT"]
            break_even_values = [
                value for value in rode_break_even.get((graph, N), [])
                if math.isfinite(value)
            ]
            rode_summary_rows.append({
                "graph": graph,
                "N": N,
                "row_split_geomean_speedup_vs_direct": geomean(row_split_speedups),
                "direct_geomean_speedup_vs_direct": geomean(
                    [float(row["speedup_vs_direct"]) for row in direct_rows if row["speedup_vs_direct"] not in (None, "")]
                ),
                "median_break_even_runs": percentile(break_even_values, 0.5) if break_even_values else math.inf,
            })
            _attach_status_counts(rode_summary_rows[-1], [row for row in graph_rows if row["path"] == "ROW_SPLIT_CUDA"], "row_split")

    tc_summary_rows: List[Dict[str, object]] = []
    for graph in sorted({str(row["graph"]) for row in tc_rows}):
        for N in DEFAULT_NS:
            graph_rows = [row for row in tc_rows if row["graph"] == graph and int(row["N"]) == N]
            if not graph_rows:
                continue
            tc_summary_rows.append({
                "graph": graph,
                "N": N,
                "csr_direct_geomean_speedup_vs_direct": geomean([
                    float(row["speedup_vs_direct"]) for row in graph_rows
                    if row["path"] == "CSR_DIRECT" and row["speedup_vs_direct"] not in (None, "")
                ]),
                "tc_reordered_geomean_speedup_vs_direct": geomean([
                    float(row["speedup_vs_direct"]) for row in graph_rows
                    if row["path"] == "TC_REORDERED" and row["speedup_vs_direct"] not in (None, "")
                ]),
                "hybrid_geomean_speedup_vs_direct": geomean([
                    float(row["speedup_vs_direct"]) for row in graph_rows
                    if row["path"] == "HYBRID_TC_CUDA" and row["speedup_vs_direct"] not in (None, "")
                ]),
            })
            _attach_status_counts(tc_summary_rows[-1], [row for row in graph_rows if row["path"] == "TC_REORDERED"], "tc_reordered")
            _attach_status_counts(tc_summary_rows[-1], [row for row in graph_rows if row["path"] == "HYBRID_TC_CUDA"], "hybrid")

    hybrid_summary_rows: List[Dict[str, object]] = []
    for graph in sorted({str(row["graph"]) for row in hybrid_rows}):
        for N in DEFAULT_NS:
            graph_rows = [row for row in hybrid_rows if row["graph"] == graph and int(row["N"]) == N]
            if not graph_rows:
                continue
            hybrid_summary_rows.append({
                "graph": graph,
                "N": N,
                "row_split_geomean_speedup_vs_direct": geomean([
                    float(row["speedup_vs_direct"]) for row in graph_rows
                    if row["path"] == "ROW_SPLIT_CUDA" and row["speedup_vs_direct"] not in (None, "")
                ]),
                "hybrid_geomean_speedup_vs_direct": geomean([
                    float(row["speedup_vs_direct"]) for row in graph_rows
                    if row["path"] == "HYBRID_TC_CUDA" and row["speedup_vs_direct"] not in (None, "")
                ]),
                "tc_reordered_geomean_speedup_vs_direct": geomean([
                    float(row["speedup_vs_direct"]) for row in graph_rows
                    if row["path"] == "TC_REORDERED" and row["speedup_vs_direct"] not in (None, "")
                ]),
            })
            _attach_status_counts(hybrid_summary_rows[-1], [row for row in graph_rows if row["path"] == "ROW_SPLIT_CUDA"], "row_split")
            _attach_status_counts(hybrid_summary_rows[-1], [row for row in graph_rows if row["path"] == "HYBRID_TC_CUDA"], "hybrid")
            _attach_status_counts(hybrid_summary_rows[-1], [row for row in graph_rows if row["path"] == "TC_REORDERED"], "tc_reordered")

    write_csv_rows(os.path.join(results_dirs["csv"], "rode_aligned_summary.csv"), rode_summary_rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "tc_aligned_summary.csv"), tc_summary_rows)
    write_csv_rows(os.path.join(results_dirs["csv"], "hybrid_aligned_summary.csv"), hybrid_summary_rows)

    write_latex_table(
        os.path.join(results_dirs["tables"], "rode_aligned.tex"),
        headers=["Graph", "N", "ROW\\_SPLIT Speedup", "Break-Even Runs"],
        rows=[[
            row["graph"],
            row["N"],
            _format_metric_or_status(row["row_split_geomean_speedup_vs_direct"], {status: int(row.get(f"row_split_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
            _format_metric_or_status(row["median_break_even_runs"], {status: int(row.get(f"row_split_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.1f}"),
        ] for row in rode_summary_rows],
        caption="Internal comparison aligned with RoDe methodology on skewed and power-law regimes.",
        label="tab:rode-aligned",
    )
    write_latex_table(
        os.path.join(results_dirs["tables"], "tc_aligned.tex"),
        headers=["Graph", "N", "CSR\\_DIRECT", "TC\\_REORDERED", "HYBRID"],
        rows=[[
            row["graph"],
            row["N"],
            f"{row['csr_direct_geomean_speedup_vs_direct']:.2f}",
            _format_metric_or_status(row["tc_reordered_geomean_speedup_vs_direct"], {status: int(row.get(f"tc_reordered_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
            _format_metric_or_status(row["hybrid_geomean_speedup_vs_direct"], {status: int(row.get(f"hybrid_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
        ] for row in tc_summary_rows],
        caption="Internal comparison aligned with FlashSparse / Acc-SpMM / DTC methodology on locality-oriented regimes.",
        label="tab:tc-aligned",
    )
    write_latex_table(
        os.path.join(results_dirs["tables"], "hybrid_aligned.tex"),
        headers=["Graph", "N", "ROW\\_SPLIT", "HYBRID", "TC\\_REORDERED"],
        rows=[[
            row["graph"],
            row["N"],
            _format_metric_or_status(row["row_split_geomean_speedup_vs_direct"], {status: int(row.get(f"row_split_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
            _format_metric_or_status(row["hybrid_geomean_speedup_vs_direct"], {status: int(row.get(f"hybrid_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
            _format_metric_or_status(row["tc_reordered_geomean_speedup_vs_direct"], {status: int(row.get(f"tc_reordered_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}),
        ] for row in hybrid_summary_rows],
        caption="Internal comparison aligned with Libra / RSH methodology on mixed-structure regimes.",
        label="tab:hybrid-aligned",
    )

    write_json(os.path.join(results_dirs["json"], "external_alignment_notes.json"), {
        "rode_aligned": "Internal comparison aligned with RoDe methodology: skewed, hub-heavy, and power-law matrices; warm execution, plan reuse, and break-even analysis.",
        "tc_aligned": "Internal comparison aligned with FlashSparse / Acc-SpMM / DTC methodology: N scaling, locality-oriented matrices, and explicit reporting of where TC_REORDERED helps or fails.",
        "hybrid_aligned": "Internal comparison aligned with Libra / RSH methodology: mixed-structure matrices and explicit reporting that HYBRID_TC_CUDA is not yet a broad winner.",
        "reuse_note": "Preprocessing overhead and break-even are reported from cold plan time and warm execution time, not from convenience rebuild wrappers.",
        "reuse_rows": len(reuse_rows),
        "rode_summary_rows": len(rode_summary_rows),
        "tc_summary_rows": len(tc_summary_rows),
        "hybrid_summary_rows": len(hybrid_summary_rows),
    })


def build_speedup_table(
    main_summary_rows: Sequence[Mapping[str, object]],
    results_dirs: Mapping[str, str],
) -> None:
    # Compare all non-CSR_DIRECT paths against CSR_DIRECT
    compare_paths = [p for p in MAIN_PATHS if p != "CSR_DIRECT"]
    rows = []
    for row in main_summary_rows:
        entry = {"category": row["group"]}
        for path in compare_paths:
            entry[f"{path.lower()}_speedup"] = row.get(path, "")
            for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]:
                entry[f"{path}_{status.lower()}"] = row.get(f"{path}_{status.lower()}", 0)
        rows.append(entry)
    write_csv_rows(os.path.join(results_dirs["csv"], "speedup_vs_csr_direct_summary.csv"), rows)
    latex_rows = []
    for row in rows:
        table_row = [row["category"]]
        for path in compare_paths:
            counts = {status: int(row.get(f"{path}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}
            table_row.append(_format_metric_or_status(row.get(f"{path.lower()}_speedup", ""), counts))
        latex_rows.append(table_row)
    latex_headers = ["Category"] + [p.replace("_CUDA", "").replace("HYBRID_TC", "HYBRID") for p in compare_paths]
    write_latex_table(
        os.path.join(results_dirs["tables"], "speedup_vs_csr_direct.tex"),
        headers=latex_headers,
        rows=latex_rows,
        caption="Geomean speedup versus CSR_DIRECT across evaluation categories.",
        label="tab:speedup-vs-direct",
    )


def run_absolute_throughput_table(
    main_rows: Sequence[Mapping[str, object]],
    results_dirs: Mapping[str, str],
    roofline_gbps: float = 936.0,
) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for group in list(CATEGORY_ORDER) + ["overall"]:
        group_rows = list(main_rows) if group == "overall" else [row for row in main_rows if row["category"] == group]
        if not group_rows:
            continue
        for path in MAIN_PATHS:
            path_rows = [
                row for row in group_rows
                if row["path"] == path and is_ok_status(row.get("status")) and row.get("exec_ms") not in (None, "")
            ]
            entry: Dict[str, object] = {
                "group": group,
                "path": path,
                "roofline_gbps": roofline_gbps,
            }
            _attach_status_counts(entry, [row for row in group_rows if row["path"] == path], path)
            if not path_rows:
                entry.update({
                    "median_exec_ms": None,
                    "median_gflops": None,
                    "median_bandwidth_gbps": None,
                    "pct_of_roofline": None,
                })
                summary_rows.append(entry)
                continue

            exec_mses = [float(row["exec_ms"]) for row in path_rows]
            gflops = [float(row["gflops"]) for row in path_rows if row.get("gflops") not in (None, "")]
            bandwidths = [
                float(row["estimated_bandwidth_gbps"])
                for row in path_rows
                if row.get("estimated_bandwidth_gbps") not in (None, "")
            ]
            median_exec_ms = percentile(exec_mses, 0.5)
            median_gflops = percentile(gflops, 0.5) if gflops else None
            median_bandwidth = percentile(bandwidths, 0.5) if bandwidths else None
            clipped_bandwidth = min(median_bandwidth, roofline_gbps) if median_bandwidth is not None else None
            entry.update({
                "median_exec_ms": median_exec_ms,
                "median_gflops": median_gflops,
                "median_bandwidth_gbps": median_bandwidth,
                "pct_of_roofline": safe_div(clipped_bandwidth, roofline_gbps) * 100.0 if clipped_bandwidth is not None else None,
                "roofline_clipped_gbps": clipped_bandwidth,
                "roofline_pct_capped": bool(median_bandwidth is not None and median_bandwidth > roofline_gbps),
            })
            summary_rows.append(entry)

    write_csv_rows(os.path.join(results_dirs["csv"], "absolute_throughput_summary.csv"), summary_rows)
    write_latex_table(
        os.path.join(results_dirs["tables"], "absolute_throughput.tex"),
        headers=["Group", "Path", "Median Exec (ms)", "Median GFLOPS", "Median GB/s", "% of 3090 Roofline"],
        rows=[[
            row["group"],
            row["path"],
            _format_metric_or_status(row["median_exec_ms"], {status: int(row.get(f"{row['path']}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.3f}"),
            _format_metric_or_status(row["median_gflops"], {status: int(row.get(f"{row['path']}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.1f}"),
            _format_metric_or_status(row["median_bandwidth_gbps"], {status: int(row.get(f"{row['path']}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.1f}"),
            _format_metric_or_status(row["pct_of_roofline"], {status: int(row.get(f"{row['path']}_{status.lower()}", 0)) for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]}, "{:.1f}\\%"),
        ] for row in summary_rows],
        caption="Absolute throughput summary for the MAIN portfolio. GB/s is compared against an RTX 3090 GDDR6X roofline of approximately 936 GB/s, and the roofline percentage is capped at 100\\% when the proxy bandwidth exceeds device DRAM bandwidth due to cache reuse.",
        label="tab:absolute-throughput",
    )
    return summary_rows


def build_profiling_artifacts(
    main_rows: Sequence[Mapping[str, object]],
    router_rows: Sequence[Mapping[str, object]],
    reuse_summary_rows: Sequence[Mapping[str, object]],
    results_dirs: Mapping[str, str],
) -> None:
    selection_counts = Counter(
        row["router_path"]
        for row in router_rows
        if is_ok_status(row.get("router_status"))
    )
    router_status_summary = status_counts(router_rows, status_key="router_status")
    bandwidth_rows = []
    for path in MAIN_PATHS:
        path_rows = [row for row in main_rows if row["path"] == path and is_ok_status(row.get("status")) and row.get("estimated_bandwidth_gbps") not in (None, "")]
        if not path_rows:
            continue
        bandwidth_rows.append({
            "path": path,
            "median_bandwidth_gbps": percentile([float(row["estimated_bandwidth_gbps"]) for row in path_rows], 0.5),
        })
    partial_datasets: Dict[str, Dict[str, int]] = {}
    for row in main_rows:
        key = f"{row['graph']}:{row['size_tag']}"
        partial_datasets.setdefault(key, {status: 0 for status in [STATUS_OK, STATUS_OOM, STATUS_SKIPPED_MEMORY, STATUS_SKIPPED_BY_MANIFEST, STATUS_ERROR]})
        status = str(row.get("status", STATUS_ERROR))
        partial_datasets[key][status] = partial_datasets[key].get(status, 0) + 1
    write_csv_rows(os.path.join(results_dirs["csv"], "bandwidth_summary.csv"), bandwidth_rows)
    write_json(os.path.join(results_dirs["json"], "profiling_summary.json"), {
        "path_selection_histogram": dict(selection_counts),
        "router_status_counts": router_status_summary,
        "bandwidth_summary": bandwidth_rows,
        "reuse_summary": list(reuse_summary_rows),
        "partial_evaluation_datasets": partial_datasets,
    })
    write_json(os.path.join(results_dirs["json"], "memory_robustness_notes.json"), {
        "warm_cold_semantics": "Warm mode still measures execution with plan reuse. Cold mode still measures plan/build plus execution.",
        "infeasible_handling": "OOM and pre-launch memory-infeasible path/case combinations are recorded explicitly instead of being dropped or silently rerouted.",
        "oracle_rule": "Oracle is computed only over paths with status == OK. If no path is OK, oracle_path is NONE and oracle_time_ms is Infinity.",
    })


def build_figure_captions(results_dirs: Mapping[str, str]) -> None:
    captions = {
        "speedup_vs_csr_direct": "Speedup versus CSR_DIRECT across evaluation categories. ROW_SPLIT_CUDA dominates skewed and hub-heavy regimes, while TC_REORDERED remains narrow and HYBRID_TC_CUDA remains conservative.",
        "router_slowdown_cdf": "CDF of warm router slowdown relative to the warm oracle. The calibrated router removes catastrophic misroutes and concentrates most cases near the oracle.",
        "performance_vs_N": "Performance versus output feature dimension N, aligned with FlashSparse-style N scaling analyses.",
        "selection_histogram": "Histogram of MAIN-path selections made by the warm-calibrated router over the evaluation suite.",
        "reuse_break_even": "Reuse break-even curves from plan time and warm execution time, aligned with preprocessing-versus-reuse analyses used in RoDe-style evaluations.",
    }
    write_json(os.path.join(results_dirs["json"], "figure_captions.json"), captions)
    with open(os.path.join(results_dirs["tables"], "figure_captions.tex"), "w") as fh:
        for key, caption in captions.items():
            fh.write(f"% {key}\n{caption}\n\n")


def maybe_generate_plots(results_dirs: Mapping[str, str]) -> None:
    from plot_paper_results import generate_all_plots

    generate_all_plots(results_dirs["root"])


def should_generate_plots(results_dirs: Mapping[str, str]) -> bool:
    root_name = os.path.basename(os.path.normpath(results_dirs["root"]))
    return not root_name.startswith("shard_gpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready SpMM evaluation suite")
    parser.add_argument("--section", default="all")
    parser.add_argument("--dataset_manifest", type=str, default=None)
    parser.add_argument("--synthetic_only", action="store_true")
    parser.add_argument("--real_only", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip_memory_heavy_paths", action="store_true")
    parser.add_argument("--memory_budget_gb", type=float, default=None)
    parser.add_argument("--memory_policy", choices=MEMORY_POLICIES, default="optimistic")
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_groups", type=str, default="")
    parser.add_argument("--include_names", type=str, default="")
    parser.add_argument("--exclude_names", type=str, default="")
    parser.add_argument("--include_sources", type=str, default="")
    args = parser.parse_args()

    if args.synthetic_only and args.real_only:
        raise ValueError("Choose at most one of --synthetic_only and --real_only")

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    include_synthetic = not args.real_only
    include_real = not args.synthetic_only
    results_dirs = ensure_results_tree(args.results_dir)
    acquire_results_lock(results_dirs["root"])
    cases = collect_cases(
        args.dataset_manifest,
        DEFAULT_NS,
        include_synthetic=include_synthetic,
        include_real=include_real,
    )
    include_groups = parse_csv_list(args.include_groups)
    include_names = parse_csv_list(args.include_names)
    exclude_names = parse_csv_list(args.exclude_names)
    include_sources = parse_csv_list(args.include_sources)
    cases = filter_cases(
        cases,
        include_groups=include_groups,
        include_names=include_names,
        exclude_names=exclude_names,
        include_sources=include_sources,
    )
    if not cases:
        raise RuntimeError("No evaluation cases available. Provide a dataset manifest or enable synthetic cases.")

    sections = set(parse_sections(args.section))
    if "all" in sections:
        sections = {
            "datasets",
            "main",
            "router",
            "external",
            "router_ablations",
            "feature_ablations",
            "kernel_ablations",
            "reuse",
            "profiling",
            "plots",
        }

    print_sep()
    print("PAPER EVALUATION SUITE")
    print_sep()
    print(f"cases: {len(cases)}")
    print(f"results_dir: {results_dirs['root']}")
    print(f"synthetic_enabled: {include_synthetic}")
    print(f"real_enabled: {include_real}")
    print(f"dataset_manifest: {args.dataset_manifest or 'none'}")
    print(f"include_groups: {','.join(include_groups) if include_groups else 'all'}")
    print(f"include_names: {','.join(include_names) if include_names else 'all'}")
    print(f"exclude_names: {','.join(exclude_names) if exclude_names else 'none'}")
    print(f"include_sources: {','.join(include_sources) if include_sources else 'all'}")
    print(f"memory_policy: {args.memory_policy}")
    print(
        "memory_budget_gb: "
        f"{args.memory_budget_gb if args.memory_budget_gb is not None else ('auto' if args.memory_policy == 'conservative' else 'unused')}"
    )
    print(f"skip_memory_heavy_paths: {args.skip_memory_heavy_paths}")
    print()

    write_dataset_inventory(cases, results_dirs)
    runner = ExperimentRunner(
        args.warmup,
        args.iters,
        portfolio="MAIN",
        seed=args.seed,
        memory_budget_gb=args.memory_budget_gb,
        memory_policy=args.memory_policy,
        skip_memory_heavy_paths=args.skip_memory_heavy_paths,
        continue_on_error=args.continue_on_error,
    )

    main_rows: List[Dict[str, object]] = []
    main_summary_rows: List[Dict[str, object]] = []
    absolute_throughput_rows: List[Dict[str, object]] = []
    external_rows: List[Dict[str, object]] = []
    external_summary_rows: List[Dict[str, object]] = []
    router_rows: List[Dict[str, object]] = []
    reuse_rows: List[Dict[str, object]] = []
    reuse_summary_rows: List[Dict[str, object]] = []

    if "main" in sections:
        print_sep()
        print("SECTION: main")
        print_sep()
        main_rows, main_summary_rows = run_main_kernel_comparison(runner, cases, results_dirs)
        build_speedup_table(main_summary_rows, results_dirs)
        build_n_scaling_summary(main_rows, results_dirs)
        absolute_throughput_rows = run_absolute_throughput_table(main_rows, results_dirs)
        print(f"main_kernel_points: {len(main_rows)}")
        print()

    if "router" in sections:
        print_sep()
        print("SECTION: router")
        print_sep()
        router_rows, router_summary_rows = run_router_vs_oracle(runner, cases, results_dirs)
        print(f"router_points: {len(router_rows)}")
        print(f"router_overall_avg_slowdown: {router_summary_rows[-1]['avg_slowdown']:.3f}x")
        print()

    if "router_ablations" in sections:
        print_sep()
        print("SECTION: router_ablations")
        print_sep()
        ablation_rows, ablation_summary_rows = run_router_ablation_study(runner, cases, results_dirs)
        print(f"router_ablation_points: {len(ablation_rows)}")
        print(f"best_ablation_avg: {min(row['avg_slowdown'] for row in ablation_summary_rows):.3f}x")
        print()

    if "feature_ablations" in sections:
        print_sep()
        print("SECTION: feature_ablations")
        print_sep()
        feature_rows, feature_summary_rows = run_feature_ablation_study(runner, cases, results_dirs)
        print(f"feature_ablation_points: {len(feature_rows)}")
        print(f"feature_ablation_variants: {len(feature_summary_rows)}")
        print()

    if "kernel_ablations" in sections:
        print_sep()
        print("SECTION: kernel_ablations")
        print_sep()
        kernel_rows, kernel_summary_rows = run_kernel_ablation_study(runner, cases, results_dirs)
        print(f"kernel_ablation_points: {len(kernel_rows)}")
        print()

    if "reuse" in sections:
        print_sep()
        print("SECTION: reuse")
        print_sep()
        reuse_rows, reuse_summary_rows = run_reuse_analysis(runner, cases, results_dirs)
        print(f"reuse_points: {len(reuse_rows)}")
        print()

    if "external" in sections:
        print_sep()
        print("SECTION: external")
        print_sep()
        if not main_rows:
            main_rows, main_summary_rows = run_main_kernel_comparison(runner, cases, results_dirs)
        external_rows, external_summary_rows = run_external_baselines(runner, cases, results_dirs)
        if not reuse_rows:
            reuse_rows, reuse_summary_rows = run_reuse_analysis(runner, cases, results_dirs)
        build_external_alignment_artifacts(main_rows, reuse_rows, results_dirs)
        print(f"external_baseline_points: {len(external_rows)}")
        for ext_path in EXTERNAL_BASELINE_PATHS:
            val = external_summary_rows[-1].get(ext_path, None)
            print(f"external_overall_{ext_path.lower()}: {val:.3f}x" if val is not None else f"external_overall_{ext_path.lower()}: N/A")
        print("external baseline and alignment artifacts written")
        print()

    if "profiling" in sections:
        print_sep()
        print("SECTION: profiling")
        print_sep()
        if not main_rows:
            main_rows, main_summary_rows = run_main_kernel_comparison(runner, cases, results_dirs)
        if not absolute_throughput_rows:
            absolute_throughput_rows = run_absolute_throughput_table(main_rows, results_dirs)
        if not router_rows:
            router_rows, _ = run_router_vs_oracle(runner, cases, results_dirs)
        if not reuse_summary_rows:
            reuse_rows, reuse_summary_rows = run_reuse_analysis(runner, cases, results_dirs)
        build_profiling_artifacts(main_rows, router_rows, reuse_summary_rows, results_dirs)
        build_figure_captions(results_dirs)
        print("profiling summaries written")
        print()

    if "plots" in sections:
        print_sep()
        print("SECTION: plots")
        print_sep()
        if should_generate_plots(results_dirs):
            try:
                maybe_generate_plots(results_dirs)
                print(f"plots written to {results_dirs['plots']}")
            except Exception as exc:
                print(f"[plot warning] plot generation skipped: {exc}")
        else:
            print("plot generation skipped for shard output; merged run will write final plots")
        print()


if __name__ == "__main__":
    main()
