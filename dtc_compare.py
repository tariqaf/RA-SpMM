import argparse
import os
from typing import List

from paper_eval_utils import ExperimentRunner, collect_cases, ensure_results_tree, write_csv_rows


TARGET_GRAPHS = {
    "roadNet-CA",
    "roadNet-TX",
    "roadNet-PA",
    "web-Google",
    "web-Stanford",
    "ogbn-proteins",
    "clustered_window",
}


def safe_ratio(num, den):
    if num in (None, "") or den in (None, ""):
        return None
    num_f = float(num)
    den_f = float(den)
    if abs(den_f) < 1e-12:
        return None
    return num_f / den_f


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused RA-oracle and TC_REORDERED vs DTC-SpMM comparisons.")
    parser.add_argument("--dataset_manifest", default="paper_datasets.json")
    parser.add_argument("--results_dir", default="results_full_after_seven_regimes")
    parser.add_argument("--output_csv", default="ra_best_vs_dtc.csv")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--memory_policy", default="optimistic")
    args = parser.parse_args()

    results_dirs = ensure_results_tree(args.results_dir)
    cases = [
        case for case in collect_cases(args.dataset_manifest, (64, 128, 256, 512), include_synthetic=True, include_real=True)
        if case.name in TARGET_GRAPHS
    ]
    runner = ExperimentRunner(
        warmup=args.warmup,
        iters=args.iters,
        portfolio="MAIN",
        seed=args.seed,
        memory_policy=args.memory_policy,
    )

    rows: List[dict] = []
    for case in cases:
        for N in case.Ns:
            print(f"[dtc] graph={case.name} size={case.size_tag} N={N}", flush=True)
            warm = runner.warm_oracle(case, N)
            tc = warm["path_results"]["TC_REORDERED"]
            dtc = runner.warm_external_baselines(case, N)["DTC_SPMM"]
            oracle_path = warm.get("oracle_path", "")
            oracle_result = warm["path_results"].get(oracle_path, {})
            tc_total = float(tc["total_ms"]) if tc.get("status") == "OK" and tc.get("total_ms") not in (None, "") else None
            oracle_total = float(oracle_result["total_ms"]) if oracle_result.get("status") == "OK" and oracle_result.get("total_ms") not in (None, "") else None
            dtc_total = float(dtc["total_ms"]) if dtc.get("status") == "OK" and dtc.get("total_ms") not in (None, "") else None
            rows.append({
                "graph": case.name,
                "source": case.source,
                "category": case.category,
                "graph_group": case.group,
                "size_tag": case.size_tag,
                "M": case.M,
                "K": case.K,
                "N": N,
                "oracle_path": oracle_path,
                "oracle_status": oracle_result.get("status", ""),
                "oracle_ms": oracle_total,
                "oracle_over_dtc_speedup": safe_ratio(dtc_total, oracle_total),
                "tc_reordered_status": tc.get("status", ""),
                "tc_reordered_ms": tc_total,
                "tc_reordered_over_dtc_speedup": safe_ratio(dtc_total, tc_total),
                "dtc_status": dtc.get("status", ""),
                "dtc_ms": dtc_total,
                "dtc_variant": dtc.get("dtc_variant", ""),
                "dtc_balance": dtc.get("dtc_balance", False),
                "oracle_time_ms": warm.get("oracle_time_ms", ""),
            })
            out_path = os.path.join(results_dirs["csv"], args.output_csv)
            write_csv_rows(out_path, rows)

    out_path = os.path.join(results_dirs["csv"], args.output_csv)
    print(f"dtc_rows: {len(rows)}")
    print(f"dtc_csv: {out_path}")


if __name__ == "__main__":
    main()
