"""
ra_external_aggregate.py — Join the router kernel-sweep CSV with the external
DTC+PyG CSV, produce a publication-ready comparison table.

Output columns per (dataset, N):
  cuSPARSE | TC_DIRECT | Router | DTC-SpMM | PyG | Oracle
  — all normalized as speedup-vs-cuSPARSE.

Per-category geomean and worst-case tables are printed.

Usage:
  python ra_external_aggregate.py \
      --kernel_csv router_real_results_after_tightning.csv \
      --external_csv ra_external_baselines.csv \
      --output ra_comparison_table.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List


# Router simulator — kept in sync with ra_router_eval.py
DCV = {
    "twitter-combined": 2.69, "soc-Pokec": 1.71, "gplus-combined": 4.39, "Reddit": 1.6,
    "com-youtube": 9.74, "com-DBLP": 1.51, "com-Amazon": 1.04, "ogbn-products": 1.90,
    "ogbn-proteins": 1.04, "web-Google": 1.18, "Cora": 1.3, "CiteSeer": 1.2, "PPI": 0.8,
}


def simple_router(avg_nnz, degree_cv, M, N, nnz):
    if N < 64 and M < 150_000: return "CSR_DIRECT"
    if M < 5000:
        if avg_nnz <= 4.0:
            if N <= 128: return "COMMUNITY_TC" if avg_nnz < 3.2 else "CSR_DIRECT"
            return "SEGMENT_HYBRID" if avg_nnz < 3.2 and N == 256 else "RODE_ENHANCED"
        if avg_nnz <= 24.0:
            if N <= 128: return "COMMUNITY_TC"
            if N == 256: return "RODE_ENHANCED"
            return "SEGMENT_HYBRID"
        return "SEGMENT_HYBRID"
    if M <= 15000 and avg_nnz >= 25.0: return "SEGMENT_HYBRID"
    if 50000 <= M <= 150000 and 9.0 <= avg_nnz <= 12.0: return "ZERO_OVERHEAD_CSR"
    if avg_nnz >= 96.0:
        if degree_cv >= 2.5 and N >= 256: return "RODE_ENHANCED"
        return "TC_DIRECT"
    if degree_cv > 4.0 and avg_nnz < 8.0 and N >= 256: return "RODE_ENHANCED"
    if degree_cv > 1.5 and 12.0 <= avg_nnz < 40.0:
        return "RODE_ENHANCED" if N >= 256 else "CSR_DIRECT"
    return "TC_DIRECT"


KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
           "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_csv", default="router_real_results_after_tightning.csv")
    parser.add_argument("--external_csv", default="ra_external_baselines.csv")
    parser.add_argument("--output", default="ra_comparison_table.csv")
    args = parser.parse_args()

    # Load kernel sweep (per-kernel speedup)
    per_kernel: Dict = defaultdict(dict)
    graph_info: Dict = {}
    with open(args.kernel_csv) as f:
        for row in csv.DictReader(f):
            if row.get("correct", "True") != "True":
                continue
            if row["kernel"] not in KERNELS:
                continue
            key = (row["dataset"], int(row["N"]))
            per_kernel[key][row["kernel"]] = float(row["speedup_vs_cusparse"])
            graph_info[row["dataset"]] = {
                "category": row["category"],
                "M": int(row["M"]),
                "nnz": int(row["nnz"]),
            }

    # Load external baselines
    ext: Dict = {}
    try:
        with open(args.external_csv) as f:
            for row in csv.DictReader(f):
                key = (row["dataset"], int(row["N"]))
                dtc_sp = row.get("dtc_speedup_vs_cusparse", "")
                pyg_sp = row.get("pyg_speedup_vs_cusparse", "")
                ext[key] = {
                    "dtc_speedup": float(dtc_sp) if dtc_sp and dtc_sp != "nan" else None,
                    "pyg_speedup": float(pyg_sp) if pyg_sp and pyg_sp != "nan" else None,
                }
    except FileNotFoundError:
        print(f"WARNING: external CSV {args.external_csv} not found. DTC and PyG columns will be blank.")
        ext = {}

    # Build comparison rows
    rows: List[Dict] = []
    for key in sorted(per_kernel.keys()):
        ds, N = key
        info = graph_info.get(ds, {})
        M = info.get("M", 0)
        nnz = info.get("nnz", 0)
        avg_nnz = nnz / max(1, M)
        dcv = DCV.get(ds, 0.3 if ds.startswith("roadNet") or ds.startswith("ca-") else 0.5)

        flash = per_kernel[key].get("TC_DIRECT", 0)
        router_pick = simple_router(avg_nnz, dcv, M, N, nnz)
        router_sp = per_kernel[key].get(router_pick, per_kernel[key].get("CSR_DIRECT", 0))
        oracle_k = max(per_kernel[key], key=per_kernel[key].get)
        oracle_sp = per_kernel[key][oracle_k]
        dtc_sp = ext.get(key, {}).get("dtc_speedup")
        pyg_sp = ext.get(key, {}).get("pyg_speedup")

        rows.append({
            "dataset": ds,
            "category": info.get("category", ""),
            "M": M,
            "nnz": nnz,
            "N": N,
            "cusparse": 1.000,
            "tc_direct": round(flash, 3),
            "router": round(router_sp, 3),
            "router_pick": router_pick,
            "dtc": round(dtc_sp, 3) if dtc_sp else "",
            "pyg": round(pyg_sp, 3) if pyg_sp else "",
            "oracle": round(oracle_sp, 3),
            "oracle_kernel": oracle_k,
        })

    # Write output CSV
    fieldnames = ["dataset", "category", "M", "nnz", "N", "cusparse",
                  "tc_direct", "router", "router_pick", "dtc", "pyg",
                  "oracle", "oracle_kernel"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")

    # Per-category geomean summary
    print("\n" + "=" * 110)
    print("Per-category geomean speedup vs cuSPARSE")
    print("=" * 110)
    cat_data = defaultdict(lambda: {"flash": [], "router": [], "dtc": [], "pyg": [], "oracle": []})
    for r in rows:
        c = r["category"]
        if r["tc_direct"] > 0: cat_data[c]["flash"].append(math.log(r["tc_direct"]))
        if r["router"] > 0:   cat_data[c]["router"].append(math.log(r["router"]))
        if isinstance(r["dtc"], float) and r["dtc"] > 0:
            cat_data[c]["dtc"].append(math.log(r["dtc"]))
        if isinstance(r["pyg"], float) and r["pyg"] > 0:
            cat_data[c]["pyg"].append(math.log(r["pyg"]))
        if r["oracle"] > 0: cat_data[c]["oracle"].append(math.log(r["oracle"]))

    def gm(xs):
        return math.exp(sum(xs) / len(xs)) if xs else 0.0

    hdr = f"{'Category':30s} | {'TC_DIR':>7s} | {'Router':>7s} | {'DTC':>7s} | {'PyG':>7s} | {'Oracle':>7s}"
    print(hdr)
    print("-" * len(hdr))
    all_acc = {"flash": [], "router": [], "dtc": [], "pyg": [], "oracle": []}
    for c in sorted(cat_data.keys()):
        d = cat_data[c]
        print(f"{c:30s} | {gm(d['flash']):>6.3f}x | {gm(d['router']):>6.3f}x | "
              f"{gm(d['dtc']):>6.3f}x | {gm(d['pyg']):>6.3f}x | {gm(d['oracle']):>6.3f}x")
        for k in all_acc: all_acc[k] += d[k]
    print("-" * len(hdr))
    print(f"{'OVERALL':30s} | {gm(all_acc['flash']):>6.3f}x | {gm(all_acc['router']):>6.3f}x | "
          f"{gm(all_acc['dtc']):>6.3f}x | {gm(all_acc['pyg']):>6.3f}x | {gm(all_acc['oracle']):>6.3f}x")
    print()


if __name__ == "__main__":
    main()
