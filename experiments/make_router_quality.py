"""
Produces router_quality_v2.csv (router vs oracle on the RTX 3090) from the baseline
sweep CSV. Reuses the paper router (ra_router_eval.simple_router) and the accurate
per-graph cv_d already measured in feature_extraction_gpu.csv.

Also copies the per-(graph,N,kernel) sweep CSV to the 4090-facing path.

Outputs (the two CSVs the RTX 4090 run needs for its cross-arch parity check):
  fgcs_results/spmm/all_graphs_results.csv
  fgcs_results/summary/router_quality_v2.csv
"""
from __future__ import annotations
import csv, math, shutil, sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from ra_router_eval import simple_router  # noqa

SWEEP = R / "fgcs_results/revision/baseline_3090/all_graphs_results.csv"
FEAT = R / "fgcs_results/revision/featbreak/feature_extraction_gpu.csv"
SPMM_OUT = R / "fgcs_results/spmm/all_graphs_results.csv"
RQ_OUT = R / "fgcs_results/summary/router_quality_v2.csv"
KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR", "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"]


def main():
    # cv_d per dataset (accurate, measured on-device)
    cvd = {}
    if FEAT.exists():
        for r in csv.DictReader(open(FEAT)):
            cvd[r["dataset"]] = float(r["cv_d"])

    pairs = defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(SWEEP)):
        if r["kernel"] not in KERNELS:
            continue
        if r.get("correct", "True") not in ("True", "true", "1"):
            continue
        try:
            sp = float(r["speedup_vs_cusparse"])
        except ValueError:
            continue
        key = (r["dataset"], int(r["N"]))
        pairs[key][r["kernel"]] = sp
        meta[key] = {"category": r.get("category", "?"), "M": int(r["M"]), "nnz": int(r["nnz"])}

    rows = []
    hits = 0
    oracle_logs, router_logs = [], []
    for (ds, N), speeds in sorted(pairs.items()):
        m = meta[(ds, N)]
        M, nnz = m["M"], m["nnz"]
        d_bar = nnz / max(1, M)
        cv = cvd.get(ds, 0.5)
        oracle_k = max(speeds, key=speeds.get)
        oracle_sp = speeds[oracle_k]
        router_k = simple_router(d_bar, cv, M, N, nnz)
        router_sp = speeds.get(router_k, 0.0)
        if router_sp <= 0:
            router_k, router_sp = "CSR_DIRECT", speeds.get("CSR_DIRECT", 1.0)
        hit = router_k == oracle_k
        hits += hit
        if oracle_sp > 0:
            oracle_logs.append(math.log(oracle_sp))
        if router_sp > 0:
            router_logs.append(math.log(router_sp))
        rows.append({
            "dataset": ds, "category": m["category"], "M": M, "nnz": nnz, "N": N,
            "d_bar": round(d_bar, 3), "cv_d": round(cv, 4),
            "oracle_kernel": oracle_k, "oracle_speedup": round(oracle_sp, 3),
            "router_kernel": router_k, "router_speedup": round(router_sp, 3),
            "hit": hit, "router_over_oracle": round(router_sp / oracle_sp, 3) if oracle_sp > 0 else 0,
        })

    RQ_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RQ_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    SPMM_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SWEEP, SPMM_OUT)

    n = len(rows)
    ogm = math.exp(sum(oracle_logs) / len(oracle_logs)) if oracle_logs else 0
    rgm = math.exp(sum(router_logs) / len(router_logs)) if router_logs else 0
    print(f"router_quality_v2.csv: {n} (graph,N) pairs")
    print(f"  router hit rate: {hits}/{n} ({100*hits/max(1,n):.1f}%)")
    print(f"  oracle geomean {ogm:.3f}x  |  router geomean {rgm:.3f}x  |  overhead {ogm/rgm:.3f}x" if rgm else "")
    print(f"Wrote {RQ_OUT}")
    print(f"Wrote {SPMM_OUT}")


if __name__ == "__main__":
    main()
