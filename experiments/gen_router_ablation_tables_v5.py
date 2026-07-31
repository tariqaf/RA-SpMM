#!/usr/bin/env python3
"""Regenerate the router ablation / feature tables from the committed
final_fair_v3.csv (single source of truth). Produces three CSVs:

  rule_ablation_v5.csv      - per-rule leave-one-out over the 7 router rules
                              (metric identical to ra_router_eval.main:
                               geomean router_speedup vs cuSPARSE, oracle-hit count)
  feature_loo_v5.csv        - leave-one-graph-out drop-one-feature study
                              (identical protocol to experiments/learned_selector.py:
                               dtree depth-6, log1p features, speedup geomean + hits)
  feature_gain_maxdeg.csv   - LOGO Router/Oracle ratio, base-4 features vs
                              base-4 + max_row_degree, for DT depth-4 and RF-200

Numbers reproduce the values printed in the paper's ablation tables exactly.
"""
from __future__ import annotations
import csv, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ra_router_eval import KERNELS, route_with_rules  # noqa: E402

RESULTS = REPO / "fgcs_results/revision/tf32/final_fair_v3.csv"
OUTDIR = REPO / "fgcs_results/revision/tf32"


def gm(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def load():
    """Return {(dataset,N): {kernel: ms}}, meta{(dataset,N): {...}} using the
    same filters as ra_router_eval: KERNELS roster, correct==True, warm."""
    pairs, meta = defaultdict(dict), {}
    for r in csv.DictReader(open(RESULTS)):
        if r["kernel"] not in KERNELS or r.get("correct", "False").lower() not in ("true", "1"):
            continue
        if not r.get("ms_warm") or not r.get("ms_cusparse_warm"):
            continue
        k = (r["dataset"], int(r["N"]))
        pairs[k][r["kernel"]] = float(r["ms_warm"])
        meta[k] = dict(cus=float(r["ms_cusparse_warm"]), M=int(r["M"]),
                       nnz=int(r["nnz"]), cv=float(r["cv_d"]),
                       maxdeg=int(r["max_nnz_per_row"]), graph=r["dataset"])
    # keep only complete pairs (all 8 kernels), exactly as ra_router_eval
    complete = {k: v for k, v in pairs.items() if set(v) == set(KERNELS)}
    return complete, meta


PAIRS, META = load()
KEYS = sorted(PAIRS)
assert len(KEYS) == 192, f"expected 192 complete pairs, got {len(KEYS)}"


# ---------------------------------------------------------------- rule ablation
def eval_rules(disabled):
    """Two hit definitions:
      hits_incl : oracle floor = min(best custom kernel, cuSPARSE); a router
                  pick of the cuSPARSE guardrail counts as a hit when cuSPARSE
                  ties/beats every custom kernel. This is the router-inclusive
                  definition the ablation table uses (guardrail is part of the
                  router), and it reproduces the per-rule vector.
      hits_kern : best custom kernel only (the paper's kernel-only headline;
                  the full-router value is 172 under this definition).
    The two differ on exactly one config (amazon-photo, N=64)."""
    speeds, hits_incl, hits_kern = [], 0, 0
    for k in KEYS:
        m = META[k]
        d = m["nnz"] / max(1, m["M"])
        pick = route_with_rules(d, m["cv"], m["M"], k[1], m["nnz"], disabled_rules=disabled)
        best = min(PAIRS[k], key=PAIRS[k].get)
        best_ms = PAIRS[k][best]
        router_ms = m["cus"] if pick == "CUSPARSE" else PAIRS[k][pick]
        speeds.append(m["cus"] / router_ms)
        hits_kern += (pick == best)
        floor = min(best_ms, m["cus"])
        hits_incl += (router_ms <= floor * (1 + 1e-9))
    return gm(speeds), hits_incl, hits_kern


def rule_ablation():
    rows = []
    full_gm, full_incl, full_kern = eval_rules(())
    rows.append({"disabled_rule": "none", "router_geomean": round(full_gm, 4),
                 "oracle_hits": full_incl, "oracle_hits_kernelonly": full_kern,
                 "delta_pct": 0.0})
    for r in range(1, 8):
        g, hi, hk = eval_rules((r,))
        rows.append({"disabled_rule": f"R{r}", "router_geomean": round(g, 4),
                     "oracle_hits": hi, "oracle_hits_kernelonly": hk,
                     "delta_pct": round((g / full_gm - 1) * 100, 2)})
    out = OUTDIR / "rule_ablation_v5.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disabled_rule", "router_geomean",
                                          "oracle_hits", "oracle_hits_kernelonly", "delta_pct"])
        w.writeheader(); w.writerows(rows)
    return rows, out


# ------------------------------------------------------------- feature LOO (LOGO)
def feature_tables():
    """Exact learned_selector.py protocol: log1p([M, d_bar, CV_d, N]) features,
    warm-oracle winner label (cuSPARSE floor when no custom kernel beats it),
    leave-one-graph-out CV, speedup geomean of the predicted kernel + hit count."""
    X, y, graphs, keys = [], [], [], []
    for k in KEYS:
        m = META[k]
        X.append([m["M"], m["nnz"] / max(1, m["M"]), m["cv"], k[1], m["maxdeg"]])
        kt = PAIRS[k]
        best = min(kt, key=kt.get)
        y.append(best if kt[best] < m["cus"] else "CUSPARSE")
        graphs.append(m["graph"]); keys.append(k)
    Xfull = np.log1p(np.array(X, dtype=np.float64))
    y = np.array(y); graphs = np.array(graphs)
    uniq = sorted(set(graphs))

    def speed(k, pick):
        m = META[k]
        if pick == "CUSPARSE" or pick not in PAIRS[k]:
            return 1.0
        return m["cus"] / PAIRS[k][pick]

    def oracle_ratio(k, pick):
        m = META[k]
        best_ms = min(PAIRS[k].values())
        pick_ms = m["cus"] if (pick == "CUSPARSE" or pick not in PAIRS[k]) else PAIRS[k][pick]
        return best_ms / pick_ms

    def logo_predict(cols, make):
        preds = {}
        for gph in uniq:
            tr = graphs != gph
            clf = make().fit(Xfull[tr][:, cols], y[tr])
            for i in np.where(~tr)[0]:
                preds[keys[i]] = clf.predict(Xfull[i:i + 1, cols])[0]
        return preds

    # --- feature LOO: dtree depth-6, drop one of the 4 base features
    base_cols = [0, 1, 2, 3]  # M, d_bar, CV_d, N
    names = {0: "M", 1: "d_bar", 2: "CV_d", 3: "N"}
    mk6 = lambda: DecisionTreeClassifier(max_depth=6, random_state=0)
    loo_rows = []
    preds = logo_predict(base_cols, mk6)
    loo_rows.append({"feature_set": "all_4", "router_geomean": round(gm([speed(k, preds[k]) for k in KEYS]), 4),
                     "oracle_hits": sum(preds[k] == y[i] for i, k in enumerate(KEYS))})
    for drop in base_cols:
        cols = [c for c in base_cols if c != drop]
        preds = logo_predict(cols, mk6)
        loo_rows.append({"feature_set": f"drop_{names[drop]}",
                         "router_geomean": round(gm([speed(k, preds[k]) for k in KEYS]), 4),
                         "oracle_hits": sum(preds[k] == y[i] for i, k in enumerate(KEYS))})
    out_loo = OUTDIR / "feature_loo_v5.csv"
    with out_loo.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_set", "router_geomean", "oracle_hits"])
        w.writeheader(); w.writerows(loo_rows)

    # --- feature gain: leave-one-graph-out Router/Oracle, base-4 vs +max_row_degree
    # Same rigorous out-of-sample protocol as the feature-LOO table above.
    maxdeg_cols = base_cols + [4]
    gain_rows = []
    for label, make in [("dtree_depth4", lambda: DecisionTreeClassifier(max_depth=4, random_state=0)),
                        ("rforest_200", lambda: RandomForestClassifier(n_estimators=200, random_state=0))]:
        pb = logo_predict(base_cols, make)
        pm = logo_predict(maxdeg_cols, make)
        ro_base = gm([oracle_ratio(k, pb[k]) for k in KEYS])
        ro_max = gm([oracle_ratio(k, pm[k]) for k in KEYS])
        gain_rows.append({"model": label, "features": "base4",
                          "router_oracle_ratio": round(ro_base, 3)})
        gain_rows.append({"model": label, "features": "base4+max_row_degree",
                          "router_oracle_ratio": round(ro_max, 3)})
    out_gain = OUTDIR / "feature_gain_maxdeg.csv"
    with out_gain.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "features", "router_oracle_ratio"])
        w.writeheader(); w.writerows(gain_rows)
    return loo_rows, out_loo, gain_rows, out_gain


if __name__ == "__main__":
    ra, ra_path = rule_ablation()
    print("=== rule_ablation_v5 ===")
    for r in ra:
        print(f"  {r['disabled_rule']:5s} {r['router_geomean']:.4f}x  {r['oracle_hits']:>3d} hits  ({r['delta_pct']:+.2f}%)")
    loo, loo_path, gain, gain_path = feature_tables()
    print("=== feature_loo_v5 ===")
    for r in loo:
        print(f"  {r['feature_set']:12s} {r['router_geomean']:.4f}x  {r['oracle_hits']:>3d} hits")
    print("=== feature_gain_maxdeg ===")
    for r in gain:
        print(f"  {r['model']:14s} {r['features']:22s} R/O {r['router_oracle_ratio']:.3f}")
    print(f"\nwrote {ra_path}\n      {loo_path}\n      {gain_path}")
