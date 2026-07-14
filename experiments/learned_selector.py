#!/usr/bin/env python3
"""Learned-selector baseline (reviewer 2.1): decision tree / random forest
over the router's four deployed features, evaluated leakage-free.

Method:
- Features per (graph, N) config: M, d_bar, CV_d, N (identical information
  budget to the deployed rule tree).
- Label: warm oracle winner among the final deployed candidate set
  (8 execution paths + cuSPARSE floor) from final_fair_v3.csv.
- Validation: leave-one-GRAPH-out (all N-configs of a graph held out
  together) - plain k-fold over configs would leak graph identity.
- Scoring: deployed metric = geomean speedup vs cuSPARSE of the PREDICTED
  kernel's measured time (cuSPARSE fallback when prediction is infeasible),
  plus hit rate. Compared against the rule router and the oracle.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ra_router_eval import KERNELS, route_with_rules  # noqa: E402

RESULTS = REPO / "fgcs_results/revision/tf32/final_fair_v3.csv"


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def load():
    pairs, meta = defaultdict(dict), {}
    for r in csv.DictReader(open(RESULTS)):
        if r["kernel"] not in KERNELS or r.get("correct") != "True":
            continue
        k = (r["dataset"], int(r["N"]))
        pairs[k][r["kernel"]] = float(r["ms_warm"])
        meta[k] = dict(cus=float(r["ms_cusparse_warm"]), M=int(r["M"]),
                       nnz=int(r["nnz"]), cv=float(r["cv_d"]))
    return pairs, meta


def main() -> None:
    pairs, meta = load()
    keys = sorted(pairs)
    X, y, graphs = [], [], []
    for k in keys:
        m = meta[k]
        X.append([m["M"], m["nnz"] / max(1, m["M"]), m["cv"], k[1]])
        kt = pairs[k]
        best = min(kt, key=kt.get)
        y.append(best if kt[best] < m["cus"] else "CUSPARSE")
        graphs.append(k[0])
    X = np.log1p(np.array(X, dtype=np.float64))  # log-scale features
    y = np.array(y)
    graphs = np.array(graphs)

    def speed(k, pick):
        m = meta[k]
        if pick == "CUSPARSE" or pick not in pairs[k]:
            return 1.0
        return m["cus"] / pairs[k][pick]

    oracle = gm([max(speed(k, min(pairs[k], key=pairs[k].get)), 1.0) for k in keys])
    rule_sp, rule_hits = [], 0
    for i, k in enumerate(keys):
        m = meta[k]
        pick = route_with_rules(m["nnz"] / max(1, m["M"]), m["cv"], m["M"], k[1], m["nnz"])
        rule_sp.append(speed(k, pick))
        rule_hits += pick == y[i]
    print(f"oracle geomean:        {oracle:.4f}")
    print(f"rule router:           {gm(rule_sp):.4f}  hit {rule_hits}/{len(keys)}")

    uniq = sorted(set(graphs))
    out_rows = []
    for label, make in [
        ("dtree_d3", lambda: DecisionTreeClassifier(max_depth=3, random_state=0)),
        ("dtree_d4", lambda: DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("dtree_d6", lambda: DecisionTreeClassifier(max_depth=6, random_state=0)),
        ("dtree_unlimited", lambda: DecisionTreeClassifier(random_state=0)),
        ("rforest_200", lambda: RandomForestClassifier(n_estimators=200, random_state=0)),
    ]:
        preds = {}
        for g in uniq:  # leave-one-graph-out
            tr = graphs != g
            clf = make().fit(X[tr], y[tr])
            for i in np.where(~tr)[0]:
                preds[keys[i]] = clf.predict(X[i:i+1])[0]
        sp = [speed(k, preds[k]) for k in keys]
        hits = sum(preds[keys[i]] == y[i] for i in range(len(keys)))
        # train-fit (leaky, upper reference)
        clf_full = make().fit(X, y)
        tr_sp = [speed(keys[i], clf_full.predict(X[i:i+1])[0]) for i in range(len(keys))]
        print(f"{label:16s} LOGO geomean {gm(sp):.4f}  hit {hits}/{len(keys)}"
              f"   (train-fit {gm(tr_sp):.4f})")
        out_rows.append({"model": label, "logo_geomean": round(gm(sp), 4),
                         "logo_hits": hits, "trainfit_geomean": round(gm(tr_sp), 4)})

    # feature ablation on the best interpretable model (LOGO, drop one feature)
    names = ["M", "d_bar", "CV_d", "N"]
    print("\nfeature ablation (dtree_d6, LOGO):")
    for j, nm in enumerate(names):
        cols = [c for c in range(4) if c != j]
        preds = {}
        for g in uniq:
            tr = graphs != g
            clf = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X[tr][:, cols], y[tr])
            for i in np.where(~tr)[0]:
                preds[keys[i]] = clf.predict(X[i:i+1, cols])[0]
        sp = [speed(k, preds[k]) for k in keys]
        print(f"  drop {nm:6s}: {gm(sp):.4f}")
        out_rows.append({"model": f"dtree_d6_drop_{nm}", "logo_geomean": round(gm(sp), 4),
                         "logo_hits": sum(preds[keys[i]] == y[i] for i in range(len(keys))),
                         "trainfit_geomean": ""})

    out = REPO / "fgcs_results/revision/tf32/learned_selector.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "logo_geomean", "logo_hits", "trainfit_geomean"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
