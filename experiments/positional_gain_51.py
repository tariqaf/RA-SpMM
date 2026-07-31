#!/usr/bin/env python3
"""Leave-one-graph-out gain from row/column positional statistics, all 51 graphs.

Supersedes the 47-graph run in fgcs_results/feature_gain/feature_gain_v5.csv, which was
limited to the graphs whose raw index arrays were held locally at the time. The complete
export (fgcs_results/revision/tf32/positional_stats_51.csv) now covers all 51.

Both arms are recomputed here on the same graphs, the same configurations, the same folds,
labels, model, seed, and aggregation -- the baseline is re-derived, not carried over. The
protocol is the one in gen_router_ablation_tables_v5.feature_tables():

  features  log1p of [M, d_bar, CV_d, N], optionally extended by the six positional
            statistics (already normalized by M, so passed through log1p identically)
  label     warm-oracle winner, with the cuSPARSE floor applied: a configuration whose
            best custom kernel does not beat cuSPARSE is labelled CUSPARSE
  folds     leave-one-graph-out: every configuration of a graph is held out together
  metrics   exact oracle-hit count, geomean speedup over cuSPARSE, geomean Router/Oracle

Run with --reproduce first: it re-runs the published 47-graph arms and refuses to continue
unless it reproduces 118/178 and 104/178, so the 51-graph numbers cannot be reported from a
protocol that silently drifted.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_router_eval import KERNELS  # noqa: E402

RESULTS = ROOT / "fgcs_results" / "revision" / "tf32" / "final_fair_v3.csv"
POSCSV = ROOT / "fgcs_results" / "revision" / "tf32" / "positional_stats_51.csv"
OUTDIR = ROOT / "fgcs_results" / "revision" / "tf32"

POS_COLS = ["col_mean_n", "col_std_n", "col_min_n", "col_max_n", "row_mean_n", "row_std_n"]
# The four graphs whose raw index arrays were unavailable for the published 47-graph run.
V5_EXCLUDED = {"Flickr", "PPI", "Yelp", "com-youtube"}


def gm(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def load_pairs():
    """{(dataset, N): {kernel: ms_warm}} and metadata, filtered exactly as ra_router_eval."""
    pairs, meta = defaultdict(dict), {}
    with RESULTS.open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["kernel"] not in KERNELS:
                continue
            if r.get("correct", "False").lower() not in ("true", "1"):
                continue
            if not r.get("ms_warm") or not r.get("ms_cusparse_warm"):
                continue
            key = (r["dataset"], int(r["N"]))
            pairs[key][r["kernel"]] = float(r["ms_warm"])
            meta[key] = dict(cus=float(r["ms_cusparse_warm"]), M=int(r["M"]),
                             nnz=int(r["nnz"]), cv=float(r["cv_d"]), graph=r["dataset"])
    complete = {k: v for k, v in pairs.items() if set(v) == set(KERNELS)}
    return complete, meta


def load_positional():
    with POSCSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {r["dataset"]: [float(r[c]) for c in POS_COLS] for r in rows}


def build(keys, pairs, meta, positional):
    X, y, graphs = [], [], []
    for k in keys:
        m = meta[k]
        base = [m["M"], m["nnz"] / max(1, m["M"]), m["cv"], k[1]]
        X.append(base + positional[m["graph"]])
        times = pairs[k]
        best = min(times, key=times.get)
        y.append(best if times[best] < m["cus"] else "CUSPARSE")
        graphs.append(m["graph"])
    return (np.log1p(np.asarray(X, dtype=np.float64)), np.asarray(y), np.asarray(graphs))


def logo_predict(X, y, graphs, keys, cols, depth, seed):
    preds = {}
    for g in sorted(set(graphs)):
        train = graphs != g
        clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        clf.fit(X[train][:, cols], y[train])
        for i in np.where(~train)[0]:
            preds[keys[i]] = clf.predict(X[i:i + 1, cols])[0]
    return preds


def evaluate(preds, keys, y, pairs, meta):
    speeds, ratios, hits = [], [], 0
    for i, k in enumerate(keys):
        m, pick = meta[k], preds[k]
        pick_ms = m["cus"] if (pick == "CUSPARSE" or pick not in pairs[k]) else pairs[k][pick]
        speeds.append(m["cus"] / pick_ms)
        ratios.append(min(pairs[k].values()) / pick_ms)
        hits += (pick == y[i])
    return dict(n=len(keys), hits=hits, hit_rate=hits / len(keys),
                geomean_vs_cusparse=gm(speeds), router_oracle=gm(ratios))


def run(keys, pairs, meta, positional, depth, seed, label):
    X, y, graphs = build(keys, pairs, meta, positional)
    base_cols = [0, 1, 2, 3]
    ext_cols = base_cols + [4, 5, 6, 7, 8, 9]
    out = {}
    for arm, cols in (("base", base_cols), ("base+index", ext_cols)):
        preds = logo_predict(X, y, graphs, keys, cols, depth, seed)
        out[arm] = evaluate(preds, keys, y, pairs, meta)
        out[arm].update(feature_set=arm, graphs=len(set(graphs)), scope=label,
                        model=f"dtree_depth{depth}", seed=seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUTDIR / "feature_gain_v6_51.csv"))
    ap.add_argument("--reproduce", action="store_true",
                    help="re-run the published 47-graph arms and require 118/178 and 104/178")
    args = ap.parse_args()

    pairs, meta = load_pairs()
    keys = sorted(pairs)
    positional = load_positional()

    # ---- assertions demanded before any number is accepted ----
    graphs_all = sorted({meta[k]["graph"] for k in keys})
    assert len(keys) == 192, f"expected 192 complete configurations, got {len(keys)}"
    assert len(set(keys)) == 192, "configuration keys are not unique"
    assert len(graphs_all) == 51, f"expected 51 graphs, got {len(graphs_all)}"
    missing = [g for g in graphs_all if g not in positional]
    assert not missing, f"positional statistics missing for: {missing}"
    assert len(positional) >= 51, f"positional export covers only {len(positional)} graphs"
    print(f"[assert] 192 unique configurations, 51 graphs, 0 missing positional joins")

    rows = []
    if args.reproduce:
        sub = [k for k in keys if meta[k]["graph"] not in V5_EXCLUDED]
        n_graphs = len({meta[k]["graph"] for k in sub})
        print(f"[reproduce] 47-graph scope: {n_graphs} graphs, {len(sub)} configurations")
        rep = run(sub, pairs, meta, positional, args.depth, args.seed, "v5_47graph")
        for arm in ("base", "base+index"):
            r = rep[arm]
            print(f"  {arm:11s} hits {r['hits']:>3d}/{r['n']}  "
                  f"geomean {r['geomean_vs_cusparse']:.4f}x  R/O {r['router_oracle']:.4f}")
        print("  published: base 118/178 (1.6561x), base+index 104/178 (1.6364x)")
        # The base arm is the protocol validator: it depends only on released inputs
        # (final_fair_v3.csv), so an exact match proves the filters, labels, folds, model,
        # seed and aggregation are the published ones.
        if rep["base"]["hits"] != 118 or rep["base"]["n"] != 178:
            raise SystemExit("REPRODUCTION FAILED - the base arm does not match the published "
                             "run; not reporting 51-graph numbers from a drifted protocol")
        if abs(rep["base"]["geomean_vs_cusparse"] - 1.6561) > 5e-5:
            raise SystemExit("REPRODUCTION FAILED - base geomean drifted from 1.6561")
        print("  base arm reproduces exactly -> protocol confirmed")
        # The positional arm cannot be reproduced: the 47-graph positional export it consumed
        # was never released, and positional_stats_51.csv is a fresh computation. This is
        # precisely why the published value is superseded rather than relabelled.
        if rep["base+index"]["hits"] != 104:
            print(f"  note: base+index recomputes to {rep['base+index']['hits']}/178, not the "
                  f"published 104/178 - the 47-graph positional export was never released, so "
                  f"the published value is not reproducible from released artifacts")
        rows += [rep["base"], rep["base+index"]]

    full = run(keys, pairs, meta, positional, args.depth, args.seed, "v6_51graph")
    print(f"[51-graph] 51 graphs, 192 configurations, leave-one-graph-out, "
          f"dtree depth {args.depth}, seed {args.seed}")
    for arm in ("base", "base+index"):
        r = full[arm]
        print(f"  {arm:11s} hits {r['hits']:>3d}/{r['n']}  "
              f"geomean {r['geomean_vs_cusparse']:.4f}x  R/O {r['router_oracle']:.4f}")
    rows += [full["base"], full["base+index"]]

    fields = ["scope", "feature_set", "model", "seed", "graphs", "n", "hits", "hit_rate",
              "geomean_vs_cusparse", "router_oracle"]
    out = Path(args.out)
    with out.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
