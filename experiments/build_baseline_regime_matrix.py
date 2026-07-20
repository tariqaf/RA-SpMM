#!/usr/bin/env python3
"""Task 6: unified per-regime baseline matrix, recomputed vs the FINAL router.

Single source of truth: router_quality_v5.router_ms (final round-5 router warm),
joined by (dataset, N). For every baseline, router_faster = baseline_ms /
router_ms (>1 => router faster). Per-regime geomean, overall geomean, and
regime-balanced geomean (mean of per-regime geomeans). Coverage and one-time
preprocessing (mean/max, host vs device) recorded per baseline. NO stale
speedup columns are trusted.
"""
from __future__ import annotations
import csv, math
from collections import defaultdict
from pathlib import Path

R = Path("/mnt/shared/development/tariq/RA-SpMM")
TF = R / "fgcs_results/revision/tf32"
FAIR = R / "fgcs_results/revision/fair"
OUT = TF / "baseline_audit"


def gm(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


# --- final router reference ---
router = {}   # (dataset,N) -> ms
regime = {}   # (dataset,N) -> category
synth = {}
for r in csv.DictReader(open(TF / "router_quality_v5.csv")):
    k = (r["dataset"], r["N"])
    router[k] = float(r["router_ms"])
    regime[k] = r["category"]
    synth[k] = r["synthetic"] == "True"

TOTAL_CONFIGS = len(router)  # 192


def per_regime(pairs):
    """pairs: list of (key, baseline_ms). Returns dict regime->geomean(router_faster), plus overall & balanced."""
    by = defaultdict(list)
    allf = []
    for k, bms in pairs:
        if k not in router or bms <= 0:
            continue
        f = bms / router[k]
        by[regime[k]].append(f)
        allf.append(f)
    out = {reg: gm(v) for reg, v in by.items()}
    out["_OVERALL"] = gm(allf)
    out["_REGIME_BALANCED"] = gm(list(out[reg] for reg in by)) if by else float("nan")
    out["_N"] = len(allf)
    # regime where router wins most = max router_faster geomean
    if by:
        best = max(by, key=lambda reg: gm(by[reg]))
        out["_ROUTER_BEST_REGIME"] = f"{best} ({gm(by[best]):.2f}x)"
    return out


def load_baseline_pairs(path, ms_col, n_filter=None, correct_col="correct",
                        status_col=None, status_ok=None):
    pairs, attempted, ran = [], 0, 0
    fails = defaultdict(int)
    for r in csv.DictReader(open(path)):
        n = r["N"]
        if n_filter and n != n_filter:
            continue
        attempted += 1
        ok = True
        if correct_col and correct_col in r:
            ok = r[correct_col] == "True"
        if status_col and status_ok and r.get(status_col) not in status_ok:
            ok = False
        ms = r.get(ms_col, "")
        if not ok or ms in ("", None, "nan", "None"):
            reason = r.get(status_col, "") or r.get("failure_class", "") or ("incorrect" if not ok else "no_ms")
            fails[reason or "no_ms"] += 1
            continue
        try:
            v = float(ms)
        except (TypeError, ValueError):
            fails["no_ms"] += 1
            continue
        if v <= 0:
            fails["no_ms"] += 1
            continue
        pairs.append(((r["dataset"], n), v))
        ran += 1
    return pairs, attempted, ran, dict(fails)


REGIMES = ["Sparse Uniform", "Sparse Skewed", "Dense Small", "Dense Large-Scale",
           "Community", "Mixed/Irregular"]

rows = []


def add(baseline, pairs, attempted, ran, fails, n_scope, preproc, host_dev, note=""):
    pr = per_regime(pairs)
    row = {"baseline": baseline, "n_scope": n_scope, "ran": ran, "attempted": attempted,
           "coverage_ran_over_scope": f"{ran}/{attempted}",
           "fail_reasons": ";".join(f"{k}={v}" for k, v in fails.items()) if fails else "",
           "overall_router_faster_geomean": round(pr.get("_OVERALL", float('nan')), 4),
           "regime_balanced_geomean": round(pr.get("_REGIME_BALANCED", float('nan')), 4),
           "router_best_regime": pr.get("_ROUTER_BEST_REGIME", ""),
           "preprocess_mean_ms": preproc[0], "preprocess_max_ms": preproc[1],
           "preprocess_host_or_device": host_dev, "note": note}
    for reg in REGIMES:
        row[f"rf_{reg}"] = round(pr[reg], 4) if reg in pr else ""
    rows.append(row)
    return pr


def col_stats(path, col, extra=None):
    """mean,max of `col` (+ optional extra col added per row) over rows that have it."""
    vals = []
    for r in csv.DictReader(open(path)):
        if r.get(col):
            v = float(r[col])
            if extra and r.get(extra):
                v += float(r[extra])
            vals.append(v)
    return (round(sum(vals)/len(vals), 1), round(max(vals), 1)) if vals else (0.0, 0.0)


# ---- DTC-noTCA ----  preprocessing = device TC-block build (no reorder)
p, a, rn, f = load_baseline_pairs(TF / "dtc_identity_full.csv", "mean_kernel_ms")
add("DTC-noTCA", p, a, rn, f, "all N (26 graphs x4)",
    col_stats(TF / "dtc_identity_full.csv", "preprocess_ms"), "device (thrust::sort on GPU)",
    "identity CSR order; DTC kernel, no reorder. preprocessing = device TC-block build only.")

# ---- DTC-TCA ----  preprocessing = HOST TCA reorder + device block build
p, a, rn, f = load_baseline_pairs(TF / "dtc_tca_full.csv", "mean_kernel_ms")
add("DTC-TCA", p, a, rn, f, "all N (26 graphs x4)",
    col_stats(TF / "dtc_tca_full.csv", "reorder_ms", extra="preprocess_ms"),
    "device kernel build + HOST TCA reorder (600-900s)",
    "TCA proper-order reorder fails on 13/26 graphs; reorder 600-900s where it runs.")

# ---- MP-SpMM (N=128) ----
p, a, rn, f = load_baseline_pairs(FAIR / "mp_spmm.csv", "ms_warm", n_filter="128",
                                  status_col="status", status_ok={"OK"})
mp_pre = [float(r["preprocess_ms"]) for r in csv.DictReader(open(FAIR / "mp_spmm_preproc.csv")) if r["preprocess_ms"]]
add("MP-SpMM", p, a, rn, f, "N=128 only", (round(sum(mp_pre)/len(mp_pre), 1), round(max(mp_pre), 1)),
    "device (match-and-pad preproc binary)",
    "2:4 structured sparse-TC (kernel sparse_mma_kernel_base). N=128 only; nnz>5M skipped (18).")

# ---- HC-SpMM (N=64) ----
p, a, rn, f = load_baseline_pairs(FAIR / "hcspmm.csv", "ms_warm", n_filter="64",
                                  status_col="status", status_ok={"OK"})
hc_pre = [float(r["preproc_ms"]) for r in csv.DictReader(open(FAIR / "hcspmm_preproc.csv")) if r["preproc_ms"]]
add("HC-SpMM", p, a, rn, f, "N=64 only", (round(sum(hc_pre)/len(hc_pre), 2), round(max(hc_pre), 2)),
    "device (block build)",
    "N=64 only; illegal-memory crash on 24 configs (12/26 real graphs), 13 incorrect (synthetic).")

# ---- cuBLAS-dense ----
p, a, rn, f = load_baseline_pairs(TF / "cublas_dense_v3.csv", "cublas_warm_ms", correct_col=None)
cub_dens = [float(r["densify_ms"]) for r in csv.DictReader(open(TF / "cublas_dense_v3.csv")) if r.get("densify_ms")]
add("cuBLAS-dense", p, a, rn, f, "covered configs only", (round(sum(cub_dens)/len(cub_dens), 2), round(max(cub_dens), 2)),
    "device (densify on GPU)",
    "dense GEMM on densified matrix; only 2/6 regimes (Dense Small, Mixed) have data.")

# ---- FlashSparse (partial: 4 graphs N=128, correctness-verified on sm_86) ----
fs_pairs = []
for r in csv.DictReader(open(OUT / "flashsparse_sm86.csv")):
    if r.get("ran") == "True" and r.get("path") == "fp16_8x8" and r.get("kernel_ms"):
        fs_pairs.append(((r["dataset"], r["N"]), float(r["kernel_ms"])))
add("FlashSparse(fp16,partial)", fs_pairs, len(fs_pairs), len(fs_pairs), {}, "N=128, 4 graphs only",
    (80.5, 80.5), "device (Block_gpu seg_sort_dequ_fs)",
    "BUILDS+RUNS on sm_86 (8/8 correct). Speed row is PARTIAL (4 graphs); full per-regime sweep NOT run.")

# --- write matrix ---
fields = (["baseline", "n_scope", "ran", "attempted", "coverage_ran_over_scope",
           "overall_router_faster_geomean", "regime_balanced_geomean", "router_best_regime"]
          + [f"rf_{reg}" for reg in REGIMES]
          + ["preprocess_mean_ms", "preprocess_max_ms", "preprocess_host_or_device",
             "fail_reasons", "note"])
outp = OUT / "baseline_regime_matrix.csv"
with outp.open("w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# console summary
print(f"router configs (192 scope): {TOTAL_CONFIGS}")
print(f"\n{'baseline':26s}{'ran':>10s}{'overall':>9s}{'balanced':>9s}  router_best_regime")
for r in rows:
    print(f"{r['baseline']:26s}{r['coverage_ran_over_scope']:>10s}"
          f"{r['overall_router_faster_geomean']:>9}{r['regime_balanced_geomean']:>9}  {r['router_best_regime']}")
print(f"\nrf_<regime> = geomean(baseline_ms/router_ms); >1 = router faster. wrote {outp}")
print("\nper-regime router_faster:")
hdr = "baseline".ljust(26) + "".join(reg[:10].rjust(11) for reg in REGIMES)
print(hdr)
for r in rows:
    print(r["baseline"].ljust(26) + "".join(str(r.get(f"rf_{reg}", "")).rjust(11) for reg in REGIMES))
