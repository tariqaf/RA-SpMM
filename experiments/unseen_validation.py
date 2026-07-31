#!/usr/bin/env python3
"""Task 1: frozen seven-rule router validation on nine previously-unused graphs.

No retuning of any kind: the committed route_with_rules is called verbatim; a
read-only mirror reports which rule fired. Warm protocol identical to the main
192-config sweep (reusable state built before timing; 50 warmup + 200 timed
CUDA-event iters). Correctness gate 1e-3*sqrt(max_row_nnz), 10x for tile paths.
Oracle includes cuSPARSE (max(1.0, max speedup)); hit = router>=oracle-1e-9.
Emits unseen_sweep.csv, unseen_router_quality.csv, unseen_manifest.json.
Every attempted graph, failure, OOM, and skipped (graph,N) is recorded.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import ra_spmm  # noqa
from ra_real_graph_eval import build_kernel_plan, run_planned_kernel, population_cv  # noqa
from ra_router_eval import route_with_rules, KERNELS  # noqa

UNSEEN = REPO / "datasets/unseen"
OUTDIR = REPO / "fgcs_results/revision/tf32/unseen"
NS = [64, 128, 256, 512]
WARMUP, TIMED = 50, 200
TILE = {"TC_DIRECT", "TC_DIRECT_TF32", "COMMUNITY_TC", "COMMUNITY_TC_TF32",
        "SEGMENT_HYBRID", "SEGMENT_HYBRID_TF32"}
# eight deployed candidate paths (RoDe retired) + cuSPARSE
CANDIDATES = ["CSR_DIRECT", "ZERO_OVERHEAD_CSR", "TC_DIRECT", "TC_DIRECT_TF32",
              "COMMUNITY_TC", "COMMUNITY_TC_TF32", "SEGMENT_HYBRID", "SEGMENT_HYBRID_TF32"]

# name -> (file, directed?, source_url)
GRAPHS = {
    "ca-AstroPh":       ("ca-AstroPh.txt.gz",       False, "https://snap.stanford.edu/data/ca-AstroPh.txt.gz"),
    "Coauthor-CS":      ("Coauthor-CS.npz",         False, "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz"),
    "email-Enron":      ("email-Enron.txt.gz",      False, "https://snap.stanford.edu/data/email-Enron.txt.gz"),
    "p2p-Gnutella31":   ("p2p-Gnutella31.txt.gz",   True,  "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz"),
    "soc-Slashdot0922": ("soc-Slashdot0922.txt.gz", True,  "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz"),  # 0922 in task == SNAP 0902 (82168/948464)
    "web-NotreDame":    ("web-NotreDame.txt.gz",    True,  "https://snap.stanford.edu/data/web-NotreDame.txt.gz"),
    "wiki-Talk":        ("wiki-Talk.txt.gz",        True,  "https://snap.stanford.edu/data/wiki-Talk.txt.gz"),
    "as-Skitter":       ("as-Skitter.txt.gz",       False, "https://snap.stanford.edu/data/as-skitter.txt.gz"),
    "cit-Patents":      ("cit-Patents.txt.gz",      True,  "https://snap.stanford.edu/data/cit-Patents.txt.gz"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_graph(name):
    """Return (csr, load_record). Undirected sources are symmetrized (the
    correct representation of an undirected graph); directed kept as-is.
    Nodes remapped to contiguous ids; edges deduplicated; unit values.
    Self-loops present in the source are preserved (count recorded)."""
    fn, directed, url = GRAPHS[name]
    path = UNSEEN / fn
    if fn.endswith(".npz"):  # Coauthor-CS adjacency (undirected)
        z = np.load(path, allow_pickle=True)
        A = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]),
                          shape=tuple(z["adj_shape"]))
        src, dst = A.tocoo().row, A.tocoo().col
    else:
        rows = []
        with gzip.open(path, "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                a, b = line.split()[:2]
                rows.append((int(a), int(b)))
        e = np.asarray(rows, dtype=np.int64)
        src, dst = e[:, 0], e[:, 1]
    nodes = np.unique(np.concatenate([src, dst]))
    remap = {int(v): i for i, v in enumerate(nodes)}
    M = len(nodes)
    s = np.fromiter((remap[int(v)] for v in src), dtype=np.int64, count=len(src))
    t = np.fromiter((remap[int(v)] for v in dst), dtype=np.int64, count=len(dst))
    self_loops = int((s == t).sum())
    if not directed:                       # symmetrize undirected
        s, t = np.concatenate([s, t]), np.concatenate([t, s])
    A = sp.csr_matrix((np.ones(len(s), np.float32), (s, t)), shape=(M, M))
    A.sum_duplicates()                     # dedup
    A.data[:] = 1.0                        # unit values
    A.sort_indices()
    rec = {"directed": directed, "symmetrized": (not directed), "deduplicated": True,
           "self_loops_in_source": self_loops, "self_loops_added": False, "unit_values": True,
           "source_url": url, "sha256": sha256(path), "loader": "npz-adj" if fn.endswith(".npz") else "snap-edgelist"}
    return A, rec


def matched_rule(d, cv, M, N, nnz):
    """Read-only mirror of route_with_rules to report the firing rule (R1-R7/default)."""
    if cv >= 5.0: return "R1"
    if cv < 0.7: return "R2"
    if cv >= 3.0: return "R3"
    if d >= 40.0: return "R4"
    if M < 20000 and d >= 25.0: return "R5"
    if d >= 25.0 and cv >= 1.8: return "R6"
    if d < 9.0 and 1.10 <= cv <= 1.45 and M >= 250000: return "R7"
    return "default"


def suite_centroids():
    """Per-category centroids in z-scored (log10 M, log10 d, cv) from the 51 labeled suite graphs."""
    seen, X, y = {}, [], []
    for r in csv.DictReader(open(REPO / "fgcs_results/revision/tf32/final_fair_v3.csv")):
        g = r["dataset"]
        if g in seen or not r.get("cv_d"):
            continue
        seen[g] = 1
        M, nnz = int(r["M"]), int(r["nnz"])
        X.append([math.log10(max(1, M)), math.log10(max(1e-9, nnz / max(1, M))), float(r["cv_d"])])
        y.append(r["category"])
    X = np.array(X); mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    cats = sorted(set(y))
    cent = {c: Z[[i for i, v in enumerate(y) if v == c]].mean(0) for c in cats}
    return cent, mu, sd


def assign_category(M, d, cv, cent, mu, sd):
    z = (np.array([math.log10(max(1, M)), math.log10(max(1e-9, d)), cv]) - mu) / sd
    return min(cent, key=lambda c: float(np.linalg.norm(z - cent[c])))


def measure(run_fn):
    for _ in range(WARMUP):
        run_fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(TIMED):
        run_fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / TIMED


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cent, mu, sd = suite_centroids()
    sweep_rows, rq_rows, skipped, load_fail, graph_records = [], [], [], [], {}

    for name in GRAPHS:
        try:
            A, rec = load_graph(name)
        except Exception as ex:
            load_fail.append({"dataset": name, "error": str(ex)[:300]}); print(f"[LOAD FAIL] {name}: {ex}"); continue
        M = A.shape[0]; nnz = int(A.nnz); d = nnz / max(1, M)
        deg = torch.tensor(np.diff(A.indptr), dtype=torch.float32)
        cv = population_cv(deg)
        cat = assign_category(M, d, cv, cent, mu, sd)
        rec.update(M=M, K=M, nnz=nnz, avg_nnz_per_row=round(d, 5), cv_d=round(cv, 5),
                   descriptor_category=cat)
        graph_records[name] = rec
        print(f"\n[{name}] M={M} nnz={nnz} d={d:.2f} cv={cv:.3f} cat={cat} "
              f"{'directed' if rec['directed'] else 'undirected'}", flush=True)
        rp_c = torch.from_numpy(A.indptr.astype(np.int32))
        ci_c = torch.from_numpy(A.indices.astype(np.int32))
        vl_c = torch.from_numpy(A.data.astype(np.float32))
        rp = rp_c.to(dev); ci = ci_c.to(dev); vl = vl_c.to(dev)
        maxrow = max(1, int(np.diff(A.indptr).max()))

        for N in NS:
            try:
                torch.manual_seed(123 + N)
                B = torch.randn(M, N, device=dev, dtype=torch.float32)
                cus_plan = ra_spmm.make_cusparse_plan(rp, ci, vl, B)
                ref = ra_spmm.run_cusparse_plan(cus_plan, B)
                ms_cus = measure(lambda: ra_spmm.run_cusparse_plan(cus_plan, B))
            except torch.cuda.OutOfMemoryError as ex:
                skipped.append({"dataset": name, "N": N, "reason": f"cusparse/ref OOM: {str(ex)[:120]}"})
                torch.cuda.empty_cache(); print(f"  N={N}: SKIP (cuSPARSE OOM)"); continue
            path_speed = {}
            for k in CANDIDATES:
                try:
                    plan = build_kernel_plan(k, rp_c, ci_c, vl_c, M, M, N)
                    run = (lambda p=plan, kk=k: run_planned_kernel(kk, p, rp, ci, vl, B))
                    out = run()
                    err = float((out.float() - ref.float()).abs().max().item())
                    tol = 1e-3 * math.sqrt(maxrow) * (10.0 if k in TILE else 1.0)
                    correct = bool(err <= tol and err < 1.0)
                    ms = measure(run)
                    sp_vs = ms_cus / ms
                    if correct:
                        path_speed[k] = sp_vs
                except torch.cuda.OutOfMemoryError as ex:
                    skipped.append({"dataset": name, "N": N, "kernel": k, "reason": f"OOM: {str(ex)[:100]}"})
                    torch.cuda.empty_cache(); ms = float("nan"); err = float("nan"); tol = float("nan"); correct = False; sp_vs = float("nan")
                except Exception as ex:
                    skipped.append({"dataset": name, "N": N, "kernel": k, "reason": f"error: {str(ex)[:120]}"})
                    ms = float("nan"); err = float("nan"); tol = float("nan"); correct = False; sp_vs = float("nan")
                sweep_rows.append({
                    "dataset": name, "M": M, "K": M, "nnz": nnz, "avg_nnz_per_row": round(d, 5),
                    "cv_d": round(cv, 5), "descriptor_category": cat, "N": N, "kernel": k,
                    "ms_warm": round(ms, 6) if ms == ms else "", "ms_cusparse_warm": round(ms_cus, 6),
                    "speedup_vs_cusparse_warm": round(sp_vs, 5) if sp_vs == sp_vs else "",
                    "correct": correct, "max_error": round(err, 6) if err == err else "",
                    "tolerance": round(tol, 6) if tol == tol else ""})
            # cuSPARSE as an explicit candidate row too (for the roofline/oracle transparency)
            sweep_rows.append({
                "dataset": name, "M": M, "K": M, "nnz": nnz, "avg_nnz_per_row": round(d, 5),
                "cv_d": round(cv, 5), "descriptor_category": cat, "N": N, "kernel": "CUSPARSE",
                "ms_warm": round(ms_cus, 6), "ms_cusparse_warm": round(ms_cus, 6),
                "speedup_vs_cusparse_warm": 1.0, "correct": True, "max_error": 0.0, "tolerance": 0.0})

            # router + oracle + hit
            router_k = route_with_rules(d, cv, M, N, nnz)
            rule = matched_rule(d, cv, M, N, nnz)
            router_ms = ms_cus if router_k == "CUSPARSE" else (
                path_speed and (ms_cus / path_speed[router_k]) if router_k in path_speed else float("nan"))
            router_sp = path_speed.get(router_k, (1.0 if router_k == "CUSPARSE" else float("nan")))
            oracle_sp = max([1.0] + list(path_speed.values()))
            oracle_k = "CUSPARSE" if oracle_sp == 1.0 else max(path_speed, key=path_speed.get)
            oracle_ms = ms_cus if oracle_k == "CUSPARSE" else ms_cus / path_speed[oracle_k]
            # feasible == at least one deployed kernel produced a valid timing.
            # When every deployed kernel OOMs, path_speed is empty, the oracle
            # collapses to the cuSPARSE floor (oracle_speedup == 1.0) and the
            # router has no timing. Such rows are NOT router misses: is_hit is
            # left empty and feasible=False so downstream readers exclude them
            # from every statistic (a numerically-valid 1.0 would otherwise drag
            # the oracle geomean down, e.g. 1.313 instead of 1.334).
            feasible = bool(path_speed)
            is_hit = (router_sp == router_sp) and (router_sp >= oracle_sp - 1e-9)
            rq_rows.append({
                "dataset": name, "M": M, "nnz": nnz, "avg_nnz_per_row": round(d, 5),
                "cv_d": round(cv, 5), "N": N, "router_kernel": router_k, "matched_rule": rule,
                "router_ms": round(router_ms, 6) if router_ms == router_ms else "",
                "router_speedup": round(router_sp, 5) if router_sp == router_sp else "",
                "oracle_kernel": oracle_k, "oracle_ms": round(oracle_ms, 6),
                "oracle_speedup": round(oracle_sp, 5),
                "router_oracle_ratio": round(router_sp / oracle_sp, 5) if router_sp == router_sp else "",
                "is_hit": (bool(is_hit) if feasible else ""), "feasible": feasible})
            print(f"  N={N}: router={router_k}({rule}) sp={router_sp if router_sp==router_sp else float('nan'):.3f} "
                  f"oracle={oracle_k} sp={oracle_sp:.3f} hit={is_hit}", flush=True)
            del B, ref, cus_plan
            torch.cuda.empty_cache()
        del rp, ci, vl
        torch.cuda.empty_cache()

    # ---- write CSVs ----
    with (OUTDIR / "unseen_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "M", "K", "nnz", "avg_nnz_per_row", "cv_d",
                                          "descriptor_category", "N", "kernel", "ms_warm",
                                          "ms_cusparse_warm", "speedup_vs_cusparse_warm",
                                          "correct", "max_error", "tolerance"])
        w.writeheader(); w.writerows(sweep_rows)
    with (OUTDIR / "unseen_router_quality.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "M", "nnz", "avg_nnz_per_row", "cv_d", "N",
                                          "router_kernel", "matched_rule", "router_ms", "router_speedup",
                                          "oracle_kernel", "oracle_ms", "oracle_speedup",
                                          "router_oracle_ratio", "is_hit", "feasible"])
        w.writeheader(); w.writerows(rq_rows)

    # ---- statistics ----
    def gm(v):
        v = [x for x in v if x and x > 0]
        return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
    hits = [r for r in rq_rows if r["router_speedup"] != ""]
    cfg_router = [float(r["router_speedup"]) for r in hits]
    cfg_oracle = [float(r["oracle_speedup"]) for r in hits]
    cfg_ratio = [float(r["router_oracle_ratio"]) for r in hits]
    # graph-macro: geomean within graph across N, then across graphs
    by_g = {}
    for r in hits:
        by_g.setdefault(r["dataset"], []).append(float(r["router_speedup"]))
    macro = gm([gm(v) for v in by_g.values()])
    by_g_ratio = {}
    for r in hits:
        by_g_ratio.setdefault(r["dataset"], []).append(float(r["router_oracle_ratio"]))
    macro_ratio = gm([gm(v) for v in by_g_ratio.values()])
    # graph bootstrap CI on router speedup (resample whole graphs)
    rng = np.random.default_rng(0)
    gkeys = list(by_g); boot = []
    for _ in range(10000):
        samp = rng.choice(len(gkeys), len(gkeys), replace=True)
        boot.append(gm([gm(by_g[gkeys[i]]) for i in samp]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    hit_n = sum(1 for r in hits if r["is_hit"]); n_hits = len(hits)
    minr = min(hits, key=lambda r: float(r["router_oracle_ratio"]))
    from collections import Counter
    rule_counts = dict(Counter(r["matched_rule"] for r in rq_rows))

    gpu = torch.cuda.get_device_name(0)
    manifest = {
        "gpu_name": gpu, "driver": _driver(), "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__, "warmup_iters": WARMUP, "timed_iters": TIMED,
        "tolerance_rule": "1e-3*sqrt(max_i nnz_i), x10 for mixed-precision tile paths",
        "router_frozen": True, "retuning_performed": False,
        "used_in_threshold_design": False, "used_in_prior_router_pilots": False,
        "provenance_method": "grep of paper_datasets.json, all repo CSV/JSON/py, and git log --all -S <name>; zero hits for every graph",
        "descriptor_category_method": "nearest per-category centroid of the 51 labeled suite graphs in z-scored (log10 M, log10 avg_nnz, cv_d); paper's explicit numeric bands not committed in-repo -- substitute if available",
        "candidate_paths": CANDIDATES + ["CUSPARSE(CUSPARSE_SPMM_CSR_ALG2)"],
        "rode_excluded": True,
        "router_source_sha256": sha256(REPO / "router/router_dispatch.cpp"),
        "router_thresholds_sha256": sha256(REPO / "ra_router_eval.py"),
        "kernel_source_sha256": {p.name: sha256(p) for p in sorted((REPO / "tc").glob("*.cu")) if p.name.startswith("ra_")},
        "kernel_source_commit": _commit(), "evaluation_harness_commit": _commit(),
        "graphs": graph_records, "skipped_configs": skipped, "load_failures": load_fail,
        "note_soc_slashdot": "task name soc-Slashdot0922 has no SNAP file; SNAP soc-Slashdot0902 (82168 nodes / 948464 edges) matches the stated size and is used",
        "statistics": {
            "feasible_configs": n_hits,
            "config_geomean_router": round(gm(cfg_router), 5),
            "config_geomean_oracle": round(gm(cfg_oracle), 5),
            "config_router_oracle": round(gm(cfg_ratio), 5),
            "graph_macro_geomean_router": round(macro, 5),
            "graph_macro_router_oracle": round(macro_ratio, 5),
            "graph_bootstrap95_router": [round(ci[0], 5), round(ci[1], 5)],
            "hit_count": hit_n, "hit_rate": round(hit_n / max(1, n_hits), 5),
            "min_ratio": float(minr["router_oracle_ratio"]),
            "min_ratio_config": f"{minr['dataset']} N={minr['N']} router={minr['router_kernel']}",
            "per_rule_firing": rule_counts,
        },
    }
    (OUTDIR / "unseen_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("\n==== SUMMARY ====")
    print(json.dumps(manifest["statistics"], indent=2))
    print(f"wrote {OUTDIR}/unseen_sweep.csv ({len(sweep_rows)} rows), "
          f"unseen_router_quality.csv ({len(rq_rows)} rows), unseen_manifest.json")


def _driver():
    try:
        import subprocess
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().split("\n")[0].strip()
    except Exception:
        return "unknown"


def _commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
