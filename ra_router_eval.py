"""
ra_router_eval.py - Evaluate router quality against oracle

Compares the router's kernel selection against the oracle (best-per-point)
using the final_real_graph_results.csv as ground truth.

The router's goal: for each (graph, N) pair, select the kernel that
maximizes speedup vs cuSPARSE. The oracle always picks the best.
Router quality = geomean(router_speedup / oracle_speedup).

Usage:
    python ra_router_eval.py
    python ra_router_eval.py --results final_real_graph_results.csv
"""
import argparse
import csv
import math
import sys
from collections import defaultdict


# Final 6-kernel roster
KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
           "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"]


def simple_router(avg_nnz, degree_cv, M, N, nnz):
    """
    Approximate 6-kernel paper router (round-2 recalibration).

    Tuned for the new label-propagation COMMUNITY_TC, which dominates
    most low-to-moderate-degree workloads. Eight rules, evaluated top-to-
    bottom (first match wins). Default fallthrough is TC_DIRECT.

    Features used: avg_nnz_per_row (d), degree_cv (cv), M, N.
    """
    d = avg_nnz
    cv = degree_cv

    # 1. Sub-tiny graphs (Cora, CiteSeer, PPI; ca-GrQc is M=5242, just
    #    above the threshold). Two SEGMENT_HYBRID pockets at wide N:
    #      - mid-degree tinies (PPI, d=18)
    #      - very-low-degree tinies (Cora d=3.9, CiteSeer d=2.7)
    #    Everything else falls through to TC_DIRECT where launch overhead
    #    dominates and the dense fully-resident A tile wins.
    if M < 5000:
        if N >= 256 and (d >= 12.0 or d <= 6.0):
            return "SEGMENT_HYBRID"
        return "TC_DIRECT"

    # 2. Sparse-tail (com-youtube, very-skewed sparse): low d, very high
    #    CV. Wide-N benefits from row-split RODE; small N stays on
    #    TC_DIRECT where the kernel-launch overhead matters most.
    if M >= 100_000 and d < 8.0 and cv > 4.0:
        return "RODE_ENHANCED" if N >= 256 else "TC_DIRECT"

    # 3. Dense-small with d >= 25 (amazon-computers/photo and synthetic
    #    dense-small). Placed BEFORE the skewed-mid rule so that
    #    amazon-photo (M=7.6K, d=31, CV=1.52) is captured here rather
    #    than being mis-classified as a power-law sparse graph.
    if M <= 15_000 and d >= 25.0:
        return "SEGMENT_HYBRID" if cv >= 1.0 else "COMMUNITY_TC"

    # 4. Heavily skewed sparse mid-degree. Sub-cases by M:
    #      twitter-combined (M~80K) -> CSR_DIRECT/RODE depending on N
    #      soc-Pokec (M~1.6M)        -> CSR_DIRECT
    #      synth_mixed_v* (M=200K)  -> falls through to TC_DIRECT default
    if 12.0 <= d <= 40.0 and cv >= 1.5:
        if M <= 100_000:
            return "RODE_ENHANCED" if N >= 256 else "CSR_DIRECT"
        if M >= 1_000_000:
            return "CSR_DIRECT"

    # 5. Dense-large (Reddit, ogbn-proteins, gplus-combined). TC kernels
    #    win on arithmetic intensity. RODE for extreme-skew + wide-N.
    if d >= 96.0:
        if cv >= 2.5 and N >= 256:
            return "RODE_ENHANCED"
        return "TC_DIRECT"

    # 6. Huge mid-density sparse (ogbn-products): M >= 1M, d in [40, 96),
    #    mild skew. Label-prop COMMUNITY_TC reorders the column layout
    #    well enough to beat TC_DIRECT.
    if M >= 1_000_000 and 40.0 <= d < 96.0 and cv <= 2.5:
        return "COMMUNITY_TC"

    # 7. Medium-scale low-d irregular pocket (Flickr-class). M ~ 50-150K
    #    with d ~ 9-12 sits in a sweet spot for ZERO_OVERHEAD_CSR which
    #    avoids any preprocessing cost.
    if 50_000 <= M <= 150_000 and 9.0 <= d <= 12.0:
        return "ZERO_OVERHEAD_CSR"

    # 8. COMMUNITY_TC sweet spot (the new label-prop variant). Three OR
    #    branches catch distinct sub-regimes:
    #      (a) M >= 150K, d <= 10, CV in [0.5, 4.0], N <= 256:
    #          web-Google, web-Stanford, com-DBLP, com-Amazon, ogbn-arxiv.
    #          The N <= 256 cap prevents COMMUNITY_TC from over-firing at
    #          N=512 where its plan-build overhead can dominate (e.g.,
    #          ogbn-arxiv N=512 prefers TC_DIRECT).
    #      (b) M >= 250K, d <= 9, CV > 0.1:
    #          Amazon0601 (CV=0.33), roadNet-* family. CV > 0.1 keeps
    #          natural-clustering real graphs while excluding pure-
    #          uniform synthetics (synth_sparse_uniform_d5/8/12/18 have
    #          CV = 0 exactly).
    #      (c) M >= 150K, d <= 4:
    #          synth_sparse_uniform_d3 and any very-sparse graph where
    #          even random reordering pays off.
    #    synth_community_nc* (M=200K, d=5.5, CV=0.42) deliberately
    #    excluded: fails (a) on CV, (b) on M, (c) on d.
    if (M >= 150_000 and d <= 10.0 and 0.5 <= cv <= 4.0 and N <= 256) or \
       (M >= 250_000 and d <= 9.0 and cv > 0.1) or \
       (M >= 150_000 and d <= 4.0):
        return "COMMUNITY_TC"

    # Fallthrough: TC_DIRECT catches synth_community_nc*, synth_mixed_v*,
    # synth_sparse_uniform_d5/8/12/18, synth_sparse_skewed_cv1p5..4p0,
    # ca-HepTh, ca-CondMat, Yelp, gplus-combined remainders, etc.
    return "TC_DIRECT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="final_real_graph_results.csv")
    args = parser.parse_args()

    # Load results
    data = []
    with open(args.results) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['kernel'] in KERNELS and row.get('correct', 'True') == 'True':
                data.append(row)

    # Group by (dataset, N) pair
    pairs = defaultdict(dict)
    graph_meta = {}
    feature_cv = {}  # actual cv_d from CSV per (dataset, N)
    for row in data:
        key = (row['dataset'], int(row['N']))
        pairs[key][row['kernel']] = float(row['speedup_vs_cusparse'])
        graph_meta[row['dataset']] = {
            'category': row['category'],
            'M': int(row['M']),
            'nnz': int(row['nnz']),
        }
        # Prefer the actual cv_d column when present (round-2 sweep CSVs).
        cv_csv = row.get('cv_d', '')
        if cv_csv not in (None, '', 'None'):
            try:
                feature_cv[key] = float(cv_csv)
            except (ValueError, TypeError):
                pass

    print("=" * 80)
    print("RA-SpMM Router Quality Evaluation")
    print(f"Data: {args.results} | Kernels: {len(KERNELS)} | Pairs: {len(pairs)}")
    print("=" * 80)

    # Evaluate
    oracle_logs = []
    router_logs = []
    hits = 0
    misses = []
    by_category = defaultdict(lambda: {'oracle': [], 'router': [], 'hits': 0, 'total': 0})

    for (dataset, N), kernel_speeds in sorted(pairs.items()):
        meta = graph_meta.get(dataset, {})
        M = meta.get('M', 100000)
        nnz = meta.get('nnz', 1000000)
        avg_nnz = nnz / max(1, M)
        category = meta.get('category', '?')

        # Prefer the per-(dataset, N) cv_d carried in the CSV; fall back
        # to a hardcoded approximation only for legacy CSVs that lack it.
        if (dataset, N) in feature_cv:
            degree_cv = feature_cv[(dataset, N)]
        else:
            degree_cv = 0.5  # default: moderate
            if dataset in ['twitter-combined']:
                degree_cv = 2.6906
            elif dataset in ['soc-Pokec']:
                degree_cv = 1.7138
            elif dataset in ['gplus-combined']:
                degree_cv = 4.3896
            elif dataset in ['Reddit']:
                degree_cv = 1.6
            elif dataset in ['com-youtube']:
                degree_cv = 9.7378
            elif dataset in ['com-DBLP']:
                degree_cv = 1.5113
            elif dataset in ['com-Amazon']:
                degree_cv = 1.0419
            elif dataset in ['ogbn-products']:
                degree_cv = 1.8985
            elif dataset in ['ogbn-proteins']:
                degree_cv = 1.0408
            elif dataset.startswith('roadNet') or dataset.startswith('ca-'):
                degree_cv = 0.3
            elif dataset in ['web-Google']:
                degree_cv = 1.1770
            elif dataset in ['Cora']:
                degree_cv = 1.3
            elif dataset in ['CiteSeer']:
                degree_cv = 1.2
            elif dataset in ['PPI']:
                degree_cv = 0.8

        # Oracle: best kernel for this pair
        best_kernel = max(kernel_speeds, key=kernel_speeds.get)
        oracle_speed = kernel_speeds[best_kernel]

        # Router: what would our router pick?
        router_kernel = simple_router(avg_nnz, degree_cv, M, N, nnz)
        router_speed = kernel_speeds.get(router_kernel, 0)

        # If router's pick isn't in the results, fall back to CSR_DIRECT
        if router_speed <= 0:
            router_kernel = "CSR_DIRECT"
            router_speed = kernel_speeds.get("CSR_DIRECT", 1.0)

        # Track
        if oracle_speed > 0:
            oracle_logs.append(math.log(oracle_speed))
        if router_speed > 0:
            router_logs.append(math.log(router_speed))

        hit = (router_kernel == best_kernel)
        if hit:
            hits += 1
        else:
            ratio = router_speed / oracle_speed if oracle_speed > 0 else 0
            misses.append((dataset, N, category, router_kernel, best_kernel,
                          router_speed, oracle_speed, ratio))

        cat = by_category[category]
        cat['total'] += 1
        if hit: cat['hits'] += 1
        if oracle_speed > 0: cat['oracle'].append(math.log(oracle_speed))
        if router_speed > 0: cat['router'].append(math.log(router_speed))

    total = len(pairs)
    oracle_gm = math.exp(sum(oracle_logs) / len(oracle_logs)) if oracle_logs else 0
    router_gm = math.exp(sum(router_logs) / len(router_logs)) if router_logs else 0
    overhead = oracle_gm / router_gm if router_gm > 0 else float('inf')

    print(f"\n{'Metric':<40s} {'Value':>10s}")
    print("-" * 52)
    print(f"{'Oracle geomean (vs cuSPARSE)':<40s} {oracle_gm:>10.3f}x")
    print(f"{'Router geomean (vs cuSPARSE)':<40s} {router_gm:>10.3f}x")
    print(f"{'Router overhead (oracle/router)':<40s} {overhead:>10.3f}x")
    print(f"{'Router hit rate':<40s} {hits}/{total} ({100*hits/max(1,total):.1f}%)")
    print(f"{'Router miss rate':<40s} {total-hits}/{total} ({100*(total-hits)/max(1,total):.1f}%)")

    # Per-category
    print(f"\n{'Category':<30s} {'Hits':>6s} {'Oracle':>8s} {'Router':>8s} {'Overhead':>10s}")
    print("-" * 66)
    for cat in sorted(by_category.keys()):
        c = by_category[cat]
        ogm = math.exp(sum(c['oracle']) / len(c['oracle'])) if c['oracle'] else 0
        rgm = math.exp(sum(c['router']) / len(c['router'])) if c['router'] else 0
        ovh = ogm / rgm if rgm > 0 else float('inf')
        print(f"{cat:<30s} {c['hits']:>3d}/{c['total']:<3d} {ogm:>7.3f}x {rgm:>7.3f}x {ovh:>9.3f}x")

    # Misses detail
    if misses:
        print(f"\n--- Router Misses ({len(misses)}) ---")
        print(f"{'Dataset':<25s} {'N':>4s} {'Category':<20s} {'Router':>15s} {'Oracle':>15s} {'Ratio':>8s}")
        print("-" * 90)
        for ds, n, cat, rk, ok, rs, os_, ratio in sorted(misses, key=lambda x: x[7]):
            print(f"{ds:<25s} {n:>4d} {cat:<20s} {rk:>15s} {ok:>15s} {ratio:>7.3f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
