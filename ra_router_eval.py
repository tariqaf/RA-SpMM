"""
ra_router_eval.py - Evaluate router quality against oracle

Compares the router's kernel selection against the oracle (best-per-point)
using a kernel timing CSV as ground truth.

The router's goal: for each (graph, N) pair, select the kernel that
maximizes speedup vs cuSPARSE. The oracle always picks the best.
Router quality = geomean(router_speedup / oracle_speedup).

Usage:
    python ra_router_eval.py
    python ra_router_eval.py --results results/spmm/all_graphs_results.csv
    python ra_router_eval.py --csv results/spmm/all_graphs_results.csv \
        --output results/router/router_quality.csv
"""
import argparse
import csv
import math
import sys
from collections import defaultdict


# Final 6-kernel roster
KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
           "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID"]
DEFAULT_RESULTS = "results/spmm/all_graphs_results.csv"


def route_with_rules(avg_nnz, degree_cv, M, N, nnz, *, disabled_rules=()):
    """
    Python mirror of the production six-kernel router.

    Tuned for the label-propagation COMMUNITY_TC path, which dominates
    most low-to-moderate-degree workloads. Eight rules, evaluated top-to-
    bottom (first match wins). Default fallthrough is TC_DIRECT.

    Features used: avg_nnz_per_row (d), degree_cv (cv), M, N.
    """
    disabled = frozenset(disabled_rules)
    d = avg_nnz
    cv = degree_cv

    # 1. Sub-tiny graphs (Cora, CiteSeer, PPI; ca-GrQc is M=5242, just
    #    above the threshold). Two SEGMENT_HYBRID pockets at wide N:
    #      - mid-degree tinies (PPI, d=18)
    #      - very-low-degree tinies (Cora d=3.9, CiteSeer d=2.7)
    #    Everything else falls through to TC_DIRECT where launch overhead
    #    dominates and the dense fully-resident A tile wins.
    if 1 not in disabled and M < 5000:
        if N >= 256 and (d >= 12.0 or d <= 6.0):
            return "SEGMENT_HYBRID"
        return "TC_DIRECT"

    # 2. Sparse-tail (com-youtube, very-skewed sparse): low d, very high
    #    CV. Wide-N benefits from row-split RODE; small N stays on
    #    TC_DIRECT where the kernel-launch overhead matters most.
    if 2 not in disabled and M >= 100_000 and d < 8.0 and cv > 4.0:
        return "RODE_ENHANCED" if N >= 256 else "TC_DIRECT"

    # 3. Dense-small with d >= 25 (amazon-computers/photo and synthetic
    #    dense-small). Placed BEFORE the skewed-mid rule so that
    #    amazon-photo (M=7.6K, d=31, CV=1.52) is captured here rather
    #    than being mis-classified as a power-law sparse graph.
    if 3 not in disabled and M <= 15_000 and d >= 25.0:
        return "SEGMENT_HYBRID" if cv >= 1.0 else "COMMUNITY_TC"

    # 4. Heavily skewed sparse mid-degree. Sub-cases by M:
    #      twitter-combined (M~80K) -> CSR_DIRECT/RODE depending on N
    #      soc-Pokec (M~1.6M)        -> CSR_DIRECT
    #      synth_mixed_v* (M=200K)  -> falls through to TC_DIRECT default
    if 4 not in disabled and 12.0 <= d <= 40.0 and cv >= 1.5:
        if M <= 100_000:
            return "RODE_ENHANCED" if N >= 256 else "CSR_DIRECT"
        if M >= 1_000_000:
            return "CSR_DIRECT"

    # 5. Dense-large (Reddit, ogbn-proteins, gplus-combined). TC kernels
    #    win on arithmetic intensity. RODE for extreme-skew + wide-N.
    if 5 not in disabled and d >= 96.0:
        if cv >= 2.5 and N >= 256:
            return "RODE_ENHANCED"
        return "TC_DIRECT"

    # 6. Huge mid-density sparse (ogbn-products): M >= 1M, d in [40, 96),
    #    mild skew. Label-prop COMMUNITY_TC reorders the column layout
    #    well enough to beat TC_DIRECT.
    if 6 not in disabled and M >= 1_000_000 and 40.0 <= d < 96.0 and cv <= 2.5:
        return "COMMUNITY_TC"

    # 7. Medium-scale low-d irregular pocket (Flickr-class). M ~ 50-150K
    #    with d ~ 9-12 sits in a sweet spot for ZERO_OVERHEAD_CSR which
    #    avoids any preprocessing cost.
    if 7 not in disabled and 50_000 <= M <= 150_000 and 9.0 <= d <= 12.0:
        return "ZERO_OVERHEAD_CSR"

    # 8. COMMUNITY_TC sweet spot (label-propagation variant). Three OR
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
    if 8 not in disabled and ((M >= 150_000 and d <= 10.0 and 0.5 <= cv <= 4.0 and N <= 256) or \
       (M >= 250_000 and d <= 9.0 and cv > 0.1) or \
       (M >= 150_000 and d <= 4.0)):
        return "COMMUNITY_TC"

    # Fallthrough: TC_DIRECT catches synth_community_nc*, synth_mixed_v*,
    # synth_sparse_uniform_d5/8/12/18, synth_sparse_skewed_cv1p5..4p0,
    # ca-HepTh, ca-CondMat, Yelp, gplus-combined remainders, etc.
    return "TC_DIRECT"


def simple_router(avg_nnz, degree_cv, M, N, nnz):
    """Return the production rule-tree decision with all eight rules enabled."""
    return route_with_rules(avg_nnz, degree_cv, M, N, nnz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "--csv", dest="results",
                        default=DEFAULT_RESULTS,
                        help="Kernel timing CSV used as router/oracle ground truth")
    parser.add_argument("--output", default=None,
                        help="Optional path for per-(dataset,N) router-quality CSV")
    parser.add_argument("--regime", choices=("warm", "cold"), default="warm")
    parser.add_argument("--expected", type=int, default=192)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    time_column = f"ms_{args.regime}"
    cusparse_column = f"ms_cusparse_{args.regime}"
    pairs = defaultdict(dict)
    graph_meta = {}
    with open(args.results) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['kernel'] not in KERNELS:
                continue
            if row.get('correct', 'False').lower() not in ('true', '1'):
                continue
            if not row.get(time_column) or not row.get(cusparse_column):
                continue
            key = (row['dataset'], int(row['N']))
            pairs[key][row['kernel']] = float(row[time_column])
            graph_meta[key] = {
                'category': row['category'], 'M': int(row['M']),
                'nnz': int(row['nnz']), 'synthetic': row.get('synthetic', 'False'),
                'cv_d': float(row['cv_d']),
                'cusparse_ms': float(row[cusparse_column]),
            }

    incomplete = {key: sorted(set(KERNELS) - set(values))
                  for key, values in pairs.items() if set(values) != set(KERNELS)}
    complete_pairs = {key: values for key, values in pairs.items() if key not in incomplete}
    if ((len(complete_pairs) != args.expected or incomplete) and not args.allow_partial):
        print(f"ROUTER QUALITY FAIL: expected {args.expected} complete pairs, "
              f"loaded {len(complete_pairs)}; incomplete={len(incomplete)}")
        sys.exit(1)
    pairs = complete_pairs

    print("=" * 80)
    print("RA-SpMM Router Quality Evaluation")
    print(f"Data: {args.results} | Regime: {args.regime} | "
          f"Kernels: {len(KERNELS)} | Pairs: {len(pairs)}")
    print("=" * 80)

    # Evaluate
    oracle_logs = []
    router_logs = []
    hits = 0
    misses = []
    by_category = defaultdict(lambda: {'oracle': [], 'router': [], 'hits': 0, 'total': 0})
    quality_rows = []
    ratios = []

    for (dataset, N), kernel_times in sorted(pairs.items()):
        meta = graph_meta[(dataset, N)]
        M = meta.get('M', 100000)
        nnz = meta.get('nnz', 1000000)
        avg_nnz = nnz / max(1, M)
        category = meta.get('category', '?')

        degree_cv = meta['cv_d']

        best_kernel = min(kernel_times, key=kernel_times.get)
        oracle_ms = kernel_times[best_kernel]
        cusparse_ms = meta['cusparse_ms']
        oracle_speed = cusparse_ms / oracle_ms

        # Router: what would our router pick?
        router_kernel = simple_router(avg_nnz, degree_cv, M, N, nnz)
        router_ms = kernel_times[router_kernel]
        router_speed = cusparse_ms / router_ms
        ratio = oracle_ms / router_ms
        ratios.append(ratio)

        # Track
        if oracle_speed > 0:
            oracle_logs.append(math.log(oracle_speed))
        if router_speed > 0:
            router_logs.append(math.log(router_speed))

        hit = (router_kernel == best_kernel)
        quality_rows.append({
            'dataset': dataset,
            'category': category,
            'N': N,
            'M': M,
            'nnz': nnz,
            'synthetic': meta.get('synthetic', 'False'),
            'regime': args.regime,
            'oracle_kernel': best_kernel,
            'oracle_ms': oracle_ms,
            'oracle_speedup': oracle_speed,
            'router_kernel': router_kernel,
            'router_ms': router_ms,
            'router_speedup': router_speed,
            'router_oracle_ratio': ratio,
            'is_hit': hit,
        })
        if hit:
            hits += 1
        else:
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
    router_oracle_gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0
    empirical_85 = sum(r >= 0.85 for r in ratios)
    empirical_99 = sum(r >= 0.99 for r in ratios)

    print(f"\n{'Metric':<40s} {'Value':>10s}")
    print("-" * 52)
    print(f"{'Oracle geomean (vs cuSPARSE)':<40s} {oracle_gm:>10.3f}x")
    print(f"{'Router geomean (vs cuSPARSE)':<40s} {router_gm:>10.3f}x")
    print(f"{'Router overhead (oracle/router)':<40s} {overhead:>10.3f}x")
    print(f"{'Router/Oracle geomean':<40s} {router_oracle_gm:>10.6f}x")
    print(f"{'Empirical ratio >= 0.85':<40s} {empirical_85:>4d}/{total:<4d}")
    print(f"{'Empirical ratio >= 0.99':<40s} {empirical_99:>4d}/{total:<4d}")
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

    if args.output:
        fieldnames = [
            'dataset', 'category', 'N', 'M', 'nnz', 'synthetic', 'regime',
            'oracle_kernel', 'oracle_ms', 'oracle_speedup', 'router_kernel',
            'router_ms', 'router_speedup', 'router_oracle_ratio', 'is_hit',
        ]
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(quality_rows)
        print(f"\nWrote {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
