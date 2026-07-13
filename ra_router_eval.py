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
    Python mirror of the production six-kernel router (recalibrated on the
    fair sweep v2 after the ME-BCRS/subwarp kernel redesign).

    Eight rules, evaluated top-to-bottom; first match wins. Default
    fallthrough is TC_DIRECT. Features: avg_nnz_per_row (d), population
    degree CV (cv), M, N, nnz.

    Preprocessing-aware tile gate: the ME-BCRS plan build measures about
    20 ms per 1e6 nnz on the reference host, so tile kernels are withheld
    whenever the estimated build exceeds the 20 s amortization budget
    (or N < 16, where the mma path is infeasible); such configs fall back
    to a CSR kernel chosen by skew.
    """
    disabled = frozenset(disabled_rules)
    d = avg_nnz
    cv = degree_cv

    tile_ok = (N >= 16) and (float(nnz) * 2e-8 <= 20.0)

    def tile(kernel):
        if tile_ok:
            return kernel
        return "CSR_DIRECT" if cv < 1.5 else "ZERO_OVERHEAD_CSR"

    # 1. Sub-tiny graphs (Cora, CiteSeer, PPI): launch-bound; the dense
    #    fully-resident window plan wins across N.
    if 1 not in disabled and M < 5000:
        return tile("TC_DIRECT")

    # 2. Extreme-skew sparse tails (com-youtube, cv ~ 9.7): the binned CSR
    #    kernel with flattened hub chunks dominates every tile variant.
    if 2 not in disabled and cv >= 5.0:
        return "ZERO_OVERHEAD_CSR"

    # 3. Uniform / near-uniform family (road networks, synthetic uniform,
    #    synthetic communities, uniform dense-small).
    if 3 not in disabled and cv < 0.7:
        if d < 4.5:
            return "CSR_DIRECT"          # roadNet-*, uniform d=3
        if cv < 0.2 and N <= 64 and d >= 25.0:
            return "CSR_DIRECT"          # uniform dense-small at narrow N
        if cv >= 0.2 and d < 7.0:
            return "CSR_DIRECT"          # synth_community_nc*
        return tile("TC_DIRECT")

    # 4. Very skewed (cv >= 3): Yelp-class large -> COMMUNITY_TC,
    #    Flickr-class medium -> balance-split SEGMENT_HYBRID.
    if 4 not in disabled and cv >= 3.0:
        return tile("COMMUNITY_TC") if M >= 250000 else tile("SEGMENT_HYBRID")

    # 5. Dense rows (Reddit, ogbn-proteins, gplus, high-d skewed synth):
    #    locality-ordered windows maximize vector reuse.
    if 5 not in disabled and d >= 40.0:
        return tile("COMMUNITY_TC")

    # 6. Small dense (amazon-photo/computers): cuSPARSE at narrow N,
    #    balance-split at mid N, locality windows at wide N.
    if 6 not in disabled and M < 20000 and d >= 25.0:
        if N < 96:
            return "CUSPARSE"
        if N < 384:
            return tile("SEGMENT_HYBRID")
        return tile("COMMUNITY_TC")

    # 7. Heavy mixed (synth_mixed_v2..v5: d >= 25, cv >= 1.8).
    if 7 not in disabled and d >= 25.0 and cv >= 1.8:
        return tile("COMMUNITY_TC")

    # 8. Web-locality sparse (web-Google, web-Stanford): large M, low d,
    #    moderate cv.
    if 8 not in disabled and d < 9.0 and 1.10 <= cv <= 1.45 and M >= 250000:
        return tile("COMMUNITY_TC")

    # Fallthrough: TC_DIRECT (arxiv, Amazon0601, com-DBLP/Amazon, Pokec,
    # twitter, mid-degree skewed/mixed synthetics, ...).
    return tile("TC_DIRECT")


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

        # Router: what would our router pick? (Rule 6 may pick CUSPARSE.)
        router_kernel = simple_router(avg_nnz, degree_cv, M, N, nnz)
        router_ms = (cusparse_ms if router_kernel == "CUSPARSE"
                     else kernel_times[router_kernel])
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
