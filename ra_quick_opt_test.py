"""
ra_quick_opt_test.py - Quick A/B test for OPT-2 + OPT-3 optimizations

Tests ONLY CSR_DIRECT and RODE_ENHANCED on 6 representative graphs (one per category)
at N=64 and N=512 (small N + large N to check N-scaling improvement).

Compares against cuSPARSE reference. Fast enough to run in ~2 minutes.

Usage:
    python ra_quick_opt_test.py
"""
import sys, math, torch, json, os
import numpy as np

try:
    import ra_spmm
except ImportError:
    print("ERROR: ra_spmm not found. Build first: python setup.py build_ext --inplace")
    sys.exit(1)

WARMUP = 30
ITERS = 100
Ns = [64, 512]  # Small N + Large N to check scaling

# 6 representative graphs — one per category, chosen for diversity
TARGET_GRAPHS = [
    "roadNet-PA",       # Sparse Uniform (M=1.09M, avg_deg=2.8)
    "Amazon0601",       # Sparse Uniform (M=403K, avg_deg=8.4)
    "twitter-combined",  # Sparse Skewed (M=81K, avg_deg=30)
    "ca-CondMat",       # Dense Small (M=23K, avg_deg=16)
    "com-DBLP",         # Community (M=317K, avg_deg=6.6)
    "Flickr",           # Mixed/Irregular (M=89K, avg_deg=10)
]

KERNELS = ["CSR_DIRECT", "RODE_ENHANCED", "TC_DIRECT", "CUSPARSE"]


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    if 'indptr' in data:
        rowptr = torch.tensor(data['indptr'].astype(np.int32), dtype=torch.int32)
        colind = torch.tensor(data['indices'].astype(np.int32), dtype=torch.int32)
        vals = torch.ones(len(colind), dtype=torch.float32)
    elif 'rowptr' in data:
        rowptr = torch.tensor(data['rowptr'], dtype=torch.int32)
        colind = torch.tensor(data['colind'], dtype=torch.int32)
        vals = torch.ones(len(colind), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown NPZ: {list(data.keys())}")
    M = int(rowptr.shape[0] - 1)
    return rowptr, colind, vals, M


def load_edge(path, symmetrize=False, one_indexed=False):
    edges = []
    max_node = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                s, d = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if one_indexed:
                s -= 1; d -= 1
            edges.append((s, d))
            max_node = max(max_node, s, d)
            if symmetrize and s != d:
                edges.append((d, s))
    n = max_node + 1
    row_counts = [0] * n
    for s, d in edges:
        if s < n: row_counts[s] += 1
    rowptr = [0]
    for i in range(n):
        rowptr.append(rowptr[-1] + row_counts[i])
    colind_arr = [0] * rowptr[n]
    cursor = list(rowptr[:-1])
    for s, d in edges:
        if s < n and cursor[s] < rowptr[s + 1]:
            colind_arr[cursor[s]] = min(d, n - 1)
            cursor[s] += 1
    # Dedup per row
    new_colind, new_rowptr = [], [0]
    for i in range(n):
        seg = sorted(set(colind_arr[rowptr[i]:rowptr[i+1]]))
        new_colind.extend(seg)
        new_rowptr.append(len(new_colind))
    return (torch.tensor(new_rowptr, dtype=torch.int32),
            torch.tensor(new_colind, dtype=torch.int32),
            torch.ones(len(new_colind), dtype=torch.float32), n)


def measure_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iters


def run_kernel(name, rowptr, colind, vals, B, cache, key):
    M = rowptr.shape[0] - 1
    N = B.shape[1]
    if name == "CSR_DIRECT":
        return ra_spmm.spmm_csr_direct(rowptr, colind, vals, B)
    elif name == "CUSPARSE":
        return ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
    elif name == "RODE_ENHANCED":
        if key not in cache:
            cache[key] = ra_spmm.make_rode_enhanced_plan(rowptr.cpu(), M, M)
        return ra_spmm.run_rode_enhanced_plan(cache[key], colind, vals, B)
    elif name == "TC_DIRECT":
        if key not in cache:
            cache[key] = ra_spmm.make_tc_direct_plan(rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, N)
        return ra_spmm.run_tc_direct_plan(cache[key], B)


def main():
    print("=" * 70)
    print("Quick OPT-2 + OPT-3 Validation Test")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernels: {KERNELS}")
    print(f"N values: {Ns}")
    print(f"Graphs: {TARGET_GRAPHS}")
    print("=" * 70)

    with open("paper_datasets.json") as f:
        manifest = json.load(f)
    ds_map = {d["name"]: d for d in manifest["datasets"]}

    results = []
    for gname in TARGET_GRAPHS:
        if gname not in ds_map:
            print(f"\n[SKIP] {gname}: not in manifest")
            continue
        entry = ds_map[gname]
        path = entry["path"]
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(__file__), entry["path"])
        if not os.path.exists(path):
            print(f"\n[SKIP] {gname}: file not found")
            continue

        fmt = entry.get("format", "edge")
        if fmt == "npz":
            rowptr, colind, vals, M = load_npz(path)
        else:
            rowptr, colind, vals, M = load_edge(
                path, symmetrize=entry.get("symmetrize", False),
                one_indexed=entry.get("one_indexed", False))

        rowptr, colind, vals = rowptr.cuda(), colind.cuda(), vals.cuda()
        nnz = int(rowptr[-1].item())
        cat = entry.get("category", "?")
        print(f"\n[{cat}] {gname}: M={M}, nnz={nnz}, avg_deg={nnz/max(1,M):.1f}")

        cache = {}
        for N in Ns:
            max_N = entry.get("max_N", 512)
            if N > max_N: continue
            B = torch.randn(M, N, device="cuda")

            # Correctness check vs cuSPARSE
            C_ref = run_kernel("CUSPARSE", rowptr, colind, vals, B, cache, "cusp")
            ms_cusp = measure_ms(lambda: run_kernel("CUSPARSE", rowptr, colind, vals, B, cache, "cusp"))

            for kname in ["CSR_DIRECT", "RODE_ENHANCED", "TC_DIRECT"]:
                ckey = f"{kname}_{N}"
                C_test = run_kernel(kname, rowptr, colind, vals, B, cache, ckey)
                max_err = (C_test - C_ref).abs().max().item()
                correct = max_err < 1.0  # hard fail threshold

                ms_val = measure_ms(lambda: run_kernel(kname, rowptr, colind, vals, B, cache, ckey))
                speedup = ms_cusp / ms_val if ms_val > 0 else 0

                status = "PASS" if correct else "FAIL"
                print(f"  N={N:3d} {kname:20s}: {ms_val:.3f}ms  "
                      f"({speedup:.2f}x vs cuSPARSE, err={max_err:.6f}) [{status}]")
                results.append((gname, cat, N, kname, ms_val, ms_cusp, speedup, correct))

            del B; torch.cuda.empty_cache()
        del cache; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Geomean speedup vs cuSPARSE")
    print("=" * 70)
    for kname in ["CSR_DIRECT", "RODE_ENHANCED", "TC_DIRECT"]:
        logs_64 = [math.log(r[6]) for r in results if r[3] == kname and r[2] == 64 and r[6] > 0]
        logs_512 = [math.log(r[6]) for r in results if r[3] == kname and r[2] == 512 and r[6] > 0]
        gm_64 = math.exp(sum(logs_64)/len(logs_64)) if logs_64 else 0
        gm_512 = math.exp(sum(logs_512)/len(logs_512)) if logs_512 else 0
        logs_all = [math.log(r[6]) for r in results if r[3] == kname and r[6] > 0]
        gm_all = math.exp(sum(logs_all)/len(logs_all)) if logs_all else 0
        print(f"  {kname:20s}: N=64 {gm_64:.3f}x | N=512 {gm_512:.3f}x | overall {gm_all:.3f}x")

    # Check for regressions
    fails = [r for r in results if not r[7]]
    if fails:
        print(f"\n⚠️  {len(fails)} CORRECTNESS FAILURES!")
        for f in fails:
            print(f"  {f[0]} N={f[2]} {f[3]}")
    else:
        print("\n✅ All correctness checks PASS")

    print("\nDone. Compare N=64 vs N=512 geomean to assess N-scaling improvement.")


if __name__ == "__main__":
    main()
