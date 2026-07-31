# RA-SpMM: Regime-Aware Sparse Matrix Multiplication for GNN Workloads

RA-SpMM is the code and measurement artifact for *"RA-SpMM: Regime-Aware Sparse Matrix
Multiplication for Graph Neural Network Workloads on GPUs"* (under review at *Future
Generation Computer Systems*, Elsevier).

No single SpMM kernel is fastest across structurally diverse graphs. RA-SpMM extracts three
structural features from the CSR row-pointer array (row count `M`, average degree `d_bar`,
degree variation `CV_d`) in a single `O(M)` pass and, together with the feature width `N`,
dispatches each call to one of **five specialized GPU kernels** through an interpretable
**seven-rule decision tree**, with a cuSPARSE guardrail for the few inputs no specialist
improves. Two kernels run on native CSR with no format build; three tile kernels use an
`8x1` vector-block tensor-core layout with router-selected FP16 or TF32 precision paths.
A conversion-aware layer defers the tile build at cold start, so the first call dispatches
a zero-conversion CSR kernel and reuse recovers full tile throughput. Python and C++
routing decisions are asserted identical on every evaluation configuration.

## Kernel portfolio

| Kernel | Source | Strategy |
|---|---|---|
| `CSR_DIRECT` | `csr/csr_direct.cu` | Subwarp-per-row CSR (W-lane subwarps chosen from N); no per-matrix setup |
| `ZERO_OVERHEAD_CSR` | `csr/ra_zero_overhead.cu` | Degree-binned CSR dispatch (lightweight degree scan) |
| `TC_DIRECT` | `tc/ra_tc_direct.cu` | Natural-order 8x1 tile packing, `mma.m16n8k8`, FP16 and TF32-direct paths |
| `COMMUNITY_TC` | `tc/ra_community_tc.cu` | Deterministic locality-ordering sort (leading-neighbor) before tile packing |
| `SEGMENT_HYBRID` | `tc/ra_segment_hybrid.cu` | Balanced tile split for mixed row lengths |

The router (rules, precision selection, preprocessing-aware tile gate, cuSPARSE guardrail)
is implemented twice and kept in lockstep: `router/router_dispatch.cpp` (deployed C++) and
the Python mirror exercised by `ra_router_parity_test.py`. The remaining kernel sources in
`csr/` and `tc/` are earlier candidates kept for completeness; among them, the RoDe-derived
kernel (`csr/ra_rode_enhanced.cu`) is evaluated in the paper's portfolio ablation and
excluded from the deployed portfolio (zero leave-one-out contribution on both axes).

## Headline results (RTX 3090; 192 configurations = 51 graphs x N in {64, 128, 256, 512})

| Metric | Value |
|---|---|
| Warm geomean speedup over FP32 cuSPARSE | 1.702x (oracle 1.706x; Router/Oracle 0.9976) |
| Oracle hit rate | 173/192 |
| Precision-matched speedup | 1.222x |
| Cold first-call speedup (51 graphs, N=128) | 2.269x, with 0.17 ms mean routing overhead (tiled paths additionally pay a one-time format build) |
| GNN training step vs cuSPARSE backend | 1.37x/1.26x/1.26x warm geomean (GCN/SAGE/GIN), up to 3.24x on Reddit |
| RTX 4090 (Ada) transfer, router unchanged | 1.515x warm; 192/192 identical routing decisions |

Every number in the paper is backed by a CSV in this repository; `RESULTS.md` maps each
table and figure to its source file and states the exact aggregation recipe.

## Repository layout

```
csr/, tc/, staged/, graph/   CUDA kernels (CSR family, tile family, generators)
router/                      feature extraction, rule tree, dispatch (C++/CUDA)
bindings/                    PyTorch extension bindings (ra_bindings.cpp)
ra_*.py, dtc_*.py            evaluation drivers (sweeps, router eval, DTC harness)
gnn_bench/                   end-to-end GCN/GIN/GraphSAGE benchmarks
experiments/                 analysis scripts that produce the released CSVs
scripts/                     dataset fetch, baseline build scripts, timing patches
fgcs_results/revision/       all measurement CSVs behind the paper (see RESULTS.md)
```

## Build

Requires CUDA 11.8+ and a matching PyTorch. The paper's primary numbers were measured with
CUDA 11.8 on an RTX 3090 (SM 86); the cross-architecture sweep with CUDA 12.1 on an
RTX 4090 (SM 89).

```
pip install -r requirements-revision.txt
python setup.py install
```

## Datasets

The 51-graph benchmark suite (26 real-world, 25 synthetic) is hosted on Zenodo
(DOI [10.5281/zenodo.19903313](https://doi.org/10.5281/zenodo.19903313)). One step:

```
bash scripts/fetch_datasets.sh
```

See `DATASETS.md` for the archive checksum, per-graph attribution, and licenses.

## Reproducing the paper numbers

1. Router parity check (Python and C++ decisions must match on all configurations):
   `python ra_router_parity_test.py`
2. Full kernel sweep over the eight deployed kernel/precision paths (warm and cold
   timings under matched lifecycle accounting): `python ra_real_graph_eval.py`
   Add `--portfolio paper-audit` to include the retired RoDe-derived candidate, which
   the correctness tables report alongside the deployed paths.
3. Router quality over the sweep (oracle floored at cuSPARSE):
   `python ra_router_eval.py`
4. Integrated GNN training-step benchmark, all three architectures:
   `bash experiments/run_gnn_corrected_8x3.sh`
5. One-command check of the whole set: `bash scripts/reproduce_fgcs_v5.sh`

`RESULTS.md` documents, for every table and figure in the paper, the source CSV under
`fgcs_results/revision/` and the aggregation that reproduces the printed value.

Baseline harnesses (DTC-SpMM, HC-SpMM, MP-SpMM) build from their public repositories via
`scripts/build_*_baseline.sh`; `scripts/patches/` holds the fair-timing patches applied for
the head-to-head measurements described in the paper.

## Citation

See `CITATION.cff`. Please cite the FGCS paper when using this code or the benchmark data.

## License

MIT for the code in this repository. Dataset licenses are listed in `DATASETS.md`.
