# RA-SpMM: Regime-Aware Sparse Matrix Multiplication for GNN Workloads

**RA-SpMM** is the artifact for *"Regime-Aware Sparse Matrix Multiplication for Graph Neural Network Workloads on GPUs"* (submitted to **Future Generation Computer Systems**, Elsevier, 2026).

RA-SpMM classifies sparse matrices into **six structural categories** using three CSR-derivable features (matrix size *M*, average row density *d̄*, and degree coefficient of variation CV<sub>d</sub>) and dispatches each SpMM call to one of six purpose-built, **preprocessing-free** GPU kernels via an interpretable 8-rule router. Python and C++ router implementations are validated to produce identical kernel choices on all 192 evaluation points.

## Headline results (RTX 3090, CUDA 12.x)

- **3.25× geomean speedup over cuSPARSE** across 26 real-world graphs (92 evaluation points)
- **99.0% of oracle performance** (Router/Oracle = 0.990× geomean)
- **91 of 92 real-graph points satisfy the bounded-regret target ≥ 0.85×**
- **Plan-phase wall-clock: 10.9 ms mean** (vs DTC-SpMM's 38.5 s mean autotuning — a ~3,500× setup-cost reduction)
- **End-to-end GNN training**: 2.55× / 1.71× / 1.55× geomean over cuSPARSE on GCN / GraphSAGE / GIN across 8 datasets
- **Cross-SKU validation**: 35 of 35 routing decisions transfer identically from RTX 3090 to RTX A6000
- **Synthetic stress-test**: combined 192-point suite (26 real + 25 synthetic) holds 0.995× Router/Oracle ratio with 86.5% oracle hit rate

## Six-kernel paper portfolio

| Kernel | Source | Strategy |
|---|---|---|
| `CSR_DIRECT` | `csr/csr_direct.cu` | Warp-per-row CSR baseline |
| `RODE_ENHANCED` | `csr/ra_rode_enhanced.cu` | Block-residual decomposition (extends RoDe) |
| `ZERO_OVERHEAD_CSR` | `csr/ra_zero_overhead.cu` | Degree-binned dispatch |
| `TC_DIRECT` | `tc/ra_tc_direct.cu` | Single-pass Tensor Core execution, 16×16×16 WMMA |
| `COMMUNITY_TC` | `tc/ra_community_tc.cu` | Label-propagation clustering + TC tile alignment |
| `SEGMENT_HYBRID` | `tc/ra_segment_hybrid.cu` | Row-level TC/CUDA partitioning by column-span compactness |

Plus `cuSPARSE` as the vendor baseline (dispatched only when no custom kernel dominates).

Legacy / ablation kernels (kept for reproducibility, not in the paper portfolio): `csr/csr_adaptive.cu`, `csr/ra_vectorized_coarse.cu`, `csr/row_split.cu`, `tc/hybrid_tc_cuda.cu`, `tc/ra_locality_tiled.cu`, `tc/tc_reordered.cu`, `tc/tc_sparse.cu`.

## Eight-rule router (Algorithm 1 in the paper)

The router evaluates rules top-to-bottom; first match wins. Default fall-through is `TC_DIRECT`.

```
Inputs: M, d̄ (= avg_nnz_per_row), CV (= degree_cv), N (= dense feature dim)

Rule 1 (sub-tiny):              M < 5,000
                                  → SEGMENT_HYBRID if N>=256 and (d>=12 or d<=6)
                                  → TC_DIRECT otherwise
Rule 2 (sparse-tail skewed):    M>=100K and d<8 and CV>4
                                  → RODE_ENHANCED if N>=256, else TC_DIRECT
Rule 3 (dense-small):           M<=15K and d>=25
                                  → SEGMENT_HYBRID if CV>=1, else COMMUNITY_TC
Rule 4 (skewed mid-degree):     12<=d<=40 and CV>=1.5
                                  → RODE / CSR_DIRECT depending on M and N
Rule 5 (dense-large):           d>=96
                                  → RODE_ENHANCED if CV>=2.5 and N>=256, else TC_DIRECT
Rule 6 (huge mid-density):      M>=1M and 40<=d<96 and CV<=2.5
                                  → COMMUNITY_TC
Rule 7 (Flickr-class):          50K<=M<=150K and 9<=d<=12
                                  → ZERO_OVERHEAD_CSR
Rule 8 (community sweet-spot):  three OR branches over (M, d, CV, N)
                                  → COMMUNITY_TC
Default:                        → TC_DIRECT
```

The Python implementation lives in `ra_router_eval.py::simple_router()`; the C++ mirror is in `router/router_dispatch.cpp::make_router_plan()`. Run `python ra_router_parity_test.py` to verify the two implementations agree on all 192 evaluation points (parity is required for any rule changes).

## Repository layout

```
RA-SpMM/
├── README.md                # this file
├── ra_common.h              # shared types and CUDA error-checking macros
├── setup.py                 # builds Python bindings via pybind11
├── csr/                     # CSR-based kernels (paper + legacy)
├── tc/                      # Tensor Core kernels (paper + legacy)
├── router/                  # 8-rule router (C++) + scoring + features
├── bindings/                # pybind11 bindings (ra_bindings.cpp)
├── bench/                   # SpMM benchmarking utilities
├── gnn_bench/               # GCN / GraphSAGE / GIN end-to-end runners
├── graph/                   # CSR I/O + dataset loading
├── ra_router_eval.py        # Python router (must match C++ via parity test)
├── ra_router_parity_test.py # Python ≡ C++ parity verifier
├── ra_real_graph_eval.py    # 92-point real-graph SpMM sweep entry point
├── ra_eval.py               # general-purpose evaluation harness
├── ra_eval_extended.py      # full 192-point combined sweep harness
├── pyg_baseline.py          # PyG torch_sparse baseline runner
├── dtc_baseline.py          # DTC-SpMM baseline runner (third-party)
├── paper_datasets.json      # 26 real-graph manifest with categories
└── results/                 # all measurement CSVs from the paper
    ├── spmm/all_graphs_results.csv          # 1344 rows: 192 (graph,N) × 7 kernels
    ├── router/router_quality.csv            # 192 rows: per-point router vs oracle
    ├── router/feature_extraction_times.csv  # plan-phase wall-clock per graph
    ├── ablation/router_ablation.csv         # 11 rows: full + 8 rule-removals + 2 feat-counts
    ├── ablation/router_ablation_real.csv    # same on 92 real-only points
    ├── dtc/dtc_speedup.csv                  # 92 rows: DTC vs cuSPARSE on 26 real graphs
    ├── dtc/dtc_autotuning_times.csv         # variant-scan wall-clock per (graph,N)
    ├── dtc/dtc_subset_analysis.csv          # 3 subsets × per-category breakdown
    ├── gnn_e2e/{gcn,graphsage,gin}_end_to_end.csv  # 9 datasets × 4 backends each
    ├── cross_gpu/                           # 10-graph A6000 portfolio + router + baselines
    └── POST_SWEEP_REPORT.md                 # narrative summary of headline numbers
```

## Build

### Prerequisites

- NVIDIA Ampere GPU (SM_86; tested on RTX 3090 and RTX A6000)
- CUDA Toolkit 12.x
- cuSPARSE 12.x (bundled with CUDA)
- PyTorch 2.x (for GNN end-to-end runs and bindings)
- NVIDIA driver 525 or newer
- Python 3.10+
- gcc/g++ 11+ (for nvcc host compilation)
- PyTorch Geometric >= 2.4 (for GNN end-to-end and dataset loaders)

### Install

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-sparse torch-scatter
pip install pybind11 pandas scipy numpy

cd bindings
python setup.py install        # builds the C++ kernel bindings
cd ..

# verify Python <-> C++ router parity (required after any rule edit)
python ra_router_parity_test.py
```

## Datasets

The 26 real-world graphs and 25 procedurally-generated synthetic graphs used in the paper are archived on Zenodo as a single tarball (~1.9 GB compressed; ~6.9 GB extracted):

- **DOI**: `10.5281/zenodo.19903312`
- **URL**: `https://zenodo.org/records/19903313`
- **Archive**: `ra_spmm_data_v1.tar.gz`

After cloning this repository:

```bash
# Option A: one-line helper (Linux/macOS)
ZENODO_RECORD=19903313 bash scripts/fetch_datasets.sh

# Option B: cross-platform Python helper
ZENODO_RECORD=19903313 python scripts/fetch_datasets.py

# Option C: manual
wget https://zenodo.org/records/19903313/files/ra_spmm_data_v1.tar.gz
tar -xzf ra_spmm_data_v1.tar.gz --strip-components=1
```

The tarball's internal layout mirrors this repository's expected paths, so `--strip-components=1` drops the top-level wrapper and files land under `./datasets/` and `./fgcs_results/synthetic/` exactly where `paper_datasets.json` and `paper_combined_datasets.json` resolve them.

For attribution to the original SNAP / OGB / PyG dataset authors and for a smaller-subset reproduction option, see [`DATASETS.md`](DATASETS.md).

## Reproducing the paper

### 1. Verify router parity (~5 seconds)

```bash
python ra_router_parity_test.py
```
Expected output: `PARITY OK 192/192`. This confirms that the 8-rule router in `ra_router_eval.py` is a faithful mirror of the C++ implementation.

### 2. Smoke test on 3 graphs (~5 minutes)

```bash
python ra_real_graph_eval.py --datasets ogbn-arxiv,Reddit,Cora --N 128 \
    --output /tmp/smoke.csv --warmup 50 --timed 200
```

### 3. Full 92-point real-graph sweep (~3 hours on a single RTX 3090)

```bash
python ra_real_graph_eval.py --datasets-json paper_datasets.json \
    --N 64,128,256,512 --output results/spmm/all_graphs_results.csv \
    --warmup 50 --timed 200
```

### 4. Router quality from any kernel-timing CSV (~1 second)

```bash
python ra_router_eval.py --csv results/spmm/all_graphs_results.csv \
    --output results/router/router_quality.csv
```

### 5. End-to-end GNN training (~30 minutes for the 8-dataset benchmark)

```bash
cd gnn_bench
python router_vs_baselines_gcn.py --datasets Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer
python router_vs_baselines_sage.py --datasets Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer
python router_vs_baselines_gin.py --datasets Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer
```

## Citation

If you use this artifact, please cite:

```bibtex
@article{afridi2026raspmm,
  title={Regime-Aware Sparse Matrix Multiplication for Graph Neural Network Workloads on GPUs},
  author={Afridi, Tariq Habib and Lee, Young-Koo},
  journal={Future Generation Computer Systems},
  year={2026},
  publisher={Elsevier}
}
```

## License

- **Code**: MIT License (see `LICENSE`).
- **Measurement results** in `results/`: CC-BY 4.0.

## Contact

Open an issue or contact `afridi@khu.ac.kr`.
