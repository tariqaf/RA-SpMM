# RA-SpMM: Regime-Aware Sparse Matrix Multiplication for GNN Workloads

**RA-SpMM** is the artifact for *"Regime-Aware Sparse Matrix Multiplication for Graph Neural Network Workloads on GPUs"* (submitted to **Future Generation Computer Systems**, Elsevier, 2026).

RA-SpMM classifies sparse matrices into six structural categories using CSR-derived features and dispatches each SpMM call to one of six GPU kernels through an interpretable eight-rule router. Three paths operate directly on CSR; three build reusable tile or segment metadata. Python and C++ routing decisions are required to match on all 192 evaluation configurations.

## Corrected evaluation status

The revision evaluation uses symmetric lifecycle accounting:

- **Warm / steady-state:** build reusable state once, then measure execute-only with 50 warmup and 200 timed CUDA-event iterations.
- **Cold / first-call:** measure state or format construction and one execution, reporting `preprocess_ms`, `cold_exec_ms`, and `ms_cold` separately.
- **Matching denominators:** warm speedups compare against warm cuSPARSE; cold speedups compare against cold cuSPARSE.
- **Strict eligibility:** a row participates in performance, oracle, and router statistics only when `max_error <= tolerance` and `max_error < 1.0`. Soft and hard failures are retained as separate diagnostics.

Corrected numbers are generated from the merged strict-gated CSV by `experiments/summarize_fair_results.py`. They are not duplicated manually in this README. See `CHANGES.md` for the audit trail and optimization log.

## Six-kernel paper portfolio

| Kernel | Source | Strategy |
|---|---|---|
| `CSR_DIRECT` | `csr/csr_direct.cu` | Warp-per-row CSR baseline |
| `RODE_ENHANCED` | `csr/ra_rode_enhanced.cu` | Block-residual decomposition (extends RoDe) |
| `ZERO_OVERHEAD_CSR` | `csr/ra_zero_overhead.cu` | Degree-binned dispatch |
| `TC_DIRECT` | `tc/ra_tc_direct.cu` | Direct tile packing and FP32 tile execution |
| `COMMUNITY_TC` | `tc/ra_community_tc.cu` | Label-propagation clustering and FP32 tile execution |
| `SEGMENT_HYBRID` | `tc/ra_segment_hybrid.cu` | Row-level segmented/direct partitioning by column-span compactness |

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

The Python implementation lives in `ra_router_eval.py::simple_router()`; the C++ mirror is in `router/router_dispatch.cpp::make_router_plan()`. The parity test fails if fewer than 192 configurations load unless `--allow-partial` is explicitly supplied.

## Repository layout

```
RA-SpMM/
├── README.md                # this file
├── ra_common.h              # shared types and CUDA error-checking macros
├── setup.py                 # builds Python bindings via pybind11
├── csr/                     # CSR-based kernels (paper + legacy)
├── tc/                      # Tile-format kernels; measured corpus uses FP32 fallback
├── router/                  # 8-rule router (C++) + scoring + features
├── bindings/                # pybind11 bindings (ra_bindings.cpp)
├── bench/                   # SpMM benchmarking utilities
├── gnn_bench/               # GCN / GraphSAGE / GIN end-to-end runners
├── experiments/             # revision experiment suite (profiling, baselines, analysis)
├── graph/                   # CSR I/O + dataset loading
├── ra_router_eval.py        # Python router (must match C++ via parity test)
├── ra_router_parity_test.py # Python ≡ C++ parity verifier
├── ra_real_graph_eval.py    # fair warm/cold SpMM sweep entry point
├── ra_eval.py               # general-purpose evaluation harness
├── ra_eval_extended.py      # full 192-point combined sweep harness
├── pyg_baseline.py          # PyG torch_sparse baseline runner
├── dtc_baseline.py          # DTC-SpMM baseline runner (third-party)
├── paper_datasets.json      # 26 real-graph manifest with categories
├── requirements-revision.txt # analysis-only dependencies
└── fgcs_results/revision/fair/ # corrected CSVs, summaries, and profiles
```

## Build

### Prerequisites

- NVIDIA Ampere GPU (SM_86; tested on RTX 3090 and RTX A6000)
- CUDA Toolkit compatible with the installed PyTorch build
- cuSPARSE from that CUDA toolkit
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
pip install -r requirements-revision.txt

python setup.py build_ext --inplace

# verify Python <-> C++ router parity (required after any rule edit)
python ra_router_parity_test.py
```

For binary PyG extensions, use the wheel index matching the installed PyTorch
and CUDA versions. This server uses:

```bash
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

The DTC baseline is kept outside Git because it is third-party code:

```bash
git clone --recursive https://github.com/HPMLL/DTC-SpMM_ASPLOS24.git \
    external/DTC-SpMM_ASPLOS24
bash scripts/build_dtc_baseline.sh
```

`ra_external_baselines.py` defaults to the reproducible, explicitly labeled
`DTC_IDENTITY_ORDER` mode. `--dtc-reorder tca` is available only after adapting
the upstream TCA script and installing its RAPIDS/MinHash dependencies; the
public upstream script otherwise fails loudly rather than silently changing the method.

MP-SpMM is pinned to its final SC'25 Zenodo artifact. The build helper verifies
the archive checksum, applies the tracked fair-timing and correctness patches,
and builds the SM86 preprocessing and execution binaries:

```bash
CUDA_ARCH=86 bash scripts/build_mpspmm_baseline.sh
```

The upstream archive remains outside Git; only the adapter patches and measured
result tables are part of this repository.

HC-SpMM is likewise pinned and built in place:

```bash
TORCH_CUDA_ARCH_LIST=8.6 bash scripts/build_hcspmm_baseline.sh
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

### 1. Verify strict correctness and parity

```bash
python ra_real_graph_eval.py --datasets-file fgcs_results/paper_combined_datasets.json \
    --correctness-only
python ra_router_parity_test.py --manifest fgcs_results/paper_combined_datasets.json \
    --expected 192
```
Expected parity output is `PARITY OK 192/192`. Either command exits unsuccessfully when its gate fails.

### 2. Smoke test on 3 graphs (~5 minutes)

```bash
python ra_real_graph_eval.py --datasets ogbn-arxiv,Reddit,Cora --N 128 \
    --datasets-file fgcs_results/paper_combined_datasets.json \
    --output /tmp/smoke.csv --warmup 50 --timed 200 --cold-iters 10
```

### 3. Full 192-configuration fair sweep

```bash
python ra_real_graph_eval.py --datasets-file fgcs_results/paper_combined_datasets.json \
    --output fgcs_results/revision/fair/fair_sweep.csv \
    --warmup 50 --timed 200 --cold-iters 10
python experiments/summarize_fair_results.py \
    --sweep fgcs_results/revision/fair/fair_sweep.csv \
    --outdir fgcs_results/revision/fair/summary
```

### 4. Router quality from any kernel-timing CSV (~1 second)

```bash
python ra_router_eval.py --csv fgcs_results/revision/fair/fair_sweep.csv \
    --regime warm --output fgcs_results/revision/fair/router_quality_warm.csv
python ra_router_eval.py --csv fgcs_results/revision/fair/fair_sweep.csv \
    --regime cold --output fgcs_results/revision/fair/router_quality_cold.csv
```

### 5. End-to-end GNN training (~30 minutes for the 8-dataset benchmark)

```bash
cd gnn_bench
python router_vs_baselines_gcn.py --datasets Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer --results_dir fgcs_results/revision/fair/gnn
python router_vs_baselines_sage.py --datasets Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer --results_dir fgcs_results/revision/fair/gnn
python router_vs_baselines_gin.py --datasets Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer --results_dir fgcs_results/revision/fair/gnn
```

### 6. Revision experiments (`experiments/`)

| Script | Produces |
|---|---|
| `profile_ncu.py` / `profile_parse.py` | Warm execute-only Nsight Compute metrics and full warp-stall breakdown |
| `bench_hcspmm*.py`, `bench_mpspmm*.py` | HC-SpMM / MP-SpMM warm, conversion, first-call, and strict correctness results |
| `bench_cublas_dense.py`, `bench_dense_gemm_rule.py` | dense cuBLAS probe + experiment-only DENSE_GEMM rule evaluation |
| `time_feature_extraction.py`, `time_conversion_pipeline.py` | production full-feature timing and strict fair-sweep feature/conversion/compute breakdown |
| `conversion_aware_routing.py` | offline per-graph amortization analysis with missing-data rejection |
| `fit_runtime_cost_model.py`, `ra_runtime_router.py` | deployed call-count-aware policy without online candidate benchmarking |
| `generate_learned_selector.py` | graph-grouped learned-selector comparison |
| `generate_extended_ablation.py` | feature-mask and leave-one-rule-out ablations |
| `verify_format_checksums.py` | byte-identity check of the parallel format build |
| `cross_arch/` | RTX 4090 sweep + FlashSparse comparison drivers |

FlashSparse versions expose different Python APIs. Its driver accepts an explicit
`--adapter module:function`; the function must return `(reusable_state, run_fn)`.
The adapter contract lets the 4090 host provide its commit-specific integration
without editing the fair timing and correctness harness.

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
