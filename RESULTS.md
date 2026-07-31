# RA-SpMM Results Guide

This document maps every quantitative table and figure in the paper to its source CSV in
this repository and states the aggregation that reproduces the printed value, so the
results can be checked independently. All paths are relative to `fgcs_results/revision/`.
"Geomean" is always the geometric mean over finite positive values.

## Measurement environment

- Primary GPU: NVIDIA GeForce RTX 3090 (Ampere, SM 86), CUDA 11.8.
- Cross-architecture GPU: NVIDIA GeForce RTX 4090 (Ada, SM 89), CUDA 12.1.
- Suite: 51 graphs (26 real, 25 synthetic) x N in {64, 128, 256, 512} = 192 configurations.
- Warm protocol: reusable state built once outside the timed region (cuSPARSE handle,
  descriptors, workspace, and `cusparseSpMM_preprocess` included), then 50 warmup + 200
  timed CUDA-event iterations, execute-only, both sides. cuSPARSE runs
  `CUSPARSE_SPMM_CSR_ALG2` as a fixed compile-time choice.
- Cold protocol: uninitialized per-matrix state through the first call, setup included,
  both sides (`preprocess_ms`, `cold_exec_ms`, `ms_cold` recorded separately).
- Correctness gate: a row enters performance statistics only when
  `max_error <= tolerance AND max_error < 1.0`. The global 1.0 cap is the stricter of
  the two on 48 of the 1,728 custom-path outputs -- all on the mixed-precision tile
  paths, where the 10x relaxation lifts the scaled tolerance past 1.0 -- and it
  rejected none of them. The largest deviation observed anywhere is 0.125.

## Definitions used by the tables

- **Oracle**: per configuration, the best option among the eight deployed kernel/precision
  paths and cuSPARSE itself. In `router_quality_v8.csv` the cuSPARSE floor is already
  applied, so `oracle_speedup` is the final oracle and never falls below 1.0; no
  post-processing is needed. `oracle_kernel` names the winning option and may be
  `CUSPARSE`, while `oracle_custom_kernel` names the best custom path regardless.
  Exactly one configuration is decided by the floor: amazon-photo at N=64, whose best
  custom path reaches 0.982 and where cuSPARSE is therefore the oracle.
- **Oracle hit**: `is_hit` is the deployed-oracle definition -- the router matches the
  floored oracle (173 of 192). `is_hit_kernelonly` is the stricter custom-path match
  (172 of 192). The two differ only on amazon-photo at N=64, where the guardrail
  correctly selects cuSPARSE.
- **Precision-matched speedup**: FP16-path router picks compared against FP16 cuSPARSE
  (`speedup_precision_matched_warm`), TF32/FP32 picks against FP32 cuSPARSE
  (`speedup_vs_cusparse_warm`), guardrail picks counted as 1.0.
- **Cold trio (deployed / CSR-kernel / oracle ceiling)**: deployed = component-wise
  single-call accounting -- the measured feature-and-route pass plus the first
  `CSR_DIRECT` execution, summed, against cuSPARSE descriptor and workspace
  construction plus its first execution (`experiments/deployed_coldstart.py`); CSR-kernel = `CSR_DIRECT`
  `ms_cusparse_cold / ms_cold`; ceiling = best cold speedup over the preprocessing-free
  CSR-family kernels (`CSR_DIRECT`, `ZERO_OVERHEAD_CSR`), no guardrail floor.

## Table-by-table map

| Paper item | Source CSV | Recipe |
|---|---|---|
| Router quality per category (router 1.702x, oracle 1.706x, hits 173/192, kernel-only 172/192, worst 0.868x) | `tf32/router_quality_v8.csv` | geomean of `router_speedup` and of `oracle_speedup` per `category` (the floor is already in the file); hits = `is_hit`; kernel-only = `is_hit_kernelonly`; worst = min `router_oracle_ratio` |
| Superseded pre-floor router-quality file | `tf32/superseded/router_quality_v5.csv` | retained for provenance only: it predates the cuSPARSE-floored oracle and lacks `oracle_custom_kernel` and `is_hit_kernelonly` |
| System comparison (PyG column, 0.417x overall; router 4.09x over PyG) | `fair/pyg_dtc.csv` | geomean `pyg_speedup_vs_cusparse_warm`; ratio vs `router_speedup` joined on (dataset, N) |
| Per-kernel comparison (TC_DIRECT 1.525x overall, etc.) | `tf32/final_fair_v3.csv` | geomean `speedup_vs_cusparse_warm` of each kernel's base-path rows (`TC_DIRECT`, `COMMUNITY_TC`, `SEGMENT_HYBRID` = FP16 tile path; TF32 variants are separate rows selected by the router's precision rule) |
| Precision-matched 1.222x (3090) / 1.053x (Ada) | `tf32/final_fair_v3.csv` + `tf32/router_quality_v8.csv`; `ada5/fair_sweep_ada.csv` + `ada5/router_quality_ada_warm.csv` | pairing rule above over the 192 router picks |
| Microarchitectural profiling table | `tf32/ncu_master_final.csv` | `mma`-launch rows at N=128 (`tensor_pct`, `hmma`, `dram_pct`, `l2_hit`, `occ_ach`, `elig_warps`); full `.ncu-rep` sessions summarized here |
| Roofline figure (supplementary) | `tf32/ncu_roofline_final.csv` | plotted directly |
| Cold start 2.269x + per-category + 0.17/0.15/1.05 ms plan cost | `tf32/deployed_coldstart.csv` | geomean `deployed_cold_speedup` (51 graphs, N=128); mean/median/max `feature_route_ms` |
| Cold trio 2.27/2.98/3.03 (3090) | `tf32/deployed_coldstart.csv`; `tf32/final_fair_v3.csv` | constructions above, all 192 configurations for the CSR-kernel and ceiling terms |
| Conversion/build cost per kernel (1 ms to 36 ms geomean; worst 1.5 s) | `tf32/final_fair_v3.csv` | geomean over positive `preprocess_ms` per kernel; max per kernel |
| Break-even reuse counts | `fair/conversion_aware/crossover_K.csv` | per (graph, N) `crossover_K_strict` |
| Plan-build model (about 20 ms per 1e6 nnz) | `fair/conversion_times.csv` | linear fit of tile-plan build time vs nnz; deployed constant in `router/router_dispatch.cpp` |
| DTC-SpMM comparison (1.598x / 1.773x on the 50-config common subset; coverage 79 and 50 of 92) | `tf32/dtc_tca_full.csv`, `tf32/dtc_identity_full.csv`, router times from `tf32/router_quality_v8.csv` | correct rows with timing; common subset intersection; geomean `mean_kernel_ms` ratios |
| DTC autotune cost (38.5 s mean, 736.8 s max at Reddit N=256, best-of-six) | `dtc_autotuning_times.csv` | mean/max `autotune_seconds` |
| DTC TCA reorder cost (250 s mean, 43 s median, 912.6 s max) | `tf32/dtc_tca_full.csv` | per-dataset `reorder_ms` |
| Integrated GNN training step, warm (GCN 1.37x, GraphSAGE 1.26x, GIN 1.26x geomean over 8 datasets) | `tf32/gnn_corrected/gnn_warm_{gcn,graphsage,gin}.csv` | geomean of `router_vs_cusparse`; PyG column from `pyg_vs_cusparse`; per-model aggregation operators, per-site resolved precision paths, and protocol in `gnn_corrected/gnn_manifest.json`. The superseded pre-correction run is retained at `tf32/gnn_v5_full/` for comparison |
| Integrated GNN training step, graph-cold first step (2.54x / 1.48x / 1.43x geomean, median of 25 trials) | `tf32/gnn_corrected/gnn_cold_{gcn,graphsage,gin}.csv` | per dataset, median(`cusparse_ms`) / median(`router_ms`); then geomean across the 8 datasets (geomean of raw medians, not of rounded values) |
| Frozen-router validation on 9 unused graphs (1.197x router, 0.897 Router/Oracle, 11/34 hits) | `tf32/unseen/unseen_router_quality.csv` | exclude the 2 rows with `router_speedup` empty (wiki-Talk and cit-Patents at N=512, where every deployed kernel exhausted memory), then geomean over the remaining 34 |
| Reuse break-even summary (median 388 calls; 18 for SEGMENT_HYBRID, 1140 for COMMUNITY_TC; 9 never amortize) | `fair/conversion_aware/crossover_K.csv` | `crossover_K_strict` grouped by `best_tc_kernel`; the literal `never` marks configs whose tiled warm time does not beat the CSR alternative |
| cuBLAS probe (18.5x geomean warm; 8 cold wins on PPI/CiteSeer) | `tf32/cublas_dense_v3.csv` | `1/cublas_vs_router_warm`; `cublas_vs_cusparse_cold > 1` rows |
| FlashSparse head-to-head (FS 1.28x over 190 common configs, FS faster 134 / router 56; per-regime splits; Reddit router 1.67x) | `ada5/flashsparse_ada_best.csv` + `ada5/router_quality_ada_warm.csv` | FS time = `min(ms_16x1, ms_8x1)` per row (the `8x1_balance` variant produces incorrect output in this usage and is excluded; the 8x1 path is nsys-verified and correct on all graphs); RA time = `router_ms`; both sides 50 warmup + 200 timed iterations in the same process. Aggregate = geomean `ra_ms/fs_ms` over the 190 (dataset, N) pairs both systems complete |
| FlashSparse block-build cost (about 80 ms, single-graph device measurement; not aggregated like our suite-wide tile-build geomean) | `tf32/baseline_audit/our_build_vs_flashsparse.csv` | `median_ms` |
| FlashSparse on SM 86 verification | `tf32/baseline_audit/flashsparse_sm86.csv` | direct |
| HC-SpMM (router 1.64x on its 14-config subset) | `tf32/baseline_audit/hc_regime.csv`, raw `fair/hcspmm.csv` + `fair/hcspmm_preproc.csv` | `OVERALL` row |
| MP-SpMM (0.84x aggregate, 1.15x on real graphs; preprocessing 2.18 s mean) | `tf32/baseline_audit/mp_regime.csv`, `fair/mp_spmm.csv`, `fair/mp_spmm_preproc.csv` | `OVERALL` / `REAL_ALL` / `SYN_ALL` rows; `preprocess_ms` stats |
| Regime matrix for all external baselines | `tf32/baseline_audit/baseline_regime_matrix.csv` | direct |
| Rule ablation (full seven-rule router 1.702x/173; per-rule rows) | `tf32/rule_ablation_v5.csv` | direct (`oracle_hits` = matches against the floored oracle; `oracle_hits_kernelonly` also provided). `tf32/rule_ablation_8rule.csv` records the earlier eight-rule candidate set, whose sub-5K-row rule changed no metric and was pruned |
| Feature leave-one-out | `tf32/feature_loo_v5.csv` | direct |
| Max-row-degree feature study | `tf32/feature_gain_maxdeg.csv` | direct (leave-one-graph-out Router/Oracle) |
| Positional-feature study, all 51 graphs / 192 configs (hits 122 to 122; Router/Oracle 0.957 to 0.941) | `tf32/feature_gain_v6_51.csv` | direct, from `experiments/positional_gain_51.py` |
| Positional-feature study, superseded 47-graph run | `../feature_gain/feature_gain_v5.csv` | retained for provenance; its 47-graph positional input was never released, so its `base+index` row is not reproducible |
| Learned-selector comparison | `tf32/learned_selector.csv`, generator `experiments/learned_selector.py` | direct (LOGO and train-fit columns) |
| Kernel leave-one-candidate-out | `tf32/final_fair_v3.csv` | oracle over the eight deployed paths with the cuSPARSE floor; remove one candidate; per-axis geomean delta |
| Byte-bound analysis | `tf32/byte_bounds_r5.csv` | direct |
| GPU feature extraction (31x median kernel-only, 1.6x with H2D) | `tf32/v5_alignment/feature_extract_gpu_vs_cpu.csv` | median/mean of the speedup columns |
| Cross-architecture table (Ada 1.515x warm, 0.942 Router/Oracle, 192/192 parity, 84/192 oracle shift; cold 2.249x / 2.286x) | `ada5/router_quality_ada_warm.csv`, `ada5/router_quality_ada_cold.csv`, `ada5/fair_sweep_ada.csv` | same recipes as the 3090 tables, joined against `tf32/router_quality_v8.csv` for parity; cold first-call CSR dispatch = `CSR_DIRECT` `ms_cusparse_cold/ms_cold`, ceiling = best CSR-family cold speedup |
| Dataset summary (M, nnz, mean degree) | `tf32/final_fair_v3.csv` feature columns; manifest `paper_datasets.json` | first row per dataset |

## Recompute example

```python
import numpy as np, pandas as pd
rq = pd.read_csv("fgcs_results/revision/tf32/router_quality_v8.csv")
gm = lambda s: float(np.exp(np.log(s[s > 0]).mean()))
print(gm(rq.router_speedup))                      # 1.702
print(gm(rq.oracle_speedup))                      # 1.706  (floor already applied)
print(gm(rq.router_speedup) / gm(rq.oracle_speedup))    # 0.9976
print(int(rq.is_hit.sum()))                       # 173  deployed oracle
print(int(rq.is_hit_kernelonly.sum()))            # 172  custom-path match
cs = pd.read_csv("fgcs_results/revision/tf32/deployed_coldstart.csv")
print(gm(cs.deployed_cold_speedup))               # 2.269
```
