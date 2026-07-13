# FGCS Revision — Quality Gates (node8, 2× RTX 3090, CUDA 11.8, torch 2.7.1+cu118)

All gates PASS as of this run.

- [x] **Setup verified: PARITY OK 192/192** (ra_router_parity_test.py; C++ router == Python router on all real+synthetic pairs).
- [x] **Profiling CSVs contain all 5 metric families per pair** — profile/profile_summary.csv + 7 `.ncu-rep`:
      (i) TC pipe util, (ii) achieved occupancy, (iii) DRAM throughput %, (iv) roofline SOL
      (Compute-SM % = achieved/peak; Memory GB/s; exact AI in the `.ncu-rep` chart), (v) warp-stall top-2.
- [x] **Every attempted baseline has a results CSV (+ _preproc) or a BUILD_NOTE:**
      HC-SpMM (hcspmm.csv + hcspmm_preproc.csv + hcspmm_BUILD_NOTE.txt),
      MP-SpMM (mp_spmm.csv + mp_spmm_preproc.csv + mpspmm_BUILD_NOTE.txt),
      DCGG (NOT_ATTEMPTED — no public repo; documented in BASELINES_SUMMARY.md).
- [x] **cuBLAS done with BOTH FLOP conventions; cuBLAS-wins flagged** — cublas_small.csv has
      gflops_truennz + gflops_padded; 3 wins flagged (PPI N=64 2.30x, PPI N=128 1.37x, Cora N=64 1.03x).
- [x] **Feature-extraction GPU-vs-CPU speedup measured; three-way split reported** —
      feature_extraction_gpu.csv (CPU vs GPU, 19.4x kernel-only / 1.5x incl. H2D, CPU/GPU match on all 26),
      conversion_times.csv, pipeline_proportion.csv (feature:conversion:compute split + 400-call amortization).
- [x] **Two 4090-facing CSVs present** (ada_pkg descoped): fgcs_results/spmm/all_graphs_results.csv +
      fgcs_results/summary/router_quality_v2.csv (also copied to revision/for_4090/). Router: 80.4% hit,
      geomean 3.312x vs oracle 3.324x (1.004x overhead).
- [x] **All CSVs use the existing column schema** so laptop integration is mechanical.

## Reviewer-hardening checks (HC-SpMM)
- [x] Crash attribution: **INTRINSIC** — HC-SpMM's own native loader crashes on amazon-computers while
      the bundled control (PROTEINS) runs; our loader is not involved (hcspmm_BUILD_NOTE.txt [1]).
- [x] Timing parity: HC-SpMM and cuSPARSE both timed by the same 50+200 CUDA-event harness (measure_ms) [2].
- [x] N coverage: HC-SpMM is N∈{32,64}-only (crashes at N≥128 even on N=64 survivors) — documented [3].

## rsync (Tariq's step — cannot run from node8)
Copy the whole `fgcs_results/revision/` tree (37 MB) plus the two named CSVs
(`fgcs_results/spmm/all_graphs_results.csv`, `fgcs_results/summary/router_quality_v2.csv`,
also mirrored in `revision/for_4090/`) back to the laptop. No ada_pkg is produced (descoped).
```
rsync -av node8:/mnt/shared/development/tariq/RA-SpMM/fgcs_results/revision/  <laptop>/fgcs_results/revision/
```

## Not done / deferred (honest)
- MP-SpMM skipped 4 graphs with nnz>20M (soc-Pokec, Reddit, ogbn-products, ogbn-proteins) — `.mtx`/preproc
  too heavy on the 3090; noted in mpspmm_BUILD_NOTE.txt. (MP-SpMM's own preprocessing is 0.25–21 s/graph.)
- HC-SpMM ran on the 14/26 graphs it does not crash on (crashes are intrinsic; the 12 are denser/skewed).
- Roofline exact AI + achieved/peak FLOP-rate: per-pair values live in the `.ncu-rep` (ncu-ui chart);
  the CSV reports the reliable SOL coordinates (Compute-SM %, DRAM %, Memory GB/s) instead of a fabricated AI.

## Post-finalization verifications (requested)
- [x] **Router CSV is not a buggy re-implementation.** After adding the measured `cv_d` column to the
      sweep CSV, `ra_router_eval.py` PRINTS **74/92 (80.4%), router geomean 3.312x, overhead 1.004x** —
      identical to router_quality_v2.csv. The earlier 72/92 was ra_router_eval's approximate cv_d fallback.
      The gap vs the paper's 166/192 (86.5%) is expected: 26-real-only (92 pairs) vs combined-192 (incl.
      easier synthetics) + CUDA-11.8/torch-2.7 environment drift. Noted in REVISION_RESULTS.md.
- [x] **MP-SpMM output verified EXACT vs cuSPARSE semantics** (2:4 match-and-pad = pad, not prune).
      B=1 ⇒ C[i][0]=degree(i) on all real rows (maxΔ=0) for amazon-photo, ca-CondMat, roadNet-PA, Cora, PPI;
      trailing rows are zero padding to a multiple of tile_M=16. ⇒ MP-SpMM does not need the 4090 run.
      (mpspmm_BUILD_NOTE.txt; reproduce via revision_harness/b3_mpspmm_verify.py)
- [x] **HC-SpMM three fairness checks** recorded in hcspmm_BUILD_NOTE.txt: crash INTRINSIC (native-loader
      control), same 50+200 CUDA-event timing as cuSPARSE, N=64-only (crashes at N≥128).

## B7 + B8 (TC conversion optimization + conversion-aware routing)
- [x] B7: flat tile packing + reordered-CSR + remap optimized; **PARITY OK 192/192** (byte-identical);
      TC outputs still match cuSPARSE. TC_DIRECT conversion 2–4× faster (median 466→216ms, ogbn-products
      19.6→6.2s); 8/26 small graphs <50ms. 100M+ nnz graphs retain the fundamental O(nnz)+tile-volume floor
      (documented). featbreak/{conversion_times_v2.csv, CONVERSION_OPT_SUMMARY.md}.
- [x] B8: conversion-aware routing (offline). Cold-start K=1 → CSR_DIRECT on 86/92 (≤11.3ms setup, 2.53× vs
      cuSPARSE first-call) = preprocessing-free; steady-state K=1000 recovers TC throughput kernels; crossover_K
      per graph; fair DTC setup reduction ~3500× cold-start (honest). convaware/*.csv + CONVAWARE_SUMMARY.md.
