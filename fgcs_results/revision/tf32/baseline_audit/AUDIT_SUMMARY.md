# Baseline re-verification vs the FINAL round-5 router (2026-07-16)

All ratios recomputed against `router_quality_v5.router_ms` (final router), joined by
(dataset, N). Stale speedup columns in the baseline CSVs were NOT trusted (MP's were
off by up to 2.0x). `router_faster = baseline_ms / router_ms` (>1 = router faster).
Regime = the `category` column (6 regimes). Deliverables listed at the bottom.

## TASK 1 (GATE) — FlashSparse on SM 86: VERDICT (c) — IT BUILDS AND RUNS.
- Built all 4 FlashSparse extensions for `sm_86` with `TORCH_CUDA_ARCH_LIST=8.6`, CUDA 11.8, on the RTX 3090. FS_SpMM imports and executes.
- Ran 4 real graphs (Cora, ca-HepTh, PPI, ogbn-arxiv) at N=128 on BOTH paths: **8/8 (graph,path) ran, all correct** (rel-Frobenius < 5% vs cuSPARSE): the fp16 8x8 path (`mma.sync.aligned.m16n8k8...f16`, sm_75+) **and** the tf32 8x4 path (`mma...m16n8k4...tf32`, sm_80+). Kernel PTX contains NO wgmma / fp8 / sm_90 instruction.
- **CONSEQUENCE: the manuscript claim "FlashSparse requires SM 89/90 / cannot run on our 3090" is FALSE and must be deleted.** Evidence: `flashsparse_sm86.csv`.

## TASK 2 — MP-SpMM (N=128 only), per regime
- Real vs synthetic DIVERGE as hypothesized: router_faster geomean REAL **1.147x** (router ahead) vs SYNTHETIC **0.599x** (MP ahead). (RA/MP: real 0.872, syn 1.668.)
- Per-regime router_faster: Sparse Uniform 0.52, Sparse Skewed 0.69, Community 0.58 (MP wins these), Dense Small 1.17, Mixed/Irregular 1.81 (router wins). Overall router_faster **0.838** (MP faster overall on its covered N=128 subset).
- Coverage: 33 OK / 141 UNSUPPORTED_FEATURE_DIM (N != 128) / 18 SKIPPED_RESOURCE (nnz>5M; threshold verified, min skipped nnz = 5,105,039). Preprocessing (match-and-pad): mean 2181 ms, median 1692 ms, max 8276 ms — seconds.
- PROFILER PROOF (ncu, amazon-computers): kernel = **`sparse_mma_kernel_base_Buint64_Cfloat4`** (2:4 structured sparse-TC MMA, half->float); `sm__inst_executed_pipe_tensor.sum = 519,536` (Tensor Cores active). MP's correctness gate = `1e-3*sqrt(max_degree)*10` — same tolerance model as ours. Evidence: `mp_profiler_evidence.txt`, `mp_regime.csv`.

## TASK 3 — HC-SpMM (N=64 only), per regime + crash map
- Router faster in EVERY regime HC runs (router_faster 1.35-2.11x; overall **1.641x**). HC never wins a regime.
- Crash map VERIFIED: at N=64, 14 OK / 24 CRASH / 13 INCORRECT / 141 UNSUPPORTED. The "12/26" claim is the REAL-graph slice (12 of 26 real graphs crash) — correct as a real-graph statement, but ALL 25 synthetic graphs also fail (12 crash + 13 incorrect). Recommend restating.
- Crashes concentrate in denser graphs (median crash nnz 6.07M vs OK 1.51M; every nnz>5.5M graph crashes) across all regimes. REAL crash error class captured: **CUDA error 700 "an illegal memory access was encountered"** (out-of-bounds in HC's hybrid kernel; masked as generic DSA text in the CSV). So failure is a genuine kernel crash, not merely the N=64 coverage limit. Preproc: mean 4.63 ms, max 14.89 ms. Evidence: `hc_crash_evidence.txt`, `hc_regime.csv`.

## TASK 4 — cuBLAS-dense, per regime, warm vs cold
- Router beats dense cuBLAS on **all 52 covered warm configs (0 cuBLAS wins)**, geomean **18.5x** (Dense Small 20.6x, Mixed 12.9x). cuBLAS's only wins are vs cuSPARSE-COLD on the two tiniest graphs (CiteSeer, PPI; M 1767-3327). Never beats the router warm or cold.
- COVERAGE CAVEAT: cuBLAS data exists for only 2 of 6 regimes (Dense Small, Mixed/Irregular), 52/192 configs. Cannot verify the other 4 regimes. Evidence: `cublas_regime.csv`.

## TASK 5 — our preprocessing vs FlashSparse (host vs device)
- OURS = HOST-side (CPU OpenMP): TC_DIRECT `build_ra_tc_direct_plan_impl` (tc/ra_tc_direct.cu:627); COMMUNITY_TC host stable-sort reorder (tc/ra_community_tc.cu:320); SEGMENT_HYBRID (tc/ra_segment_hybrid.cu:369). FlashSparse = DEVICE-side: `seg_sort_dequ_fs` (Block_gpu/block_kernel.cu:784, thrust::sort + device kernels).
- At matching statistic vs FlashSparse's 80.5 ms: TC_DIRECT median **78.9 ms ~= parity (0.98x)**, mean 144 ms (outlier-inflated). COMMUNITY_TC heavier — median 267 ms (3.4x TC_DIRECT) due to the host community reorder. SEGMENT_HYBRID much lighter (median 7 ms).
- FLAG: the "80.5 ms" provenance is UNVERIFIED in-repo (no doc ties it to FlashSparse's block build); cannot confirm which statistic/graph it is. Evidence: `our_build_vs_flashsparse.csv`.

## TASK 6 — unified per-regime matrix
`baseline_regime_matrix.csv` — per-regime router_faster geomean, overall + regime-balanced, coverage (ran/attempted + reasons), one-time preprocessing (mean/max + host/device), and the regime where the router wins most, for all 6 baselines.

| baseline | ran | overall rf | balanced rf | router-best regime | preproc mean/max ms |
|---|---|---|---|---|---|
| DTC-noTCA | 79/92 | 1.61 | 1.65 | Mixed 2.22x | 11 / 179 (device) |
| DTC-TCA | 50/92 | 1.60 | 1.33 | Mixed 2.38x | 250438 / 912584 (host reorder) |
| MP-SpMM | 33/51 | 0.84 | 0.85 | Mixed 1.81x | 2181 / 8276 (device) |
| HC-SpMM | 14/51 | 1.64 | 1.63 | Mixed 2.11x | 4.6 / 14.9 (device) |
| cuBLAS-dense | 52/52* | 18.52 | 16.32 | Dense Small 20.6x | 12.8 / 148.8 (device) |
| FlashSparse(fp16) | 4/4** | 1.20 | 1.04 | Mixed 1.84x | 80.5 (device) |

\* cuBLAS only covers 2/6 regimes. \*\* FlashSparse speed row is PARTIAL (4 graphs, N=128); full per-regime sweep NOT run — flagged.

## Could NOT verify (flagged, not filled in)
1. FlashSparse full per-regime SPEED sweep — only the build/run/correctness gate (Task 1) was completed; the 4-graph N=128 speed row is partial. The *hardware-reach* claim is settled (it runs); the *speed* comparison across regimes is future work.
2. cuBLAS coverage for Community / Sparse Uniform / Sparse Skewed / Dense Large-Scale (no data).
3. FlashSparse's 80.5 ms preprocessing anchor provenance (undocumented in-repo).
4. Thin regimes: MP Sparse Skewed (n=2), HC Community (n=1), HC has zero OK rows in Sparse Skewed & Dense Large-Scale.

## Deliverables (fgcs_results/revision/tf32/baseline_audit/)
baseline_regime_matrix.csv (Task 6) · flashsparse_sm86.csv (Task 1) · mp_regime.csv + mp_profiler_evidence.txt (Task 2) · hc_regime.csv + hc_crash_evidence.txt (Task 3) · cublas_regime.csv (Task 4) · our_build_vs_flashsparse.csv (Task 5)
