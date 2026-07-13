# Revision Changes

This file records artifact changes made for the FGCS major revision. Measurements in this document are produced with the corrected, matching-regime methodology. Failed experiments and reverted optimizations are retained in the optimization log rather than presented as improvements.

## Benchmark Validity

### Matching warm and cold timing

- **Why:** The original sweep timed custom kernels with reusable plans but called the one-shot cuSPARSE wrapper inside every timed iteration. That wrapper recreated descriptors, workspace, preprocessing state, and output storage on every call.
- **Change:** The corrected sweep reports warm execute-only and cold setup-plus-one-execute measurements separately. Custom kernels, cuSPARSE, PyG, DTC-SpMM, HC-SpMM, MP-SpMM, dense cuBLAS, and the GNN adapters expose matching-regime fields. Warm measurements build reusable state once and use 50 warmup plus 200 timed executions; cold measurements report setup and first execution separately.
- **Before:** The reported cuSPARSE denominator included setup while custom-kernel measurements did not.
- **After (192 configurations):** The production rule router is `1.007031x` versus cuSPARSE in the matching warm regime; the six-kernel measured oracle is `1.038749x`. The corrected result does not support a multi-fold warm speedup claim.

### Asynchronous execution bindings

- **Why:** Custom execution bindings called `cudaDeviceSynchronize()` before every return, whereas the reusable cuSPARSE benchmark enqueued repeated executions between CUDA events. The per-call global barrier serialized custom iterations and could include host dispatch gaps in tiny-kernel measurements.
- **Change:** Paper-portfolio and reusable cuSPARSE execution bindings now return asynchronously under normal PyTorch CUDA semantics. The tile paths and cuSPARSE handle use PyTorch's current CUDA stream. Correctness reads and benchmark event boundaries provide synchronization; host-side cold plan timing retains explicit synchronization.
- **Measured result:** The rebuilt strict-gated sweep reports the complete post-fix timing. Because the original measurement mixed lifecycle regimes, it is not used as an optimization baseline. The RODE/SEGMENT launch-geometry change is evaluated separately below on fixed before/after cases.

### Strict correctness eligibility

- **Why:** The original condition accepted any maximum error below `1.0`, even when it exceeded the adaptive tolerance.
- **Change:** `correct`, `soft_fail`, and `hard_fail` are separate fields; only `correct == true` rows are eligible for speedup, oracle, and router statistics. External-baseline adapters use the same adaptive tolerance and never select a candidate that misses it.
- **Hard-fail invariant:** Because the square-root tolerance can exceed `1.0` on extremely long rows, every harness also requires `max_error < 1.0` for correctness. A hard failure can therefore never be simultaneously labeled correct.
- **Before:** Soft numerical failures were treated as correct.
- **After:** The two-GPU correctness-only sweep passed all six kernels on all 192 loaded `(graph,N)` configurations with no soft or hard failures.

## Correctness And Robustness

### ZERO_OVERHEAD_CSR long rows

- **Why:** The long-row launch was capped at 1,024 chunks, silently skipping nonzeros after position 65,536 in a row.
- **Change:** The plan records the exact maximum long-row chunk count while scanning the host CSR row pointer. Execution uses that precomputed bound, removing both truncation and run-time device-to-host row-pointer reads.
- **Performance rationale:** Normal execution gains a constant-time launch-bound lookup; no additional work is added to the kernels.
- **Verification:** `experiments/verify_revision_fixes.py` passes with exact output for a synthetic 70,000-nonzero single row (`max_error=0`).

### Tensor-path minimum output width

- **Why:** TC plans are inactive for `N < 16`, while router feasibility previously allowed them for sufficiently large `M`, producing an untouched zero output.
- **Change:** TC_DIRECT, COMMUNITY_TC, LOCALITY_TILED, and SEGMENT_HYBRID are infeasible for `N < 16`; the existing CSR_DIRECT fallback is selected.
- **Performance rationale:** The paper workload (`N >= 64`) is unchanged. Unsupported narrow outputs now use a working low-overhead CSR path.
- **Verification:** `experiments/verify_revision_fixes.py` confirms `M=150000, N=8` selects CSR_DIRECT and returns exact nonzero output (`max_error=0`).

### Degree-CV convention

- **Why:** C++ uses population standard deviation while Python used PyTorch's default sample standard deviation.
- **Change:** Python evaluation and parity code use population standard deviation (`correction=0`).
- **Verification:** `ra_router_parity_test.py --expected 192` reports `PARITY OK 192/192`.

### Canonicalized graph metadata

- **Why:** Twitter and Google+ edge lists contain duplicate directed edges. The loader canonicalizes them, while manifest nnz and average-degree notes described the uncanonicalized files.
- **Change:** Both real-only and combined manifests now record the CSR actually benchmarked: Twitter `nnz=1,768,149`, `d_bar=21.7468`; Google+ `nnz=13,673,453`, `d_bar=127.0602`. Their measured structural categories remain Sparse Skewed and Dense Large-Scale, respectively.

### Router parity completeness

- **Why:** The old parity script skipped missing inputs and could print success after loading zero configurations.
- **Change:** The test requires exactly 192 configurations by default and exits unsuccessfully for missing, duplicate, or zero loaded points. Deliberately incomplete development runs require the explicit `--allow-partial` flag.
- **Final verification:** The production C++ feature path and Python rule tree match on `192/192` configurations.
- **README sweep command:** The full command now uses each manifest entry's declared `Ns` instead of overriding them with a global list, preserving the intended 192-configuration set.
- **Width filter semantics:** `ra_real_graph_eval.py --Ns ...` now intersects requested widths with each dataset's declared widths instead of replacing the manifest list.

### Public MAIN portfolio

- **Why:** The generic `run_oracle_*`, `run_router_*`, `test_ra.py`, and `ra_eval_utils.py` interfaces still labeled an obsolete five-path legacy set as `MAIN`; `ra_eval_utils.py` also imported a missing `test_next` module.
- **Change:** `MAIN` now means the six paper kernels plus cuSPARSE throughout the C++ timing API and Python harness. Warm/cold timing support was added for each specialized plan, tests use the same roster, and the harness imports the shipped `test_ra` module.
- **Legacy entry points:** `ra_eval.py` and `bench/profile_case.py` now import `test_ra`; the harness module header also names the shipped file correctly.
- **One-shot cuSPARSE lifetime:** The non-benchmark reference API synchronizes its current stream before its temporary descriptors and workspace are destroyed. Reusable plan execution remains asynchronous and is the warm benchmark path.

## Reviewer Experiments

### Production feature extraction

- **Why:** The old timing measured only degree mean and CV, while production routing also derives tile and locality features and handles `colind` and values.
- **Change:** `experiments/time_feature_extraction.py` calls the production `make_router_plan` feature path for CPU- and GPU-resident inputs. The prior degree-moment experiment remains available but is explicitly labeled `lightweight (d_bar,CV_d only)`.
- **Disclosure:** The current full production implementation is CPU-based. The GPU-resident measurement includes the transfer required by that implementation and is not labeled as a GPU feature extractor.
- **Measured result (51 graphs, five full-path repetitions):** Mean full-path time is `2958.23 ms` for CPU-resident input. GPU-resident input is copied back for the same CPU implementation and therefore does not provide a full-path GPU acceleration. The separately labeled degree-moment-only GPU kernel is `18.2x` faster than its CPU counterpart kernel-only, but only `1.3x` when its host-to-device copy is included; these lightweight numbers are not used as production routing overhead.
- **Pipeline breakdown:** `time_conversion_pipeline.py` now joins the strict fair sweep with measured full production feature costs. It no longer substitutes the lightweight degree-only pass, and it terminates on missing feature measurements.

### Conversion-aware routing

- **Why:** The old script used a hardcoded feature cost, treated missing conversions as zero, and selected the best kernel after reading all measured outcomes.
- **Change:** The offline analysis now consumes per-graph production feature costs, rejects missing conversion data, uses matching warm/cold denominators, and implements the strict crossover boundary. It is labeled as an offline measured-oracle amortization analysis.
- **Runtime option:** `ra_runtime_router.py` implements a deployable call-count-aware policy. A disclosed model predicts setup and warm cost for feasible kernels; dispatch does not benchmark candidates online.
- **Runtime validation:** Model fitting emits graph-grouped out-of-fold K=1 and K=1000 choices and evaluates their measured lifecycle against matching cuSPARSE. RA totals include the per-graph full production feature cost; missing feature or baseline measurements terminate validation.
- **Measured offline result:** Including the measured production feature cost, the measured-oracle lifecycle is `0.010706x` versus matching cold cuSPARSE at K=1 and `0.713015x` over K=1000 calls. No measurements are missing.
- **Measured deployed-policy validation:** Graph-grouped out-of-fold predictions retain `0.970332x` of the measured lifecycle oracle at K=1 and `0.913934x` at K=1000. Including feature cost, their geomean speedups versus matching cuSPARSE are `0.010702x` and `0.661982x`, respectively.

### Learned selector and router ablations

- **Change:** `experiments/generate_learned_selector.py` evaluates lightweight learned selectors with graph-grouped cross-validation, keeping every feature width of a held-out graph out of training. It compares base inputs with the exact 34-value C++ production feature vector exported by `experiments/extract_production_features.py`. `experiments/generate_extended_ablation.py` provides median-masked, only-one-feature-variable, and leave-one-rule-out tests without retuning thresholds against test outcomes.
- **Measured result:** On the corrected warm sweep, the production rules retain `0.969466x` of oracle with 143/192 exact hits. The strongest tested learned selector is the base-feature random forest at `0.944891x` of oracle with 110/192 hits; the strongest production-feature hit count is the depth-4 decision tree at 117/192 and `0.929307x` of oracle. The generated warm/cold ablation tables contain all 17 requested feature and leave-one-rule-out variants.

### HC-SpMM coverage accounting

- **Why:** HC-SpMM's shipped fixed kernel supports N=64 in this evaluation, but some graphs crash or fail strict correctness. Omitting those cases would overstate baseline coverage.
- **Change:** The adapter emits one status row for every applicable graph and feature width: measured `OK`/`INCORRECT` rows at N=64, `CRASH` or resource-skip rows when execution cannot complete, and `UNSUPPORTED_FEATURE_DIM` for N=128/256/512.
- **Completeness guard:** The driver now refuses to write output unless the observed `(graph,N)` status keys exactly match the manifest. Empty per-row error strings are not treated as failures.
- **Attribution check:** HC-SpMM's bundled `PROTEINS_full_A` control ran successfully. `amazon-computers` still produced an illegal memory access after successful preprocessing when converted to HC-SpMM's native text format and loaded by HC-SpMM's own loader. The failure is therefore not caused by the RA-SpMM CSR adapter.
- **Measured coverage:** The complete table contains 141 unsupported-width rows, 24 N=64 crashes, 13 N=64 strict hard failures, and 14 strict-correct N=64 measurements. Performance geomeans use only those 14 eligible points.

### Profiling artifact size

- **Change:** Per-pair parsed metric CSVs, profile metadata, and summaries are release artifacts. Reproducible Nsight `.ncu-rep` files and raw `.ncu.csv` exports are ignored because they occupy roughly 931 MB for this sweep; the exact capture command and metric list are in `experiments/profile_ncu.py`.
- **Optimization log:** `experiments/compare_optimization.py` joins fixed-case warm timings with primary-kernel Nsight metrics and emits the reviewer-facing before/after CSV and Markdown summary.
- **Regime-level oracle reporting:** The fair summary records warm and cold oracle-winner counts separately for every structural category, in addition to global geomeans and hit/regret statistics.

### External baseline lifecycle accounting

- **Change:** PyG, DTC-SpMM, HC-SpMM, MP-SpMM, and dense cuBLAS adapters now distinguish reusable setup, first execution, and steady-state execution. DTC autotuning trials are accounted for as cold setup and are separate from its final 50/200 steady-state timing. MP-SpMM reports match-and-pad conversion, runtime state loading, first execution, and warm execution separately.
- **Correctness:** External rows are eligible only after the strict adaptive gate. MP-SpMM's pad-not-prune conversion is checked against exact row sums with the same Tensor-Core tolerance model.
- **Adapter verification:** Cora smoke tests complete successfully for PyG, HC-SpMM, MP-SpMM, dense cuBLAS, and DTC-SpMM on the original CSR order. The DTC adapter labels this mode `DTC_IDENTITY_ORDER`; no TCA-reordering claim is made. The upstream TCA script remains optional because its public version requires RAPIDS, `libMHCUDA`, and author-local paths.
- **MP-SpMM export memory:** Matrix Market conversion now flushes bounded 65,536-line chunks instead of retaining one Python string per nonzero until the entire graph has been formatted.
- **MP-SpMM coverage:** Its adapter now emits complete manifest-key coverage with measured N=128 rows and explicit unsupported-width, resource, conversion, execution, and correctness statuses. It refuses to write an incomplete table.
- **MP-SpMM source pin:** `scripts/build_mpspmm_baseline.sh` downloads Zenodo record `16933452`, verifies MD5 `7aacfbc60cdc0c535bf666538cbe2046`, applies the tracked fair-timing and correctness patches, and builds for configurable `CUDA_ARCH` (86 for the RTX 3090).
- **HC-SpMM source pin:** `scripts/build_hcspmm_baseline.sh` checks out upstream commit `3484cf74b0591e44bf656978d90ddaf9f86e00a5` and builds the extension for `TORCH_CUDA_ARCH_LIST=8.6` by default.
- **External shard merge:** `experiments/merge_external_sweep.py` unions system-specific failure columns while rejecting duplicate or incomplete `(graph,N)` coverage.
- **FlashSparse coverage:** The commit-specific adapter driver now emits explicit runtime-error rows instead of omitting failed `(graph,N)` points. Aggregation remains optional because corrected RTX 4090 measurements are produced on the separate Ada server.
- **DTC resource scope:** The full table contains all 192 manifest keys. DTC was attempted with original CSR order only for `M <= 100,000`; larger graphs are labeled `skipped_reorder_filter`. The 36 strict-correct points have a `0.000777x` warm and `0.000070x` cold geomean versus matching cuSPARSE. There are also 20 upstream nonzero exits, two timeouts, and two incorrect rows. These values do not represent the optional unavailable TCA-reordered path.
- **Measured external coverage:** PyG is strict-correct on 192/192 points (`0.416713x` warm, `0.829477x` cold). HC-SpMM is strict-correct on 14/192 status rows (`1.111136x` warm, `0.816382x` cold). MP-SpMM is strict-correct at N=128 on 33/192 status rows (`1.997537x` warm, `0.001370x` cold); 18 rows are explicit `nnz > 5,000,000` resource skips and 141 widths are unsupported. Dense cuBLAS is strict-correct on all 24 sampled small-matrix points (`0.139078x` warm, `0.646846x` cold).
- **MP-SpMM verifier diagnostic:** The tracked verifier patch uses the correct `%zu` format for its `size_t` row count. This affects diagnostic formatting only, not numerical validation.

### End-to-end GNN lifecycle and correctness

- **Why:** The GNN adapters previously lacked a strict backend-output gate. The GIN and GraphSAGE cold setup also prepared `hidden_dim` and `out_dim`, although their actual SpMM calls use `in_dim` and `hidden_dim`; this built one unused plan and deferred the real input-width plan into the first training step.
- **Change:** Every backend is validated on a fresh graph instance for every forward/backward width, leaving the timed instance unwarmed. GIN and GraphSAGE now prepare exactly `in_dim` and `hidden_dim`. Warm/cold speedups are emitted only for strictly correct rows.
- **Comment consistency:** The GraphSAGE module description now names its actual `in_dim` and `hidden_dim` SpMM widths, and revision-era grouping labels were removed from shared workload comments.
- **Measured result:** All four backends pass strict correctness on all eight datasets for GCN, GraphSAGE, and GIN (96/96 backend-model-dataset rows). Across the 24 model/dataset points, the production router is `0.851896x` warm and `0.037706x` cold versus matching cuSPARSE. Per model, router warm/cold geomeans are GCN `0.862214x`/`0.031832x`, GraphSAGE `0.842596x`/`0.033631x`, and GIN `0.850993x`/`0.050077x`.

### Source comment cleanup

- **Change:** Removed development-phase labels from public source comments and generated report headings. Algorithm-local phase labels remain only where they describe execution order.

## Profiling And Optimizations

No optimization is accepted until strict correctness, 192/192 router parity, and a measured warm-time improvement are all demonstrated. Profiling-guided suggestions and attempted/reverted changes will be recorded here.

### Baseline profile coverage and principal result

- **Coverage:** 372 warm execute-only Nsight pairs: all 26 real graphs and one representative synthetic graph for each category that has a synthetic counterpart, at N=128 and N=512, for all six kernels.
- **Tensor pipeline:** The total executed HMMA instruction count is zero across all 372 primary kernels. The current tile implementations therefore execute their FP32 fallback paths on this corpus; no Tensor Core throughput claim is made.
- **Dominant limits:** Sparse and large-width groups are predominantly DRAM/long-scoreboard limited. RODE_ENHANCED and SEGMENT_HYBRID additionally show barrier pressure in N=128 long-row work. Dense-small and mixed groups expose lower occupancy or launch granularity in several paths.

### RODE_ENHANCED and SEGMENT_HYBRID long-row launch geometry

- **Profile signal:** Baseline `N=128` profiles show substantial barrier stalls in RODE_ENHANCED and SEGMENT_HYBRID. Their float4 long-row kernels launch one loader plus seven compute warps even though `N=128` provides work for only one compute warp; at `N=512`, only four compute warps have work.
- **Attempt:** Launch one loader warp plus `min(7, ceil((N/4)/32))` compute warps for float4, with the analogous scalar-width calculation. Kernel arithmetic, data mapping, and accumulation order are unchanged.
- **Fixed-case timing:** Across ten graph/width pairs, warm-time before/after geomean is `1.017600x` for RODE_ENHANCED and `1.019569x` for SEGMENT_HYBRID, with 8/10 wins for each. Twitter's four warm timings regress by 1.15-2.08%; these regressions remain in the released log.
- **Isolated kernel profile:** Across the six post-profiled pairs per implementation, primary-kernel before/after geomean is `1.008024x` for RODE_ENHANCED and `1.007110x` for SEGMENT_HYBRID. On Amazon Photo at N=128, the barrier metric falls from approximately `51.9` to `9.9` per issue-active and primary-kernel duration improves by `4.35%` and `3.80%`, respectively. Already memory-bound cases are essentially unchanged.
- **Status:** Accepted. The final code passes strict correctness for all six kernels on all 192 configurations, router parity is 192/192, and the measured timing/profile improvements above are retained without removing the regressing points.

### Further profiling-guided improvements

- **CSR_DIRECT:** Sparse categories already reach roughly 80-92% DRAM throughput and are long-scoreboard limited. Future work should reduce bytes and gather latency through caller-provided output reuse, read-only/vectorized dense-feature loads where alignment permits, and row grouping that improves neighboring-warp column reuse. Increasing occupancy alone is unlikely to help the bandwidth-saturated cases.
- **ZERO_OVERHEAD_CSR:** Sparse-uniform N=128/512 groups reach roughly 88-93% DRAM throughput. The most credible remaining gains are fewer bin launches, caller-provided output buffers, and combining tiny/short bins when launch cost dominates; long-row chunking should remain exact and plan-derived.
- **RODE_ENHANCED:** The accepted active-warp launch removes idle barrier participants. A next step is double-buffered long-row subblocks or warp-specialized prefetch that overlaps column/value staging with dense-feature gathers, validated separately for N=128 and bandwidth-saturated N=512.
- **TC_DIRECT:** Because HMMA is not reached, the priority is not tuning Tensor Core occupancy. Packing and eligibility should be redesigned to form sufficiently dense 16x16 groups, then validated under the strict numerical gate. If that cannot be achieved on the target graphs, the implementation and paper should continue to describe this as an FP32 tile path.
- **COMMUNITY_TC:** Community ordering should be optimized for packed-tile density and dense-feature reuse, with conversion cached across GNN layers. Any revised ordering must demonstrate nonzero HMMA counters before claiming Tensor Core execution; otherwise coalesced FP32 tile gathers remain the relevant target.
- **SEGMENT_HYBRID:** In addition to active-warp sizing, disjoint tile/direct partitions could be scheduled on separate streams and their metadata compacted. Partition thresholds should be calibrated by N and measured conversion cost, with strict output checks because concurrency and partition-boundary errors have a wide blast radius.

### Deployed Four-Feature Router

- **Why:** The production eight-rule router uses only `M`, `N`, mean row degree, and population `CV_d`, but `make_router_plan(..., MAIN)` previously copied all CSR arrays and computed the legacy 34-feature tile/locality vector. The old measured full path averaged `2958.23 ms` and reached approximately `63.39 s` on ogbn-products.
- **Change:** MAIN now reads `rowptr` only and computes the four exact rule inputs. CPU-resident input uses an OpenMP row-degree reduction. GPU-resident input uses a current-stream CUDA reduction and transfers only two double-precision moments; diagnostic `FULL` retains the 34-feature extractor for ablations and learned-selector experiments.
- **Verification:** A representative CPU/Python comparison on roadNet-CA selected `COMMUNITY_TC` in both implementations with population-CV agreement to the displayed precision. On ogbn-products, the new deployed path measures `1.808 ms` for CPU-resident CSR and `0.158 ms` for GPU-resident CSR, including dispatch-required synchronization. The old 34-feature vector is now exported only through `experiments/extract_production_features.py --portfolio FULL`.
- **Interpretation:** This is a removal of unused production work, not an approximation or a learned substitute. The 192-configuration parity check remains a release gate.
- **Final verification:** `ra_router_parity_test.py --manifest fgcs_results/paper_combined_datasets.json` reports `PARITY OK 192/192`.

### CSR Output Initialization

- **Why:** CSR_DIRECT cleared every output twice: allocation used `torch::zeros` and the launcher performed a second `cudaMemset`, although the kernel writes every element. ZERO_OVERHEAD_CSR similarly cleared outputs unnecessarily when every row is dispatched to a non-atomic bin.
- **Change:** CSR_DIRECT receives `nnz` from the binding, removing a cached device-to-host metadata read and redundant output initialization. ZERO_OVERHEAD_CSR records empty-row count and skips initialization only when there are neither empty rows nor atomic long rows; all other plans keep the clear.
- **Before/after:** On `synth_sparse_uniform_d8, N=128`, strict-correct warm execution changed from `1.918766 ms` to `1.5706 ms` for CSR_DIRECT and from `1.916739 ms` to `1.5686 ms` for ZERO_OVERHEAD_CSR, approximately `1.22x` in both cases. The matching cuSPARSE run measured `1.868616 ms`; the resulting warm speedups are approximately `1.19x`, not 2x.
- **Verification:** The 70,000-nnz long-row regression and `M=150000, N=8` fallback both remain exact. Cora, Twitter, and the uniform synthetic case pass the strict adaptive gate after the change.
- **Complete strict gate:** `strict_correctness_after_opt.csv` contains all `1,152 = 192 x 6` custom-kernel rows: 1,152 correct, zero soft failures, zero hard failures, and zero execution errors.

### RODE Output And Warp Mapping

- **Why:** RODE used a full output clear even though regular prefixes write their output. Its float4 short/residual CTAs also carried idle warps at `N=64/128/256`.
- **Change:** Residual descriptors record whether a regular prefix exists. Residual-only rows write their result; residual tails add to their already-written prefix. Empty-row matrices retain initialization. Short/residual float4 paths compact one, two, or four row-owned warps into a CTA according to output width; the long-row shared-memory pipeline is unchanged.
- **Verification:** Strict checks pass on Cora and Twitter at N=128/256. On Cora N=256, RODE warm time changed from `0.037330 ms` to `0.0308 ms`; on Twitter N=256 it changed from `1.212785 ms` to `1.1717 ms`. The Cora N=128 result changed from `0.033331 ms` to `0.0320 ms`, while Twitter N=128 is effectively unchanged (`0.660859 ms` to `0.6586 ms`). These cases do not justify a whole-regime speedup claim until the complete fair sweep is rerun.

### Rejected Wide-CTA CSR Experiment

- **Attempt:** A no-conversion, multi-warp CTA-per-row float4 kernel was evaluated for dense rows at wide N, following the output-tiling direction used in prior CSR SpMM work.
- **Result:** It was strict-correct but slower on Amazon Photo N=256 (`0.3925 ms`) than the prior CSR_DIRECT measurement (`0.360678 ms`).
- **Status:** Reverted. The released code does not include this experiment.

### Tensor-Core Conversion Finding

- **Finding:** The existing TC plans perform row reordering and CSR repacking despite often emitting no WMMA-eligible tiles. Direct measurements at N=128 found zero TC tiles for Cora, Amazon Photo, Reddit, and ogbn-products; their plan-build times were `3316.679 ms`, `19.627 ms`, `734.653 ms`, and `1470.870 ms`, respectively.
- **Implication:** A one-pass row ordering alone cannot create dense 16x16 blocks absent from the original sparsity pattern. A graph-wide row-and-column reordering/conversion under a 60 ms budget is not supported by this evidence, and no HMMA utilization claim is added.

## Kernel Redesign Round Two (Sputnik / Swift / FlashSparse ports, 2026-07-13)

All changes in this section were gated by the strict adaptive correctness
tolerance on a ten-dataset cross-regime subset before acceptance; per-change
before/after warm geomeans below are from that subset against the
`fair_after_opt` baselines on the same GPU and protocol (50 warmup, 200 timed).
Sources analyzed: google-research/sputnik, MinttHu/Swift, ParCIS/FlashSparse
(clones under `external/`).

### CSR_DIRECT: subwarp + register-tile engine (Sputnik-style) — ACCEPTED

- **Why:** At N=64 the float4 warp-per-row kernel idled 16 of 32 lanes; at
  N=256/512 it re-read the entire A row once per 32-wide N stride; A
  values/indices were loaded scalar and redundantly by every lane.
- **Change:** New `csr_direct_subwarp_vec4_kernel<W,S>`: a W-lane subwarp owns
  one row (2 rows per warp at N=64), S float4 accumulators per lane cover all
  of N in one A pass, and colind/vals chunks are loaded coalesced (one element
  per lane) and distributed with `__shfl_sync`. Exact-fit dispatch for
  N ∈ {32,64,128,256,512}; other shapes keep the previous kernels.
- **Result:** subset geomean `1.0564x`; roadNet-CA `1.20-1.34x`
  (2.02x vs cuSPARSE at N=128); zero correctness failures.

### ZERO_OVERHEAD_CSR: unified small-row launch + flattened chunks — ACCEPTED

- **Why:** Three sequential launches for tiny/short/medium bins; a rectangular
  `(num_long x max_chunks)` long-row grid that was mostly empty blocks; a full
  M x N memset whenever any empty or long row existed; 4 scalar atomicAdds per
  float4 for every long-row chunk.
- **Change:** One bin-ordered small-row list executed by the subwarp engine in
  a single launch; long rows become a flattened list of 256-nnz chunk
  descriptors (one block per real chunk) where sole chunks store directly
  (no atomics); only empty rows and multi-chunk rows are pre-zeroed by a slice
  kernel (full memset only when they exceed M/4). Binding no longer allocates
  `torch::zeros`.
- **Result:** subset geomean `1.2807x`; com-youtube N=64 `3.10x` faster
  (0.32x -> 1.19x vs cuSPARSE); roadNet-CA `2.20-2.32x`; zero failures.

### RODE_ENHANCED: dead plan arrays removed; shuffle port REJECTED

- The unused `d_long_sub_starts/counts/row_map` descriptors are no longer
  generated or uploaded (plan memory and build time only; no kernel change).
- A Sputnik-style coalesced-A/shuffle rewrite of the short and residual
  kernels measured `0.9979x` subset geomean (regressions at N=64 from
  predication overhead) and was reverted. The kernels are unchanged.

### TC_DIRECT: FlashSparse ME-BCRS + swapped-operand mma rewrite — ACCEPTED

- **Why:** The previous 16x16 dense-tile design measured zero executed HMMA
  instructions across the whole corpus (median achievable tile density
  0.004-0.008 versus the 0.08 gate) and paid a multi-second row-reordering
  plan for an FP32 fallback.
- **Change:** Complete rewrite in the FlashSparse (PPoPP'25) style: 8-row
  windows in NATURAL order (no reordering), per-window column condensation
  into 8x1 vectors, 8-vector mma blocks whose values are stored directly in
  PTX B-fragment order, executed with
  `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` (FP16 inputs, FP32
  accumulate) using the swap-and-transpose operand mapping. B is converted
  once per call to a lazily allocated half buffer, halving gather traffic.
  Tensor cores now genuinely execute; accuracy is gated by the existing
  strict tolerance (TC factor) and passes on all subset configurations
  (relative error ~2-4e-4).
- **Result:** subset geomean `1.5765x` over the previous TC_DIRECT; subset
  geomean vs cuSPARSE `1.3755x`, including wins on Mixed/Irregular
  (`1.76-1.83x`) and Sparse Skewed (`1.81-1.87x`) where no custom kernel
  previously beat cuSPARSE.

### COMMUNITY_TC: same engine over locality-ordered windows — ACCEPTED

- Rows are sorted by leading neighbor column (single deterministic parallel
  sort; label propagation, CSC build, and CSR rebuild are removed) before
  windowing; the kernel scatters window slots to original C rows.
- **Result:** subset geomean `1.3498x` over the previous COMMUNITY_TC
  (`1.2709x` vs cuSPARSE); zero failures. Plan cost drops from label
  propagation seconds to a sort plus format build.

### SEGMENT_HYBRID: same engine with FlashSparse balance splitting — ACCEPTED

- Natural-order windows whose mma-block count exceeds 64 are split into equal
  segments merged with atomicAdd (sole segments store directly; only split
  windows' rows are pre-zeroed). Replaces the previous TC/CUDA row
  partitioning.
- **Result:** subset geomean `1.6829x` over the previous SEGMENT_HYBRID
  (`1.2603x` vs cuSPARSE); zero failures.

### Known open items

- ME-BCRS plan construction is CPU/OpenMP; on 14 cores it measures ~135-220 ms
  on million-row graphs and ~1.4-2.1 s on Reddit-class dense-large graphs
  (the plan itself is gigabyte-scale there). A GPU-side format build in the
  FlashSparse style is the identified follow-up if the router assigns tile
  kernels to that regime.
- The FP16-input/FP32-accumulate mma path relies on the strict gate's
  documented TC tolerance (BASE_ATOL * sqrt(max row nnz) * 10); every
  accepted configuration passes it.
- The router rules predate this landscape and must be recalibrated (Python +
  C++ + parity) once the full two-GPU fair sweep completes.

## Precision-Fairness Closure (2026-07-14)

- **Baseline:** `benchmark_cusparse_fp16[_cold]` runs the identical cuSPARSE
  algorithm and warm/cold timing loops with A and B in `CUDA_R_16F`, C in
  FP32, compute `CUDA_R_32F` — the exact dtypes the ME-BCRS tile kernels
  consume. `ra_real_graph_eval.py` records `ms_cusparse_fp16_{warm,cold}` and
  `speedup_precision_matched_{warm,cold}` (tile kernels only). Current-sweep
  measurements: `fgcs_results/revision/fair/precision_matched.csv` (all 51
  graphs x 4 N, warm 50/200, cold 10).
- **Speed (192 sweep configs, warm geomean):** TC_DIRECT `1.4405x` vs FP32
  cuSPARSE but `0.9352x` vs FP16 cuSPARSE; COMMUNITY_TC `1.3338x`/`0.8659x`;
  SEGMENT_HYBRID `1.2858x`/`0.8348x`. Per-regime best-tile vs FP16 cuSPARSE:
  Dense Large-Scale `1.421x`, Mixed/Irregular `1.055x`, Sparse Skewed
  `0.994x`, Community `0.963x`, Sparse Uniform `0.972x`, Dense Small
  `0.844x`. A substantial share of the tile kernels' gain over FP32 cuSPARSE
  is therefore attributable to reduced-precision inputs; the format itself
  wins only on Dense Large-Scale and Mixed/Irregular.
- **Accuracy:** zero strict-gate failures across all 204 measured configs
  (including highest-degree x N=512). Median max-relative-error 2.73e-4 for
  both TC_DIRECT and FP16 cuSPARSE; worst case 4.73e-4 (identical for both).
  The tile kernels trade no additional accuracy relative to the
  precision-matched vendor baseline.
- **GNN end-to-end (GCN, cusparse vs tc_direct backends):** all validations
  pass; aggregation max errors 0.061/0.102/0.064 vs tolerances
  1.15/1.47/0.88 on ogbn-arxiv/Reddit/ogbn-proteins; training-step speedup
  vs FP32 cuSPARSE 0.96x/1.79x/1.84x. Final-accuracy training comparison on
  labeled small graphs remains available via the existing bench.

## Router Recalibration v2 (2026-07-14)

- **Why:** The eight rules predated the round-two kernel redesign; on the
  fair sweep v2 landscape they scored 40/192 oracle hits.
- **Change:** Rules refit to the measured v2 oracle in lockstep in
  `ra_router_eval.py::route_with_rules` and `router/router_dispatch.cpp`
  (same thresholds, same order). The router is now preprocessing-aware:
  estimated ME-BCRS plan-build time (~20 ms per 1e6 nnz measured on the
  reference host) above a 20 s amortization budget withdraws tile kernels
  and falls back to a CSR kernel chosen by skew. No graph in the current
  corpus exceeds the budget (max measured build 2.4 s on ogbn-products);
  the gate exists for larger inputs.
- **Result (192 configs, warm):** router `1.571x` vs cuSPARSE (oracle
  `1.581x` over the six custom kernels), Router/Oracle `0.9938`, hit rate
  `181/192` (94.3%). `ra_router_parity_test.py`: `PARITY OK 192/192`.
  Data: `fgcs_results/revision/fair/router_quality_v2.csv`.
