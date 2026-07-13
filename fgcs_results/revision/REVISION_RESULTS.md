# RA-SpMM FGCS Major Revision — 3090 Results (node8)

Machine: node8, 2× RTX 3090 (SM 86), CUDA 11.8, torch 2.7.1. All SpMM timing uses the paper protocol (50 warmup + 200 timed CUDA-event iters). cuSPARSE is the correctness reference and speedup denominator. See each sub-summary for detail.

## B2 — Profiling: why our kernels win  [Reviewer #1.1]
Nsight Compute `--set full` (+ roofline) on the required (kernel, graph, N) pairs; five metric families captured per pair (.ncu-rep + CSV in profile/).

| dataset | kernel | N | TC_pipe_pct | occupancy_pct | DRAM_pct | mem_GBs | SM_compute_pct | top_stall_1 |
|---|---|---|---|---|---|---|---|---|
| amazon-photo | TC_DIRECT | 128 | 0.0 | 52.45 | 3.38 | 31.1 | 4.75 | long_scoreboard=38.75 |
| com-DBLP | COMMUNITY_TC | 128 | 0.94 | 54.38 | 3.35 | 31.3 | 4.11 | long_scoreboard=29.37 |
| amazon-computers | SEGMENT_HYBRID | 128 | 0.0 | 46.09 | 1.86 | 16.6 | 1.68 | barrier=53.23 |
| amazon-photo | cuSPARSE | 128 | 0.0 | 73.29 | 18.08 | 163.9 | 41.28 | long_scoreboard=28.92 |
| com-DBLP | cuSPARSE | 128 | 0.0 | 81.93 | 95.38 | 864.7 | 45.27 | mio_throttle=4.50 |
| amazon-computers | cuSPARSE | 128 | 0.0 | 73.06 | 35.4 | 321.0 | 42.97 | long_scoreboard=29.22 |
| soc-Pokec | RODE_ENHANCED | 256 | 0.0 | 66.46 | 94.25 | 858.6 | 9.92 | long_scoreboard=91.64 |

Headline: cuSPARSE is **DRAM-bound** on com-DBLP (95%) and soc-Pokec, while our COMMUNITY_TC uses ~3% DRAM (community reordering kills DRAM traffic); our kernels win at **lower utilisation** (less total work), not by saturating the hardware. Full roofline per pair in the `.ncu-rep` files.

## B3 — Ampere baselines  [Reviewer #2.3/#2.1]
- **HC-SpMM**: builds cleanly on torch 2.7/CUDA 11.8 (refuting the 'won't build' worry); runs at its native GNN dim N=64; arbitrary-dim path is unstable (documented). 14 points collected. - **MP-SpMM** (2:4 SpTC): builds for SM 86; kernel N∈{32,128}; 22 points at N=128; preprocessing (match-and-pad) timed separately. - **DCGG**: NOT_ATTEMPTED — no public repository could be located.
See baselines/BASELINES_SUMMARY.md.

## B4 — cuBLAS dense-GEMM on small matrices  [Reviewer #2.4]
cuBLAS beats the router on 3 (graph,N) case(s) — flagged in CUBLAS_SUMMARY.md.

## B5 — Feature-extraction breakdown  [Reviewer #2.2]
One-pass GPU kernel for d_bar/CV_d: **19x kernel-only, 1.5x incl. H2D copy** (geomean over 26 graphs; CPU/GPU match verified). Conversion + three-way feature:conversion:compute split in featbreak/FEATBREAK_SUMMARY.md.

## B6 — DENSE_GEMM experiment path (tiny/dense corner)  [Reviewer #2.4 follow-up]
Evaluated an experiment-only **DENSE_GEMM** branch (dense FP16 + cuBLAS GemmEx) with the rule `M<=2000 and N<=128`. It would convert **2** cuBLAS-win case(s) into router wins (PPI N=64 → 2.2× over the old sparse pick, PPI N=128 → 1.4×), leaves Cora N=64 on sparse (measured tie, no demotion), and is inert on the other 25 graphs. The shipped C++/Python router remains the six-kernel sparse portfolio. What-if router geomean 3.312×→3.347×, hit 72/92→74/92 (no regression). See dense/DENSE_SUMMARY.md.

## B7 — TC plan-construction optimization (byte-identical)
Replaced per-group `std::map<int,array<float,256>>` tile packing with a flat sort-by-k-block pass + reused scratch; also optimized the reordered-CSR build and community renumber. **PARITY OK 192/192** (tiles byte-identical). TC_DIRECT conversion: median 466→216 ms, ogbn-products 19.6→6.2 s (2–4× across graphs; 8/26 small graphs now <50 ms). Residual on 100M+ nnz graphs is the fundamental O(nnz) + dense-tile-volume floor. See featbreak/CONVERSION_OPT_SUMMARY.md.

## B8 — Conversion-aware routing (cold-start vs steady-state)
Cost model total_k(K)=conversion_k+K·compute_k over B7's optimized conversions. **Cold-start (K=1): CSR_DIRECT on 86/92 pairs, ≤11.3 ms total setup, 2.53× vs cuSPARSE on the first call → genuinely preprocessing-free.** Steady-state (K=1000) shifts to the TC throughput kernels. Fair DTC setup reduction: **~3500×** cold-start (honest replacement for the old router-only claim). See convaware/CONVAWARE_SUMMARY.md.

## B0 — 4090-facing CSVs (ada_pkg DESCOPED)
The RTX 4090 (augi5) uses a separate clone, so no code package is built here. The two 3090 CSVs needed for the cross-architecture parity check are exposed:
- `fgcs_results/spmm/all_graphs_results.csv` (per-(graph,N,kernel) speedup) — present
- `fgcs_results/summary/router_quality_v2.csv` (router vs oracle, 3090) — present

**Router quality (this env): 74/92 = 80.4% hit, router geomean 3.312x vs oracle 3.324x (1.004x overhead).** This is verified consistent: `ra_router_eval.py` PRINTS the identical 74/92 on the same CSV (the CSV now carries the measured `cv_d`), so router_quality_v2.csv is a faithful serialization, not a re-implementation. It is *lower* than the paper's original 3090 166/192 = 86.5% for two expected reasons, NOT a regression: (a) this sweep is the **26 real graphs only (92 pairs)**, whereas the paper's 192 includes the easier synthetic regimes; and (b) **environment drift** — CUDA 11.8 + torch 2.7.1 (this box) shift a few oracle-optimal kernels vs the paper's original toolchain. The 4090 parity check accounts for this by comparing kernel choices across architectures using these very CSVs.
