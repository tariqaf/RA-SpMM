# RA-SpMM Round-3 Optimization Plan (decided order)

Decided after SURVEY_MINING.md review + external review feedback. Constraint
that overrides everything: **the paper story is near-zero preprocessing
(streaming graphs / GPU inference)**. Any technique that adds O(V+E)
reordering or heavy per-matrix preprocessing is out, regardless of
steady-state gain. One-time *format conversion* on large graphs is defendable
only if we also shrink it.

## Dropped (with reasons)

- **RABBIT++ / Rabbit / DTC 2-level / Acc-SpMM affinity reordering** — all add
  O(V+E)+ preprocessing; not defendable under the streaming/inference story.
  Replaced by N3 (schedule swizzle, zero data movement).
- **Half-B caching across calls** — GNN layers change B every call; only helps
  repeated-identical-B microbenchmarks. Would be benchmark gaming.
- **8-bit ME-TCF local indices (as-is)** — does not transfer: ME-BCRS windows
  have unbounded column spread (no column tiling), and 16-bit atox fails on
  giants (K > 65536). Indices are only ~20% of plan bytes anyway. Replaced by
  N2 (zero-compressed values, the 80% of bytes).
- Previously listed do-not-try items stand (2:4 SpTC, Voltrix TMA, MaxK/AES,
  SMaT, full Swift, INT8).

## Corrections adopted into the plan

1. TF32 uses `mma.m16n8k8.row.col.f32.tf32.tf32.f32` (sm_80+) — same K=8
   shape, no k4 split. Work = tf32 B-fragment value ordering in the builder
   (FP32 values) + kernel variant; +3 regs/thread.
2. **TF32 byte math is regime-dependent**: deletes the 6 B/elem convert pass
   but doubles gather to 4 B/touch. Wins on bytes only when effective B-row
   reuse r (post-L2) < ~3; wins on time wherever the gather is latency-bound
   rather than byte-bound. Expectation: likely win on low-degree /
   launch-sensitive regimes, possibly a loss on Dense-Large. This is an A/B
   measurement, and the per-regime outcome feeds N1.
3. Intra-window B reads are read-once by construction (columns deduplicated).
   The staging win is **sector efficiency** (scattered 2 B `__ldg`s in
   fragment order ≈ 50% wasted sector bytes → block-wide float4 stage = 100%)
   plus `cp.async` latency hiding — not reuse. Cross-window reuse is L2-level
   (→ N3, E4).

## Novel methods (ours, not from the survey) — paper contribution candidates

- **N1. Regime-aware precision-path routing.** The router already picks a
  kernel per regime; extend it to pick the *precision path* (FP16+convert vs
  TF32-direct) per regime from the same features, using the measured
  crossover (convert overhead ∝ 1/d̄ vs gather-byte doubling). Extends the
  paper's central thesis into the precision dimension; no other system routes
  precision by sparsity regime. Even a mixed TF32 result becomes a
  contribution this way.
- **N2. ZC-BCRS: zero-compressed vector values with in-register expansion.**
  ME-BCRS pads every 8×1 vector to 8 stored values; at our measured fill most
  of those bytes are zeros, and values are ~80% of plan bytes. Store packed
  nonzeros + per-vector 8-bit fill mask; expand to the mma fragment in
  registers. Cuts steady A-side DRAM ~10–15% and giant plan build+upload
  ~3–4× (attacks the 2.5–6.1 s cold cost *without* the GPU-side-build risk).
  Deterministic. First step: instrument the plan builder to report actual
  avg fill per graph (one print, zero risk) to size the win before building.
- **N3. Locality-aware window scheduling (reordering's benefit without
  reordering).** Permute only the *block schedule* — process windows in
  min-column order per N-strip so temporally adjacent blocks touch
  overlapping B rows while they are L2-resident. Data, outputs, and row order
  untouched; preprocessing = one O(W log W) sort of window IDs (microseconds
  to ms). Targets the Skewed/Mixed L2≈20% collapse within the
  low-preprocessing story.
- **N4. Multi-window fused staging for COMMUNITY_TC.** After its existing
  leading-neighbor row sort, consecutive windows share many columns. Process
  2–4 consecutive windows per block and stage the union of their B rows once
  in smem — real B reuse (not just coalescing) with no new preprocessing.

## Decided experiment order

Branches per external-review suggestion; serial where they touch the same
engine file. Discipline unchanged: strict correctness gate, fair warm/cold
A/B vs current round-2 kernels AND both cuSPARSE baselines, ncu before/after
on 3 probe graphs (one Community, one Skewed, one Mixed), notes appended to
this file.

### Shared TC engine (fs_tile — serves TC_DIRECT, COMMUNITY_TC, SEGMENT_HYBRID)

| # | Branch | Experiment | Gate / expected Nsight movement |
|---|---|---|---|
| E1 | `exp-tf32` | TF32 m16n8k8 path: builder stores FP32 vals in tf32 fragment order, kernel variant reads raw FP32 B, convert kernel + half-B buffer deleted on this path. TC_DIRECT first, then CT/SH. A/B per regime at N=128 and 512 | strict gate + GNN accuracy (TF32 ≥ FP16 accuracy expected: same 10-bit mantissa, FP32 exponent range); convert launch gone from nsys; per-regime win/loss table → N1 router gate |
| E2a | `exp-b-staging` | Plain smem B staging: block-cooperative float4 stage of each block's 8 B-row × 64-col strip, mma gathers from smem. No cp.async yet | sector efficiency ↑ (l1tex sectors/req), long_scoreboard ↓, runtime ↓; if flat, stop here |
| E2b | same branch | Add cp.async (16 B `ca`) double-buffer: prefetch next block's strip during current mma | long_scoreboard ↓ further, eligible warps ↑ |
| E3 | `exp-zc-values` | N2 ZC-BCRS. Step 0: fill-factor instrumentation only. Step 1: packed values + fill mask + in-register expand | plan bytes ↓ (report), giant cold s ↓, steady DRAM ↓; strict gate (bit-identical math, order preserved) |
| E4 | after E2 | N4 multi-window fused staging (COMMUNITY_TC only) | L2 hit ↑ on Community/Dense-Large |
| E5 | cheap, anytime after E2 | N3 window-schedule swizzle (TC_DIRECT; CT already sorted) | L2 hit ↑ on Skewed/Mixed |

### SEGMENT_HYBRID-specific

| # | Experiment | Gate |
|---|---|---|
| S1 | Replace split-hub-window `atomicAdd` merge (ra_segment_hybrid.cu:159-164) with warp-shuffle partial reduction + single store (HR-SpMM/BRP-SpMM style) | atomic count ↓, Skewed/Mixed runtime; strict gate (watch fp add order → tolerance unchanged) |
| S2 | Manuscript wording fix (do now, costs nothing): claim = "no atomics across the TC/CUDA partition boundary (rows disjoint); split hub windows use intra-path atomic accumulation" — a blanket "no atomics" claim is false in the current code | paper consistency |

### CSR side

| # | Experiment | Gate |
|---|---|---|
| C1 | CSR_DIRECT short-row packing (Rgs-SpMM style): pack multiple short rows per warp using the bin infrastructure already built for ZERO_OVERHEAD | eligible-warps 0.10 → ↑ on Community; strict gate |
| C2 | Only if C1 insufficient: GE-SpMM warp-merge B reuse | L1/L2 hit ↑ |
| Z1 | ZERO_OVERHEAD: coalescing audit only (Accel-GCN warp→column mapping) — round-2 rewrite likely already covers it | low priority |
| — | RODE_ENHANCED: no work (not top-2 in any regime; round-2 shuffle port measured flat) | — |

### Last, after kernels settle

| # | Experiment |
|---|---|
| R1 | Router refit on new sweep (mandatory after any kernel change) + N1 precision-path dimension + AutoSAGE-style cuSPARSE guardrail. Parity Python↔C++ must pass again |

### Deferred, not dead

- GPU-side ME-BCRS build (FlashSparse Block_gpu port): revisit **only if** E3
  leaves giant cold cost indefensible. E3 may make it unnecessary.

## Stopping rule ("no more optimization possible")

A kernel/regime pair is done when: two consecutive experiments move its fair
geomean < 2%, AND ncu shows either DRAM ≥ ~90% of peak with high sector
efficiency (bandwidth floor) or long_scoreboard minimized with eligible
warps ≥ ~2 (latency floor). Record the closing ncu snapshot here when a pair
is closed.

## Findings log

### E1 — TF32 m16n8k8 path (branch `exp-tf32`, 2026-07-14)

Implementation: `fs_tile_spmm_tf32_kernel` in tc/ra_tc_direct.cu (+ ct_/sh_
variants), exposed as TC_DIRECT_TF32 / COMMUNITY_TC_TF32 / SEGMENT_HYBRID_TF32.
Fragment mapping verified by standalone microtest before integration. Design
choice vs the review feedback: **sparse A values stay FP16 in the plan**
(fp16→tf32 is exact — both 10-bit mantissa), converted in-register; the kernel
remaps the two half slots per lane to the tf32 B-fragment layout. So the SAME
plan serves both precision paths (plan bytes unchanged, zero extra build cost,
exactly what router-selected precision needs). Dense B loads use cvt.rna.
No k4 split needed: m16n8k8.tf32 exists on sm_80+.

Full fair sweep (192 configs, 51 graphs, declared Ns, same 50/200 warm +
10-cold protocol): `fgcs_results/revision/tf32/e1_ab_full.csv`.

**Gate results:**
- Strict correctness: **192/192 pass**, zero soft/hard fails. Worst max_error
  identical to fp16 path (0.125 vs 0.125); no config differs >5% either way.
- Convert launch: **gone** (ncu shows single spmm launch per call).
- ≥5% win on a low-degree regime: **yes** (Community +9.6% geomean).
- No harm to Dense-Large/Mixed: **only if router-gated** (see below).

**Warm geomean tf32/fp16 (>1 = TF32 faster):** Community 1.096 ·
Dense-Small 0.940 · Sparse-Uniform 0.900 · Mixed 0.789 · Dense-Large 0.625 ·
Sparse-Skewed 0.626 · ALL 0.851. Cold ≈ 0.99 everywhere (same plan build).

**The real separator is average degree, not regime** (N-independent,
0.845–0.865 at every N):
- deg < 3: 12/12 win, gm 1.408 (roads +37–42%)
- deg 3–5: 8/8 win, gm 1.200 (Cora/CiteSeer up to +64%)
- deg 5–8: 29/51 win, gm 1.095 (size decides: ca-GrQc M=5k +49%,
  com-Amazon M=335k −15%)
- deg ≥ 8: 5/121 win (only tiny ca-CondMat M=23k)

**ncu root cause** (`fgcs_results/revision/tf32/ncu*`): winners — convert
time deleted outright, spmm +20–28% (doubled gather bytes), net win; DRAM%
rises 80→90 (single kernel uses bandwidth better), tensor-pipe 6.5→10.9%.
Losers — L2 hit collapses (com-Amazon 32.2→22.0%) because FP32 B halves
effective L2 residency; spmm +61% > convert saved. Confirms the pre-registered
byte model (win iff post-L2 reuse small).

**Decision: adopt as router-selected precision path (N1), not a replacement.**
Candidate rule from the sweep: `TF32 if d̄<5, or (M≤25k and d̄<9)` → 0 mispicks,
family geomean ×1.03–1.05; final threshold fit happens at R1 together with the
CT/SH tf32 numbers. TF32-vs-cuSPARSE on its win domain: Community tile path
rises 1.380→1.513 vs FP32 cuSPARSE, and vs the FP16-precision-matched baseline
0.952→1.044 (the tile format now beats even precision-matched cuSPARSE there).

CT/SH tf32 variants implemented and strict-gate-verified; full A/B sweep
queued. GNN end-to-end accuracy run in progress.

### E3 Step 0 — vector fill factors (2026-07-14)

`fgcs_results/revision/tf32/zc_fill_report.csv`, all 51 graphs. **Average
vector fill ≈ 1.0/8 slots on every graph** (zero padding 81–89% of value
bytes; value bytes ≈ 80% of plan bytes everywhere). ZC-BCRS estimated plan
shrink **2.5–2.9× universally**, including giants: ogbn-products 2496→872 MB,
Reddit 2234→788 MB, ogbn-proteins 1064→424 MB. The external-review precondition
("padding must dominate") holds everywhere → E3 Step 1 (packed values +
8-bit fill mask + in-register expand) is GO after E2.

Side observation: fill≈1 means most mma vectors carry a single nonzero — the
format's win is scheduling/coalescing, not tile density; DRAM-side padding
cost is the tax, which ZC removes.

### S2 — manuscript wording (action for Tariq; FGCS .tex not in this repo)

Replace any blanket "SEGMENT_HYBRID has no atomics" claim with: "no atomics
across the TC/CUDA partition boundary (rows are disjoint); split hub windows
use intra-path atomic accumulation" (code: ra_segment_hybrid.cu split-segment
atomicAdd merge).
