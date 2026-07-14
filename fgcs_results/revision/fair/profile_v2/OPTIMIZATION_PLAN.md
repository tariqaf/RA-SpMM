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

## Final R1 report specification (required contents, per Tariq 2026-07-14)

Not just overall geomean. Must include:
- overall ORACLE geomean vs cuSPARSE; overall ROUTER geomean vs cuSPARSE;
  Router/Oracle ratio; oracle hit rate; WORST single-config router/oracle
  ratio; per-regime router geomeans; per-regime oracle kernels
- selection counts: how often each kernel/variant is picked, TF32 selection
  count, ZC selection count, cuSPARSE guardrail count
- cold-start geomean AND warm geomean; correctness pass count (n/192);
  Python/C++ router parity (must be 192/192)
- "Diff from corrected Round 2" table: corrected R2 router geomean ~1.57×,
  then after E1 TF32 → after E3 ZC → after E2a staging → after S1 atomic-free
  SH → final R1 router. (Attribution note: E1/E2a/S1 deltas come from paired
  same-plan A/Bs; cumulative numbers come from final_fair_v3 + R1 refit —
  mark measured vs derived honestly.)

Terminology caution (paper): do NOT call ZC-BCRS "the default format" until
proven never worse. Two distinct things:
1. ZC-transport + on-device expansion — safe to state plainly: produces
   bit-identical plans and unchanged runtime kernels (never-worse warm by
   construction), only the build gets faster.
2. Resident ZC-BCRS execution — a ROUTED format choice; paper phrasing:
   "the router selects the compressed or uncompressed tile format based on
   the measured regime."

## Round 4 — closure (2026-07-14)

Goal was >2× overall warm vs FP32 cuSPARSE; success metric per review
feedback = movement vs the PRECISION-MATCHED baseline (yardstick computed
first: matched router 1.2218×, matched oracle 1.3802×; per-regime router:
Community 1.494, DenseLarge 1.554, Uniform 1.254, Mixed 1.204,
DenseSmall 1.079, Skewed 1.030).

**All levers measured; all closed. Kernels frozen at round-3 state (1.702×).**
- Adaptive SH segment split (Dense-Small small-grid theory): flat ±4%. Reverted.
- E2b cp.async double-buffer: −6..−45% (B tiles are L1/L2-resident; cp.async
  bypasses L1 + per-iteration commit/wait drain). Reverted.
- E4 multi-window fusion: closed by instrumentation — adjacent CT-window
  column overlap 0.1–4.8% (17% best case); fill≈1 implies windows are
  near-disjoint. Never implemented (gate worked as intended).
- C1 concurrent short-row packing: already implemented in round 2
  (csr_direct_subwarp: 32/W rows per warp concurrently); synth_d3 runs it at
  94.7% DRAM = bandwidth wall on structureless graphs. Closed per criterion.
- ZC2 token format (4B/vector: mask+inline fp16+spill offset; fill=1 is
  70–94% everywhere): bit-identical, plans 2.2–2.35× smaller, but warm
  fp16 0.48–0.97×, tf32 ~1.0. Reverted.
- Staged-TF32 on high-degree Dense-Small: still 0.64–0.80×; precision gate
  unchanged (ca-CondMat, which it routes, is at parity as predicted).

**Meta-finding (paper-worthy negative result):** three independent
compressed/pipelined alternatives (ZC v1, cp.async, ZC2) all lose warm —
the padded fragment-order value array trades cheap streaming bytes for
zero decode latency/divergence and is empirically optimal at this
operating point. This quantitatively justifies the format design.

Round-4 deliverables that DID land: the precision-matched yardstick table,
E4/E2b/C1 closure evidence, and the routed-path GNN accuracy gate (below).

## Findings log

### Round 5 — CSR-kernel investigation: RODE coverage fix, bounds, and verdict (2026-07-15)

Method upgrade (external review, adopted): high DRAM% is not proof of
minimum work. Stop-rule now requires measured traffic ~ cache-capacity-aware
lower bound AND no plausible decomposition improving runtime.

1. Long-pipe defect fixed (commit 8909e00): flattened 256-nnz chunks replace
   the one-CTA-per-row pipeline (was 4-35% occupancy, barrier-bound, 11% DRAM
   on com-youtube). RODE full-sweep geomean 0.943 -> 0.995 vs cuSPARSE
   (192/192 correct); com-youtube 1.93x.
2. Path composition: at d-bar~5, 43-60% of nnz flow through RODE's residual
   path (98% of youtube rows) — a design-coverage property of the 32-aligned
   block/residual split, not a physical limit.
3. Threshold sweep T in {32..256} x N in {64,128,512} x 6 graphs: FLAT
   (max 4% on one config class; geomean <<1%). Coverage != performance,
   confirmed. Default T=128 retained.
4. Three-tier byte bounds (perfect-reuse floor / no-reuse edgewise /
   LRU cache-capacity-aware, model optimistic on random columns):
   measured/cache-LB = 1.19-1.20 (com-youtube: near realistic limit),
   1.6-1.7 (arxiv), 1.9-2.0 (twitter: real headroom, needs locality
   scheduling), cv2p5 bracketed [746MB..edgewise 5.46GB], measured ~ edgewise.
5. Head-to-head ZO vs post-fix RODE (24 configs): ZO wins 19 (up to 1.88x),
   ties 3, RODE wins only cv>=2.5 synthetics at N=64 by ~5% — configs whose
   oracle is TC_DIRECT at ~2x regardless. ZO's whole-row binned-subwarp +
   chunk architecture IS the combined design; RODE's split costs an extra
   C read-modify-write on every row with both parts.
6. Leave-one-out oracle (round-5 RODE times included): RODE unique wins = 0,
   LOO delta = +0.00%. Every other kernel has measurable unique value
   (TC_TF32 -2.47%, TC -2.34%, CT -2.19%, CT_TF32 -0.69%, CSR -0.54%,
   SH -0.28%, ZO -0.13%).
7. Cold budget preserved: RODE build max 52 ms, ZO max 30 ms (O(M) only).

VERDICT (empirical phrasing, per protocol): after a fair optimization round
(coverage analysis, defect fix worth 1.93x, threshold sweep, byte bounds),
RODE_ENHANCED is empirically dominated on the evaluated suite - zero unique
oracle wins warm or cold, dominated 19/24 by ZERO_OVERHEAD_CSR head-to-head.
Recommendation: retire RODE from the deployed roster (retain in-tree as the
R1.5 ablation evidence); ZO already embodies the merged CSR-specialist
design. No physics-limited claim is made except com-youtube, where measured
traffic is within 20% of the cache-aware bound.


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

### E1b — CT/SH TF32 sweeps (2026-07-14)

`fgcs_results/revision/tf32/e1_ctsh_full.csv`, zero gate failures. All three
tile kernels show the SAME degree-gated pattern (warm tf32/fp16 geomean per
degree bucket):

| deg bucket | TC_DIRECT | COMMUNITY_TC | SEGMENT_HYBRID |
|---|---|---|---|
| <3   | 1.408 (12/12) | 1.323 (12/12) | 1.295 (12/12) |
| 3–5  | 1.200 (8/8)   | 1.203 (8/8)   | 1.178 (8/8)   |
| 5–8  | 1.095         | 1.083         | 1.077         |
| 8–15 | 0.850 (4/24)  | 0.934 (8/24)  | 0.853 (4/24)  |
| ≥15  | 0.68–0.69     | 0.69–0.70     | 0.69–0.70     |

The uniform rule `deg<5 or (M<=25000 and deg<9)` routes with **0 mispicks on
all three kernels**: family geomeans ×1.0468 / ×1.0433 / ×1.0363. The
external-review hypothesis that COMMUNITY_TC's sort widens the TF32 win
domain is visible (deg 8–15: 8/24 wins vs 4/24) but too small to justify
per-kernel thresholds — R1 deploys ONE rule for the precision dimension.
Community-regime tile paths vs cuSPARSE improve: CT 1.130→1.180, SH
1.159→1.255 (tf32-routed).

### E3 Step 1 — ZC-BCRS verification (2026-07-14)

`experiments/zc_verify.py`: ZC output is **exactly bit-identical** to the
baseline kernels on 12/12 graph×N checks, both fp16 and tf32 paths (same mma
fragments → same accumulation order). Measured plan shrink 2.33–2.70×
(e.g. roadNet-TX 77.2→29.4 MB). A/B benchmark sweep + ncu (regs, occupancy,
DRAM, L2, stalls, per external review) in progress.

### E2a — staged smem B gather (2026-07-14)

`fs_tile_spmm_staged_kernel`(+tf32) in tc/ra_tc_direct.cu: warp-local 8×16 B
tile staged with one aligned 8-byte load per lane (vs 4 scattered 2-byte
fragment loads), 24-element row stride (conflict-free fragment reads), active
when N%64==0, RA_TC_STAGED=0 reverts. Correction to the mining doc's premise:
intra-block sector waste was already absorbed by L1 (all 4 warps touch the
full 128B row strip), so the win is load-instruction width/count, not sector
efficiency — hence moderate gains, not the hoped-for big lever.

Paired same-plan A/B, all 51 graphs (`revision/tf32/e2a_staged_ab.csv`):
staged/unstaged warm geomean **fp16 ×1.042, tf32 ×1.011** (DenseSmall +8.7%,
DenseLarge +8.1%, Mixed +7.0%, Skewed +3.8%, Community +2.4%, Uniform −0.5%
fp16 / +1.2% tf32). Only regression (roads fp16 −3..−5%) disappears under
precision routing (roads → tf32 where staged +1.9%). **Decision: staged is
the default.** E2b (cp.async double-buffer) gate met; queued after S1/C1.
CT/SH staged port queued with S1 edits.

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
