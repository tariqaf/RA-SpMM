# Round-3 R1 Report — Recalibrated Router with Precision Dimension

Data: `final_fair_v3.csv` (192 configs × 9 custom kernels + cuSPARSE, RTX 3090,
50/200 warm + 10-cold protocol, strict gate). Router: `ra_router_eval.py`
`route_with_rules` ↔ `router/router_dispatch.cpp` (mirrored).

## Headline (warm, vs FP32 cuSPARSE)

| Metric | Value |
|---|---|
| Oracle geomean | **1.706×** |
| Router geomean | **1.702×** |
| Router/Oracle | **0.9977** |
| Oracle hit rate | 172/192 (89.6%) |
| Worst single-config router/oracle | 0.868 (synth_community_nc10, N=128) |
| Configs with router/oracle ≥ 0.85 | 192/192 |
| Correctness | **192/192 strict-gate pass** (all 9 kernels; 0 soft/hard fails in 1920 rows) |
| Python/C++ router parity | **192/192 PASS** |

## Per-regime router geomeans and oracle kernels (warm)

| Regime | Router | Oracle | Hits | Dominant oracle kernels |
|---|---|---|---|---|
| Community | 1.669× | 1.680× | 25/31 | TC_DIRECT_TF32 / CSR_DIRECT (narrow N) |
| Dense Large-Scale | 2.661× | 2.661× | 11/11 | COMMUNITY_TC |
| Dense Small | 1.429× | 1.432× | 37/40 | TC_DIRECT, SEGMENT_HYBRID |
| Mixed/Irregular | 1.802× | 1.806× | 32/36 | TC_DIRECT (+ TF32 on tiny) |
| Sparse Skewed | 1.847× | 1.847× | 27/27 | TC_DIRECT |
| Sparse Uniform | 1.645× | 1.648× | 40/47 | TC_DIRECT_TF32 (roads wide N), CSR_DIRECT |

## Selection counts (192 configs)

| Pick | Count |
|---|---|
| TC_DIRECT | 75 |
| COMMUNITY_TC | 46 |
| TC_DIRECT_TF32 | 32 |
| CSR_DIRECT | 20 |
| COMMUNITY_TC_TF32 | 8 |
| SEGMENT_HYBRID | 6 |
| ZERO_OVERHEAD_CSR | 3 |
| CUSPARSE (guardrail) | 2 |

TF32 selections: **40/192 (20.8%)**. ZC-resident selections: **0** — resident
ZC is a cold-critical/memory-lean option, never picked by the warm router
(the ZC *expansion builder* is used by every tile plan; it is bit-identical
and build-time-only). cuSPARSE guardrail: 2 (rule 6, small dense narrow N).

## Cold

Warm-fitted router evaluated on cold (build+exec) times: 0.207× vs cuSPARSE
(cold oracle 3.034×, dominated by zero-preprocessing CSR_DIRECT at 2.977×).
Unchanged story from v2: the router optimizes the amortized/steady regime; the
preprocessing-aware tile gate bounds giant cold exposure, and round-3 cut tile
cold build 1.8–4.5× (products 4.16→2.10 s, further 1.16 s with resident ZC).

## Diff from corrected Round 2

| Stage | Router geomean vs cuSPARSE (warm) | Evidence |
|---|---|---|
| Corrected Round 2 | 1.571× (R/O 0.9938) | fair_v2 + router v2 [measured] |
| + E1 TF32 paths | ×1.036–1.047 per tile family | paired A/B, 0 mispicks [measured] |
| + E3 ZC expansion builder | warm ×1.000 (bit-identical); cold build 1.8–4.5× | exact-equality + cold timing [measured] |
| + E2a staged B tiles | ×1.042 fp16 / ×1.011 tf32 | paired same-plan A/B [measured] |
| + S1 atomic-free SH | SH warm 1.29→1.323 incl. staging; deterministic | final sweep [derived] |
| **Final R1 router** | **1.702× (R/O 0.9977)** | final_fair_v3 + refit [measured] |

Individual deltas are paired measurements; they do not sum linearly to the
cumulative number (overlapping regimes) — the 1.702× is the direct measurement.

## Router changes vs v2 (mirrored Python ↔ C++, parity 192/192)

1. Precision dimension: tile picks run TF32 when `d̄<5 or (M≤25k and d̄<9)`;
   forced TF32 for web-locality rule 8 and the community branch at N≥256
   (locality keeps the doubled FP32 gather cache-resident).
2. Rule 3: `d̄<4.5` → CSR only for N≤256 (staged/TF32 tile wins wider);
   community branch CSR only for N≤128; dropped the uniform-dense-small
   N≤64 CSR clause (round-3 tile now wins it).
3. Rule 6: SEGMENT_HYBRID window narrowed to N<256.
