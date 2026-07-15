# DTC-SpMM four-way comparison (2026-07-16)

Warm, kernel-only, EXE_TIME-normalized. Router = RA-SpMM deployed rule router.
cuSPARSE = FP32 warm baseline. Two DTC modes, same fair TC tolerance
(1e-3·√max_row_nnz·10), same per-variant subprocess isolation and selection:
- **DTC-noTCA**: original CSR ordering (identity), the DTC kernel with no reorder.
- **DTC-TCA**: DTC's TCA proper-order reorder (its best published configuration).

## Timing correction applied
DTC's bindings run the kernel `EXE_TIME=1000` times per call
(`DTCSpMM_kernel.cu:18`) and return only the output; all mean/std/cold/e2e
kernel times are divided by 1000 to per-op. Every pre-2026-07-15 DTC absolute
time was 1000× too large. The TCA-vs-noTCA *ratio* is unaffected.

## Coverage roll-up (92 configs = 26 graphs × {64,128,256,512}, M≤3M)
| Mode | ran+correct | ran-incorrect | failed |
|---|---|---|---|
| DTC-noTCA | 79 | 2 | 11 (timeouts + all-variants-failed on Reddit/ogbn-products) |
| DTC-TCA | 50 | 2 | 40 (**all 40 = TCA reorder tool failure**, not the kernel) |

DTC-TCA reorder failures (13 graphs): Reddit, Yelp, com-youtube, gplus-combined,
ogbn-arxiv, ogbn-products, ogbn-proteins, roadNet-{CA,PA,TX}, soc-Pokec,
web-{Google,Stanford}. roadNet times out or aborts rc=-6; large graphs exceed
the 30-min reorder budget.

Both modes: same 2 incorrect configs — amazon-computers N=64/128
(max_error ≈ 40–60), genuine DTC numerical failures, correctly gated out.

## Geomeans

### Common set (50 configs where BOTH DTC modes ran) — apples-to-apples
| Comparison | Geomean |
|---|---|
| DTC-noTCA vs cuSPARSE | 0.904× |
| DTC-TCA vs cuSPARSE | 1.003× |
| **TCA benefit (TCA/noTCA)** | **1.109×** |
| Router vs cuSPARSE | 1.603× |
| **Router vs DTC-TCA** | **1.598× (router wins 38/50)** |
| **Router vs DTC-noTCA** | **1.773× (router wins 50/50)** |

### Each mode over its own full valid set
| Comparison | Geomean |
|---|---|
| DTC-noTCA vs cuSPARSE (n=79) | 1.021× |
| Router vs DTC-noTCA (n=79) | 1.614× (router wins 79/79) |
| DTC-TCA vs cuSPARSE (n=50) | 1.003× |
| Router vs DTC-TCA (n=50) | 1.598× (router wins 38/50) |

### "Kernel works, TCA fails" isolation subset (29 configs, 11 graphs where TCA reorder failed)
DTC-noTCA kernel vs cuSPARSE **1.260×**, Router vs cuSPARSE 1.729×.
→ On these graphs DTC's **kernel runs fine**; only its **TCA preprocessing tool**
cannot produce an ordering. Cleanly separates kernel capability from pipeline fragility.

## Where DTC-TCA beats the router (12 configs, all large power-law graphs)
com-Amazon (1.37–1.50×), Amazon0601 (1.30–1.33×), com-DBLP (1.17–1.21×) —
DTC's design sweet spot. But the reorder that unlocks these costs
**600–900 s per graph** (Amazon0601 913 s, Flickr 619 s), vs the router's
millisecond planning. Millions of SpMMs needed to amortize.

## Paper-ready statement
On the evaluated suite, RA-SpMM's router outperforms DTC-SpMM by **1.60×
geomean with TCA** and **1.61–1.77× without TCA**, winning **38/50** (TCA) and
**all** no-TCA configs. DTC-TCA is competitive only on large power-law graphs,
and only after a 600–900 s offline reorder its own tool fails to complete on
13/26 graphs. Kernel-only, not counting DTC's preprocessing at all.
