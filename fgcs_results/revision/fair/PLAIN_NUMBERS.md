# Plain Numbers Summary

All ratios use strict-correct rows and matching lifecycle regimes. Warm means reusable state plus execute-only timing after 50 warmups over 200 CUDA-event-timed executions. Cold means setup plus first execution.

## Core Sweep

- Coverage: 192 graph/width configurations, 1,344 rows, six RA-SpMM kernels plus cuSPARSE, zero soft failures, zero hard failures.
- Warm production router vs cuSPARSE: `1.007031x`.
- Warm six-kernel oracle vs cuSPARSE: `1.038749x`.
- Warm router/oracle: `0.969466x`; exact hits `143/192`; ratio >= 0.85 on `181/192`; ratio >= 0.99 on `164/192`.
- Cold production throughput router vs cuSPARSE: `0.061975x`.
- Cold six-kernel oracle vs cuSPARSE: `2.340051x`; CSR_DIRECT wins `178/192` cold configurations.

| Kernel | Warm vs cuSPARSE | Cold vs cuSPARSE | Best warm category |
|---|---:|---:|---|
| CSR_DIRECT | 0.817430x | 2.269110x | Sparse Skewed: 0.960296x |
| RODE_ENHANCED | 0.790025x | 0.947005x | Sparse Skewed: 0.927454x |
| ZERO_OVERHEAD_CSR | 0.805705x | 1.355221x | Sparse Uniform: 0.924082x |
| TC_DIRECT | 0.927541x | 0.048603x | Sparse Uniform: 1.156603x |
| COMMUNITY_TC | 0.943182x | 0.017104x | Sparse Uniform: 1.239344x |
| SEGMENT_HYBRID | 0.789102x | 0.568678x | Dense Large-Scale: 0.918081x |

## Lifecycle And Routing

- Production full-feature extraction over 51 graphs: mean `2958.233 ms`, median `691.938 ms`, maximum `63392.662 ms` on ogbn-products.
- Offline measured-oracle lifecycle including feature cost: K=1 `0.010706x`; K=1000 `0.713015x` versus matching cuSPARSE.
- Deployed model policy, graph-grouped out-of-fold: router/oracle K=1 `0.970332x`, K=1000 `0.913934x`; including feature cost, cuSPARSE ratios are `0.010702x` and `0.661982x`.
- Warm production rules beat every tested graph-grouped learned selector: rules `0.969466x` of oracle and 143 hits; strongest learned ratio is random forest `0.944891x` with 110 hits.

## External Systems

| System | Strict-correct coverage | Warm vs cuSPARSE | Cold vs cuSPARSE |
|---|---:|---:|---:|
| PyG | 192/192 | 0.416713x | 0.829477x |
| HC-SpMM | 14/192 | 1.111136x | 0.816382x |
| MP-SpMM | 33/192 | 1.997537x | 0.001370x |
| DTC identity order | 36/192 | 0.000777x | 0.000070x |
| dense cuBLAS | 24/24 | 0.139078x | 0.646846x |

DTC is scoped to M <= 100,000 and MP-SpMM to nnz <= 5,000,000; all skipped, failed, incorrect, and unsupported points remain explicit status rows. Corrected FlashSparse/Ada results require the separate RTX 4090 host and are not claimed here.

## End-To-End GNN

- Strict correctness: 96/96 backend/model/dataset rows.
- Production router across 24 model/dataset points: warm `0.851896x`, cold `0.037706x` versus cuSPARSE.
- GCN router: warm `0.862214x`, cold `0.031832x`.
- GraphSAGE router: warm `0.842596x`, cold `0.033631x`.
- GIN router: warm `0.850993x`, cold `0.050077x`.

## Profiling And Optimization

- Nsight coverage: 372 kernel/graph/width profiles; executed HMMA instructions: zero.
- Dominant limits: DRAM throughput and long-scoreboard stalls; RODE_ENHANCED and SEGMENT_HYBRID also showed N=128 barrier pressure.
- Accepted active-warp launch change: fixed-case warm geomean before/after `1.017600x` for RODE_ENHANCED and `1.019569x` for SEGMENT_HYBRID. On Amazon Photo N=128, barrier pressure fell from about 51.9 to 9.9 and primary-kernel time improved 4.35% and 3.80%.
