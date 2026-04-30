# Post-sweep report — FGCS 170-pt evaluation

Source CSV: `fgcs_results/spmm/all_graphs_results.csv`  
Total rows: 1344  
Real graphs: 26, Synthetic graphs: 25  
Total (graph, N) pairs: 192

## Per-kernel geomean speedup vs cuSPARSE

| Kernel | Real (pts) | Synth (pts) | Combined |
|---|---:|---:|---:|
| CSR_DIRECT | 2.459× (92) | 2.042× (100) | 2.232× |
| RODE_ENHANCED | 2.426× (92) | 1.808× (100) | 2.082× |
| ZERO_OVERHEAD_CSR | 1.392× (92) | 0.875× (100) | 1.093× |
| TC_DIRECT | 2.829× (92) | 2.187× (100) | 2.474× |
| COMMUNITY_TC | 2.957× (92) | 2.163× (100) | 2.513× |
| SEGMENT_HYBRID | 2.415× (92) | 1.808× (100) | 2.077× |

## Router quality

- Oracle geomean speedup vs cuSPARSE: **2.663×**  
- Router geomean speedup vs cuSPARSE: **2.649×**  
- Router overhead (oracle / router): **1.005×**  
- Router hit rate: **166/192 (86.5%)**

### Per-category breakdown

| Category | Hits | Oracle | Router | Overhead | (real / synth) |
|---|---:|---:|---:|---:|---|
| Community | 31/31 | 2.349× | 2.349× | 1.000× | 11 / 20 |
| Dense Large-Scale | 10/11 | 1.200× | 1.194× | 1.005× | 11 / 0 |
| Dense Small | 28/40 | 6.871× | 6.842× | 1.004× | 20 / 20 |
| Mixed/Irregular | 32/36 | 3.163× | 3.156× | 1.002× | 16 / 20 |
| Sparse Skewed | 22/27 | 1.336× | 1.335× | 1.000× | 7 / 20 |
| Sparse Uniform | 43/47 | 2.026× | 1.998× | 1.014× | 27 / 20 |

### Router misses (router pick != oracle pick)

| Dataset | N | Category | Router pick | Oracle pick | Ratio |
|---|---:|---|---|---|---:|
| Yelp | 128 | Sparse Uniform | TC_DIRECT | CSR_DIRECT | 0.525× |
| amazon-photo | 64 | Dense Small | SEGMENT_HYBRID | COMMUNITY_TC | 0.929× |
| gplus-combined | 128 | Dense Large-Scale | TC_DIRECT | CSR_DIRECT | 0.943× |
| Cora | 128 | Mixed/Irregular | TC_DIRECT | COMMUNITY_TC | 0.963× |
| Cora | 64 | Mixed/Irregular | TC_DIRECT | COMMUNITY_TC | 0.967× |
| ca-HepTh | 512 | Dense Small | TC_DIRECT | COMMUNITY_TC | 0.977× |
| ca-HepTh | 256 | Dense Small | TC_DIRECT | COMMUNITY_TC | 0.981× |
| ca-HepTh | 128 | Dense Small | TC_DIRECT | COMMUNITY_TC | 0.987× |
| synth_dense_small_d50 | 512 | Dense Small | COMMUNITY_TC | TC_DIRECT | 0.988× |
| synth_dense_small_d30 | 512 | Dense Small | COMMUNITY_TC | TC_DIRECT | 0.988× |
| synth_dense_small_d50 | 64 | Dense Small | COMMUNITY_TC | TC_DIRECT | 0.993× |
| synth_dense_small_d30 | 128 | Dense Small | COMMUNITY_TC | TC_DIRECT | 0.995× |
| synth_dense_small_d120 | 128 | Dense Small | COMMUNITY_TC | TC_DIRECT | 0.996× |
| roadNet-PA | 64 | Sparse Uniform | COMMUNITY_TC | TC_DIRECT | 0.997× |
| synth_sparse_skewed_cv4p0 | 512 | Sparse Skewed | TC_DIRECT | RODE_ENHANCED | 0.997× |
| synth_sparse_skewed_cv3p0 | 64 | Sparse Skewed | TC_DIRECT | COMMUNITY_TC | 0.997× |
| synth_sparse_skewed_cv2p5 | 256 | Sparse Skewed | TC_DIRECT | COMMUNITY_TC | 0.998× |
| roadNet-CA | 64 | Sparse Uniform | COMMUNITY_TC | TC_DIRECT | 0.998× |
| CiteSeer | 512 | Mixed/Irregular | SEGMENT_HYBRID | RODE_ENHANCED | 0.998× |
| synth_sparse_skewed_cv2p5 | 64 | Sparse Skewed | TC_DIRECT | COMMUNITY_TC | 0.999× |
| PPI | 64 | Mixed/Irregular | TC_DIRECT | COMMUNITY_TC | 0.999× |
| amazon-computers | 128 | Dense Small | SEGMENT_HYBRID | RODE_ENHANCED | 0.999× |
| amazon-photo | 512 | Dense Small | SEGMENT_HYBRID | RODE_ENHANCED | 1.000× |
| synth_sparse_uniform_d18 | 256 | Sparse Uniform | TC_DIRECT | COMMUNITY_TC | 1.000× |
| amazon-computers | 256 | Dense Small | SEGMENT_HYBRID | RODE_ENHANCED | 1.000× |
| synth_sparse_skewed_cv2p5 | 512 | Sparse Skewed | TC_DIRECT | COMMUNITY_TC | 1.000× |
