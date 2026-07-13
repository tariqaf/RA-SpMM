# Profiling Bottlenecks

Means are grouped by implementation, feature width, and structural category. Stall values preserve Nsight's `warps_issue_stalled_*_per_issue_active` metric and are not percentages.

| Kernel | N | Category | Profiles | Occupancy % | DRAM % | AI | HMMA | Dominant stalls |
|---|---:|---|---:|---:|---:|---:|---:|---|
| COMMUNITY_TC | 128 | Community | 4 | 79.79 | 74.04 | 0.762 | 0 | long_scoreboard 54.7; wait 1.9; selected 1.0 |
| COMMUNITY_TC | 128 | Dense Large-Scale | 4 | 78.12 | 72.81 | 1.151 | 0 | long_scoreboard 77.0; wait 1.6; lg_throttle 1.4 |
| COMMUNITY_TC | 128 | Dense Small | 6 | 63.44 | 21.03 | 4.006 | 0 | long_scoreboard 39.7; wait 1.8; selected 1.0 |
| COMMUNITY_TC | 128 | Mixed/Irregular | 5 | 52.41 | 27.04 | 2.388 | 0 | long_scoreboard 50.7; wait 1.9; imc_miss 1.3 |
| COMMUNITY_TC | 128 | Sparse Skewed | 3 | 80.69 | 75.86 | 0.874 | 0 | long_scoreboard 102.4; wait 1.7; selected 1.0 |
| COMMUNITY_TC | 128 | Sparse Uniform | 9 | 85.13 | 88.04 | 0.868 | 0 | long_scoreboard 56.2; wait 1.9; selected 1.0 |
| COMMUNITY_TC | 512 | Community | 4 | 77.67 | 76.67 | 0.601 | 0 | long_scoreboard 81.4; wait 1.8; selected 1.0 |
| COMMUNITY_TC | 512 | Dense Large-Scale | 4 | 78.67 | 87.18 | 0.650 | 0 | long_scoreboard 108.3; lg_throttle 2.7; wait 1.6 |
| COMMUNITY_TC | 512 | Dense Small | 6 | 65.08 | 41.11 | 1.382 | 0 | long_scoreboard 58.7; wait 1.7; selected 1.0 |
| COMMUNITY_TC | 512 | Mixed/Irregular | 5 | 49.85 | 30.49 | 2.001 | 0 | long_scoreboard 61.0; wait 1.8; selected 1.0 |
| COMMUNITY_TC | 512 | Sparse Skewed | 3 | 80.87 | 84.04 | 0.597 | 0 | long_scoreboard 121.6; wait 1.7; selected 1.0 |
| COMMUNITY_TC | 512 | Sparse Uniform | 9 | 85.30 | 89.17 | 0.729 | 0 | long_scoreboard 81.6; wait 1.8; selected 1.0 |
| CSR_DIRECT | 128 | Community | 4 | 75.13 | 76.06 | 0.540 | 0 | long_scoreboard 76.1; wait 1.9; selected 1.0 |
| CSR_DIRECT | 128 | Dense Large-Scale | 4 | 60.22 | 75.23 | 1.106 | 0 | long_scoreboard 68.2; wait 1.5; selected 1.0 |
| CSR_DIRECT | 128 | Dense Small | 6 | 47.67 | 21.73 | 3.803 | 0 | long_scoreboard 35.0; wait 1.7; selected 1.0 |
| CSR_DIRECT | 128 | Mixed/Irregular | 5 | 31.61 | 26.74 | 2.409 | 0 | long_scoreboard 37.8; wait 1.7; selected 1.0 |
| CSR_DIRECT | 128 | Sparse Skewed | 3 | 65.17 | 81.16 | 0.928 | 0 | long_scoreboard 78.2; wait 1.5; selected 1.0 |
| CSR_DIRECT | 128 | Sparse Uniform | 9 | 78.99 | 89.79 | 0.555 | 0 | long_scoreboard 76.8; wait 1.8; selected 1.0 |
| CSR_DIRECT | 512 | Community | 4 | 75.35 | 77.65 | 0.465 | 0 | long_scoreboard 104.9; wait 1.8; selected 1.0 |
| CSR_DIRECT | 512 | Dense Large-Scale | 4 | 59.82 | 88.98 | 0.636 | 0 | long_scoreboard 84.9; wait 1.4; selected 1.0 |
| CSR_DIRECT | 512 | Dense Small | 6 | 49.53 | 40.93 | 1.249 | 0 | long_scoreboard 49.8; wait 1.6; selected 1.0 |
| CSR_DIRECT | 512 | Mixed/Irregular | 5 | 31.03 | 30.41 | 1.961 | 0 | long_scoreboard 45.1; wait 1.7; selected 1.0 |
| CSR_DIRECT | 512 | Sparse Skewed | 3 | 62.60 | 88.77 | 0.649 | 0 | long_scoreboard 89.4; wait 1.5; selected 1.0 |
| CSR_DIRECT | 512 | Sparse Uniform | 9 | 79.72 | 91.88 | 0.480 | 0 | long_scoreboard 106.4; wait 1.8; selected 1.0 |
| RODE_ENHANCED | 128 | Community | 4 | 41.19 | 70.30 | 0.600 | 0 | long_scoreboard 24.1; barrier 21.9; wait 2.0 |
| RODE_ENHANCED | 128 | Dense Large-Scale | 4 | 82.66 | 56.03 | 1.497 | 0 | barrier 83.0; long_scoreboard 11.3; wait 1.7 |
| RODE_ENHANCED | 128 | Dense Small | 6 | 33.44 | 32.96 | 3.164 | 0 | barrier 17.5; long_scoreboard 15.1; wait 1.8 |
| RODE_ENHANCED | 128 | Mixed/Irregular | 5 | 21.50 | 19.79 | 0.796 | 0 | barrier 41.0; long_scoreboard 17.4; wait 1.6 |
| RODE_ENHANCED | 128 | Sparse Skewed | 3 | 33.64 | 86.27 | 0.759 | 0 | long_scoreboard 38.0; wait 1.5; selected 1.0 |
| RODE_ENHANCED | 128 | Sparse Uniform | 9 | 34.31 | 90.86 | 0.533 | 0 | long_scoreboard 28.7; wait 2.0; selected 1.0 |
| RODE_ENHANCED | 512 | Community | 4 | 93.64 | 93.95 | 0.503 | 0 | long_scoreboard 106.9; wait 1.6; selected 1.0 |
| RODE_ENHANCED | 512 | Dense Large-Scale | 4 | 86.52 | 91.77 | 0.719 | 0 | long_scoreboard 83.9; barrier 43.3; wait 1.4 |
| RODE_ENHANCED | 512 | Dense Small | 6 | 75.17 | 63.07 | 1.309 | 0 | long_scoreboard 71.7; barrier 7.1; wait 1.5 |
| RODE_ENHANCED | 512 | Mixed/Irregular | 5 | 61.66 | 54.67 | 0.670 | 0 | long_scoreboard 74.7; barrier 6.0; wait 1.6 |
| RODE_ENHANCED | 512 | Sparse Skewed | 3 | 96.13 | 92.19 | 0.680 | 0 | long_scoreboard 129.5; wait 1.3; selected 1.0 |
| RODE_ENHANCED | 512 | Sparse Uniform | 9 | 92.41 | 93.87 | 0.495 | 0 | long_scoreboard 109.9; wait 1.6; selected 1.0 |
| SEGMENT_HYBRID | 128 | Community | 4 | 41.24 | 70.31 | 0.600 | 0 | long_scoreboard 24.1; barrier 21.9; wait 2.0 |
| SEGMENT_HYBRID | 128 | Dense Large-Scale | 4 | 82.65 | 56.02 | 1.496 | 0 | barrier 83.0; long_scoreboard 11.3; wait 1.7 |
| SEGMENT_HYBRID | 128 | Dense Small | 6 | 33.18 | 33.05 | 3.166 | 0 | barrier 17.5; long_scoreboard 15.0; wait 1.8 |
| SEGMENT_HYBRID | 128 | Mixed/Irregular | 5 | 21.48 | 19.79 | 0.796 | 0 | barrier 41.0; long_scoreboard 17.4; wait 1.6 |
| SEGMENT_HYBRID | 128 | Sparse Skewed | 3 | 33.64 | 86.58 | 0.749 | 0 | long_scoreboard 38.0; wait 1.5; selected 1.0 |
| SEGMENT_HYBRID | 128 | Sparse Uniform | 9 | 34.29 | 91.02 | 0.529 | 0 | long_scoreboard 28.7; wait 2.0; selected 1.0 |
| SEGMENT_HYBRID | 512 | Community | 4 | 93.02 | 93.71 | 0.503 | 0 | long_scoreboard 106.5; wait 1.6; selected 1.0 |
| SEGMENT_HYBRID | 512 | Dense Large-Scale | 4 | 86.60 | 91.77 | 0.719 | 0 | long_scoreboard 84.2; barrier 43.2; wait 1.4 |
| SEGMENT_HYBRID | 512 | Dense Small | 6 | 75.51 | 63.28 | 1.306 | 0 | long_scoreboard 71.8; barrier 7.1; wait 1.5 |
| SEGMENT_HYBRID | 512 | Mixed/Irregular | 5 | 61.60 | 54.44 | 0.670 | 0 | long_scoreboard 75.2; barrier 6.0; wait 1.6 |
| SEGMENT_HYBRID | 512 | Sparse Skewed | 3 | 95.82 | 92.37 | 0.670 | 0 | long_scoreboard 129.4; wait 1.3; selected 1.0 |
| SEGMENT_HYBRID | 512 | Sparse Uniform | 9 | 92.44 | 93.90 | 0.491 | 0 | long_scoreboard 109.3; wait 1.6; selected 1.0 |
| TC_DIRECT | 128 | Community | 4 | 82.78 | 77.03 | 0.669 | 0 | long_scoreboard 67.7; wait 1.9; selected 1.0 |
| TC_DIRECT | 128 | Dense Large-Scale | 4 | 84.00 | 77.03 | 1.121 | 0 | long_scoreboard 86.2; lg_throttle 2.5; wait 1.6 |
| TC_DIRECT | 128 | Dense Small | 6 | 65.50 | 22.93 | 3.942 | 0 | long_scoreboard 40.1; wait 1.8; selected 1.0 |
| TC_DIRECT | 128 | Mixed/Irregular | 5 | 53.59 | 27.53 | 2.391 | 0 | long_scoreboard 52.5; wait 1.9; imc_miss 1.4 |
| TC_DIRECT | 128 | Sparse Skewed | 3 | 82.87 | 80.90 | 0.779 | 0 | long_scoreboard 103.6; wait 1.7; selected 1.0 |
| TC_DIRECT | 128 | Sparse Uniform | 9 | 83.55 | 90.31 | 0.708 | 0 | long_scoreboard 60.7; wait 1.9; selected 1.0 |
| TC_DIRECT | 512 | Community | 4 | 81.95 | 78.65 | 0.544 | 0 | long_scoreboard 96.9; wait 1.8; selected 1.0 |
| TC_DIRECT | 512 | Dense Large-Scale | 4 | 85.05 | 88.99 | 0.643 | 0 | long_scoreboard 117.2; lg_throttle 3.7; wait 1.6 |
| TC_DIRECT | 512 | Dense Small | 6 | 67.57 | 42.59 | 1.315 | 0 | long_scoreboard 59.6; wait 1.7; selected 1.0 |
| TC_DIRECT | 512 | Mixed/Irregular | 5 | 51.51 | 30.83 | 2.004 | 0 | long_scoreboard 63.3; wait 1.8; selected 1.0 |
| TC_DIRECT | 512 | Sparse Skewed | 3 | 83.83 | 87.19 | 0.568 | 0 | long_scoreboard 127.6; wait 1.7; selected 1.0 |
| TC_DIRECT | 512 | Sparse Uniform | 9 | 83.41 | 90.79 | 0.623 | 0 | long_scoreboard 85.6; wait 1.8; selected 1.0 |
| ZERO_OVERHEAD_CSR | 128 | Community | 4 | 73.61 | 74.00 | 0.678 | 0 | long_scoreboard 60.9; wait 2.1; selected 1.0 |
| ZERO_OVERHEAD_CSR | 128 | Dense Large-Scale | 4 | 54.00 | 36.05 | 1.338 | 0 | long_scoreboard 37.4; wait 2.8; selected 1.0 |
| ZERO_OVERHEAD_CSR | 128 | Dense Small | 6 | 22.28 | 13.59 | 3.156 | 0 | long_scoreboard 20.7; wait 2.0; imc_miss 1.4 |
| ZERO_OVERHEAD_CSR | 128 | Mixed/Irregular | 5 | 25.78 | 38.32 | 1.078 | 0 | long_scoreboard 36.5; imc_miss 2.2; wait 1.8 |
| ZERO_OVERHEAD_CSR | 128 | Sparse Skewed | 3 | 42.44 | 57.21 | 0.835 | 0 | long_scoreboard 36.6; wait 2.5; selected 1.0 |
| ZERO_OVERHEAD_CSR | 128 | Sparse Uniform | 9 | 82.02 | 88.36 | 0.590 | 0 | long_scoreboard 74.6; wait 1.9; selected 1.0 |
| ZERO_OVERHEAD_CSR | 512 | Community | 4 | 82.20 | 84.79 | 0.570 | 0 | long_scoreboard 89.5; wait 1.9; selected 1.0 |
| ZERO_OVERHEAD_CSR | 512 | Dense Large-Scale | 4 | 92.62 | 75.59 | 0.745 | 0 | long_scoreboard 91.8; wait 2.2; selected 1.0 |
| ZERO_OVERHEAD_CSR | 512 | Dense Small | 6 | 65.95 | 71.38 | 0.911 | 0 | long_scoreboard 50.3; wait 1.8; selected 1.0 |
| ZERO_OVERHEAD_CSR | 512 | Mixed/Irregular | 5 | 40.80 | 48.96 | 0.648 | 0 | long_scoreboard 57.2; wait 1.8; selected 1.0 |
| ZERO_OVERHEAD_CSR | 512 | Sparse Skewed | 3 | 96.50 | 90.57 | 0.641 | 0 | long_scoreboard 131.8; wait 1.7; selected 1.0 |
| ZERO_OVERHEAD_CSR | 512 | Sparse Uniform | 9 | 88.06 | 93.09 | 0.514 | 0 | long_scoreboard 110.7; wait 1.8; selected 1.0 |
