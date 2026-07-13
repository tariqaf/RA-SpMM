# Optimization Before/After

Ratios above 1.0 favor the after version. Timing includes the corrected asynchronous Python execution API; kernel duration is the primary Nsight kernel and isolates the launch-shape change. Stall values retain Nsight's per-issue-active units.

| Kernel | Timing pairs | Timing geomean | Timing wins | Profile pairs | Kernel geomean | Kernel wins |
|---|---:|---:|---:|---:|---:|---:|
| RODE_ENHANCED | 10 | 1.017600x | 8/10 | 6 | 1.008024x | 5/6 |
| SEGMENT_HYBRID | 10 | 1.019569x | 8/10 | 6 | 1.007110x | 3/6 |

See the CSV for per-graph occupancy, DRAM, barrier, and long-scoreboard changes.
