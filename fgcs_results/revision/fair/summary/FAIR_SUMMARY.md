# Corrected Fair Evaluation

## Warm

- Router policy: production rule tree (no measured-oracle or lifecycle look-ahead)
- Router geomean vs cuSPARSE: 1.007031x
- Oracle geomean vs cuSPARSE: 1.038749x
- Router/Oracle geomean: 0.969466x
- Exact oracle hits: 143/192 (74.48%)
- Empirical Router/Oracle >=0.85: 181/192
- Empirical Router/Oracle >=0.99: 164/192

- CSR_DIRECT: 0.817430x
- RODE_ENHANCED: 0.790025x
- ZERO_OVERHEAD_CSR: 0.805705x
- TC_DIRECT: 0.927541x
- COMMUNITY_TC: 0.943182x
- SEGMENT_HYBRID: 0.789102x

### Oracle Winners By Structural Regime

- Community: TC_DIRECT=20, COMMUNITY_TC=8, ZERO_OVERHEAD_CSR=3
- Dense Large-Scale: TC_DIRECT=5, RODE_ENHANCED=3, COMMUNITY_TC=2, SEGMENT_HYBRID=1
- Dense Small: COMMUNITY_TC=23, ZERO_OVERHEAD_CSR=9, TC_DIRECT=8
- Mixed/Irregular: TC_DIRECT=13, ZERO_OVERHEAD_CSR=10, SEGMENT_HYBRID=6, COMMUNITY_TC=3, CSR_DIRECT=2, RODE_ENHANCED=2
- Sparse Skewed: TC_DIRECT=10, COMMUNITY_TC=5, CSR_DIRECT=5, SEGMENT_HYBRID=4, RODE_ENHANCED=2, ZERO_OVERHEAD_CSR=1
- Sparse Uniform: COMMUNITY_TC=26, TC_DIRECT=21

## Cold

- Router policy: production rule tree (no measured-oracle or lifecycle look-ahead)
- Router geomean vs cuSPARSE: 0.061975x
- Oracle geomean vs cuSPARSE: 2.340051x
- Router/Oracle geomean: 0.026484x
- Exact oracle hits: 9/192 (4.69%)
- Empirical Router/Oracle >=0.85: 9/192
- Empirical Router/Oracle >=0.99: 9/192

- CSR_DIRECT: 2.269110x
- RODE_ENHANCED: 0.947005x
- ZERO_OVERHEAD_CSR: 1.355221x
- TC_DIRECT: 0.048603x
- COMMUNITY_TC: 0.017104x
- SEGMENT_HYBRID: 0.568678x

### Oracle Winners By Structural Regime

- Community: CSR_DIRECT=30, ZERO_OVERHEAD_CSR=1
- Dense Large-Scale: CSR_DIRECT=9, ZERO_OVERHEAD_CSR=2
- Dense Small: CSR_DIRECT=34, ZERO_OVERHEAD_CSR=6
- Mixed/Irregular: CSR_DIRECT=31, ZERO_OVERHEAD_CSR=5
- Sparse Skewed: CSR_DIRECT=27
- Sparse Uniform: CSR_DIRECT=47
