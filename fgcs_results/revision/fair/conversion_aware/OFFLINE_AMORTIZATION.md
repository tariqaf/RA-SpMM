# Offline Conversion-Aware Amortization

This is a post-hoc analysis over all measured correct kernels, not the deployed router. Each lifecycle compares matching costs: first-call setup plus execution, followed by warm execution.

- Cold K=1 geomean vs cold cuSPARSE: 0.010706x (192 points)
- K=1000 lifecycle geomean vs the matching cuSPARSE lifecycle: 0.713015x (192 points)
- Missing or incorrect measurement records: 0
