# External Systems

Only strict-correct points contribute. Coverage is reported because native-width, crash, timeout, and memory limitations differ by system.

| System | Correct / attempted | Warm vs cuSPARSE | Cold vs cuSPARSE |
|---|---:|---:|---:|
| HC-SpMM | 14 / 192 | 1.111136x | 0.816382x |
| MP-SpMM | 33 / 192 | 1.997537x | 0.001370x |
| cuBLAS dense | 24 / 24 | 0.139078x | 0.646846x |
| PyG | 192 / 192 | 0.416713x | 0.829477x |
| DTC | 36 / 192 | 0.000777x | 0.000070x |
