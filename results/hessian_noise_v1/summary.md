# Is GPTQ's Hessian noisy enough to explain the g64 instability?

Reference statistics from 524,288 tokens of separate text; 4 disjoint draws at each of 4 calibration sizes. Codebooks are deterministic fits over weights, so the Hessian is the only thing that varies.

Reference contrast (g64 against g128): -0.002714.

| tokens | mean ‖ΔH‖/‖H‖ | mean ‖ΔU‖/‖U‖ | contrast range | contrast sd |
|---:|---:|---:|---:|---:|
| 16,384 | 0.30675 | 0.39711 | 0.009269 | 0.003906 |
| 32,768 | 0.20851 | 0.30326 | 0.014760 | 0.006139 |
| 65,536 | 0.14364 | 0.23676 | 0.003991 | 0.001996 |
| 131,072 | 0.11774 | 0.19541 | 0.004366 | 0.001813 |

## The three predictions

1. Hessian deviation against tokens, log-log slope **-0.468** (compensation operator -0.343); finite-sample noise predicts $-0.5$.
2. Contrast spread against tokens, log-log slope of the standard deviation: **-0.494**.
3. Within-size correlation between compensation-operator deviation and contrast deviation: **-0.142** over 16 draws.
