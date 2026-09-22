# Controlled recurrent-memory quantization results

Scope: trained quantizers on three sampled, frozen linear systems; no task-model, Qwen or quantum-hardware training. All evaluations use exported and decoded operators. Relative output MSE is lower-is-better; predicting zero gives exactly 1. Values are mean ± sample SD across three systems, not significance estimates.

| Method | 4-bit, L=128 | 4-bit, L=512 | 8-bit, L=128 | 8-bit, L=512 |
|---|---:|---:|---:|---:|
| FP32 reference | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| Cartesian nearest | 7.51e+33 ± 6.56e+33 | 1.17e+148 ± 1.01e+148 | 1.31e+03 ± 637 | 1.94e+19 ± 1.35e+19 |
| Stable Cartesian weight | 1.14 ± 0.185 | 1.3 ± 0.221 | 0.841 ± 0.0222 | 0.998 ± 0.0736 |
| Stable Cartesian memory | 0.852 ± 0.0452 | 0.936 ± 0.0312 | 0.789 ± 0.0261 | 0.922 ± 0.0267 |
| Normalized Cartesian memory | 1.11 ± 0.0338 | 1.31 ± 0.0695 | 0.35 ± 0.0921 | 0.998 ± 0.177 |
| Phase nearest | 1.5 ± 0.188 | 1.77 ± 0.213 | 0.117 ± 0.0241 | 0.939 ± 0.0904 |
| Phase weight | 1.61 ± 0.109 | 1.78 ± 0.206 | 0.094 ± 0.0306 | 0.811 ± 0.177 |
| Phase memory | 0.967 ± 0.0736 | 1.25 ± 0.0825 | 0.0943 ± 0.0327 | 0.791 ± 0.198 |

## Interpretation

At four index bits, phase-memory calibration improves over phase-weight calibration, but it does not outperform the stable Cartesian memory control at L=512. Phase-memory output error exceeds 1 at that horizon, so it is worse than predicting zero. This is evidence against claiming a useful long-context phase advantage from this pilot. See the eight-bit results above to assess the higher-rate setting separately.

The unstable Cartesian nearest row is a diagnostic, not a competitive practical baseline. The fitted Cartesian weight and memory controls start from the same stable scale and enforce candidate radii no larger than the reference radii. The normalized Cartesian control also preserves radii.

At four bits: operator files 304 bytes vs FP32 524; complete system including unchanged projections 4,400 bytes vs 4,620. At eight bits: operator files 336 bytes, whole system 4,432. Whole-system savings are about 4.8% and 4.1%, respectively. This study does not demonstrate whole-model low-bit quantization or inference acceleration.

The pilot varies system seed; Gaussian probes are fresh and deterministic for each horizon. It has no learned free codebook, nonlinear hybrid model, or task accuracy result. Calibration and candidate search budgets are fixed, not tuned using test outcomes.

Version 1 is retained as a numerical diagnostic with unconstrained fitted Cartesian controls; version 2 stabilizes these controls and is the version tabulated here. The phase-family results are unchanged between versions at four bits. Full raw metrics and training histories are retained in each run directory.
