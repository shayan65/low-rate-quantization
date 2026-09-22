# Qwen3.5-0.8B: codec ladder and self-consistent calibration

All 18 DeltaNet `in_proj_qkv` projections replaced simultaneously; 261,248 WikiText-2 validation targets; equal 4-bit index budget.
BF16 reference NLL 3.436670 (ppl 31.083).

| Method | NLL | ppl | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|:--|
| real4_uniform | 3.459241 | 31.793 | +0.022571 | [+0.021615, +0.023511] |
| real4_lloyd | 3.451448 | 31.546 | +0.014778 | [+0.013951, +0.015582] |
| polar_legacy | 3.456446 | 31.704 | +0.019776 | [+0.018821, +0.020732] |
| polar_fitted | 3.452687 | 31.585 | +0.016016 | [+0.015169, +0.016830] |
| vq2d | 3.446460 | 31.389 | +0.009789 | [+0.009125, +0.010448] |
| real4_lloyd+scf | 3.451302 | 31.541 | +0.014631 | [+0.013804, +0.015436] |
| polar_fitted+scf | 3.452739 | 31.587 | +0.016068 | [+0.015219, +0.016903] |
| vq2d+scf | 3.449180 | 31.475 | +0.012509 | [+0.011853, +0.013153] |
