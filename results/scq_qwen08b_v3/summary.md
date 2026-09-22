# Qwen3.5-0.8B: codec ladder and self-consistent calibration

All 18 DeltaNet `in_proj_qkv` projections replaced simultaneously; 261,248 WikiText-2 validation targets; equal 4-bit index budget.
BF16 reference NLL 3.436670 (ppl 31.083).

| Method | NLL | ppl | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|:--|
| real4_uniform | 3.458469 | 31.768 | +0.021798 | [+0.020841, +0.022752] |
| real4_lloyd | 3.451102 | 31.535 | +0.014431 | [+0.013626, +0.015235] |
| real4_lloyd+seq | 3.451124 | 31.536 | +0.014454 | [+0.013641, +0.015255] |
| polar_legacy | 3.456446 | 31.704 | +0.019776 | [+0.018821, +0.020732] |
| polar_legacy+seq | 3.456446 | 31.704 | +0.019776 | [+0.018821, +0.020732] |
| polar_fitted | 3.452258 | 31.572 | +0.015588 | [+0.014736, +0.016421] |
| polar_fitted+seq | 3.451662 | 31.553 | +0.014992 | [+0.014156, +0.015820] |
| vq2d | 3.445063 | 31.345 | +0.008393 | [+0.007720, +0.009069] |
| vq2d+seq | 3.445670 | 31.364 | +0.009000 | [+0.008321, +0.009679] |
