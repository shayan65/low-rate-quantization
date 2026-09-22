# Qwen3.5-0.8B: codec ladder and self-consistent calibration

All 18 DeltaNet `in_proj_qkv` projections replaced simultaneously; 261,248 WikiText-2 validation targets; equal 4-bit index budget.
BF16 reference NLL 3.436670 (ppl 31.083).

| Method | NLL | ppl | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|:--|
| real4_uniform | 3.459277 | 31.794 | +0.022607 | [+0.021638, +0.023569] |
| real4_lloyd | 3.451387 | 31.544 | +0.014717 | [+0.013909, +0.015520] |
| polar_legacy | 3.456446 | 31.704 | +0.019776 | [+0.018821, +0.020732] |
| polar_fitted | 3.451877 | 31.560 | +0.015206 | [+0.014359, +0.016055] |
| vq2d | 3.446239 | 31.382 | +0.009568 | [+0.008895, +0.010235] |
| real4_lloyd+scf | 3.451387 | 31.544 | +0.014717 | [+0.013909, +0.015520] |
| polar_fitted+scf | 3.451530 | 31.549 | +0.014860 | [+0.014012, +0.015717] |
| vq2d+scf | 3.445654 | 31.364 | +0.008983 | [+0.008340, +0.009620] |
