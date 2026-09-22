# Selection-fairness check: the earlier draft's protocol, applied to every codec

Per-layer candidate selection by calibration NLL (16 WikiText-2 training windows), then cumulative application; 261,248 validation targets.
BF16 NLL 3.436670.

| Method | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| polar_legacy_nllsel | 3.452573 | +0.015902 | [+0.015006, +0.016825] |
| real4_uniform_nllsel | 3.460834 | +0.024164 | [+0.023088, +0.025239] |
| real4_lloyd_nllsel | 3.445209 | +0.008539 | [+0.007662, +0.009403] |
| vq2d_nllsel | 3.445815 | +0.009145 | [+0.008325, +0.009950] |

## Paired head-to-head differences

| Comparison | ΔNLL | 95% CI |
|---|---:|:--|
| polar_legacy_nllsel − real4_uniform_nllsel | -0.008261 | [-0.009618, -0.006945] |
| polar_legacy_nllsel − real4_lloyd_nllsel | +0.007364 | [+0.006332, +0.008376] |
| polar_legacy_nllsel − vq2d_nllsel | +0.006757 | [+0.005703, +0.007801] |
| real4_uniform_nllsel − real4_lloyd_nllsel | +0.015625 | [+0.014411, +0.016816] |
| real4_uniform_nllsel − vq2d_nllsel | +0.015019 | [+0.013759, +0.016255] |
| real4_lloyd_nllsel − vq2d_nllsel | -0.000607 | [-0.001639, +0.000431] |
