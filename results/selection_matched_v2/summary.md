# Selection-fairness check: the earlier draft's protocol, applied to every codec

Per-layer candidate selection by calibration NLL (16 WikiText-2 training windows), then cumulative application; 261,248 validation targets.
BF16 NLL 3.436670.

| Method | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| polar_legacy_nllsel | 3.452573 | +0.015902 | [+0.015006, +0.016825] |
| real4_lloyd_nllsel | 3.445209 | +0.008539 | [+0.007662, +0.009403] |
| vq2d_nllsel | 3.445815 | +0.009145 | [+0.008325, +0.009950] |
| vq2d_h_nllsel | 3.444363 | +0.007693 | [+0.007029, +0.008398] |

## Paired head-to-head differences

| Comparison | ΔNLL | 95% CI |
|---|---:|:--|
| polar_legacy_nllsel − real4_lloyd_nllsel | +0.007364 | [+0.006332, +0.008376] |
| polar_legacy_nllsel − vq2d_nllsel | +0.006757 | [+0.005703, +0.007801] |
| polar_legacy_nllsel − vq2d_h_nllsel | +0.008210 | [+0.007205, +0.009186] |
| real4_lloyd_nllsel − vq2d_nllsel | -0.000607 | [-0.001639, +0.000431] |
| real4_lloyd_nllsel − vq2d_h_nllsel | +0.000846 | [-0.000169, +0.001822] |
| vq2d_nllsel − vq2d_h_nllsel | +0.001452 | [+0.000556, +0.002327] |
