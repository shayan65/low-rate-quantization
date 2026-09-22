# Selection-fairness check: the earlier draft's protocol, applied to every codec

Per-layer candidate selection by calibration NLL (4 WikiText-2 training windows), then cumulative application; 261,248 validation targets.
BF16 NLL 3.436670.

| Method | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| polar_legacy_nllsel | 3.447186 | +0.010515 | [+0.009586, +0.011440] |
| real4_lloyd_nllsel | 3.446527 | +0.009857 | [+0.009025, +0.010678] |

## Paired head-to-head differences

| Comparison | ΔNLL | 95% CI |
|---|---:|:--|
| polar_legacy_nllsel − real4_lloyd_nllsel | +0.000659 | [-0.000385, +0.001745] |
