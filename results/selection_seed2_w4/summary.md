# Selection-fairness check: the earlier draft's protocol, applied to every codec

Per-layer candidate selection by calibration NLL (4 WikiText-2 training windows), then cumulative application; 261,248 validation targets.
BF16 NLL 3.436670.

| Method | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| polar_legacy_nllsel | 3.447631 | +0.010960 | [+0.010040, +0.011906] |
| real4_lloyd_nllsel | 3.450867 | +0.014196 | [+0.013347, +0.015026] |

## Paired head-to-head differences

| Comparison | ΔNLL | 95% CI |
|---|---:|:--|
| polar_legacy_nllsel − real4_lloyd_nllsel | -0.003236 | [-0.004271, -0.002195] |
