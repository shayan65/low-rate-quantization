# Is coverage damage additive? A frozen-artifact interaction test

One codebook and one quantization pass over 90 tensors (dim 4, K=81, 1.7250 bits/weight), then the dequantized weights installed in three combinations. Every tensor carries bit-identical weights in every arm that includes it, so the interaction is a clean paired quantity rather than a comparison between separately fitted runs.

BF16 NLL 3.436710 over 261,284 validation targets.

| arm | tensors | params | ΔNLL | 95% CI |
|---|---:|---:|---:|:--|
| A_qkv | 18 | 113,246,208 | +0.085350 | [+0.082522, +0.088216] |
| B_mlp | 72 | 264,241,152 | +0.548660 | [+0.540311, +0.557142] |
| AB_both | 90 | 377,487,360 | +0.637829 | [+0.628446, +0.647323] |

## Interaction

`L_AB - L_A - L_B + L_BF16` = **+0.003818** [+0.000769, +0.006835], excludes zero.

Additive reference +0.634011 against a measured +0.637829.
