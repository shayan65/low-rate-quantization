# Step 2: ternary rate with activation weighting and error compensation

Qwen3.5-0.8B, all 18 DeltaNet input projections, **261,284 validation targets** (complete stream, token-weighted).
Calibration: 65,536 tokens from the *train* split.
BF16 NLL 3.436710. Every arm is rotated, stores FP16 group scales,
asserts decode(encode(w)) bit-exact, and measures bits/weight from real bytes.

| Arm | bits/wt | weight MSE | act-weighted err | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| vq8_rot_gptq | 1.7324 | 7.9194e-05 | 4.9996e-05 | 3.513771 | +0.077061 | [+0.074362, +0.079868] |
| vq8_rot_gptq_seq | 1.7324 | 7.9711e-05 | 4.8564e-05 | 3.511881 | +0.075171 | [+0.072454, +0.077936] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| vq8_rot_gptq_seq vs vq8_rot_gptq | -0.001890 | [-0.003421, -0.000315] | 1.0000 | excludes 0 |
