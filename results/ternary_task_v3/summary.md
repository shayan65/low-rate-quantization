# Step 1 (corrected): ternary rate, weights decoded from stored bytes

Qwen3.5-0.8B, all 18 DeltaNet input projections, **261,284 validation targets** (complete stream, token-weighted).
BF16 NLL 3.436710. `bits/wt` is measured from real packed bytes; every arm
asserts decode(encode(w)) is bit-exact.

| Arm | bits/wt | stored MB | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|:--|
| scalar_ternary_g128 | 1.7250 | 24.42 | 3.878234 | +0.441525 | [+0.436018, +0.447092] |
| scalar_ternary_g128_rot | 1.7250 | 24.42 | 3.793309 | +0.356599 | [+0.352028, +0.361286] |
| scalar_ternary_g64_rot | 1.8500 | 26.19 | 3.790474 | +0.353765 | [+0.348936, +0.358561] |
| vq_dim4_g128_rot | 1.7251 | 24.42 | 3.694866 | +0.258156 | [+0.254283, +0.262020] |
| vq_dim8_g128_rot | 1.7324 | 24.52 | 3.676890 | +0.240180 | [+0.236367, +0.243903] |
| vq_dim8_g128_norot | 1.7324 | 24.52 | 3.683285 | +0.246575 | [+0.242897, +0.250227] |
| vq_dim4_g128_rot_w | 1.7251 | 24.42 | 3.700681 | +0.263971 | [+0.259985, +0.268005] |
| vq_dim8_g128_rot_w | 1.7324 | 24.52 | 3.698379 | +0.261669 | [+0.257760, +0.265709] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| scalar_ternary_g128 vs scalar_ternary_g128_rot | +0.084926 | [+0.080518, +0.089412] | 1.0000 | excludes 0 |
| scalar_ternary_g128 vs scalar_ternary_g64_rot | +0.087760 | [+0.083219, +0.092314] | 0.9324 | excludes 0 |
| scalar_ternary_g128_rot vs scalar_ternary_g64_rot | +0.002834 | [+0.000630, +0.005013] | 0.9324 | excludes 0 |
| scalar_ternary_g64_rot vs scalar_ternary_g128_rot | -0.002834 | [-0.005013, -0.000630] | 1.0725 | excludes 0 |
| vq_dim4_g128_rot vs scalar_ternary_g128_rot | -0.098443 | [-0.102171, -0.094707] | 1.0000 | excludes 0 |
| vq_dim4_g128_rot vs scalar_ternary_g64_rot | -0.095608 | [-0.099443, -0.091955] | 0.9325 | excludes 0 |
| vq_dim8_g128_rot vs scalar_ternary_g128_rot | -0.116419 | [-0.120206, -0.112588] | 1.0043 | excludes 0 |
| vq_dim8_g128_rot vs scalar_ternary_g64_rot | -0.113585 | [-0.117539, -0.109645] | 0.9364 | excludes 0 |
| vq_dim8_g128_norot vs scalar_ternary_g128_rot | -0.110023 | [-0.113995, -0.106188] | 1.0043 | excludes 0 |
| vq_dim8_g128_norot vs scalar_ternary_g64_rot | -0.107189 | [-0.111286, -0.103210] | 0.9364 | excludes 0 |
| vq_dim4_g128_rot_w vs scalar_ternary_g128_rot | -0.092628 | [-0.096457, -0.088790] | 1.0000 | excludes 0 |
| vq_dim4_g128_rot_w vs scalar_ternary_g64_rot | -0.089794 | [-0.093740, -0.085863] | 0.9325 | excludes 0 |
| vq_dim8_g128_rot_w vs scalar_ternary_g128_rot | -0.094930 | [-0.098628, -0.091153] | 1.0043 | excludes 0 |
| vq_dim8_g128_rot_w vs scalar_ternary_g64_rot | -0.092095 | [-0.095924, -0.088259] | 0.9364 | excludes 0 |
