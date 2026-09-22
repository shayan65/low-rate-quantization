# Step 1 (corrected): ternary rate, weights decoded from stored bytes

Qwen3.5-0.8B, all 18 DeltaNet input projections, **261,284 validation targets** (complete stream, token-weighted).
BF16 NLL 3.436710. `bits/wt` is measured from real packed bytes; every arm
asserts decode(encode(w)) is bit-exact.

| Arm | bits/wt | stored MB | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|:--|
| scalar_ternary_g128 | 1.7250 | 24.42 | 3.878057 | +0.441348 | [+0.435807, +0.446899] |
| scalar_ternary_g128_rot | 1.7250 | 24.42 | 3.793590 | +0.356880 | [+0.352322, +0.361525] |
| scalar_ternary_g64_rot | 1.8500 | 26.19 | 3.789856 | +0.353146 | [+0.348339, +0.357969] |
| vq_dim4_g128_rot | 1.7251 | 24.42 | 3.695006 | +0.258296 | [+0.254405, +0.262194] |
| vq_dim8_g128_rot | 1.7324 | 24.52 | 3.677154 | +0.240444 | [+0.236589, +0.244183] |
| vq_dim8_g128_norot | 1.7324 | 24.52 | 3.683029 | +0.246319 | [+0.242681, +0.249955] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| scalar_ternary_g128 vs scalar_ternary_g128_rot | +0.084468 | [+0.080038, +0.088955] | 1.0000 | excludes 0 |
| scalar_ternary_g128 vs scalar_ternary_g64_rot | +0.088202 | [+0.083682, +0.092802] | 0.9324 | excludes 0 |
| scalar_ternary_g128_rot vs scalar_ternary_g64_rot | +0.003734 | [+0.001552, +0.005923] | 0.9324 | excludes 0 |
| scalar_ternary_g64_rot vs scalar_ternary_g128_rot | -0.003734 | [-0.005923, -0.001552] | 1.0725 | excludes 0 |
| vq_dim4_g128_rot vs scalar_ternary_g128_rot | -0.098584 | [-0.102295, -0.094849] | 1.0000 | excludes 0 |
| vq_dim4_g128_rot vs scalar_ternary_g64_rot | -0.094850 | [-0.098649, -0.091197] | 0.9325 | excludes 0 |
| vq_dim8_g128_rot vs scalar_ternary_g128_rot | -0.116436 | [-0.120254, -0.112646] | 1.0043 | excludes 0 |
| vq_dim8_g128_rot vs scalar_ternary_g64_rot | -0.112702 | [-0.116676, -0.108742] | 0.9364 | excludes 0 |
| vq_dim8_g128_norot vs scalar_ternary_g128_rot | -0.110561 | [-0.114509, -0.106722] | 1.0043 | excludes 0 |
| vq_dim8_g128_norot vs scalar_ternary_g64_rot | -0.106827 | [-0.110910, -0.102838] | 0.9364 | excludes 0 |
