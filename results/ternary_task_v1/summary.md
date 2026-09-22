# Step 1: ternary rate, does the weight-MSE gap convert?

Qwen3.5-0.8B, all 18 DeltaNet input projections, 261,248 WikiText-2 validation targets. BF16 NLL 3.436670.
Codebook storage is charged in `bits/wt`.

| Arm | bits/wt | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|:--|
| scalar_ternary_g128 | 1.710 | 3.877938 | +0.441267 | [+0.435534, +0.446983] |
| scalar_ternary_g128_rot | 1.710 | 3.795474 | +0.358803 | [+0.354169, +0.363503] |
| scalar_ternary_g64_rot | 1.835 | 3.790168 | +0.353497 | [+0.348844, +0.358319] |
| vq_dim4_g128_rot | 1.711 | 3.713675 | +0.277004 | [+0.272871, +0.281136] |
| vq_dim8_g128_rot_shared | 1.717 | 3.658186 | +0.221516 | [+0.217679, +0.225362] |
| vq_dim8_g128_rot_pertensor | 1.843 | 3.645873 | +0.209203 | [+0.205581, +0.212775] |
| vq_dim8_g128_norot_shared | 1.717 | 3.682985 | +0.246314 | [+0.242474, +0.250071] |

## Head-to-head vs byte-matched scalar controls

| Comparison | ΔNLL | 95% CI | |
|---|---:|:--|:--|
| scalar_ternary_g128 − scalar_ternary_g128_rot | +0.082464 | [+0.077881, +0.087093] | excludes 0 |
| scalar_ternary_g128 − scalar_ternary_g64_rot | +0.087770 | [+0.083150, +0.092273] | excludes 0 |
| scalar_ternary_g128_rot − scalar_ternary_g64_rot | +0.005306 | [+0.003140, +0.007590] | excludes 0 |
| scalar_ternary_g64_rot − scalar_ternary_g128_rot | -0.005306 | [-0.007590, -0.003140] | excludes 0 |
| vq_dim4_g128_rot − scalar_ternary_g128_rot | -0.081799 | [-0.085453, -0.078153] | excludes 0 |
| vq_dim4_g128_rot − scalar_ternary_g64_rot | -0.076493 | [-0.080247, -0.072829] | excludes 0 |
| vq_dim8_g128_rot_shared − scalar_ternary_g128_rot | -0.137287 | [-0.141188, -0.133457] | excludes 0 |
| vq_dim8_g128_rot_shared − scalar_ternary_g64_rot | -0.131981 | [-0.135933, -0.128104] | excludes 0 |
| vq_dim8_g128_rot_pertensor − scalar_ternary_g128_rot | -0.149600 | [-0.153500, -0.145770] | excludes 0 |
| vq_dim8_g128_rot_pertensor − scalar_ternary_g64_rot | -0.144294 | [-0.148256, -0.140316] | excludes 0 |
| vq_dim8_g128_norot_shared − scalar_ternary_g128_rot | -0.112489 | [-0.116395, -0.108715] | excludes 0 |
| vq_dim8_g128_norot_shared − scalar_ternary_g64_rot | -0.107183 | [-0.111295, -0.103301] | excludes 0 |
