# Mixed-precision allocation under a fixed extra-byte budget

Qwen3.5-0.8B, 90 tensors (`in_proj_qkv` + all MLP), 377,487,360 params = 50.2% of the model.
**261,284 validation targets**, token-weighted. BF16 NLL 3.436710.
Low rate dim-4 K=81 (1.600 bits of index), high rate dim-4 K=243 (2.000).
Every promotion arm spends the same **5,662,224 extra bytes** — exactly what
promoting all 18 projections costs.

| Arm | bits/wt | stored MB | promoted params | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| all_low | 1.7250 | 81.40 | 0.0M | 4.074539 | +0.637829 | [+0.628446, +0.647323] |
| promote_proj | 1.8451 | 87.06 | 113.2M | 4.031360 | +0.594650 | [+0.585831, +0.603636] |
| promote_greedy_mlp | 1.8417 | 86.90 | 110.1M | 3.970255 | +0.533546 | [+0.525270, +0.542197] |
| promote_greedy_all | 1.8417 | 86.90 | 110.1M | 4.035570 | +0.598860 | [+0.590043, +0.607873] |
| promote_random | 1.8451 | 87.06 | 113.2M | 3.974326 | +0.537616 | [+0.529333, +0.546178] |
| all_high | 2.1250 | 100.27 | 377.5M | 3.787829 | +0.351119 | [+0.344776, +0.357529] |
| all_low_d8 | 1.7272 | 81.50 | 0.0M | 4.032455 | +0.595745 | [+0.586526, +0.604849] |
| greedy_all_d8_low | 1.8439 | 87.01 | 110.1M | 3.999993 | +0.563283 | [+0.554577, +0.572138] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| all_low vs promote_proj | +0.043179 | [+0.040853, +0.045614] | 0.9349 | excludes 0 |
| all_low vs promote_random | +0.100213 | [+0.097418, +0.103060] | 0.9349 | excludes 0 |
| all_low vs all_low_d8 | +0.042084 | [+0.037910, +0.046297] | 0.9987 | excludes 0 |
| promote_proj vs all_low | -0.043179 | [-0.045614, -0.040853] | 1.0696 | excludes 0 |
| promote_proj vs promote_random | +0.057034 | [+0.053814, +0.060232] | 1.0000 | excludes 0 |
| promote_proj vs all_low_d8 | -0.001095 | [-0.005282, +0.003122] | 1.0682 | **includes 0** |
| promote_greedy_mlp vs all_low | -0.104284 | [-0.107092, -0.101563] | 1.0677 | excludes 0 |
| promote_greedy_mlp vs promote_proj | -0.061105 | [-0.064430, -0.057804] | 0.9982 | excludes 0 |
| promote_greedy_mlp vs promote_random | -0.004070 | [-0.006991, -0.001160] | 0.9982 | excludes 0 |
| promote_greedy_mlp vs all_low_d8 | -0.062200 | [-0.066439, -0.058022] | 1.0663 | excludes 0 |
| promote_greedy_all vs all_low | -0.038969 | [-0.041249, -0.036743] | 1.0677 | excludes 0 |
| promote_greedy_all vs promote_proj | +0.004210 | [+0.002400, +0.006041] | 0.9982 | excludes 0 |
| promote_greedy_all vs promote_random | +0.061244 | [+0.058083, +0.064344] | 0.9982 | excludes 0 |
| promote_greedy_all vs all_low_d8 | +0.003115 | [-0.001049, +0.007285] | 1.0663 | **includes 0** |
| promote_random vs all_low | -0.100213 | [-0.103060, -0.097418] | 1.0696 | excludes 0 |
| promote_random vs promote_proj | -0.057034 | [-0.060232, -0.053814] | 1.0000 | excludes 0 |
| promote_random vs all_low_d8 | -0.058129 | [-0.062320, -0.053798] | 1.0682 | excludes 0 |
| all_high vs all_low | -0.286710 | [-0.292029, -0.281440] | 1.2319 | excludes 0 |
| all_high vs promote_proj | -0.243531 | [-0.248489, -0.238702] | 1.1517 | excludes 0 |
| all_high vs promote_random | -0.186497 | [-0.190949, -0.181975] | 1.1517 | excludes 0 |
| all_high vs all_low_d8 | -0.244626 | [-0.249646, -0.239600] | 1.2303 | excludes 0 |
| all_low_d8 vs all_low | -0.042084 | [-0.046297, -0.037910] | 1.0013 | excludes 0 |
| all_low_d8 vs promote_proj | +0.001095 | [-0.003122, +0.005282] | 0.9361 | **includes 0** |
| all_low_d8 vs promote_random | +0.058129 | [+0.053798, +0.062320] | 0.9361 | excludes 0 |
| greedy_all_d8_low vs all_low | -0.074546 | [-0.078928, -0.070328] | 1.0689 | excludes 0 |
| greedy_all_d8_low vs promote_proj | -0.031367 | [-0.035551, -0.027103] | 0.9994 | excludes 0 |
| greedy_all_d8_low vs promote_random | +0.025668 | [+0.021325, +0.029851] | 0.9994 | excludes 0 |
| greedy_all_d8_low vs all_low_d8 | -0.032462 | [-0.034571, -0.030393] | 1.0676 | excludes 0 |
