# Mixed-precision allocation under a fixed extra-byte budget

Qwen3.5-0.8B, 90 tensors (`in_proj_qkv` + all MLP), 377,487,360 params = 50.2% of the model.
**261,284 validation targets**, token-weighted. BF16 NLL 3.436710.
Low rate dim-4 K=81 (1.600 bits of index), high rate dim-4 K=243 (2.000).
Every promotion arm spends the same **5,662,224 extra bytes** — exactly what
promoting all 18 projections costs.

| Arm | bits/wt | stored MB | promoted params | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| all_low_d8 | 1.7272 | 81.50 | 0.0M | 4.032455 | +0.595745 | [+0.586526, +0.604849] |
| greedy_all_d8_low | 1.8439 | 87.01 | 110.1M | 3.999993 | +0.563283 | [+0.554577, +0.572138] |
| greedy_mlp_d8_low | 1.8439 | 87.01 | 110.1M | 3.951222 | +0.514512 | [+0.506307, +0.522731] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| greedy_all_d8_low vs all_low_d8 | -0.032462 | [-0.034571, -0.030393] | 1.0676 | excludes 0 |
| greedy_mlp_d8_low vs all_low_d8 | -0.081233 | [-0.084066, -0.078503] | 1.0676 | excludes 0 |
