# Scale transfer: the section 5c recipe on Qwen3.8-27B

All 48 `linear_attn.in_proj_qkv` tensors (10240x5120 each), 2,516,582,400 params = 9.4% of the model.
**65,536 validation targets** over 512 blocks, token-weighted. BF16 NLL 2.579685.
Hadamard block 1024; calibration 32,768 tokens from the train split. Every arm asserts decode(encode(w)) bit-exact.

| Arm | bits/wt | stored MB | weight MSE | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| scalar3_rot_gptq | 1.7250 | 542.64 | 7.4163e-05 | 2.589900 | +0.010215 | [+0.007008, +0.013455] |
| vq4_rot_gptq | 1.7250 | 542.64 | 6.3995e-05 | 2.586125 | +0.006439 | [+0.003473, +0.009490] |
| vq8_rot_gptq | 1.7253 | 542.74 | 5.9472e-05 | 2.585944 | +0.006258 | [+0.003408, +0.009206] |
| scalar3_g64_rot_gptq | 1.8500 | 581.96 | 7.3881e-05 | 2.587680 | +0.007994 | [+0.004835, +0.011247] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| scalar3_rot_gptq vs scalar3_g64_rot_gptq | +0.002220 | [-0.000004, +0.004450] | 0.9324 | **includes 0** |
| vq4_rot_gptq vs scalar3_rot_gptq | -0.003776 | [-0.006495, -0.001045] | 1.0000 | excludes 0 |
| vq4_rot_gptq vs scalar3_g64_rot_gptq | -0.001555 | [-0.004232, +0.001122] | 0.9324 | **includes 0** |
| vq8_rot_gptq vs scalar3_rot_gptq | -0.003957 | [-0.006805, -0.001083] | 1.0002 | excludes 0 |
| vq8_rot_gptq vs scalar3_g64_rot_gptq | -0.001736 | [-0.004444, +0.000923] | 0.9326 | **includes 0** |
| scalar3_g64_rot_gptq vs scalar3_rot_gptq | -0.002220 | [-0.004450, +0.000004] | 1.0725 | **includes 0** |
