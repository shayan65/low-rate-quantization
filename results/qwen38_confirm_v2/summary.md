# Scale transfer: the section 5c recipe on Qwen3.8-27B

All 48 `linear_attn.in_proj_qkv` tensors (10240x5120 each), 2,516,582,400 params = 9.4% of the model.
**261,284 validation targets** over 2042 blocks, token-weighted. BF16 NLL 2.691117.
Hadamard block 1024; calibration 65,536 tokens from the train split. Every arm verifies decode(encode(w)) per tensor and re-decodes
the reloaded artifact from disk.

| Arm | bits/wt | stored MB | weight MSE | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| scalar3_rot_gptq | 1.7250 | 542.64 | 7.0566e-05 | 2.702075 | +0.010958 | [+0.009366, +0.012561] |
| vq4_rot_gptq | 1.7250 | 542.64 | 6.0863e-05 | 2.698988 | +0.007871 | [+0.006436, +0.009331] |
| vq8_rot_gptq | 1.7253 | 542.74 | 5.6535e-05 | 2.694954 | +0.003837 | [+0.002420, +0.005250] |
| scalar3_g64_rot_gptq | 1.8500 | 581.96 | 7.0277e-05 | 2.702790 | +0.011673 | [+0.010078, +0.013264] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| scalar3_rot_gptq vs scalar3_g64_rot_gptq | -0.000714 | [-0.001828, +0.000375] | 0.9324 | **includes 0** |
| scalar3_rot_gptq vs vq4_rot_gptq | +0.003087 | [+0.001807, +0.004364] | 1.0000 | excludes 0 |
| vq4_rot_gptq vs scalar3_rot_gptq | -0.003087 | [-0.004364, -0.001807] | 1.0000 | excludes 0 |
| vq4_rot_gptq vs scalar3_g64_rot_gptq | -0.003801 | [-0.005149, -0.002448] | 0.9324 | excludes 0 |
| vq8_rot_gptq vs scalar3_rot_gptq | -0.007121 | [-0.008513, -0.005725] | 1.0002 | excludes 0 |
| vq8_rot_gptq vs scalar3_g64_rot_gptq | -0.007836 | [-0.009247, -0.006356] | 0.9326 | excludes 0 |
| vq8_rot_gptq vs vq4_rot_gptq | -0.004034 | [-0.005334, -0.002715] | 1.0002 | excludes 0 |
| scalar3_g64_rot_gptq vs scalar3_rot_gptq | +0.000714 | [-0.000375, +0.001828] | 1.0725 | **includes 0** |
| scalar3_g64_rot_gptq vs vq4_rot_gptq | +0.003801 | [+0.002448, +0.005149] | 1.0725 | excludes 0 |
