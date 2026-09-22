# MLP feasibility: does the ternary VQ recipe transfer off the projections?

Qwen3.5-0.8B, all 72 MLP tensors (264,241,152 params, 35.1% of the model); `in_proj_qkv` left at BF16.
**261,284 validation targets**, token-weighted. Calibration: 65,536 tokens from the train split.
BF16 NLL 3.436710. `down_proj` uses a 512-element Hadamard block (3584 is not a multiple of 1024).

| Arm | bits/wt | weight MSE | act-weighted err | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| mlp_scalar3_rot | 1.7250 | 2.3917e-05 | 3.5607e-05 | 5.556430 | +2.119720 | [+2.104286, +2.134513] |
| mlp_scalar3_rot_gptq | 1.7250 | 3.3370e-05 | 1.1247e-05 | 4.094717 | +0.658007 | [+0.648510, +0.667584] |
| mlp_vq4_rot_gptq | 1.7250 | 2.8850e-05 | 9.5374e-06 | 3.989267 | +0.552558 | [+0.544087, +0.561040] |
| mlp_vq8_rot_gptq | 1.7282 | 2.6786e-05 | 8.7428e-06 | 3.937487 | +0.500777 | [+0.492819, +0.508608] |
| mlp_vq8_rot_gptq_pertype | 1.7345 | 2.6874e-05 | 8.7749e-06 | 3.958673 | +0.521963 | [+0.514011, +0.529926] |
| mlp_scalar3_g64_rot_gptq | 1.8500 | 3.3273e-05 | 1.1166e-05 | 4.093212 | +0.656502 | [+0.647276, +0.665713] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| mlp_scalar3_rot vs mlp_scalar3_rot_gptq | +1.461714 | [+1.450678, +1.472360] | 1.0000 | excludes 0 |
| mlp_scalar3_rot vs mlp_scalar3_g64_rot_gptq | +1.463218 | [+1.452235, +1.473864] | 0.9324 | excludes 0 |
| mlp_scalar3_rot vs mlp_vq8_rot_gptq | +1.618943 | [+1.607292, +1.630316] | 0.9982 | excludes 0 |
| mlp_scalar3_rot_gptq vs mlp_scalar3_rot | -1.461714 | [-1.472360, -1.450678] | 1.0000 | excludes 0 |
| mlp_scalar3_rot_gptq vs mlp_scalar3_g64_rot_gptq | +0.001504 | [-0.002309, +0.005386] | 0.9324 | **includes 0** |
| mlp_scalar3_rot_gptq vs mlp_vq8_rot_gptq | +0.157230 | [+0.152538, +0.161835] | 0.9982 | excludes 0 |
| mlp_vq4_rot_gptq vs mlp_scalar3_rot | -1.567163 | [-1.578687, -1.555401] | 1.0000 | excludes 0 |
| mlp_vq4_rot_gptq vs mlp_scalar3_rot_gptq | -0.105449 | [-0.110025, -0.100968] | 1.0000 | excludes 0 |
| mlp_vq4_rot_gptq vs mlp_scalar3_g64_rot_gptq | -0.103945 | [-0.108309, -0.099690] | 0.9324 | excludes 0 |
| mlp_vq4_rot_gptq vs mlp_vq8_rot_gptq | +0.051780 | [+0.047825, +0.055881] | 0.9982 | excludes 0 |
| mlp_vq8_rot_gptq vs mlp_scalar3_rot | -1.618943 | [-1.630316, -1.607292] | 1.0018 | excludes 0 |
| mlp_vq8_rot_gptq vs mlp_scalar3_rot_gptq | -0.157230 | [-0.161835, -0.152538] | 1.0018 | excludes 0 |
| mlp_vq8_rot_gptq vs mlp_scalar3_g64_rot_gptq | -0.155725 | [-0.160014, -0.151299] | 0.9342 | excludes 0 |
| mlp_vq8_rot_gptq_pertype vs mlp_scalar3_rot | -1.597757 | [-1.609060, -1.586550] | 1.0055 | excludes 0 |
| mlp_vq8_rot_gptq_pertype vs mlp_scalar3_rot_gptq | -0.136044 | [-0.140901, -0.131262] | 1.0055 | excludes 0 |
| mlp_vq8_rot_gptq_pertype vs mlp_scalar3_g64_rot_gptq | -0.134539 | [-0.139278, -0.130084] | 0.9376 | excludes 0 |
| mlp_vq8_rot_gptq_pertype vs mlp_vq8_rot_gptq | +0.021186 | [+0.017102, +0.025162] | 1.0037 | excludes 0 |
| mlp_scalar3_g64_rot_gptq vs mlp_scalar3_rot | -1.463218 | [-1.473864, -1.452235] | 1.0725 | excludes 0 |
| mlp_scalar3_g64_rot_gptq vs mlp_scalar3_rot_gptq | -0.001504 | [-0.005386, +0.002309] | 1.0725 | **includes 0** |
| mlp_scalar3_g64_rot_gptq vs mlp_vq8_rot_gptq | +0.155725 | [+0.151299, +0.160014] | 1.0705 | excludes 0 |
