# Step 2: ternary rate with activation weighting and error compensation

Qwen3.5-0.8B, all 18 DeltaNet input projections, **261,284 validation targets** (complete stream, token-weighted).
Calibration: 65,536 tokens from the *train* split.
BF16 NLL 3.436710. Every arm is rotated, stores FP16 group scales,
asserts decode(encode(w)) bit-exact, and measures bits/weight from real bytes.

| Arm | bits/wt | weight MSE | act-weighted err | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| scalar3_rot | 1.7250 | 7.1845e-05 | 2.3293e-04 | 3.793309 | +0.356599 | [+0.352028, +0.361286] |
| scalar3_rot_actw | 1.7250 | 7.1851e-05 | 2.3292e-04 | 3.797793 | +0.361084 | [+0.356485, +0.365764] |
| scalar3_rot_gptq | 1.7250 | 9.8886e-05 | 6.4700e-05 | 3.549889 | +0.113179 | [+0.109919, +0.116351] |
| vq8_rot | 1.7324 | 5.7165e-05 | 1.7333e-04 | 3.676890 | +0.240180 | [+0.236367, +0.243903] |
| vq8_rot_actw | 1.7324 | 5.8663e-05 | 1.5524e-04 | 3.664537 | +0.227827 | [+0.224049, +0.231700] |
| vq8_rot_gptq | 1.7324 | 7.9194e-05 | 4.9996e-05 | 3.513771 | +0.077061 | [+0.074362, +0.079868] |
| vq8_rot_gptq_actwfit | 1.7324 | 7.9306e-05 | 5.0085e-05 | 3.513977 | +0.077267 | [+0.074642, +0.079910] |
| vq4_rot_gptq | 1.7251 | 8.5366e-05 | 5.4554e-05 | 3.525221 | +0.088512 | [+0.085730, +0.091370] |
| scalar3_g64_rot_gptq | 1.8500 | 9.8627e-05 | 6.4136e-05 | 3.548073 | +0.111363 | [+0.108285, +0.114369] |
| vq8_rot_gptq_seq | 1.7324 | 7.9711e-05 | 4.8564e-05 | 3.511881 | +0.075171 | [+0.072454, +0.077936] |

## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)

| Comparison | ΔNLL | 95% CI | byte ratio | |
|---|---:|:--|---:|:--|
| scalar3_rot vs scalar3_rot_gptq | +0.243420 | [+0.239658, +0.247197] | 1.0000 | excludes 0 |
| scalar3_rot vs scalar3_g64_rot_gptq | +0.245235 | [+0.241356, +0.249101] | 0.9324 | excludes 0 |
| scalar3_rot_actw vs scalar3_rot | +0.004485 | [+0.003883, +0.005107] | 1.0000 | excludes 0 |
| scalar3_rot_actw vs scalar3_rot_gptq | +0.247905 | [+0.244113, +0.251700] | 1.0000 | excludes 0 |
| scalar3_rot_actw vs scalar3_g64_rot_gptq | +0.249720 | [+0.245825, +0.253613] | 0.9324 | excludes 0 |
| scalar3_rot_gptq vs scalar3_rot | -0.243420 | [-0.247197, -0.239658] | 1.0000 | excludes 0 |
| scalar3_rot_gptq vs scalar3_g64_rot_gptq | +0.001815 | [-0.000172, +0.003794] | 0.9324 | **includes 0** |
| vq8_rot vs scalar3_rot | -0.116419 | [-0.120206, -0.112588] | 1.0043 | excludes 0 |
| vq8_rot vs scalar3_rot_gptq | +0.127001 | [+0.123792, +0.130302] | 1.0043 | excludes 0 |
| vq8_rot vs scalar3_g64_rot_gptq | +0.128816 | [+0.125502, +0.132215] | 0.9364 | excludes 0 |
| vq8_rot_actw vs scalar3_rot | -0.128771 | [-0.132491, -0.124906] | 1.0043 | excludes 0 |
| vq8_rot_actw vs scalar3_rot_gptq | +0.114649 | [+0.111241, +0.118066] | 1.0043 | excludes 0 |
| vq8_rot_actw vs scalar3_g64_rot_gptq | +0.116464 | [+0.113094, +0.119877] | 0.9364 | excludes 0 |
| vq8_rot_gptq vs scalar3_rot | -0.279538 | [-0.283580, -0.275433] | 1.0043 | excludes 0 |
| vq8_rot_gptq vs scalar3_rot_gptq | -0.036118 | [-0.038414, -0.033768] | 1.0043 | excludes 0 |
| vq8_rot_gptq vs scalar3_g64_rot_gptq | -0.034303 | [-0.036660, -0.031971] | 0.9364 | excludes 0 |
| vq8_rot_gptq_actwfit vs scalar3_rot | -0.279331 | [-0.283499, -0.275171] | 1.0043 | excludes 0 |
| vq8_rot_gptq_actwfit vs scalar3_rot_gptq | -0.035911 | [-0.038447, -0.033412] | 1.0043 | excludes 0 |
| vq8_rot_gptq_actwfit vs scalar3_g64_rot_gptq | -0.034096 | [-0.036567, -0.031560] | 0.9364 | excludes 0 |
| vq4_rot_gptq vs scalar3_rot | -0.268087 | [-0.272015, -0.264113] | 1.0000 | excludes 0 |
| vq4_rot_gptq vs scalar3_rot_gptq | -0.024667 | [-0.027041, -0.022348] | 1.0000 | excludes 0 |
| vq4_rot_gptq vs scalar3_g64_rot_gptq | -0.022852 | [-0.025130, -0.020564] | 0.9325 | excludes 0 |
| scalar3_g64_rot_gptq vs scalar3_rot | -0.245235 | [-0.249101, -0.241356] | 1.0725 | excludes 0 |
| scalar3_g64_rot_gptq vs scalar3_rot_gptq | -0.001815 | [-0.003794, +0.000172] | 1.0725 | **includes 0** |
| vq8_rot_gptq_seq vs scalar3_rot | -0.281428 | [-0.285406, -0.277450] | 1.0043 | excludes 0 |
| vq8_rot_gptq_seq vs scalar3_rot_gptq | -0.038008 | [-0.040328, -0.035591] | 1.0043 | excludes 0 |
| vq8_rot_gptq_seq vs scalar3_g64_rot_gptq | -0.036192 | [-0.038490, -0.033845] | 0.9364 | excludes 0 |
