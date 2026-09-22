# Polar codec ablation (weight reconstruction only)

Ratio columns are MSE relative to the equal-byte real-4 control; < 1.000 favours polar.

| Model | Tensor | real4 MSE | legacy polar | +exact NN | +fitted grid | real4+Lloyd | polar/real4-Lloyd |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-metadata | in_proj_qkv L0 | 3.0410e-06 | 0.979 | 0.979 | 0.905 | 0.843 | 1.073 |
| Qwen3.8-27B-metadata | q_proj L3 | 4.0363e-06 | 0.948 | 0.948 | 0.871 | 0.822 | 1.060 |
| Qwen3.8-27B-metadata | up_proj L3 | 1.3820e-06 | 0.995 | 0.995 | 0.925 | 0.856 | 1.081 |
