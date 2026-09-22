# Which single layer is worth leaving in BF16?

Qwen3.5-0.8B, vq2d codec, 261,248 validation targets. BF16 NLL 3.436670.

| Arm | ΔNLL vs BF16 | 95% CI | damage recovered | payload vs all-18 |
|---|---:|:--|---:|---:|
| all_18 | +0.008393 | [+0.007720, +0.009069] | — | 1.000x |
| protect_fisher (L0) | +0.003716 | [+0.003197, +0.004245] | 55.7% | 1.167x |
| protect_weightmse (L20) | +0.008061 | [+0.007401, +0.008729] | 4.0% | 1.167x |
| protect_random (L10) | +0.008382 | [+0.007733, +0.009025] | 0.1% | 1.167x |
| all_18_k512 (4.5 bit) | +0.004872 | [+0.004348, +0.005402] | 41.9% | 1.125x |
| all_18_k1024 (5 bit) | +0.002950 | [+0.002555, +0.003346] | 64.9% | 1.250x |
