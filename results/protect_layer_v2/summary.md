# Which single layer is worth leaving in BF16?

Qwen3.5-0.8B, vq2d codec, 65,536 validation targets. BF16 NLL 3.295255.

| Arm | ΔNLL vs BF16 | 95% CI | damage recovered | payload vs all-18 |
|---|---:|:--|---:|---:|
| all_18 | +0.008883 | [+0.007492, +0.010293] | — | 1.000x |
| protect_fisher (L0) | +0.003759 | [+0.002719, +0.004817] | 57.7% | 1.167x |
| protect_weightmse (L20) | +0.008609 | [+0.007244, +0.009991] | 3.1% | 1.167x |
| protect_random (L10) | +0.009002 | [+0.007689, +0.010343] | -1.3% | 1.167x |
| all_18_k512 (4.5 bit) | +0.004881 | [+0.003912, +0.005861] | 45.1% | 1.125x |
| all_18_k1024 (5 bit) | +0.002877 | [+0.002027, +0.003713] | 67.6% | 1.250x |
