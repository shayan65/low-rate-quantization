# Mixed polar/GPTQ codec with paired confidence intervals

| Method | NLL | Delta NLL (95% paired block CI) | Export bytes |
|---|---:|---:|---:|
| bf16 | 2.314341 | +0.000000 [+0.000000, +0.000000] | 0 |
| task_real4 | 2.322563 | +0.008222 [+0.007428, +0.009030] | 57,065,490 |
| task_polar3_5 | 2.319747 | +0.005406 [+0.004666, +0.006148] | 57,065,490 |
| block_gptq_real4 | 2.323978 | +0.009637 [+0.009123, +0.010181] | 60,162,066 |
| mixed_polar_gptq | 2.315292 | +0.000951 [+0.000243, +0.001658] | 57,925,650 |

{"complete": true, "seconds": 982.6569699440151, "polar_layers": 13, "gptq_layers": 5, "mixed_beats_polar": true, "mixed_beats_gptq": true}
