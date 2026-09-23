# Embedding precision against weight rate: where the bytes should go

Qwen3.5-0.8B. The embedding is 254,279,680 parameters and is **tied** to the output head, so quantizing it perturbs both the input lookup and every logit. 186 non-embedding matrices (497,614,848 params) carry the weight codec. BF16 model 1505 MB, BF16 NLL 3.436710 over 261,284 targets.

Embedding quantization here is deliberately naive — per-row or per-group scaling, no rotation, no compensation — so the comparison is conservative against the weight codec, which has both.

| weights | embedding | model MB | eff. bpw | shrink | ΔNLL | 95% CI |
|---|---|---:|---:|---:|---:|:--|
| bf16 | bf16 | 1504.8 | 16.000 | 1.00x | +0.000000 | [+0.000000, +0.000000] |
| bf16 | fp8_e4m3 | 1251.0 | 13.302 | 1.20x | +0.001434 | [+0.001076, +0.001810] |
| bf16 | int8_g128 | 1254.5 | 13.339 | 1.20x | +0.000244 | [+0.000062, +0.000424] |
| bf16 | int4_g128 | 1127.3 | 11.987 | 1.33x | +0.032871 | [+0.031549, +0.034191] |
| r1600_k81 | bf16 | 616.9 | 6.559 | 2.44x | +1.000022 | [+0.986485, +1.013520] |
| r1600_k81 | fp8_e4m3 | 363.1 | 3.860 | 4.14x | +1.000291 | [+0.986681, +1.013811] |
| r1600_k81 | int8_g128 | 366.5 | 3.897 | 4.11x | +1.000814 | [+0.987253, +1.014304] |
| r1600_k81 | int4_g128 | 239.4 | 2.546 | 6.29x | +1.053793 | [+1.039956, +1.067295] |
| r2000_k255 | bf16 | 641.7 | 6.823 | 2.34x | +0.470715 | [+0.462434, +0.478752] |
| r2000_k255 | fp8_e4m3 | 388.0 | 4.125 | 3.88x | +0.471673 | [+0.463402, +0.479723] |
| r2000_k255 | int8_g128 | 391.4 | 4.162 | 3.84x | +0.470934 | [+0.462682, +0.478959] |
| r2000_k255 | int4_g128 | 264.3 | 2.810 | 5.69x | +0.514113 | [+0.505586, +0.522352] |
| r2667_k1625 | bf16 | 683.2 | 7.264 | 2.20x | +0.171490 | [+0.167185, +0.175860] |
| r2667_k1625 | fp8_e4m3 | 429.4 | 4.566 | 3.50x | +0.172940 | [+0.168601, +0.177312] |
| r2667_k1625 | int8_g128 | 432.9 | 4.603 | 3.48x | +0.171893 | [+0.167593, +0.176261] |
| r2667_k1625 | int4_g128 | 305.8 | 3.251 | 4.92x | +0.208926 | [+0.204311, +0.213554] |

## Frontier (cheapest model at each new best loss)

| cell | model MB | ΔNLL |
|---|---:|---:|
| r1600_k81|int4_g128 | 239.4 | +1.053793 |
| r2000_k255|int4_g128 | 264.3 | +0.514113 |
| r2667_k1625|int4_g128 | 305.8 | +0.208926 |
| r2667_k1625|fp8_e4m3 | 429.4 | +0.172940 |
| r2667_k1625|int8_g128 | 432.9 | +0.171893 |
| r2667_k1625|bf16 | 683.2 | +0.171490 |
| bf16|int4_g128 | 1127.3 | +0.032871 |
| bf16|fp8_e4m3 | 1251.0 | +0.001434 |
| bf16|int8_g128 | 1254.5 | +0.000244 |
| bf16|bf16 | 1504.8 | +0.000000 |
