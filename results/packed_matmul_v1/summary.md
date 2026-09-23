# Compressed matmul: correctness, resident bytes, peak memory, latency

Shape 10240x5120 (Qwen3.8-27B QKV), dimension 8, K=6561, group 128.

## Storage layout is not runtime layout

| layout | bits/weight of index |
|---|---:|
| storage (5 codes per uint64) | 1.600 |
| runtime (1 code per int16) | 2.000 |

Resident: 104.9 MB BF16 against 14.0 MB compressed, a factor of 7.47. That includes FP16 group scales (0.125 bits/weight) and the amortized codebook (0.016).

## Correctness

Streamed path against `decode` plus a dense product: relative error 3.27e-07.

Triton path: relative error 3.49e-07.

## Peak allocation during one product

Dense BF16 811.7 MB, streamed 775.1 MB.

## Latency, batch one

| path | ms |
|---|---:|
| dense_bf16 | 0.132 |
| streamed | 5.408 |
| triton | 0.466 |
