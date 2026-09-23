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

Each arm is measured with only its own state resident: the FP32 reference is freed first, and the codebook and scales are parked on the host while the dense arm runs. `baseline` is what was allocated when the peak counter was reset, so `transient` is what the product itself asked for.

| path | baseline MB | peak MB | transient MB | peak vs dense |
|---|---:|---:|---:|---:|
| dense_bf16 | 113.4 | 113.5 | 0.0 | -- |
| streamed | 23.5 | 91.7 | 68.2 | 1.24x |
| triton | 23.5 | 23.6 | 0.0 | 4.81x |

The dense arm's transient is near zero because cuBLAS reuses a workspace already allocated during the correctness check; that workspace (~8.5 MB) sits in every baseline here and is a cost of running any matmul, not of either format.

The two compressed arms differ in what they do with the weights they never store. The Triton kernel reads codes and accumulates, so its transient is nil and its peak is essentially its resident footprint. The streamed path materializes one decoded FP32 column tile at a time, and at `tile_cols=512` those tiles cost more than the compressed weights themselves -- its peak is a property of that tile size, not of the format:

| tile_cols | peak MB | ms |
|---|---:|---:|
| 128 | 40.6 | 5.769 |
| 512 | 91.7 | 5.436 |
| 2048 | 296.2 | 5.304 |

An earlier version of this benchmark held the FP32 reference through both measurements and reported 811.7 MB against 775.1 MB, a 1.05x ratio that described the harness rather than either path.

## Latency, batch one

| path | ms |
|---|---:|
| dense_bf16 | 0.132 |
| streamed | 5.437 |
| triton | 0.467 |
