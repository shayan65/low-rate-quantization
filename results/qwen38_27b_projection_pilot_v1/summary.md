# Qwen3.8-27B equal-payload projection pilot

Frozen BF16 model, official Qwen3.8-27B weights, real WikiText-2 validation data. Each quantization changes only one projection; all other weights remain BF16. Real-4 uses optimized per-row scales. Polar uses 3 magnitude bits and 5 phase bits per complex pair with a geometry-preselected pairing. Both use four index bits per real weight plus one FP32 scale per row and a one-byte pairing ID. These are decoded BF16 weights for the task-quality screen, not packed-kernel inference or an end-to-end compressed checkpoint.

Lower NLL is better. The table shows polar NLL minus real-4 NLL on three disjoint, contiguous slices. Negative values favor polar.

| Tensor | First 2,048 targets | Next 2,048 targets | Additional 8,192 targets | Direction |
| --- | ---: | ---: | ---: | --- |
| DeltaNet layer 0 QKV | +0.001020 | -0.001236 | — | Mixed |
| Attention layer 3 Q | -0.001466 | -0.000532 | -0.001848 | Polar in all three |
| MLP layer 3 up | +0.001375 | +0.001006 | — | Real-4 in both |

On the 8,192-target attention check, BF16 NLL was 2.440645, real-4 was 2.439876, and polar was 2.438028. The polar advantage over real-4 is small in absolute NLL. The first two screens took about 85 seconds each; the attention-only check took 139 seconds. No full-dataset or cumulative-quantization claim follows from these isolated screens.

The next useful test is an attention-focused replication across more attention layers, followed by a larger disjoint validation slice if the direction holds. Quantizing all layer types together is not justified by these results. A full WikiText-2 pass would take roughly 23 minutes per method at the measured batch-8 rate, plus quantization and loading overhead, so this should remain gated.
