# Qwen3.5-0.8B small confirmation

Frozen official Qwen3.5-0.8B-Base, layer 0 `linear_attn.in_proj_qkv.weight` only. Three independent fixed samples × 2,048 real WikiText-2 validation targets. Adjacent real weights form complex pairs. Each quantized method exports 3,170,304 bytes: eight index bits per pair plus one FP32 row scale. No training, test-set access, or additional-layer quantization.

| Method | Sample 3101 ΔNLL | Sample 3102 ΔNLL | Sample 3103 ΔNLL | Mean ΔNLL |
|---|---:|---:|---:|---:|
| Real4 row-wise | +0.008926 | +0.010263 | +0.010475 | +0.009888 |
| Polar magnitude4 + phase4 | +0.005052 | -0.001652 | +0.008457 | +0.003952 |
| **Polar magnitude3 + phase5** | **-0.001164** | **+0.003549** | **+0.001394** | **+0.001260** |
| Polar magnitude5 + phase3 | +0.020493 | +0.019757 | +0.017779 | +0.019343 |

The 3+5 polar split best preserves NLL on average and every confirmation sample has at least one polar split better than equal-byte real4. The sample-wise best polar split varies. This is a one-projection feasibility result; the complex pairing is arbitrary and basis-dependent, and real4 is not yet a strong Hessian-aware baseline.
