# Qwen3.5-0.8B one-projection screen

Frozen model, one DeltaNet input projection, 2,048 real WikiText-2 validation targets. This is not full-dataset perplexity.

| Method | NLL | Perplexity | Delta NLL | Payload bytes |
|---|---:|---:|---:|---:|
| bf16 | 3.696935 | 40.324 | +0.000000 | 12,582,912 |
| real4_row | 3.707198 | 40.740 | +0.010263 | 3,170,304 |
| polar_mag4_phase4_row | 3.695283 | 40.257 | -0.001652 | 3,170,304 |
| polar_mag3_phase5_row | 3.700484 | 40.467 | +0.003549 | 3,170,304 |
| polar_mag5_phase3_row | 3.716692 | 41.128 | +0.019757 | 3,170,304 |

{"complete": true, "seconds": 9.340891700005159, "best_quantized": "polar_mag4_phase4_row", "polar_beats_equal_bit_real4": true, "action": "No additional layer or training launched."}
