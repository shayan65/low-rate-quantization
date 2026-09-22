# Qwen3.5-0.8B one-projection screen

Frozen model, one DeltaNet input projection, 2,048 real WikiText-2 validation targets. This is not full-dataset perplexity.

| Method | NLL | Perplexity | Delta NLL | Payload bytes |
|---|---:|---:|---:|---:|
| bf16 | 3.483973 | 32.589 | +0.000000 | 12,582,912 |
| real4_row | 3.492899 | 32.881 | +0.008926 | 3,170,304 |
| polar_mag4_phase4_row | 3.489024 | 32.754 | +0.005052 | 3,170,304 |
| polar_mag3_phase5_row | 3.482809 | 32.551 | -0.001164 | 3,170,304 |
| polar_mag5_phase3_row | 3.504466 | 33.264 | +0.020493 | 3,170,304 |

{"complete": true, "seconds": 9.272515380987898, "best_quantized": "polar_mag3_phase5_row", "polar_beats_equal_bit_real4": true, "action": "No additional layer or training launched."}
