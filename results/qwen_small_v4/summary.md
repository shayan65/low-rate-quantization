# Qwen3.5-0.8B one-projection screen

Frozen model, one DeltaNet input projection, 1,024 real WikiText-2 validation targets. This is not full-dataset perplexity.

| Method | NLL | Perplexity | Delta NLL | Payload bytes |
|---|---:|---:|---:|---:|
| bf16 | 3.649283 | 38.447 | +0.000000 | 12,582,912 |
| real4_row | 3.649109 | 38.440 | -0.000174 | 3,170,304 |
| polar_mag4_phase4_row | 3.646218 | 38.329 | -0.003065 | 3,170,304 |
| polar_mag3_phase5_row | 3.654348 | 38.642 | +0.005065 | 3,170,304 |
| polar_mag5_phase3_row | 3.663235 | 38.987 | +0.013951 | 3,170,304 |

{"complete": true, "seconds": 5.6496611459879205, "best_quantized": "polar_mag4_phase4_row", "polar_beats_equal_bit_real4": true, "action": "No additional layer or training launched."}
