# Qwen3.5-0.8B one-projection screen

Frozen model, one DeltaNet input projection, 2,048 real WikiText-2 validation targets. This is not full-dataset perplexity.

| Method | NLL | Perplexity | Delta NLL | Payload bytes |
|---|---:|---:|---:|---:|
| bf16 | 3.518749 | 33.742 | +0.000000 | 12,582,912 |
| real4_row | 3.529224 | 34.097 | +0.010475 | 3,170,304 |
| polar_mag4_phase4_row | 3.527206 | 34.029 | +0.008457 | 3,170,304 |
| polar_mag3_phase5_row | 3.520144 | 33.789 | +0.001394 | 3,170,304 |
| polar_mag5_phase3_row | 3.536528 | 34.347 | +0.017779 | 3,170,304 |

{"complete": true, "seconds": 9.222881692985538, "best_quantized": "polar_mag3_phase5_row", "polar_beats_equal_bit_real4": true, "action": "No additional layer or training launched."}
