# Full cumulative Qwen3.5 DeltaNet projection screen

All 18 DeltaNet input projections are quantized simultaneously. Evaluation covers the complete tokenized WikiText-2 validation stream with fixed context resets; test data is untouched.

| Method | NLL | Perplexity | Delta NLL | Export bytes |
|---|---:|---:|---:|---:|
| bf16 | 3.436621 | 31.082 | +0.000000 | 0 |
| cumulative_task_real4 | 3.458566 | 31.771 | +0.021945 | 57,065,490 |
| cumulative_task_polar3_5 | 3.445954 | 31.373 | +0.009333 | 57,065,490 |
| cumulative_adjacent_polar3_5 | 3.452048 | 31.565 | +0.015427 | 57,065,490 |

{"complete": true, "seconds": 813.2516749870265, "task_polar_beats_task_real4": true, "task_pairing_beats_adjacent": true, "action": "No training or test-set evaluation launched."}
