# Qwen3.5 cross-layer polar screen

Each row changes one frozen DeltaNet projection independently and evaluates the same 2,048 real WikiText-2 validation targets. BF16 baseline NLL 3.557637.

| Layer | Method | Delta NLL | Perplexity | Weight MSE |
|---:|---|---:|---:|---:|
| 0 | real4_rtn | +0.009285 | 35.407 | 7.21e-06 |
| 0 | real4_optimized_clip | +0.005817 | 35.285 | 4.25e-06 |
| 0 | polar3_5_adjacent_optimized | +0.004158 | 35.226 | 3.96e-06 |
| 0 | polar3_5_split_half_optimized | +0.008692 | 35.386 | 3.88e-06 |
| 8 | real4_rtn | +0.000373 | 35.093 | 6.07e-06 |
| 8 | real4_optimized_clip | +0.000052 | 35.082 | 3.81e-06 |
| 8 | polar3_5_adjacent_optimized | +0.001323 | 35.127 | 3.71e-06 |
| 8 | polar3_5_split_half_optimized | +0.000511 | 35.098 | 3.63e-06 |
| 16 | real4_rtn | +0.002598 | 35.171 | 1.03e-05 |
| 16 | real4_optimized_clip | +0.001907 | 35.147 | 6.25e-06 |
| 16 | polar3_5_adjacent_optimized | +0.001562 | 35.135 | 5.95e-06 |
| 16 | polar3_5_split_half_optimized | +0.002187 | 35.157 | 5.87e-06 |

{"complete": true, "jobs": 12, "planned_jobs": 12, "seconds": 21.829371138010174, "polar_wins_vs_optimized_real4_by_layer": {"polar3_5_adjacent_optimized": 2, "polar3_5_split_half_optimized": 0}, "promote": false, "action": "No cumulative, all-layer, or training run launched."}
