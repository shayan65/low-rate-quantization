# Strong classical controls for cumulative Qwen DeltaNet projection quantization

These are transparent local implementations on Qwen3.5 custom projections, not official AutoAWQ/GPTQModel backend runs. Evaluation uses the same complete 261,284-target WikiText-2 validation stream.

| Method | NLL | Perplexity | Delta NLL | Export bytes |
|---|---:|---:|---:|---:|
| cumulative_task_real4 | 3.458566 | 31.771 | +0.021945 | 57,065,490 |
| cumulative_task_polar3_5 | 3.445954 | 31.373 | +0.009333 | 57,065,490 |
| awq_style_real4 | 3.465755 | 32.001 | +0.029134 | 57,139,200 |
| block_gptq_real4 | 3.449311 | 31.479 | +0.012689 | 60,162,048 |

{"complete": true, "seconds": 387.06589291204, "task_polar_beats_awq_style": true, "task_polar_beats_block_gptq": true, "action": "No training or test evaluation launched."}
