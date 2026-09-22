# Matched-protocol TinyStories pilot

Exactly 261,284 contiguous validation targets and 128-target resets, matching the WikiText-2 protocol.

| Method | NLL | Perplexity | Delta NLL | Export bytes |
|---|---:|---:|---:|---:|
| bf16 | 2.314341 | 10.118 | +0.000000 | 0 |
| task_real4 | 2.322563 | 10.202 | +0.008222 | 57,065,490 |
| task_polar3_5 | 2.319747 | 10.173 | +0.005406 | 57,065,490 |
| block_gptq_real4 | 2.323978 | 10.216 | +0.009637 | 60,162,048 |

{"complete": true, "seconds": 870.0528143109987, "polar_beats_block_gptq": true, "polar_beats_task_real4": true, "action": "If polar still loses GPTQ, revise to a layer-wise mixed codec."}
