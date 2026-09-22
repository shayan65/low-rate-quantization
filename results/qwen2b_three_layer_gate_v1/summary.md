# Qwen3.5-2B three-layer gate

BF16 NLL 3.350234.

| Layer | Method | Delta NLL | Bytes |
|---:|---|---:|---:|
| 0 | task_real4 | +0.002222 | 6,316,032 |
| 0 | task_polar3_5 | +0.002452 | 6,316,032 |
| 0 | block_gptq_real4 | +0.000643 | 6,684,672 |
| 8 | task_real4 | -0.000158 | 6,316,032 |
| 8 | task_polar3_5 | +0.000817 | 6,316,032 |
| 8 | block_gptq_real4 | +0.000953 | 6,684,672 |
| 16 | task_real4 | -0.000006 | 6,316,032 |
| 16 | task_polar3_5 | +0.000614 | 6,316,032 |
| 16 | block_gptq_real4 | +0.000640 | 6,684,672 |

{"complete": true, "seconds": 35.34778849798022, "winners": {"0": "block_gptq_real4", "8": "task_real4", "16": "task_real4"}, "promote": false}
