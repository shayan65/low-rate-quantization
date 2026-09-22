# TinyStories second-dataset replication

Complete official validation split; train-only pairing/Hessian calibration; test split untouched.

| Method | NLL | Perplexity | Delta NLL | Export bytes |
|---|---:|---:|---:|---:|
| bf16 | 1.717422 | 5.570 | +0.000000 | 0 |
| task_polar3_5 | 1.722666 | 5.599 | +0.005245 | 57,065,490 |
| block_gptq_real4 | 1.719900 | 5.584 | +0.002479 | 60,162,048 |

{"complete": true, "seconds": 703.6778967830469, "polar_beats_block_gptq": false, "action": "Proceed to second-checkpoint pilot only if complete."}
