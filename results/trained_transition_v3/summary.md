# Frozen trained recurrent-transition screen

Real WikiText-103 text drives a frozen learned recurrence from tiny sample-trained models. Calibration takes train windows; held-out original validation windows are distinct. No synthetic metrics, no test split, and no new model training. The Cartesian control has 16 distinct 2+2-bit complex grid points and quantizes pole radius as well as angle. Both families use the same 292-byte exported mode record; phase retains exact FP32 learned decay. The 512-token probe cycles an existing 128-position embedding and is isolated-layer output MSE, not LM perplexity.

| Equal-byte 4-bit method | Seeds | Relative MSE 128 | Relative MSE 512 |
|---|---:|---:|---:|
| phase_nearest | 3 | 0.2365 ± 0.111 | 0.268 ± 0.129 |
| phase_weight_offset | 3 | 0.2185 ± 0.115 | 0.2395 ± 0.125 |
| phase_memory_offset | 3 | 0.1975 ± 0.0877 | 0.2194 ± 0.103 |
| cartesian_nearest | 3 | 0.3279 ± 0.171 | 0.3425 ± 0.174 |
| cartesian_memory_offset | 3 | 0.283 ± 0.106 | 0.3007 ± 0.116 |

{"completed_jobs": 15, "planned_jobs": 15, "all_seeds_complete": true, "seconds": 1.141681872017216, "phase_memory_beats_cartesian_memory_at_512": true, "action": "No long run launched. A positive diagnostic warrants a trained task-model follow-up, not a paper claim."}
