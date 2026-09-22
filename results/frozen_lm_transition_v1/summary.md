# Frozen language-model transition quantization screen

Only the first recurrent transition angle changes. Metrics use 16,384 fixed held-out WikiText-103 validation targets; the test split is untouched.

| Method | Seeds | Validation NLL | Perplexity | ΔNLL vs FP32 |
|---|---:|---:|---:|---:|
| fp32 | 3 | 6.510978 | 672.488 | +0.000000 |
| phase_nearest | 3 | 6.514951 | 675.168 | +0.003974 |
| phase_weight_offset | 3 | 6.514543 | 674.894 | +0.003566 |
| phase_response_offset | 3 | 6.513567 | 674.233 | +0.002589 |
| circular_response_offset | 3 | 6.513567 | 674.233 | +0.002589 |
| square_direction_response_offset | 3 | 6.516995 | 676.554 | +0.006017 |

{"completed_jobs": 18, "planned_jobs": 18, "complete": true, "seconds": 1.8709593339590356, "phase_equals_circular": true, "promote": true, "action": "No long run launched; this gate only decides whether to build a stronger small task-level QAT experiment."}
