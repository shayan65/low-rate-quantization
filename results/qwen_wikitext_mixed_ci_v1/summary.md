# WikiText-2 mixed polar/GPTQ replication

| Method | NLL | Delta NLL | Export bytes |
|---|---:|---:|---:|
| bf16 | 3.436621 | +0.000000 | 0 |
| task_polar3_5 | 3.445954 | +0.009333 | 57,065,490 |
| block_gptq_real4 | 3.450313 | +0.013691 | 60,162,066 |
| mixed_polar_gptq | 3.447188 | +0.010567 | 57,925,650 |

Direct contrasts:
```json
{
  "task_polar3_5_minus_bf16": {
    "mean_delta": 0.009332850658255391,
    "ci95": [
      0.008380224426789852,
      0.010303287388305098
    ],
    "bootstrap_reps": 5000,
    "unit": "128-target contiguous block"
  },
  "block_gptq_real4_minus_bf16": {
    "mean_delta": 0.01369139191181668,
    "ci95": [
      0.012865750553731872,
      0.01451060467507829
    ],
    "bootstrap_reps": 5000,
    "unit": "128-target contiguous block"
  },
  "mixed_polar_gptq_minus_bf16": {
    "mean_delta": 0.010566894916217841,
    "ci95": [
      0.009605482798445746,
      0.011489891415953008
    ],
    "bootstrap_reps": 5000,
    "unit": "128-target contiguous block"
  },
  "mixed_minus_polar": {
    "mean_delta": 0.0012340442579624506,
    "ci95": [
      0.00036787009124875287,
      0.002087128181847799
    ],
    "bootstrap_reps": 5000,
    "unit": "128-target contiguous block"
  },
  "mixed_minus_gptq": {
    "mean_delta": -0.0031244969955988382,
    "ci95": [
      -0.004023326008790684,
      -0.002242528998326623
    ],
    "bootstrap_reps": 5000,
    "unit": "128-target contiguous block"
  }
}
```
