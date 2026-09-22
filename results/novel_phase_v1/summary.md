# Equal-budget novel phase screen

All quantized methods use 292-byte decoded-and-evaluated transition exports.

| Method | NLL | Perplexity | Delta NLL |
|---|---:|---:|---:|
| fp32 | 6.510978 | 672.488 | +0.000000 |
| uniform_response | 6.513691 | 674.317 | +0.002713 |
| warped_response | 6.513952 | 674.493 | +0.002975 |
| adaptive_5_3_response | 6.516453 | 676.181 | +0.005475 |

{"complete": true, "jobs": 12, "seconds": 2.2399733829661272, "winner": "uniform_response", "novel_method_beats_uniform_all_seeds": false, "action": "No Qwen or long training is automatically launched."}
