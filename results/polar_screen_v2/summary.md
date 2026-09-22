# Small real-data magnitude/phase screen

0.6M-scale custom hybrid, real WikiText-103 source blocks, train-only 8K tokenizer. Only 262,144 selected training targets and 16,384 selected validation targets. Test set untouched. These are screening results, not full-dataset perplexity.

| Method | Seeds completed | Validation sample perplexity mean ± SD | Deployment tensor bytes |
|---|---:|---:|---:|
| real_fp32 | 3 | 538.626 ± 4.232 | 2,493,184 |
| real8 | 3 | 538.983 ± 4.200 | 2,272,024 |
| polar_fp32 | 3 | 588.170 ± 5.299 | 2,788,352 |
| mag4phase4 | 3 | 591.058 ± 3.609 | 2,272,280 |

Gate: {"completed_jobs": 12, "planned_jobs": 12, "budget_exhausted": false, "this_invocation_wall_seconds": 77.84237461799057, "mean_paired_nll_gaps": {"real8": 0.09223876396814983, "polar_fp32": 0.004913727442423503}, "passed": false, "action": "No full-data job is launched by this screen."}

Matching 8 index bits does not imply identical parameter count, complex state capacity or optimization. Shared FP32 clip scales and unchanged components are counted in export bytes. A passing small-sample gate only warrants another limited test; it does not establish novelty, superiority or justify an automatic long run.
