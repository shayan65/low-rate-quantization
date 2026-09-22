# Small real-data magnitude/phase screen

0.6M-scale custom hybrid, real WikiText-103 source blocks, train-only 8K tokenizer. Only 262,144 selected training targets and 16,384 selected validation targets. Test set untouched. These are screening results, not full-dataset perplexity.

| Method | Seeds completed | Validation sample perplexity mean ± SD | Deployment tensor bytes |
|---|---:|---:|---:|
| real_fp32 | 3 | 693.153 ± 6.260 | 2,493,184 |
| complex_fp32 | 3 | 672.488 ± 2.885 | 2,788,352 |
| polar_fp32 | 3 | 770.840 ± 3.375 | 2,788,352 |

Gate: {"completed_jobs": 9, "planned_jobs": 9, "budget_exhausted": false, "this_invocation_wall_seconds": 58.59712695796043, "mean_paired_nll_gaps": {"real_fp32": 0.10625152786572774, "complex_fp32": 0.13649741808573404}, "passed": false, "action": "No full-data job is launched by this screen."}

Matching 8 index bits does not imply identical parameter count, complex state capacity or optimization. Shared FP32 clip scales and unchanged components are counted in export bytes. A passing small-sample gate only warrants another limited test; it does not establish novelty, superiority or justify an automatic long run.
