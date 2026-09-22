# Full-data WikiText-103 sweep

Completed 7 / 51 runs. Pending runs have no reported metrics.

All reported results come from full real WikiText-103 splits. BPE token perplexity uses the train-only 8K tokenizer and is not directly comparable to a different tokenizer. All training is floating-point QAT.

| Architecture | Method | Seeds completed | Test perplexity mean ± SD | Deployment tensor bytes |
|---|---|---:|---:|---:|
| hybrid | continuous | 1 | 46.223 (one seed) | 52,346,952 |
| hybrid | restricted180 | 1 | 43.669 (one seed) | 28,459,080 |
| hybrid | full180 | 1 | 46.230 (one seed) | 28,459,080 |
| hybrid | full256 | 1 | 46.227 (one seed) | 28,459,080 |
| hybrid | real_fp32 | 1 | 24.122 (one seed) | 52,342,272 |
| hybrid | real2 | 1 | 27.695 (one seed) | 22,482,504 |
| hybrid | real4 | 1 | 24.788 (one seed) | 24,473,160 |

Real and polar models differ in effective state capacity; the unconstrained complex model also has more parameters. These comparisons cannot attribute all differences to quantization. The matched phase variants isolate codebook coverage. Temporal calibration and downstream Qwen validation are separate, uncompleted experiments.
