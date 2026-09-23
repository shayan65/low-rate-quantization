# Full-data findings and research decisions

Snapshot UTC: 2026-09-12T13:23:37.545545+00:00. Five of 51 runs completed; all completed comparisons currently use seed 0. See the synced CSV and immutable per-run source/config snapshots.

## Observed WikiText-103 results

| Hybrid representation | Test token perplexity | Deployment tensor payload (decimal MB) |
|---|---:|---:|
| continuous | 46.223 | 52.347 |
| restricted180 | 43.669 | 28.459 |
| full180 | 46.230 | 28.459 |
| full256 | 46.227 | 28.459 |
| real_fp32 | 24.122 | 52.342 |

All completed runs use three full training epochs, a train-only 8K BPE tokenizer, full validation-selected checkpointing and 322,999 real test next-token targets. Every training pass contains 135,350,274 targets. The data audit verified every source row in all splits. Synthetic operator studies and small-subset smoke values are not included here. Token perplexity cannot be compared directly across tokenizers.

## Interpretation

The signed 1–90-degree codebook improves seed-0 test perplexity by 5.53% relative to continuous shared-radius phases. Full-circle 180 and 256 codebooks are nearly identical to continuous phases. Restriction therefore merits replication as a possible inductive bias, rather than an advantage from finer phase precision. Selection of this hypothesis after seed-0 inspection is exploratory; seeds 1/2 and further datasets must be treated as replication, not evidence that the hypothesis was preregistered.

The real FP32 hybrid is substantially better: 24.122 perplexity versus 43.669, so the restricted polar version has 81.0% higher token perplexity. Earlier emphasis on the 5.53% within-polar improvement must not obscure this stronger baseline. Parameter counts are nearly equal, but real and complex state capacity and nonlinear computations differ; the comparison does not isolate a single quantization mechanism.

The quantized phase exports reduce complete deployment tensor payload by 45.63% against continuous polar weights, from 52.347 MB to 28.459 MB. This includes unchanged FP32 components and projection scales; it is not a claim of an 8-bit entire model. All these training runs execute floating-point QAT. No training-memory advantage or optimized low-bit inference speedup is established. A valid accuracy/storage comparison must include pending real 8-bit QAT.

## Next discriminating comparisons

1. Finish real 2/4/8-bit controls. Compare actual bytes and perplexity; a compressed real model may dominate the polar variants.
2. Finish the unconstrained complex FP32 baseline. If it matches real FP32 while polar variants lag, shared-radius/phase constraints become the leading concern. If it also lags, investigate initialization, complex nonlinearities and optimization. These are diagnostic hypotheses, not deductions from current results.
3. Finish Transformer controls and seeds 1/2 to determine whether the restricted-phase effect depends on complex recurrence and whether it reproduces.
4. Before launching modified models, define a validation-only follow-up protocol for group scales, residual real components or amplitude codebooks. Preserve this first sweep unchanged. Do not choose modifications by repeatedly inspecting test scores.
5. Only after a competitive representation is established, test temporal-response calibration against weight/activation reconstruction and learned-codebook controls on trained models. The present language-model sweep does not implement that proposed contribution.

## Prior-art and paper boundaries

iFairy already explores four-direction complex LLM quantization: https://arxiv.org/abs/2508.05571. Fairy2i explores conversion of pretrained real LLMs and complex phase quantization: https://arxiv.org/abs/2512.02901. These are required related work; our phase4 implementation is a codebook control, not a faithful reproduction of either complete training framework. Angular restriction alone has no verified novelty.

The existing paper/main.tex remains the earlier explicitly labeled sampled-operator draft. These real-data findings must not be described as validation of its memory-calibration objective, because that objective is absent from this sweep. No quantum hardware, entanglement or quantum advantage result exists. A future paper should lead with the final method that survives controls rather than assert a contribution before evidence.
