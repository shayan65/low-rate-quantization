# Experiment and paper review — 2026-09-22

Reviewed the saved Qwen3.8 confirmation, ternary v3/v4, and rate-curve results;
the confirmation runner, shared quantizer/decoder and bootstrap; and the current
LaTeX paper and low-rate proposal. This is a source/results audit, not an
independent GPU reproduction. No new GPU experiments were launched.

## Recommended paper direction

The strongest current story is a controlled empirical study of low-rate
quantization in hybrid language models: how codebook dimension, compensation,
rate, and tensor coverage affect loss under measured storage budgets. The
27B result strengthens that story, but supports a comparison on DeltaNet QKV
projections, not a fully quantized model or a causal scaling law.

Suggested title: **Low-Rate Quantization in Hybrid Language Models: Codebook
Dimension, Error Compensation, and Coverage**.

`main.tex` still presents “Two Negative Results” and says all LM experiments
use 0.8B and four index bits. It does not incorporate the latest experiments.
Choose a single paper thesis before adding more results: for the low-rate
paper, put the polar/SCF history in a concise motivation/appendix, retain its
lessons about controls, and organize the main text by experimental question.

## Claims to correct before circulation

1. **Withdraw the full-coverage 27B four-bit quality claim.** Section 2 of
   `ternary_proposal.md` already correctly identifies the historical +0.008393
   NLL result as a small-model projection-subset measurement. That is 0.24%
   of baseline NLL, but **0.843% relative perplexity**, not 0.24% perplexity.
   Sections 10–11 incorrectly turn it into a full-coverage 27B result. Neither
   quality nor runtime feasibility of that conversion was measured here.
   Label model-size calculations as estimates, count scale/codebook/header
   overhead and retained modules, reconcile 27.78B with the runner's
   26.896B text-model census, and distinguish GB from GiB. Packed weight
   storage fitting VRAM does not establish that execution fits: cache,
   recurrent state, activations, workspaces and decoding buffers also matter.

2. **Compare the same recipe across checkpoints.** The quoted −0.113585
   small-model storage-dominance margin belongs to `ternary_task_v3`, without
   GPTQ. Under the v4 GPTQ recipe, VQ8 versus scalar g64 is **−0.034303
   [−0.036660, −0.031971]**. Use this alongside the 27B −0.001736 interval.
   Also disclose calibration changes: v4 reports 65,536 training tokens;
   the 27B runner defaults to 32,768, and does not save the actual setting.

3. **Replace “tie” with “inconclusive,” and “buys nothing” with an interval.**
   Failure to reject zero is not an equivalence result. At 27B, scalar g64
   reduces damage by 0.002220 NLL, about 21.7% of scalar g128 damage; its
   interval [−0.004450, +0.000004] is almost entirely favorable. This is
   weak evidence for a benefit, not replicated evidence of no benefit.
   A claim of practical equivalence needs a justified, prespecified margin.
   These experiments also share evaluation text and implementation, so “three
   independent replications” is too strong.

4. **Describe approximate byte matching accurately.** VQ8 stores 542,743,056
   bytes versus scalar's 542,638,086: **0.01934% more**. VQ4 differs by only
   642 bytes. “Near-equal payload storage” is supported; literal parity or
   “at or below” is not. At 0.8B VQ8's overhead is about 0.43%.
   `Packed.payload_bytes()` counts indices and scales, and the runner adds a
   calculated FP16 codebook size. It does not persist a complete artifact or
   count a serialized manifest/rotation description. Distinguish this from
   complete-file byte accounting and independent file reload.

5. **Present smaller effects as an observation, not an explained cause.**
   The two checkpoints differ in generation/training, target coverage, tensor
   dimensions, calibration, and evaluated text extent. “The advantage also
   appears on this larger checkpoint” is defensible. “Scale caused the margins
   to shrink” and “the only limitation is evaluation resolution” are not
   established. Use actual parameter counts if stating a scale ratio.

6. **Narrow the rate-curve conclusion.** Five distinct rates from 1.6 to 3.2
   index bits on one model and 50.2% coverage support an approximately
   exponential decline over that interval. They cannot establish that no
   experiment could find a knee or that rate is purely a budget decision.
   The stored-byte range is 156.95/81.40 = **1.93×**, not 2.6×. A four-bit
   point on the same tensors is missing; the old QKV-only point cannot fill it.

## Statistical interpretation and the proposed longer evaluation

The primary near-equal-byte VQ8/scalar contrast is supported by the saved
paired interval. The 38.7% damage reduction is a descriptive ratio of point
estimates; give the absolute −0.003957 NLL contrast first. Bootstrap the ratio
jointly if reporting uncertainty on relative damage reduction.

For VQ8 versus scalar g64, the present half-width is 0.002684. Assuming
unchanged variance and exchangeable blocks, extending from 65,536 to 261,284
targets predicts a half-width of **0.001344**. If the effect stayed exactly
−0.001736, that projected interval would exclude zero. This is not a promise
that the experiment will settle the comparison: a normal approximation gives
only about **72% power** at that effect and sample size, and about **320k
targets** for 80% power under the same assumptions. These are rough planning
calculations based on a selected pilot estimate, not prospective guarantees.

Adjacent 128-target text blocks can be dependent even when model state resets.
Retain document/block identity and compare the current block bootstrap with
document-cluster or longer moving-block resampling. The current intervals
measure text-sampling variability conditional on one fitted quantizer; they
do not include calibration/codebook-seed variability.

Use the full validation stream as a fixed descriptive extension, not a
sequence of tests stopped when zero disappears. Since validation has guided
many decisions, freeze the recipe and primary contrasts before a final
untouched test or second-corpus evaluation. Report new/unseen text separately
from the original prefix as well as in the aggregate.

## Runner issues to address before spending more GPU time

- `run_qwen38_confirm_v1.py:175–176` creates only full 129-token windows.
  Increasing `--eval-blocks` cannot include the final partial block. Derive
  complete coverage from this tokenizer's actual token count rather than
  assuming the small model's 261,284 count applies.
- `_ev` is excluded from intermediate output and removed from final output.
  Save per-block sums, counts, identifiers and BF16 block losses to permit
  independent CI reconstruction, VQ8–VQ4 comparisons and dependence checks.
- Save exact arguments, checkpoint/tokenizer revision or hashes, data hashes,
  source revision, library/CUDA versions, RNG seeds, device map and GPU budget.
  The saved result does not itself attest to the reported 15 GiB configuration.
- Persist packed tensors, rounded codebook and rotation specification, then
  evaluate an independent reload. Record both payload and complete-file bytes.
- The default GPU budget remains **17 GiB**, the reported failing setting.
  Change/document the reproduction command to use the passing setting.
- Mismatches are logged per tensor but asserted only after the entire arm.
  Assert finite values and exact equality immediately, save the first failing
  case, and stop before installing a failed tensor. `max(maxdiff, NaN)` is not
  a robust rejection mechanism; explicit finiteness/equality checks are safer.

## Highest-value next experiments, in order

1. **Localize the exactness defect with a saved real failing tensor.** Capture
   indices, FP16 scales/codebook, reference reconstruction, first differing
   coordinates, strides/devices and allocation/device-map state. Compare:
   (a) unpacked indices to original indices; (b) unpacked scales to originals;
   (c) an independent CPU indexed-codebook-times-scale reconstruction to both
   GPU decode and the quantizer reference. Replay identical inputs in fresh
   processes under both budgets; add synchronized diagnostic runs and memory
   checking if needed. This separates packing, reconstruction and mutable
   reference corruption from changes in fitted quantization. Budget dependence
   is an observation, not evidence that memory budget changes arithmetic.
   A synthetic shape test and a clean layer 0 do not exclude data-dependent
   or later-layer faults. Preserve the assertion and passing-run results;
   do not relax tolerance to hide a 0.036 mismatch.

2. **Run one frozen full-stream confirmation after repairing logging/coverage.**
   Keep BF16, scalar g128, scalar g64, VQ4 and VQ8 if affordable. Prespecify
   VQ8–scalar g128 as primary; storage dominance and VQ8–VQ4 as secondary.
   Freeze artifacts so extended evaluation does not refit the method.

3. **Test whether eight dimensions are needed at 27B.** VQ4 captures about
   **95.4%** of VQ8's point-estimated improvement over scalar. VQ8 improves on
   VQ4 by just **0.000181 NLL**, with no saved paired interval for that contrast.
   This is a more informative dimensionality question than selecting VQ8
   because it has the smallest point estimate. Report measured fitting and
   quantization costs; packed inference costs still need a runtime.

4. **Add one generalization/mechanism check instead of many minor variants.**
   Use a frozen recipe on independent text with a longer evaluation context,
   and a few calibration/codebook seeds on the cheaper model. The 128-target
   reset protocol cannot establish robustness of recurrent state over long
   contexts. A targeted family-wise damage comparison (QKV versus MLP versus
   attention) at matched rate/coverage would directly test the proposed
   explanation for why coverage dominates codec refinements.

5. **If deployment remains a paper claim, measure it directly.** First add
   a matched four-bit reference on the existing 90-tensor small-model sweep.
   Then measure a credible full-coverage 27B four-bit baseline, quality,
   peak runtime memory and latency under stated batch/context settings.
   Until then, deployment remains motivation and an estimate.

## Positioning and proposed main figures

Generic “VQ plus Hessian compensation” and “Hadamard plus eight-dimensional
quantization” are established ideas: see [GPTVQ](https://arxiv.org/abs/2402.15319)
and [QuIP#](https://arxiv.org/abs/2402.04396). Position the contribution as
controlled evidence about hybrid-model projections and storage/coverage
tradeoffs. Include a credible published baseline if claiming method superiority;
do not infer a new algorithm from a local vector extension of GPTQ.

Build the main paper around (1) a protocol/coverage table; (2) scalar/VQ4/VQ8
paired effects across both checkpoints, with exact byte ratios; (3) the
rotation/compensation/dimension ablation; (4) the measured rate curve, with
coverage fixed and uncertainty; and (5) family/context generalization.
Keep codec correctness and the unresolved budget-sensitive defect explicit
in reproducibility/limitations. Generate reported tables from raw results to
prevent further mixing of versions.

## Suggested replacement for the headline claim

> On the 48 DeltaNet QKV projections of Qwen3.8-27B (9.36% of text-model
> parameters), eight-dimensional VQ with rotation and GPTQ-style compensation
> reduces validation NLL relative to a learned three-level scalar code by
> 0.003957 (paired block-bootstrap 95% CI: 0.001083–0.006805), using 0.019%
> more encoded payload bytes. This reproduces the direction of the small-model
> comparison under the evaluated protocol. Its advantage over the higher-storage
> scalar g64 control remains inconclusive. These are partial-model, short-context
> quality and storage results; full-model deployment quality and runtime memory
> remain unmeasured.
