# Model selection and polar-quantization failure analysis (2026-09-20)

> **Superseded in part (2026-09-20).** The polar-versus-real-4 comparisons in
> this document used a real-4 control with a *uniform* codebook. Giving the
> scalar control the same per-tensor Lloyd-Max fitting reverses the ranking, and
> the polar family is structurally capped above it. See
> [`polar_retirement.md`](polar_retirement.md) for the measurements that
> supersede these conclusions. The protocol notes, negative findings, and
> mechanism hypotheses below remain valid.



## Model choice

The first Transformer-only test should use `Qwen/Qwen3-0.6B-Base` (28 full-attention layers, 1,024 hidden width). It is close enough in scale to the existing Qwen3.5-0.8B hybrid experiment for a tractable RTX 3090 comparison. Use the base checkpoint and the same WikiText-2 and TinyStories text protocols, but compare degradation within each checkpoint rather than raw perplexity across different tokenizers.

If the 0.6B screen passes, replicate on `Qwen/Qwen3-1.7B-Base` (28 layers, 2,048 hidden width). `Qwen/Qwen3-4B-Base` is a later scale check, not the initial screen. Retain Qwen3.5-0.8B as the hybrid anchor and Qwen3.5-2B as an explicit negative checkpoint. The current Qwen3.8 collection begins with 27B and much larger models; these are not suitable for the first mechanistic experiment on a 24 GB RTX 3090.

For Transformer-only models, pair real coefficients in each selected attention projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and then MLP projections (`gate_proj`, `up_proj`, `down_proj`). No complex states or recurrent dynamics are needed: the method is a classical vector quantizer of two real weights. Compare one tensor at a time before cumulative replacement, and use identical output bytes, calibration windows, and target tokens for all codecs. Group attention and MLP results separately. Avoid comparing raw perplexity across checkpoints with different tokenizers.

Qwen3.8-Flash-Next has an important distinction: Qwen reports 125B language-model parameters with 6B activated per token, plus 51B n-gram-embedding and 4B MTP parameters. The official repository is about 360 GB. Sparse activation explains its high quality per unit of computation, but all expert weights must still be stored or streamed for arbitrary routing. It is a relevant eventual MoE/hybrid compression target; a full BF16 calibration reference on one 24 GB RTX 3090 is a different and much larger systems problem. A narrowly scoped expert-weight study could download selected shards, but it would need an independent reference method for routing and task loss and would not establish whole-model quality. Official source: [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and [repository files](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/main).

For a new-generation Qwen study, choose **Qwen3.8-27B** as the primary architecture target: it is dense, with 64 layers arranged in 16 Gated-DeltaNet/full-attention groups. The official BF16 language-model weight index totals 55.56 GB. That exceeds GPU memory and nearly fills the workstation's 60 GiB RAM, so begin with shard-local weight analysis and do not label it a full-model validation. The official FP8 variant is useful for deployment but is already quantized and cannot serve as an unquantized reference for our four-bit comparison. Qwen3.8-Flash-Next is a later MoE target, and Qwen3.8-2.4T-A95B is outside this workstation's scope.

### First Qwen3.8-27B analysis: complete tensors from shard 1

The official BF16 shard contains layer-0 DeltaNet `in_proj_qkv` (10,240×5,120), layer-3 attention `q_proj` (12,288×5,120), and layer-3 MLP `up_proj` (17,408×5,120). We processed **all rows** of each tensor. Equal-byte real4 and polar magnitude3+phase5 each store eight index bits per two real weights plus one FP32 scale per row and one candidate byte. Each codec tunes ten row-scale ratios by weight MSE; polar also selects the best of four deterministic pairings by weight MSE. This is a weight-reconstruction screen, not activation or NLL evaluation.

| Tensor | Real4 MSE | Best polar MSE | Polar improvement | Best pairing |
|---|---:|---:|---:|---|
| DeltaNet QKV | 3.0410e-6 | 2.9777e-6 | 2.1% | split-half |
| Attention Q | 4.0363e-6 | 3.8263e-6 | 5.2% | adjacent |
| MLP up | 1.3820e-6 | 1.3753e-6 | 0.5% | split-half |

The highest-magnitude quartile contributes 48.0%, 47.2%, and 43.4% of polar squared error in these tensors, respectively. This motivates magnitude-sensitive bit allocation and group scales, but does not establish a causal task-loss mechanism. Results: [`results/qwen38_27b_weight_geometry_full_v2/summary.md`](../results/qwen38_27b_weight_geometry_full_v2/summary.md). Next, obtain actual activations and paired task loss on a full-model reference before claiming polar helps Qwen3.8 performance.

## Measured observations

- On Qwen3.5-0.8B, calibration-selected polar beats equal-byte real4 and local block-GPTQ on matched all-token WikiText-2 and TinyStories validation.
- The preferred pairing differs by corpus: WikiText-2 selects reverse-half in 6 layers and adjacent in 5; TinyStories selects adjacent in 8 and reverse-half in 3. This establishes pairing sensitivity, not why it occurs.
- On TinyStories last-token-per-story evaluation, GPTQ beats polar. Context and target selection change the ranking.
- On the Qwen3.5-2B three-layer screen, polar wins none of layers 0, 8, and 16. This is checkpoint sensitivity under a small sample.
- A global phase-lattice rotation improves only one of three screened layers in each of 0.8B and 2B. A single global offset is insufficient.

## Mechanisms to test, not assume

1. **Pairing geometry.** Polar quantizes two weights jointly. Arbitrary pairing may combine coefficients with different activation importance, creating large functional error despite low weight MSE. Measure the covariance and activation-weighted reconstruction error for each candidate pairing.
2. **Bit allocation and angular sensitivity.** Three magnitude bits and five phase bits are fixed for every pair. Near-zero weights have unstable angles; large-magnitude pairs incur larger absolute angular error. Compute loss by magnitude and phase bins, and test 2+6, 3+5, 4+4, and 5+3 splits at the same eight index bits per pair.
3. **Outliers and group scales.** One FP32 scale per row can sacrifice typical weights to preserve a few large pairs. Plot magnitude quantiles, clipping rate, and quantization error by row; compare matched group sizes while charging every additional scale byte.
4. **Curvature and error propagation.** GPTQ uses activation Hessians and within-block error feedback; the polar codec does not. Compare weight MSE, activation-weighted output MSE, and task NLL on the same tensors. A lower MSE with worse NLL would implicate downstream sensitivity or correlated errors.
5. **Evaluation objective.** Last-token-per-story and all-token objectives weight positions differently. Stratify NLL degradation by token position and context length using paired examples, with the same quantized weights and no retuning.
6. **Cumulative interaction.** Per-layer improvements do not add linearly. Record single-layer and cumulative loss changes in a fixed order, and measure whether errors reinforce or cancel. The failed mixed and greedy selectors already suggest that local NLL is an imperfect surrogate for global NLL.

## Bounded next experiment

Run a single-layer diagnostic on Qwen3.5-0.8B layers 0, 8, and 16 and Qwen3-0.6B-Base attention layers 0, 9, and 18. For each tensor compare polar 3+5, real4, and local GPTQ on 2,048 fixed held-out WikiText-2 targets, with 512 train-only calibration targets. Save packed bytes, plain and activation-weighted MSE, angle/magnitude-bin errors, and position-stratified NLL. Promote to cumulative full-validation only if polar wins at least two of three Transformer layers against both controls and the mechanism diagnostics explain the ranking. If it fails, revise the quantizer before scaling.

Official model cards: [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base), [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), and [Qwen collections](https://huggingface.co/Qwen/collections).
