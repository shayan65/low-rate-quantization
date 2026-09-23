# Proposal: Shared low-rate vector quantization under measured storage budgets

**Revised:** 2026-09-21, following implementation review.
**Status:** historical rate hypothesis failed its gate. A corrected, bounded
Qwen3.5-0.8B feasibility experiment is authorized; 27B deployment remains gated.
**Previous draft:** `ternary_proposal_pre_review_20260921.md` is retained as a
historical record, including claims withdrawn below.

## 1. Question and scope

Can one shared eight-dimensional weight codebook improve language-model loss
relative to a learned scalar code while using no more serialized bytes, on the
18 Gated DeltaNet input projections of Qwen3.5-0.8B-Base?

This is classical post-training vector quantization (VQ). A vector codebook
with 3^8 = 6,561 entries has an ideal index rate of log2(3) per weight, but its
reconstructed components are arbitrary codebook values. It is **not literal
ternary weights**, and it does not implement a qubit or quantum computation.
Literal ternary weights, reconstructed as {-s, 0, +s}, are a separate control.
No model parameters are trained with language-model gradients in this study.

The first checkpoint has 24 layers: 18 Gated DeltaNet and six full-attention
blocks. Quantizing only the 18 QKV input projections does not produce an
entire 0.8B model at the reported target-tensor bit rate. Untouched parameters
remain BF16 and must be charged when estimating whole-model storage.

## 2. What the earlier experiments establish

The historical four-bit experiment measured +0.008393 NLL relative to a
3.436670 baseline in this small-model projection subset. That is about 0.24%
of baseline NLL, or exp(0.008393)-1 = 0.84% relative perplexity. It does not
establish that every four-bit method, every layer, or every 27B model is an
easy regime. Weak controls and failed mechanisms also contributed to the
previous negative results.

The first rate sweep is retained in `results/rate_sweep_v1/`. Its fitted
codebooks and group scales were evaluated at FP32 precision and its nominal
rates excluded codebook and packing overhead. These are **equal-cardinality
reconstruction comparisons**, not matched-byte deployment measurements.

A review found that aggregation used tensor names without model identity.
The layer-0 names collide across the two checkpoints. Corrected analysis uses
(model, tensor) throughout, validates full coverage, and writes a separate
`results/rate_sweep_v1_reanalysis/` without changing the original measurements.
Across all six distinct tensors, the mean post-rotation MSE reductions are:

| Ideal index bits/weight | Dimension 2 | Dimension 4 | Dimension 8 |
|---:|---:|---:|---:|
| log2(3) | 4.80% | 13.91% | 23.19% |
| 2 | 8.55% | 18.57% | not measured |
| 3 | 13.38% | 24.16% | not measured |
| 4 | 11.69% | not measured | not measured |

At the three-level rate, dimension-8 gains average 20.65% on the three 27B
tensors and 25.73% on the three 0.8B tensors. The original requirement of a
15% gain and at least twice the four-bit gain on every tested tensor fails.
That decision stays **STOP for the original rate mechanism**. It is not a
proof that VQ cannot work at low rates.

These fits are best observed results under a specific optimizer, sample,
initialization and dimension cap, not an information-theoretic ceiling.
K-means may converge to a local minimum. Product-grid containment guarantees
an optimum no worse than the product grid only for the same objective and
precision; it does not guarantee a random fit finds that optimum. A randomized
Hadamard transform is orthogonal but is not generally a whitening transform.
Rotation and VQ may be complementary. Classical asymptotic shaping gains do
not supply a universal finite-rate ceiling for these learned quantizers.

## 3. Preliminary language-model results, with limitations

A newer legacy run exists at `results/ternary_task_v1/`. Preserve it rather
than relabeling it as an execution of the corrected protocol:

| Legacy arm | NLL | Change versus BF16 |
|---|---:|---:|
| BF16 | 3.436670 | 0 |
| Learned scalar, g128, rotated | 3.795474 | +0.358803 |
| Learned scalar, g64, rotated | 3.790168 | +0.353497 |
| Shared dimension-8 VQ, g128, rotated | 3.658186 | +0.221516 |
| Per-tensor dimension-8 VQ, g128, rotated | 3.645873 | +0.209203 |

The shared arm's observed difference versus the g64 scalar arm is -0.131981
NLL. This is encouraging preliminary evidence. However, the legacy run used
FP32 quantizer metadata, reported ideal rates rather than actual files,
omitted the final 36 validation targets (261,248 rather than 261,284), and did
not retain the per-block losses needed to recompute its paired intervals.
Its `quantize_seconds` also includes evaluation. Withdraw the prior claims of
byte matching, a complete validation stream, a Bonsai reproduction, and
"Step 3 justified." Do not claim the corrected implementation reproduces
these numbers until its separate run finishes.

## 4. Corrected codec and storage contract

All arms use the same fixed block-1024 randomized Hadamard transform along
input columns. Sign information needed for decoding is included in the
artifact. Group scales and learned codebooks are rounded to FP16 before
assignment and reconstruction; the stored precision is the evaluated precision.
Files include codec version, tensor shape, group size, dimension, index format,
rotation information, and byte counts. A shared book is stored and charged
once per arm. Evaluation reloads the artifacts from disk.

| Arm | Representation | Group size | Payload and scale cost, before other metadata |
|---|---|---:|---:|
| Genuine ternary | {-s, 0, +s}; five trits per byte | 128 | 26 index bytes + 2 scale bytes per group = 1.750 bpw |
| Learned scalar3 | three arbitrary FP16 levels; five trits per byte | 128 | 1.750 bpw, plus codebooks |
| Learned scalar3, stronger storage control | same, with finer scales | 64 | 13 index bytes + 2 scale bytes per group = 1.875 bpw, plus codebooks |
| Shared VQ8 | K=6,561 FP16 vectors; 13-bit indices | 128 | 1.625 index bpw + 0.125 scale bpw = 1.750 bpw, plus shared book |

The shared VQ8 book costs 6,561 x 8 x 2 = 104,976 bytes, about 0.007416 bpw
when amortized over 18 tensors of 6144 x 1024 weights. The exact reported rate
is 8 times all codec artifact bytes divided by all encoded weights, including
headers, sign storage, and alignment. No ideal-logarithm rate is called an
actual size. The primary comparison is a **storage/quality dominance test**
against scalar g64, not a claim of equal bytes. The g128 scalar remains a
useful neighboring point on the storage-quality curve.

For learned codes, fit normalized vectors with each vector weighted by its
FP16 group scale squared, so the objective matches original-weight MSE.
Use a bounded sample, deterministic seeds, product initialization/fallback
and limited restarts. Score candidate books at their stored precision. Record
fit settings. Memory for nearest-neighbor distances must be bounded as a
function of codebook size, with deadline checks inside fitting loops.

## 5. Corrected experiment and decision rule

**Model:** cached Qwen3.5-0.8B-Base, frozen, text path, all 18 DeltaNet QKV
projections. Check architecture and tensor shapes rather than assuming them.
**Arms:** BF16, genuine ternary g128, learned scalar3 g128, learned scalar3
g64, shared VQ8 g128; all compressed arms rotated identically.
**Data:** WikiText-2. Fit codebooks on model weights; use a small training-text
slice only for the nonreportable implementation smoke test. The quality run
uses every token target in the validation stream, including the final partial
block, under the established 128-target context-reset protocol. Test remains
untouched. Record split hash, token stream hash, checkpoint provenance,
software versions and exact run configuration.

Retain each block's loss sum and target count. Compute token-weighted NLL and
perplexity, and paired block-bootstrap intervals with the same target-count
weighting. These intervals describe variation over this corpus and do not
establish robustness to calibration seeds, domains, or other checkpoints.

Freeze one primary comparison before the corrected run: shared VQ8 minus
learned scalar3 g64. A feasibility pass requires all of:

1. VQ uses no more **actual serialized target-tensor bytes** than scalar g64.
2. VQ improves NLL by at least 0.005 nats per target.
3. The upper endpoint of the paired 95% interval is below zero.
4. It recovers at least 10% of scalar g64's positive NLL damage versus BF16.

The numerical thresholds are engineering choices for this revised study,
informed by the preliminary run. This remains exploratory, not a fresh
confirmatory test. Secondary comparisons are descriptive. A passed small-model
gate permits planning independent seed/domain/model checks; it does not
automatically authorize a multi-day 27B run or establish publishable novelty.

**Execution:** unit and serialization checks first; then a training-only smoke
with a short hard cap; then one full small-model run only if smoke succeeds.
A process lock prevents overlapping runs, and a hard wall-clock deadline
bounds compute. Refuse nonempty output directories to prevent accidental
mixing of runs. Preserve partial status on failure. Separate quantization,
decode and evaluation timing and measure peak allocated/reserved GPU memory.
Do not interpret BF16 evaluation throughput as packed-kernel speed.

## 5b. Corrected experiment: results (2026-09-21)

> **Byte-matching convention (revised 2026-09-22).** Comparisons described as
> byte-matched are *near-equal payload bytes*, not literal parity. Dimension-4
> VQ costs +0.0029% over scalar ternary at 0.8B and +0.00012% at 27B;
> dimension-8 costs +0.43% and +0.0193%. Payload counts indices plus FP16
> group scales plus the shared codebook; they are not complete-file byte
> counts, and §11's v2 runner reports both.

Raw: `results/ternary_task_v3/`. The corrected protocol of section 4 has now
been executed. Every arm asserts `decode(encode(w))` is bit-exact, bits/weight
is computed from `len(bytes)`, and the evaluation covers the complete
**261,284-target** stream, token-weighted so the ragged 36-target block cannot
count as a full one.

| Arm | bits/wt | stored MB | weight MSE | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|:--|
| learned scalar3 g128 | 1.7250 | 24.42 | 7.642e-05 | +0.441525 | [+0.437, +0.446] |
| learned scalar3 g128, rotated | 1.7250 | 24.42 | 7.185e-05 | +0.356599 | [+0.352, +0.361] |
| learned scalar3 g64, rotated | 1.8500 | 26.19 | 7.143e-05 | +0.353765 | [+0.349, +0.359] |
| **dim-4 VQ g128, rotated** | **1.7251** | **24.42** | 6.183e-05 | **+0.258156** | [+0.254, +0.262] |
| **dim-8 VQ g128, rotated** | **1.7324** | **24.52** | 5.717e-05 | **+0.240180** | [+0.237, +0.244] |
| dim-8 VQ g128, unrotated | 1.7324 | 24.52 | 5.831e-05 | +0.246575 | [+0.243, +0.250] |
| dim-4 VQ, scale²-weighted fit | 1.7251 | 24.42 | 6.184e-05 | +0.263971 | [+0.260, +0.268] |
| dim-8 VQ, scale²-weighted fit | 1.7324 | 24.52 | 5.729e-05 | +0.261669 | [+0.258, +0.266] |

**Storage/quality dominance is established.** Against `scalar3 g64` — a control
that stores **6.4% more** bytes than the dim-8 VQ arm — dim-8 VQ is $-0.113585$
NLL, CI $[-0.117539,-0.109645]$. It wins while storing strictly less.

**At near-equal payload bytes** (dim-4 VQ stores 714 bytes more than scalar3
g128, +0.0029%), dim-4 VQ is
$-0.098443$, CI $[-0.102171,-0.094707]$: **27.6% less damage for identical
bytes.** dim-8 VQ reaches 32.6% at +0.43% bytes.

### Packing: two improvements over the section-4 contract

Section 4 assumes per-group padding (26 bytes per 128-weight group) and 13-bit
VQ indices, putting both formats at 1.750 bpw. Packing across the tensor rather
than per group, and packing $K$-ary codes $c$ at a time where $K^c<2^{64}$,
reaches **exactly 1.600 bpw of index for both formats**:

| | scheme | index bpw |
|---|---|---:|
| ternary | 5 trits/byte, tensor-wide | 1.60000 |
| dim-8 VQ (K=6561) | 5 codes/uint64 | 1.60001 |
| dim-4 VQ (K=81) | 10 codes/uint64 | 1.60001 |

Measured index bytes differ by 4 out of 1,258,292 per tensor. Fixing the
codes-per-word at 5 would have inflated dim-4 to 3.2 bpw, so the word packing
must adapt to $K$.

### The scale²-weighted fit does not work, on its own objective

Section 4 prescribes weighting each normalized vector by its FP16 group scale
squared so the fit matches original-weight MSE. Implemented and measured, it
**fails to improve even weight MSE** (+0.0% at dim-4, +0.2% at dim-8) while
costing $+0.0058$ and $+0.0215$ NLL respectively.

The likely mechanism is that per-group normalization has already removed most
of the scale variation the weighting is meant to correct, so the reweighting
buys nothing while collapsing the effective sample size of the fit — which
matters far more for a 6561-point codebook than an 81-point one, exactly the
observed pattern. We recommend dropping it from the contract.

### Weight MSE is predictive at ternary rate, unlike at 4 bits

> **Narrowed by §5c.** What follows holds *within* the stripped recipe only.
> Under GPTQ error compensation the relationship inverts: weight MSE rises 38%
> while NLL damage falls 68%. Read this section as scoped to a fixed recipe.

Across the six main arms the weight-MSE ordering matches the NLL ordering
exactly (7.642 > 7.185 > 7.143 > 6.183 > 5.831 > 5.717 against 0.4415 > 0.3566
> 0.3538 > 0.2582 > 0.2466 > 0.2402). At 4 bits this project repeatedly found
weight error failing to predict task damage; at ternary rate, where damage is
an order of magnitude larger, it tracks. The exception is the scale²-weighted
pair, which has near-identical weight MSE but materially worse NLL — so the
metric is predictive *across codec families* but not *across fitting
procedures*.

### Rotation: confirms Step 0, and retracts a claim made from the flawed run

| | value of rotation |
|---|---:|
| on learned scalar3 | $-0.084926$ NLL |
| on dim-8 VQ | $-0.006395$ NLL |

Rotation is worth 13x more to the scalar code than to the dim-8 code, which is
what Step 0's weight-MSE analysis predicted (rotation closes 62.5% of the dim-2
gap but only 13.2% of the dim-8 gap). An earlier claim, made from the flawed
run, that weight MSE *understated* rotation and that rotation and VQ were
complementary, is **withdrawn**: it was an artifact of FP32 scale metadata.

### What the corrections cost the earlier headline

The legacy run claimed 38.3% for dim-8; corrected, it is 32.6%. The difference
is FP16 scale storage, which cost the dim-8 arm $+0.019$ NLL while leaving the
scalar arm unchanged at $+0.0001$ — a 6561-point codebook fitted in normalized
space is far more sensitive to scale quantization than a 3-point one. That
sensitivity is invisible unless the evaluated weights are decoded from stored
bytes, which is the case for doing it that way.

## 6. Deployment and 27B gate

The implementation produces reloadable compressed projection artifacts and
installs their decoded BF16 weights in the existing model for evaluation.
It does not yet provide resident packed inference, a compressed KV cache, or
a fully quantized 27B checkpoint. Actual total model storage must include every
untouched tensor; target-tensor bpw alone cannot predict it.

A fully decoded 27B model cannot fit in 24 GiB. Before any larger conversion,
choose either measured tensor/layer streaming with explicit offload costs or
a packed execution implementation. Prototype the **dimension-8, 6,561-entry**
decode path that produced the gain; a dimension-2 nine-entry kernel measures
a different method. Establish peak memory and end-to-end latency before a
speed or 3090-residency claim. Quantization must stream input shards and write
output incrementally. Broader coverage also needs explicit policies for
attention, MLP, embeddings, heads and sensitive recurrent parameters.

## 7. External baseline and related work

The [Bonsai model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
reports custom ternary artifacts derived from Qwen3.8-27B, Hadamard transforms,
and a benchmark-score retention figure. Its compressed file sizes do not
predict our decoded runtime footprint. Its formats require the supported
[PrismML runtime fork](https://github.com/PrismML-Eng/llama.cpp); stock compatibility
must not be assumed. Reproducing perplexity would evaluate a different axis
from its reported thinking-task average. Our learned scalar control is not
its released ternary model. Use BF16 as the primary reference; label Q8
separately if used, since it introduces its own quantization error.

The basic combination is not new:

- [QuIP#](https://arxiv.org/abs/2402.04396) combines randomized Hadamard
  incoherence processing and eight-dimensional lattice vector quantization.
- [GPTVQ](https://arxiv.org/abs/2402.15319) studies dimensionality, compression,
  accuracy and efficient VQ reconstruction with Hessian-aware quantization.
- [AQLM](https://arxiv.org/abs/2401.06118) uses additive codebooks for extreme
  compression. An explicit unstructured 16^8-entry book is not the only way
  to represent eight-dimensional vectors at four bits per weight.

A potential contribution is a controlled measurement of shared-codebook
storage/quality/runtime tradeoffs across recurrent and attention projections
in hybrid models, compared with credible published methods. Neither a new
checkpoint nor a rate sweep alone establishes novelty. A stronger paper
needs independent replications and a mechanism or practical advantage beyond
these precedents. Retain negative results without turning them into universal
claims about all scalar or vector quantizers.

## 5c. A real recipe: activation weighting and error compensation (2026-09-21)

Sections 5/5b compared codecs under a deliberately stripped recipe: nearest
neighbour assignment against a codebook fitted to the weights alone, using no
calibration activations. That is a fair codec comparison but not a competitive
pipeline, and it leaves open the objection that vector quantization only buys
back error that error compensation would have removed anyway. `run_ternary_task_v4.py`
tests that objection directly by adding the two things every serious low-rate
pipeline has and §5b had neither of.

**Activation weighting.** Per target tensor, accumulate $H=\mathbb{E}[xx^{\mathsf T}]$
over 65,536 calibration tokens from the wikitext2 *train* split (evaluation
remains the full validation stream, so no activation statistic is fitted on the
evaluation data). Use $h=\operatorname{diag}H$ to weight both the codebook fit
and the assignment.

**GPTQ-style error compensation.** Quantize in column order and push each
block's residual onto the not-yet-quantized columns through the inverse
Cholesky factor of $H$.

### The vector generalization of GPTQ, and its test

Extending GPTQ to a codec that quantizes $d$ columns jointly is the one piece
that is not standard. For a block of columns $J$, the cost charged by the GPTQ
derivation and the compensation applied to the remainder are

$$\|(W_J-Q_J)\,U_{JJ}^{-1}\|_F^2,\qquad W_{\text{rest}}\;{-}{=}\;(W_J-Q_J)\,U_{JJ}^{-1}U_{J,\text{rest}},$$

with $U$ the upper Cholesky factor of $H^{-1}$. So the within-block decision is
a *metric* nearest-neighbour search with $M=U_{JJ}^{-1}$: transform the points
and the codebook by $M$ and search as usual. At $d=1$ this must collapse to
plain nearest-level with the error divided by $U_{jj}$ — ordinary scalar GPTQ.

`test_gptq_v4.py` checks that collapse against an independently written
textbook implementation and finds it **bit-exact** (max $|\Delta|=0$), checks
the transformed search against brute-force metric nearest-neighbour, checks
that the rotated Hessian equals the Hessian of rotated activations (relative
error $2.5\times10^{-7}$), and checks that `decode(encode(w))` stays bit-exact
under compensation at $d\in\{1,4,8\}$. The uncompensated path reproduces the
§5b quantizer exactly, and the `scalar3_rot` and `vq8_rot` arms reproduce the
§5b numbers to all printed digits — so v4 is a superset of v3, not a
replacement for it.

Storage is untouched by both additions: they change *which* codes are chosen,
never the format. Bits/weight is therefore identical to §5b by construction and
the comparison stays byte-for-byte.

### Results

| Arm | bits/wt | weight MSE | act-weighted err | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|:--|
| scalar3_rot | 1.7250 | 7.184e-05 | 2.3293e-04 | $+0.356599$ | $[+0.352028, +0.361286]$ |
| scalar3_rot_actw | 1.7250 | 7.185e-05 | 2.3292e-04 | $+0.361084$ | $[+0.356485, +0.365764]$ |
| scalar3_rot_gptq | 1.7250 | 9.889e-05 | 6.4700e-05 | $+0.113179$ | $[+0.109919, +0.116351]$ |
| scalar3_g64_rot_gptq | 1.8500 | 9.863e-05 | 6.4136e-05 | $+0.111363$ | $[+0.108285, +0.114369]$ |
| vq8_rot | 1.7324 | 5.717e-05 | 1.7333e-04 | $+0.240180$ | $[+0.236367, +0.243903]$ |
| vq8_rot_actw | 1.7324 | 5.866e-05 | 1.5524e-04 | $+0.227827$ | $[+0.224049, +0.231700]$ |
| vq4_rot_gptq | 1.7251 | 8.537e-05 | 5.4554e-05 | $+0.088512$ | $[+0.085730, +0.091370]$ |
| **vq8_rot_gptq** | **1.7324** | 7.919e-05 | 4.9996e-05 | $\mathbf{+0.077061}$ | $[+0.074362, +0.079868]$ |
| vq8_rot_gptq_actwfit | 1.7324 | 7.931e-05 | 5.0085e-05 | $+0.077267$ | $[+0.074642, +0.079910]$ |
| **vq8_rot_gptq_seq** | **1.7324** | 7.971e-05 | 4.8564e-05 | $\mathbf{+0.075171}$ | $[+0.072454, +0.077936]$ |

### Error compensation matters far more than the codec

GPTQ removes $-0.243420$ $[-0.247197,-0.239658]$ from scalar ternary — **68.3%
of the damage**, at identical bytes. That is more than twice what the entire
§5b vector-quantization result bought. Stated plainly: the largest single
improvement in this arc comes from a 2022 method the earlier runs simply did
not have, not from anything novel here.

### The vector-quantization margin survives, at nearly the same relative size

| | scalar3 | vq8 | margin | relative |
|---|---:|---:|---:|---:|
| stripped recipe (§5b) | $+0.356599$ | $+0.240180$ | $-0.116419$ | 32.6% |
| with GPTQ | $+0.113179$ | $+0.077061$ | $-0.036118$ | 31.9% |

The paired interval for the compensated comparison is
$[-0.038414,-0.033768]$, excluding zero. The absolute margin shrinks roughly
threefold because everything shrinks threefold; **the relative margin is
essentially unchanged**. So the two mechanisms are close to independent, and
the objection that VQ was only recovering what compensation recovers is
answered: it is not.

At near-equal payload bytes (+0.0029%), `vq4_rot_gptq` beats `scalar3_rot_gptq`
by $-0.024667$ $[-0.027041,-0.022348]$, or 21.8% less damage — down from 27.6%
in the stripped recipe, so dimension-4 does lose some ground to compensation
even though dimension-8 does not.

### The storage/quality argument gets sharper, not weaker

Under GPTQ, spending 6.4% more bytes on finer scalar groups buys **nothing
measurable**: `scalar3_g64_rot_gptq` vs `scalar3_rot_gptq` is $-0.001815$
$[-0.003794,+0.000172]$, an interval that **includes zero**. The same budget
spent on codebook dimension instead — an extra 0.43% of bytes — buys
$-0.036118$. The dominance claim therefore holds in its strongest form:
`vq8_rot_gptq` beats a control that stores 6.4% *more* by $-0.034303$
$[-0.036660,-0.031971]$.

### Diagonal activation weighting is not where the value is

On scalar ternary it is actively harmful: $+0.004485$ $[+0.003883,+0.005107]$.
This is mechanically clear — at $d=1$ a positive scalar metric cannot change an
argmin, so `actw` only perturbs the codebook fit, moving it off the unweighted
Lloyd optimum in exchange for a metric that never gets used. On dimension-8 VQ,
where it changes the assignment too, it helps by $-0.012353$. But once GPTQ is
present the weighted fit adds nothing ($+0.000206$, a tie), because the block
metric $U_{JJ}^{-1}$ already carries the diagonal and the cross terms besides.

Together with §5b's scale²-weighted result, that is two independent weighting
schemes that looked principled and delivered nothing. The value in $H$ is in
the off-diagonal structure, which only compensation exploits.

### Weight MSE inverts under a real recipe — §5b's claim is narrowed

§5b reported that weight-MSE ordering matched NLL ordering exactly at ternary
rate. That holds *within* the stripped recipe and **fails across recipes**:
GPTQ *raises* weight MSE by 38% ($7.18\to9.89\times10^{-5}$) while cutting NLL
damage by 68%. Ranked by weight MSE, the best arm in this table looks like one
of the worst. The §5b claim should be read as scoped to a fixed recipe, not as
a statement about the metric.

The activation-weighted error $\operatorname{tr}(\Delta W\,H\,\Delta W^{\mathsf T})/n$
does track: across all ten arms its ordering matches the NLL ordering with a
single adjacent inversion, and that inversion is between two arms whose values
differ by 0.004%. The caveat is that this quantity is in-sample for the GPTQ
arms — it is the objective they optimize — so it is a diagnostic, not an
independent predictor. Its rank agreement *across* recipe families is
nonetheless the first proxy in this project to survive a setting where plain
weight error reverses.

### Sequential statistics

Statistics are captured once from the BF16 model rather than re-captured after
each layer is quantized. Rather than assume the difference is negligible, it is
measured: `vq8_rot_gptq_seq` re-captures $H$ for each target against already
quantized earlier layers, at 3.5x the quantization cost.

Paired against the static arm on a rerun of both (`results/ternary_task_v4_seq`),
sequential capture is worth $-0.001890$ $[-0.003421,-0.000315]$ — a real effect,
excluding zero, but 2.5% of the remaining damage for 3.5x the quantization time.
This is the retired SCF question appearing once more in a setting where the
statistics genuinely do feed back: the forward-only dependency is real, it is
resolved by a single ordered pass, and its magnitude is small. Static capture is
the right default; the sequential variant is available and cheap enough to
enable at 27B if the budget allows.

### Where this leaves the 27B gate

| | NLL | perplexity | vs BF16 |
|---|---:|---:|---:|
| BF16 | 3.436710 | 31.09 | — |
| §5b best (`vq8_rot`) | 3.676890 | 39.52 | $+27.2\%$ |
| §5c best (`vq8_rot_gptq_seq`) | 3.511881 | 33.51 | $+7.8\%$ |

The recipe improvement is 68.7% of the remaining damage. A ternary conversion
that costs 7.8% perplexity on the target tensors is a materially more credible
thing to scale than one that costs 27.2%, and the §6 gate should be read
against the new number. The §6 engineering conditions are unchanged: none of
this provides resident packed inference, a compressed KV cache, or measured
peak memory and latency at 27B.

## 8. MLP feasibility: does the recipe transfer off the projections? (2026-09-21)

Every result in §§1–5c was measured on `linear_attn.in_proj_qkv` alone — 15.1%
of the parameters at 0.8B and **9.1% at 27B**. The MLPs are **63.5% of
Qwen3.8-27B** and were entirely unmeasured, so the §6 conversion plan rested on
an untested assumption: that a recipe fitted to DeltaNet input projections
transfers to `gate/up/down`. `run_mlp_task_v1.py` is the cheap falsification,
run before any 27B time is spent.

All 72 MLP tensors (264,241,152 params, 35.1% of the 0.8B) are quantized while
`in_proj_qkv` is held at BF16, isolating the MLP question. `down_proj` is
(1024, 3584) and 3584 is not a multiple of the 1024-element Hadamard block, so
it uses a 512-element block (3584 = 7×512); both block sizes were verified to
round-trip and to satisfy the rotated-Hessian identity before the run.
`gate_proj` and `up_proj` read the same tensor, and their independently
captured Hessians agree to **exactly zero**, which validates the capture path.

| Arm | bits/wt | weight MSE | act-weighted err | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|:--|
| mlp_scalar3_rot | 1.7250 | 2.392e-05 | 3.561e-05 | $+2.119720$ | $[+2.106,+2.133]$ |
| mlp_scalar3_rot_gptq | 1.7250 | 3.337e-05 | 1.125e-05 | $+0.658007$ | $[+0.652,+0.664]$ |
| mlp_scalar3_g64_rot_gptq | 1.8500 | 3.327e-05 | 1.117e-05 | $+0.656502$ | $[+0.651,+0.662]$ |
| mlp_vq4_rot_gptq | 1.7250 | 2.885e-05 | 9.537e-06 | $+0.552558$ | $[+0.547,+0.558]$ |
| **mlp_vq8_rot_gptq** | 1.7282 | 2.679e-05 | 8.743e-06 | $\mathbf{+0.500777}$ | $[+0.495,+0.506]$ |
| mlp_vq8_rot_gptq_pertype | 1.7345 | 2.687e-05 | 8.775e-06 | $+0.521963$ | $[+0.516,+0.528]$ |

### The codec ordering transfers

`mlp_vq8_rot_gptq` beats `mlp_scalar3_rot_gptq` by $-0.157230$
$[-0.161835,-0.152538]$ — **23.9% less damage**, against 31.9% on the
projections. Smaller, but far outside the interval. Storage dominance holds in
the same strong form: it also beats `mlp_scalar3_g64_rot_gptq`, which stores
6.4% *more* bytes, by $-0.155725$ $[-0.160014,-0.151299]$. At near-equal bytes
(1.7250 both), dim-4 wins by $-0.105449$ $[-0.110025,-0.100968]$, 16.0%.

GPTQ's contribution is almost identical on both families: it removes **69.0%**
of MLP damage against 68.3% on the projections. And the §5c finding that finer
scalar groups buy nothing under compensation **replicates independently**:
`scalar3_g64` vs `scalar3` is $+0.001504$ $[-0.002309,+0.005386]$, an interval
**including zero**, for 6.4% more bytes.

The go/no-go condition for §6 was whether the codec ordering survives off the
projections. It does. But the absolute numbers change what §6 should expect.

### Per-type codebooks are a negative result

Fitting a separate codebook for `gate`/`up`/`down` is **worse**, by $+0.021186$
$[+0.017102,+0.025162]$, *and* costs 0.37% more bytes — dominated on both axes.
The motivating intuition was that `down_proj` reads the activated intermediate
and so needs its own book. The measurement says the loss of effective sample
size in each fit outweighs whatever distributional difference exists. One
shared codebook across all 72 tensors is the right choice, which is also the
cheaper one.

### The number that should change expectations for 27B

| | params | ΔNLL | per 1e9 params |
|---|---:|---:|---:|
| `in_proj_qkv` (§5c) | 113.2 M | $+0.077$ | 0.680 |
| MLP (§8) | 264.2 M | $+0.501$ | **1.895** |

MLP weights are **2.79x more damaging per parameter** at ternary rate. In
perplexity: the projections alone cost $+8.0\%$, the MLPs alone cost
$+65.0\%$, and if the two were additive a joint conversion would cost
$+78.2\%$ — at 0.8B, where MLP is only 35.1% of the model. At 27B the MLP share
is 63.5%, so the mix is worse, not better.

**The conclusion is two-sided and both sides matter.** The codec result
generalizes: shared-codebook dim-8 VQ beats learned scalar ternary on MLP
weights, at or below parity bytes, under a competitive recipe, with the same
storage-dominance property. But post-training quantization alone will not
produce a *good* ternary 27B. A 65% perplexity cost on the MLPs is not a
deployable model, and no amount of further codec work closes a gap that size —
§5c already showed error compensation is worth 3x what the codec is worth, and
compensation is already applied here.

What this implies for §6, in order of expected value:

1. **Mixed precision is now the obvious lever, not an afterthought.** The
   2.79x per-parameter asymmetry is a direct instruction about where to spend
   bits. A rate allocation across families, measured rather than assumed, is
   likely worth more than anything left in codec design.
2. **Quantization-aware training or distillation** is what the published
   ternary models almost certainly rely on; §7 should not compare our PTQ
   numbers against their trained ones as if they were the same procedure.
3. The §6 engineering conditions are unchanged and still unmet: no resident
   packed inference, no compressed KV cache, no measured peak memory or
   latency at 27B.

## 9. Mixed-precision allocation, and the failure of the sensitivity proxy (2026-09-21)

Section 8 measured that MLP weights are 2.79x more damaging per parameter than
the input projections, and concluded that rate allocation was the obvious
remaining lever. `run_mixed_alloc_v1.py` tests that, over all 90 target tensors
(`in_proj_qkv` + every MLP projection; 377,487,360 params, **50.2% of the
model**).

The question is posed so it can be answered fairly. Not "which family deserves
more bits" — that cannot be byte-matched, because the families differ in size —
but: **given a fixed number of extra bytes, which tensors should receive
them?** Every promotion arm spends exactly 5,662,224 extra bytes, which is
precisely what promoting all 18 projections costs, so `promote_proj` is one
candidate answer rather than the reference. Both rates are dimension-4 vector
codes (K=81 at 1.600 bits of index, K=243 at 2.000), so promotion changes the
rate and nothing else; mixing dimension-8 into the ladder would have confounded
rate with dimension.

Tensors are ranked without any extra model evaluations, by quantizing each at
both rates and scoring the summed activation-weighted error
$\operatorname{tr}(\Delta W H \Delta W^{\mathsf T})$, then promoting greedily by
error removed per extra byte. `promote_random` spends the identical budget on
randomly chosen tensors and is the control that decides whether the proxy is
worth anything.

| Arm | bits/wt | stored MB | promoted | ΔNLL vs BF16 |
|---|---:|---:|---:|---:|
| all_low (dim-4 K=81) | 1.7250 | 81.40 | — | $+0.637829$ |
| promote_proj | 1.8451 | 87.06 | 113.2 M | $+0.594650$ |
| promote_greedy_all | 1.8417 | 86.90 | 110.1 M | $+0.598860$ |
| promote_random | 1.8451 | 87.06 | 113.2 M | $+0.537616$ |
| **promote_greedy_mlp** | 1.8417 | 86.90 | 110.1 M | $\mathbf{+0.533546}$ |
| all_low_d8 (dim-8 K=6561) | 1.7272 | 81.50 | — | $+0.595745$ |
| all_high (everything K=243) | 2.1250 | 100.27 | 377.5 M | $+0.351119$ |

### The proxy is worse than random

The greedy ranking over all 90 tensors promoted **14 of 18 `in_proj_qkv`, 6 of
24 `gate_proj`, and none of the 48 `up_proj`/`down_proj`** — it sent the budget
almost entirely to the projections. Against the random control on identical
bytes it loses by $+0.061244$ $[+0.058083,+0.064344]$. `promote_proj` likewise
loses to random, by $+0.057034$. **A principled, measured, Hessian-derived
ranking performs significantly worse than choosing tensors at random.**

The mechanism is visible in numbers already reported. Per parameter, the
projections carry $5.00\times10^{-5}$ of activation-weighted error against the
MLPs' $8.74\times10^{-6}$ — 5.7x *more* — while causing 2.79x *less* NLL
damage. Across families the proxy is anti-correlated with what it is meant to
predict, by roughly a factor of 16 in the wrong direction.

Why: $\operatorname{tr}(\Delta W H \Delta W^{\mathsf T})$ measures error
injected at a layer's *output*. It carries no information about how that error
propagates to the loss, which is a backward quantity. This project has now
failed to rank layers twice by two different forward-side statistics — the
Fisher diagnostic found Spearman $+0.015$, and this proxy is worse than
chance. Those are consistent findings, not two accidents.

### Narrowing a claim from §5c

§5c reported that the activation-weighted error "is the first proxy in this
project to survive a setting where plain weight error reverses." That was
measured **within one tensor family, across recipes**, and in that setting it
holds. Used **across families** it fails outright. The §5c sentence should be
read with that scope, and the general claim is **withdrawn**.

### What does work: family-level allocation, from measured damage

`promote_greedy_mlp` — the budget confined to MLP tensors — is the best
promotion arm, beating `promote_proj` by $-0.061105$
$[-0.064430,-0.057804]$ and `all_low` by $-0.104284$ for 6.8% more bytes. But
the credit belongs to the *family* decision, which came from §8's measured
NLL-per-parameter, not from the ranking. Random already places about 70% of its
budget on MLP simply because MLP is 70% of the target parameters, and
`greedy_mlp` beats it by only $-0.004070$ $[-0.006991,-0.001160]$.

Interpolating the family effect linearly between `promote_proj` (0% of budget
to MLP) and `promote_random` (~70%), an allocation at 100% should land near
$+0.513$. The observed $+0.533546$ is **$0.020$ worse**, so the within-family
ranking appears mildly harmful too, consistent with its sign across families.
The honest summary: *choose the family by measured damage per parameter, then
distribute within it uniformly.*

### Dimension and allocation compose, sub-additively

The first pass tested composition with `greedy_all`, which turned out to be the
inferior allocation; a rerun (`results/mixed_alloc_v1b`, in which both original
arms reproduce to all printed digits) adds the allocation that won:

| Arm | bits/wt | ΔNLL vs BF16 |
|---|---:|---:|
| all_low (dim-4 K=81) | 1.7250 | $+0.637829$ |
| all_low_d8 (dim-8 K=6561) | 1.7272 | $+0.595745$ |
| promote_greedy_mlp (dim-4 low) | 1.8417 | $+0.533546$ |
| **greedy_mlp_d8_low** | 1.8439 | $\mathbf{+0.514512}$ |

Against `all_low_d8` on identical rates elsewhere, MLP-first promotion is worth
$-0.081233$ $[-0.084066,-0.078503]$. Taken from `all_low`, the dimension change
alone is worth $-0.042084$, the allocation alone $-0.104283$, and the two
together $-0.123317$ — **84% of the additive sum**. They reinforce rather than
overlap, but not fully: once the MLP tensors are promoted to dimension-4 K=243,
the dimension-8 gain applies only to what remains, and it halves.

The best configuration measured in this project is therefore dimension-8
ternary on unpromoted tensors with the MLP family promoted to 2.000 bits:
$+0.514512$ at 1.8439 bits/weight over 50.2% of the model.

### The levers, ranked by what they actually buy

| Lever | damage cut | byte cost |
|---|---:|---:|
| GPTQ error compensation (§5c, §8) | 68–69% | 0% |
| Rate 1.600 → 2.000 bits of index | 45.0% | $+23.2\%$ |
| Family allocation (MLP-first) | 16.3% | $+6.8\%$ |
| Codec dimension, dim-4 → dim-8 | 6.6% | $+0.1\%$ |
| Per-tensor proxy ranking | **negative** | — |

The rate row is the one that should change the 27B plan most. A 0.4-bit
increase in index rate removes 45% of the damage — ternary sits on a steep part
of the rate–distortion curve, and the distance from 1.6 to 2.0 bits buys
considerably more than every codec refinement in this project combined. The
codec contribution is real, replicated and essentially free in bytes, but it is
a 6.6% effect sitting on top of a 68% one.

## 10. The rate–distortion curve above ternary, and a premise worth re-examining (2026-09-22)

Section 9's ladder stopped at 2.000 bits because that was all the allocation
study needed, leaving the region between 2 and 4 bits unmeasured — which is
where a *usable* model is most likely to live. `run_rate_curve_v1.py` sweeps
uniform rate over the same 90 tensors (50.2% of the model) with everything else
fixed: dimension-4 codes, rotation, FP16 group scales, GPTQ compensation,
storage measured from packed bytes, weights decoded back from them.

Rungs come from the packing itself. With $c$ codes per `uint64` at dimension
$d$ the rate is exactly $64/(cd)$, and the best code at that rate takes the
largest $K$ with $K^{c}<2^{64}$. Every rung's packed rate matches
$\log_2 K/4$ to three decimals, so the container wastes essentially nothing and
the curve is a property of the codec.

| K | index bpw | total bpw | stored MB | weight MSE | ΔNLL vs BF16 | perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 81 | 1.600 | 1.7250 | 81.40 | 4.589e-05 | $+0.637829$ | $+89.2\%$ |
| 84 | 1.600 | 1.7250 | 81.40 | 4.511e-05 | $+0.629249$ | $+87.6\%$ |
| 255 | 2.000 | 2.1250 | 100.27 | 2.700e-05 | $+0.340476$ | $+40.6\%$ |
| 565 | 2.286 | 2.4108 | 113.76 | 1.850e-05 | $+0.230274$ | $+25.9\%$ |
| 1625 | 2.667 | 2.7920 | 131.74 | 1.111e-05 | $+0.121974$ | $+13.0\%$ |
| 7131 | 3.200 | 3.3262 | 156.95 | 5.533e-06 | $+0.060716$ | $+6.3\%$ |
| **65535** | **4.000** | 4.1361 | 195.17 | 1.522e-06 | $\mathbf{+0.018572}$ | $+1.9\%$ |

### No knee is visible over the measured interval

Damage halves every **0.451, 0.506, 0.416, 0.530 and 0.468 bits** across the
five intervals — a mean of 0.474 with no systematic drift over a $2.40\times$
range of stored bytes (81.40 to 195.17 MB) and 2.4 bits of rate. Over this
range the decline is well described by

$$\Delta\text{NLL}\;\approx\;0.629\cdot 2^{-(r-1.600)/0.474}.$$

Within the measured interval there is no visible knee, so on this checkpoint at
this coverage the rate choice behaves like a budget decision at a roughly
constant exchange rate. Scope: six rates on one model, one 50.2% target set and
one corpus. That supports "approximately exponential over this interval" and
nothing stronger; it does not establish that no knee exists outside it, at other
coverages or on other checkpoints, and an earlier draft overreached by saying no
experiment could find one.

**The four-bit point, measured on these tensors.** The review noted that a
four-bit rung was missing and that the historical QKV-only measurement could not
fill the gap. It is now measured: $+0.018572$ NLL, $+1.87\%$ perplexity, at
4.136 total bits/weight over 50.2% of parameters. The historical figure is
$+0.008393$ ($+0.84\%$ perplexity) over 15.1% of parameters — a different target
set, roughly a third of the coverage, and not a substitute for this point. The
earlier omission was also not the judgement call the first draft claimed: a
fixed 200,000-point chunk in the nearest-neighbour search asks for 52 GB of
distance buffer at $K=65535$, and the rung became affordable only once that
buffer was sized against the codebook. The reduction triggers above 6 GB, so
all six smaller rungs reproduce **bit-identically** ($\Delta = 0$ to nine
decimals).

A smaller free observation: K=84 beats K=81 by $-0.008581$ at *identical*
1.600 bits/weight, because 84 is the largest codebook the 10-codes-per-word
packing admits. Section 9's K=81 left that on the table.

### A premise worth testing — but not yet tested

The 27B target has always been a 24 GiB card, and ternary was adopted to reach
it. That motivation deserves scrutiny, but the arithmetic below is a **storage
estimate, not a measured deployment result**, and an earlier draft of this
section presented it as though it were the latter.

Using the safetensors census (27.78 B parameters including the vision tower;
the runner's text-model census is 26.896 B, and the two are not reconciled
here), with embeddings retained at FP16 and *ignoring* scale, codebook and
header overhead as well as any retained modules:

| rate | estimated weight bytes | under 24 GiB? |
|---:|---:|:--|
| 1.600 | 10.1 GB (9.4 GiB) | yes |
| 2.000 | 11.4 GB (10.6 GiB) | yes |
| 2.667 | 13.5 GB (12.6 GiB) | yes |
| 3.200 | 15.2 GB (14.2 GiB) | yes |
| 4.000 | 17.7 GB (16.5 GiB) | yes |

Three things this does **not** establish. First, packed weights fitting in VRAM
is not execution fitting: the KV cache, DeltaNet recurrent state, activations,
kernel workspaces and decoding buffers are all unmeasured here. Second, the
quality of a full-coverage conversion at any of these rates has never been
measured — §8 measured MLP damage only at ternary on a 0.8B model, and §11
measured 27B damage only on the 9.36% QKV subset. Third, and most importantly:

> **Withdrawn.** An earlier draft asserted that four-bit weights give a
> deployable 27B at 0.24% damage. The +0.008393 NLL figure behind that number
> is a **0.8B, projection-subset** measurement (§2). It is 0.24% *of baseline
> NLL* and 0.84% *relative perplexity* — two different quantities that the
> draft conflated — and it is not a 27B result, not a full-coverage result,
> and not a runtime result.

What survives is narrower and still worth saying: the storage arithmetic gives
no reason to believe ternary is *forced* by a 24 GiB budget, and a matched
four-bit measurement on the §9 90-tensor set would be cheap and would settle
far more than another low-rate variant. Until that is run, deployment remains
motivation and estimate, not result.

That does not retract the codec result. Dimension-8 vector quantization beating
learned scalar ternary at parity bytes is real, replicated on two tensor
families, and robust to a competitive recipe — it is a genuine finding about
low-rate codecs. But it should be presented as a finding about codecs at
extreme rates, not as the route to a deployable 27B model on this hardware. For
that goal the evidence now points somewhere much duller: quantize at 3 to 4
bits, where damage is 0.06 NLL or less, and spend the engineering effort on the
packed inference kernel that §6 still lacks.

## 11. Scale transfer: frozen confirmation on Qwen3.8-27B (2026-09-22)

`run_qwen38_confirm_v2.py` applies the §5c recipe to all 48
`linear_attn.in_proj_qkv` tensors of Qwen3.8-27B (10240x5120, 2,516,582,400
params, **9.36%** of the runner's 26.896 B text-model census) and evaluates the
**complete** 261,284-target validation stream. Calibration is 65,536 training
tokens, matched to the 0.8B recipe. Contrasts were prespecified before the run
and recorded in `manifest.json`: VQ8 vs scalar g128 primary; storage dominance
and VQ8 vs VQ4 secondary.

Protocol repairs relative to the pilot: the ragged final block is no longer
dropped, per-block losses and identifiers are saved, the encoded model is
written to disk and **re-decoded from the reloaded files** (exact for all four
arms), and the manifest records arguments, source and checkpoint hashes, device
map and GPU budget.

| Arm | bits/wt | payload B | artifact file B | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|---:|---:|:--|
| scalar3_rot_gptq | 1.7250 | 542,638,086 | 542,669,401 | $+0.010958$ | $[+0.009366,+0.012561]$ |
| scalar3_g64_rot_gptq | 1.8500 | 581,959,686 | 581,991,000 | $+0.011673$ | $[+0.010078,+0.013264]$ |
| vq4_rot_gptq | 1.7250 | 542,638,728 | 542,670,686 | $+0.007871$ | $[+0.006436,+0.009331]$ |
| **vq8_rot_gptq** | 1.7253 | 542,743,056 | 542,879,296 | $\mathbf{+0.003837}$ | $[+0.002420,+0.005250]$ |

BF16 NLL 2.691117.

### All three prespecified contrasts resolve

| Contrast | ΔNLL | 95% CI | payload ratio |
|---|---:|:--|---:|
| **primary** VQ8 vs scalar g128 | $-0.007121$ | $[-0.008513,-0.005725]$ | 1.00019 |
| VQ8 vs scalar g64 (*more* bytes) | $-0.007836$ | $[-0.009247,-0.006356]$ | 0.93261 |
| VQ8 vs VQ4 | $-0.004034$ | $[-0.005334,-0.002715]$ | 1.00019 |
| g64 vs g128 (context) | $+0.000714$ | $[-0.000375,+0.001828]$ | 1.07246 |

The primary contrast holds. **Storage dominance, inconclusive in the pilot, now
resolves**: VQ8 beats the scalar control while storing 6.7% *fewer* payload
bytes, interval excluding zero. And **dimension 8 earns its place**: VQ8 beats
VQ4 by $-0.004034$, more than half the size of VQ8's entire advantage over
scalar. The pilot had suggested the opposite, with VQ4 capturing 95.4% of the
gain and VQ8 ahead by only $0.000181$.

Finer scalar groups remain unsupported: g64 spends 7.25% more bytes for
$+0.000714$ $[-0.000375,+0.001828]$, an interval including zero whose point
estimate is now the *wrong sign* for g64. This is still not an equivalence
result, but the earlier "21.7% benefit" reading from the pilot does not survive.

### The pilot was misleading, and not because of sampling

The 512-block pilot gave $-0.003957$ (primary) and $-0.001736$ (dominance). The
frozen run gives $-0.007121$ and $-0.007836$. Two checks separate the possible
explanations.

**The corpus is homogeneous.** Splitting the frozen run into the pilot's first
512 blocks and the 1,530 blocks never previously evaluated
(`split_prefix_check.py`) gives consistent intervals on every contrast:

| Contrast | prefix (65,536) | remainder (195,748) |
|---|---:|---:|
| VQ8 vs scalar | $-0.005912$ | $-0.007526$ |
| VQ8 vs g64 | $-0.007681$ | $-0.007887$ |
| VQ8 vs VQ4 | $-0.003309$ | $-0.004277$ |

So the prefix is not unrepresentative, and extending coverage is not what moved
the numbers.

**The evaluation path is identical.** BF16 NLL restricted to the pilot's blocks
is 2.579685 in the frozen run and 2.579685 in the pilot — equal to all printed
digits. The two runs therefore differ only in the weights they produced.

**What differs is calibration.** The pilot used 32,768 calibration tokens; the
frozen run uses 65,536. On *identical evaluation blocks* the dominance contrast
moves from $-0.001736$ $[-0.004444,+0.000923]$ to $-0.007681$
$[-0.010467,-0.004984]$ — intervals that do not overlap.

Doubling calibration data changes $H$, hence the Cholesky factor, hence the
compensation and every arm's chosen codes. A comparison of two calibration
sizes cannot separate a size effect from calibration-sample variability, and
the 64-block set is a prefix of the 128-block set rather than an independent
draw. The defensible conclusion is narrower and more uncomfortable: **at this
rate these contrasts are sensitive to the calibration configuration, and every
interval in this document is conditional on one fitted quantizer.** Reported
intervals capture text-sampling variability only; calibration and codebook-seed
variability are not in them. Repeating the frozen run across calibration draws
and codebook seeds is the obvious missing measurement.

### What this says about the programme

Converting 9.36% of this checkpoint to ternary costs $+1.10\%$ perplexity with
the scalar code and $+0.38\%$ with VQ8. The codec choice moves about 0.7
percentage points of perplexity on this subset. Whether that matters for a
deployed model is still unmeasured: no full-coverage conversion at any rate has
been evaluated at 27B for quality or runtime footprint, and there is no packed
inference kernel.

### An exactness failure observed once and not reproduced

An earlier draft of this document described a "budget-sensitive defect" and
speculated that memory pressure changed arithmetic. That outran the evidence.

**What was observed.** One run of the v1 confirmation script at a 17 GiB GPU
weight budget raised the per-tensor bit-exactness check with a
decoded-versus-reference difference of $0.036$ — far too large to be rounding.
Runs at 15 GiB were exact.

**What reproduction found.** Seven further runs at 17 GiB, across two script
versions and in fresh processes, all terminated with an identical CUDA
out-of-memory error (200 MiB requested, 196 MiB free) *before* reaching the
state in question: four with the strict check (two of them using a
reduced-footprint host-side comparison specifically to survive longer) and
three re-running the exact original v1 invocation. Runs at 15 and 16 GiB are
exact. **The corruption was never reproduced and no failing tensor was
captured**, so the intended localization — comparing unpacked indices, unpacked
scales, an independent CPU reconstruction, the GPU decode and the quantizer
reference — could not be run at all.

**What can be said.** At a 17 GiB weight budget this configuration sits within
about 200 MiB of the card's capacity and does not complete there. One
observation of a wrong value stands against seven of a clean failure. That is
consistent with a transient fault, with machine state differing between the two
occasions, or with a genuine narrow-regime bug, and the evidence does not
distinguish them. The honest status is **one unexplained event, not
reproduced**, and the earlier framing is withdrawn.

**Why it does not affect the reported results.** Every reported run used 15 GiB
and every tensor in every reported arm passed the strict per-tensor check —
explicit finiteness and exact equality, evaluated before the tensor is
installed into the model, with the failing case dumped on violation. The frozen
confirmation additionally writes each arm's artifact to disk and re-decodes it
from the reloaded files, exact in all four arms. The check is cheap, it is the
only reason the anomaly was noticed, and it should be kept rather than relaxed.

## 12. Refit variability: what the published intervals leave out (2026-09-22)

Every interval in this document is a paired block bootstrap over evaluation
text, conditional on one fitted quantizer. §11 showed that conditioning is not
harmless — doubling calibration tokens moved a contrast across non-overlapping
intervals on identical evaluation blocks — but one nested pair on one
checkpoint could not separate a size effect from draw-to-draw variability.
`run_calib_variability_v1.py` separates them on the 0.8B QKV projections,
holding the recipe fixed and varying three things independently: three disjoint
calibration draws at 65,536 tokens, three k-means seeds at fixed calibration,
and calibration size from 16k to 131k tokens.

Eight refits, complete validation stream, four arms each.

| Contrast | median half-width (text bootstrap) | range over 3 draws | over 3 seeds | over 4 sizes |
|---|---:|---:|---:|---:|
| VQ8 vs scalar | 0.002416 | 0.004255 | 0.005057 | 0.004474 |
| VQ8 vs scalar g64 | 0.002416 | 0.007078 | 0.005057 | 0.007110 |
| VQ8 vs VQ4 | 0.002324 | 0.003893 | 0.006350 | 0.011189 |
| g64 vs g128 | 0.001983 | 0.005548 | 0.000000 | 0.007441 |

### Two statistics, reported side by side rather than divided

An earlier draft divided these columns and called the quotient an uncertainty
inflation factor, claiming the intervals "understate uncertainty two to five
fold". **That is withdrawn.** A range over three point estimates and a
bootstrap confidence-interval half-width are different statistics: one is the
observed spread of a handful of refits, the other a quantile interval for a
single estimate. Their ratio is not an inflation factor and carries no coverage
interpretation. The two belong beside each other:

* **conditional text-bootstrap intervals** — what every table in this document
  reports, describing sampling of the evaluation text given one fitted
  quantizer;
* **observed effects across refits** — the ranges above.

The calibration-size column is different in kind again. Going from 16k to 131k
tokens changes the experimental condition; it is not a further draw from the
same distribution, and the earlier draft's grouping of it with draw and seed as
though all three were "refit noise" was a second error. It is a condition
effect, and the monotone trend below is what a condition effect looks like.

What the columns do support, without the arithmetic: **the effects observed
across refits are of the same order as, and in several cases larger than, the
text-bootstrap half-width.** A single six-decimal margin from one refit is
quoted more precisely than the procedure warrants.

The single zero remains a harness check: `g64 vs g128` compares two scalar arms
fitted by a deterministic Lloyd pass, so a k-means seed cannot move it, and the
measured range is exactly $0.000000$.

### What survives, and what does not

| Contrast | range over 8 refits | sign |
|---|---|---|
| VQ8 vs scalar | $-0.037809$ to $-0.031061$ | negative 8/8 |
| VQ8 vs scalar g64 | $-0.040478$ to $-0.027224$ | negative 8/8 |
| VQ8 vs VQ4 | $-0.018548$ to $-0.005101$ | negative 8/8 |
| g64 vs g128 | $-0.004638$ to $+0.005035$ | **flips, 5/8 negative** |

The three headline claims are **directionally robust**: the vector code beats
scalar ternary, beats the higher-storage scalar control, and beats
dimension-4, in all eight refits. Their *magnitudes* are not — VQ8 vs VQ4 moves
by a factor of 3.6 across refits — so effect sizes should be quoted as ranges
over refits, not as a single interval.

The `g64 vs g128` contrast **changes sign** across refits. That retrospectively
explains every inconclusive reading of it in §§5c, 8 and 11: it was never a
small effect measured imprecisely, it is a refit-dependent one. Any statement
about finer scalar groups, in either direction, is unsupported.

### Calibration size has a direction

Along the nested size axis, VQ8's advantage over scalar grows monotonically
with calibration: $-0.033335$ (16k), $-0.035775$ (32k), $-0.036118$ (65k),
$-0.037809$ (131k). VQ8 vs VQ4 trends the same way, less smoothly. This
reproduces the direction of the 27B observation in §11 — more calibration, a
larger vector-code advantage — on an independent checkpoint, which is weak
evidence that the 27B movement was a size effect rather than a draw effect.
Four nested sizes on one draw cannot establish it.

### Consequence for the paper

Reported intervals stay, labelled as text-sampling variability conditional on
one fitted quantizer, with refit ranges beside them rather than folded into
them. The honest headline is a direction and a range: on this target set,
eight-dimensional VQ reduces damage relative to learned scalar ternary by
**0.031 to 0.038 NLL across eight refits**, always in the same direction. A
comparable measurement at 27B was not run, at roughly eight times the cost of
the frozen confirmation, so the 27B intervals carry the same conditioning with
their refit variability unmeasured.

### Why finer scalar groups have no stable sign: a hypothesis

The `g64 vs g128` contrast changes sign across refits and, in §13, across
evaluation domains. The leading explanation is a mismatch between the local
quantization objective and the final language-model loss, and it is specific to
how the control is built rather than mysterious.

Moving from g128 to g64 does not enlarge a solution space that contains the
g128 solution. Group scales are *derived* (the absolute maximum of each group),
not free parameters, so a g64 configuration cannot in general represent a g128
one; and the shared three-level codebook is refit in g64-normalized space, so
the reconstruction alphabet changes too. GPTQ then makes greedy sequential
decisions against calibration activations on top of both changes. Spending more
bytes on scale metadata therefore carries no guarantee of lower reconstruction
error, let alone lower held-out loss.

Calibration-dependent and domain-dependent sign changes are *consistent* with
this account. They do not prove it. A direct test would hold the codebook fixed
across group sizes and measure reconstruction error and held-out loss
separately, which this project has not done.

## 13. Generalization: fixed quantizer, varied evaluation (2026-09-22)

Every contrast so far was evaluated on wikitext-2 validation in 128-token
blocks with state reset per block. That text has guided many decisions and is
no longer held out, and a 128-token reset cannot speak to how a quantized
recurrent model behaves over long contexts. §12 varied the fit and held the
evaluation fixed; `run_generalization_v1.py` does the reverse — **one
calibration draw, one seed, one set of codes**, with only the evaluation
changing — so the two sensitivities are separated rather than confounded.

| Evaluation | corpus | context | blocks | targets | BF16 NLL |
|---|---|---:|---:|---:|---:|
| wt2_val_128 | wikitext-2 val | 128 | 2,042 | 261,284 | 3.436710 |
| wt2_test_128 | wikitext-2 **test** | 128 | 2,048 | 262,143 | 3.394295 |
| tiny_128 | **TinyStories** | 128 | 2,048 | 262,143 | 2.314432 |
| wt2_test_512 | wikitext-2 test | 512 | 512 | 262,143 | 2.870202 |
| wt2_test_2048 | wikitext-2 test | 2048 | 128 | 262,143 | 2.567222 |

Calibration is always wikitext-2 train, which is the realistic setting and what
makes the TinyStories column informative. `wt2_val_128` reproduces §5c to six
decimals on all four contrasts, so it is a harness check, not evidence.

### The ordering holds everywhere tested

All three primary contrasts exclude zero on **all five** evaluation sets:

| Contrast | wikitext range | TinyStories |
|---|---|---:|
| VQ8 vs scalar | $-0.038806$ to $-0.031832$ | $-0.043775$ |
| VQ8 vs scalar g64 | $-0.039543$ to $-0.031779$ | $-0.054717$ |
| VQ8 vs VQ4 | $-0.015220$ to $-0.011451$ | $-0.015677$ |

Across the four wikitext sets the VQ8-vs-scalar contrast varies by $0.006974$,
about 2.7x a typical bootstrap half-width — real variation, though these are
different texts with different baselines rather than repeated measurements of
one quantity, so this is not an error term in the sense §12's refit spread is.

### Context extension to 2048 tokens: no reversal, but the advantage shrinks

This was the worry that motivated the check, and the ordering survives: VQ8
beats scalar at every context length, excluding zero at 2048. But the earlier
draft's "does not degrade" was too strong. VQ8's advantage falls from
$0.038806$ at 128 tokens to $0.031832$ at 2048 — about **18% smaller** — while
its own damage rises slightly, $0.082309$ to $0.084152$. Aggregate damage is
roughly flat (scalar ternary $0.121115 \to 0.115984$).

The defensible statement is narrow: **no ordering reversal and no large
increase in aggregate quantization damage was observed through 2048 tokens.**
That is useful evidence and it is not evidence of absence of recurrent-state
drift. Aggregate NLL over a long block averages over positions and would hide a
drift that grows with position. Position-resolved loss, or direct comparison of
recurrent state between the BF16 and quantized models, would address the
mechanism; neither was run.

A correction to the earlier draft: it claimed that because BF16 loss falls with
context, the same absolute damage is "a larger relative cost at 2048". That is
wrong. The relative perplexity penalty is $\exp(\Delta\mathrm{NLL}) - 1$ and
depends on $\Delta\mathrm{NLL}$ alone — $12.88\%$ at 128 tokens and $12.30\%$
at 2048. Only when expressed as a percentage *of baseline NLL* does a lower
baseline inflate it, and that is a different quantity.

### Domain shift: absolute advantage grows, proportional advantage shrinks

On TinyStories every arm is hurt far more than on wikitext. Both framings are
legitimate measurements and they point in opposite directions, so both belong
in the table:

| | WikiText test | TinyStories |
|---|---:|---:|
| scalar damage | 0.121115 | 0.182354 |
| VQ8 damage | 0.082309 | 0.138579 |
| VQ8 absolute advantage | 0.038806 | **0.043775** |
| fraction of scalar damage removed | **32.0%** | 24.0% |

The earlier draft reported only the first of these and concluded that the
vector code "absorbs domain shift better". **That wording is withdrawn.** The
absolute advantage grows; the proportional advantage shrinks from 32.0% to
24.0%.

A second overreach: higher damage on TinyStories does not establish that
*calibration mismatch* caused it. TinyStories may simply be more sensitive to
these weight perturbations for reasons unrelated to where the Hessians came
from. Isolating mismatch needs two calibration domains crossed with two
evaluation domains, everything else fixed — four cells, all cheap at 0.8B, and
not run here. Until then the observation is that damage is higher
out-of-domain, with the cause unattributed.

### Scope

One model, one calibration corpus, a single fitted quantizer, and five
evaluation configurations that are **two corpora and three context lengths, not
five independent replications** — the three wikitext-test rows share their text
and differ only in blocking. It does not establish that the ordering holds on
other model families, other calibration domains, or at 27B, where neither the
domain nor the context axis was measured.

## 14. Three mechanism tests (2026-09-22)

§§12–13 reported three things without causes. Each is cheap to resolve on the
0.8B model, and each resolution changes what the earlier section should say.

### A. Domain damage tracks calibration mismatch

§13 found higher damage on TinyStories under a wikitext-calibrated quantizer
and declined to attribute it, since intrinsic sensitivity was an equally good
explanation. Crossing two calibration domains with two evaluation domains
settles it. ΔNLL against BF16:

| calibration \\ evaluation | wikitext-2 test | TinyStories |
|---|---:|---:|
| wikitext, scalar | **0.121115** | 0.183486 |
| wikitext, VQ8 | **0.082309** | 0.140296 |
| TinyStories, scalar | 0.288304 | **0.036675** |
| TinyStories, VQ8 | 0.199492 | **0.029201** |

Damage is minimized on the diagonal in every cell, and the off-diagonal
penalties are large: calibrating on TinyStories and evaluating on wikitext
costs $0.288304$ against $0.121115$ matched. **Calibration matching accounts
for the direction of §13's effect**, which was the question.

A later review was right that the original wording went further than this. Two
sentences are withdrawn. "Intrinsic sensitivity is refuted" claimed too much:
what these four cells rule out is the specific alternative reading of §13 —
that TinyStories is harder for this recipe *whatever* it is fitted on — since
matched TinyStories costs $0.036675$ against matched wikitext's $0.121115$, a
factor of 3.3 the other way. That is a statement about two corpora, one model
and one recipe, not a general claim that corpora do not differ in sensitivity.
And "the high TinyStories figure was entirely an artifact" asserts a
decomposition that was never performed: the design separates matched from
mismatched, it does not partition the mismatched number into a mismatch part
and a sensitivity part.

VQ8 wins in all four cells, removing 20.4% to 32.0% of scalar damage. Its
proportional advantage is largest where damage is largest (32.0% and 30.8% in
the two worst cells, 20.4% in the mildest), so the vector code earns most in
the regimes that hurt most.

### B. Damage depends on position within the block, and the dependence is specific to the vector code

§13 reported flat aggregate damage through 2048 tokens and correctly refused to
call that evidence against recurrent-state drift. Resolving loss by position
inside the block shows why the refusal was right.

Mean loss over 127 blocks of 2048 tokens, wikitext-2 test:

| positions | BF16 | scalar damage | VQ8 damage | VQ8 advantage |
|---|---:|---:|---:|---:|
| 0–256 | 3.0776 | 0.117398 | 0.079553 | 0.037845 |
| 256–512 | 2.6192 | 0.115513 | 0.082541 | 0.032973 |
| 512–1024 | 2.5238 | 0.116048 | 0.082780 | 0.033268 |
| 1024–1536 | 2.4587 | 0.113739 | 0.084449 | 0.029290 |
| 1536–2048 | 2.4480 | 0.118204 | 0.088811 | 0.029393 |

Scalar damage is flat with position. VQ8 damage rises from $0.079553$ to
$0.088811$ — about 12% — so VQ8's advantage erodes from $0.037845$ to
$0.029393$, roughly 22%, across the block. The aggregate hid this because it
averages the two ends.

> **Superseded by §18.** These are means over 127 blocks with no measure of
> spread, and the original wording ("rises monotonically") does not survive the
> rerun that retains per-block losses: with intervals, VQ8's damage increase
> across the block does *not* exclude zero, and it is not monotone over eight
> buckets. What survives is the erosion of VQ8's *advantage*, which does exclude
> zero. See §18 for the corrected reading.

**This is position-dependent damage, not identified drift.** The original
heading called it recurrent-state drift, which the design cannot support: later
positions hold different tokens as well as more accumulated state, so a
position effect and a content effect are confounded here, and rising loss with
position is consistent with drift without identifying it. Separating them needs
a direct state comparison — the same tokens evaluated with and without a state
reset — which was not run. §16 of this document retains the weaker claim.

The ordering is preserved at every position, so the headline claim stands. But
the earlier statement that context extension leaves the result undisturbed is
now too strong: there *is* a position-dependent effect, it is specific to the
vector code, and it points the same way as the 18% aggregate shrinkage from 128
to 2048 tokens. Naively extrapolating to much longer contexts is unwarranted on
five buckets.

Limitation, now addressed: per-block per-position losses were not stored, so
these are means over 127 blocks with no measure of block-to-block spread and no
intervals. §18 reruns this with the full matrix retained, so the bucket
contrasts get paired intervals over blocks and the trend can be tested rather
than read off.

### C. Weight MSE is anti-correlated with loss inside the scalar family

The hypothesis for the unstable g64 contrast was a mismatch between the local
quantization objective and the language-model loss, with two separable causes:
derived scales and a refitted codebook. Holding the codebook fixed across group
sizes separates them, and the result is stronger than the hypothesis.

| variant | weight MSE | ΔNLL |
|---|---:|---:|
| g64, g128's codebook | **9.71810e-05** (best) | **0.135133** (worst) |
| g64, own codebook | 9.86269e-05 | 0.121852 |
| g128, own codebook | 9.88856e-05 | 0.121115 |
| g128, g64's codebook | **1.04436e-04** (worst) | **0.115284** (best) |

The weight-MSE ordering is the **exact reverse** of the loss ordering across
all four variants — Spearman $-1.0$. Lower reconstruction error means higher
loss, monotonically, within this family at fixed rate.

**The original reading of these contrasts was wrong, and is withdrawn.** It
said: "changing the group size alone moves nothing measurable ($+0.000737$);
changing which codebook is used moves 0.013 to 0.014 — roughly twenty times as
much." The $+0.000737$ contrast is `g64 own book` against `g128 own book`, and
each arm fits its codebook in its own normalized space, so that comparison
changes grouping *and* codebook together. It is a diagonal of the 2x2, not a
main effect, and its near-zero value is a cancellation rather than an absence.

Read as the factorial it is:

| contrast | holds fixed | ΔNLL | 95% CI |
|---|---|---:|:--|
| grouping, g128 book | codebook | $+0.014019$ | $[+0.012133,+0.015908]$ |
| grouping, g64 book | codebook | $+0.006568$ | *not computed in this run* |
| codebook, g128 grouping | grouping | $-0.005831$ | $[-0.007809,-0.003792]$ |
| codebook, g64 grouping | grouping | $-0.013281$ | $[-0.015222,-0.011297]$ |
| *diagonal (both change)* | nothing | $+0.000737$ | $[-0.001213,+0.002710]$ |

Main effect of grouping $+0.010293$, main effect of moving to the g64-fitted
codebook $-0.009556$, interaction $+0.007451$. The two factors are of the
**same order**, they point in opposite directions, and they interact.

So the corrected statement is: moving from g128 to g64 at a fixed alphabet
*hurts* — by $0.014$ or $0.007$ depending on which alphabet is held — and
refitting the codebook in the new normalized space recovers most of that
damage. Grouping is not irrelevant; the near-zero end-to-end contrast is the
sum of a harmful grouping change and a compensating codebook refit. What
survives from the original claim is the part that matters for §12: the
end-to-end g64 contrast is unstable because it is a small difference between
two larger opposed effects, and weight MSE mispredicts every one of them.

The fourth contrast never had an interval because it was never computed. §18
reruns the 2x2 with all six pairwise contrasts and paired intervals.

One tempting number deserves a warning. The best cell, g128 quantization with
the codebook fitted in g64-normalized space, beats the standard configuration
by $-0.005831$ with an interval excluding zero. But §12 measured refit-to-refit
ranges of about $0.005$ for scalar-family contrasts, so a single refit cannot
establish this as a real recipe improvement. It is a lead, not a result, and
the honest test is the §12 protocol applied to these four variants.

### What these change upstream

* §13's refusal to attribute domain damage is replaced by an attribution:
  calibration mismatch, with intrinsic sensitivity refuted.
* §13's "no large increase in aggregate damage through 2048 tokens" stands, but
  "does not degrade" must not be read into it: VQ8's advantage erodes with
  position within the block.
* §12's hypothesis is upheld in a stronger form — the local objective is not
  merely imperfectly aligned with loss here, it is inverted — and relocated
  from the scale derivation to the codebook.

## 15. The 27B axes that were stated as limitations (2026-09-22)

§11 measured one fitted quantizer, on one corpus, at one context length, and
listed the rest as unmeasured. `run_qwen38_extend_v1.py` measures them. Both
parts use a 512-block prefix rather than the complete stream — justified by
§11's prefix/remainder agreement, and paid for in wider intervals.

### D. Domain and context transfer at 27B

| Evaluation | BF16 | scalar | VQ8 | VQ8 vs scalar | 95% CI |
|---|---:|---:|---:|---:|:--|
| wikitext-2 test, 128 | 2.626645 | 0.008636 | 0.002581 | $-0.006056$ | $[-0.008697,-0.003446]$ |
| TinyStories, 128 | 1.962926 | 0.022162 | 0.013393 | $-0.008768$ | $[-0.010684,-0.006899]$ |
| wikitext-2 test, 2048 | 1.932252 | 0.011318 | 0.005776 | $-0.005542$ | $[-0.010030,-0.000981]$ |

The ordering transfers on both axes: all three intervals exclude zero, the
second corpus included. The TinyStories column is again the more damaged one
under wikitext calibration ($0.022162$ against $0.008636$), matching the 0.8B
pattern that §14 attributed to calibration mismatch — though the 2x2 that
established the attribution was not repeated here.

### E. Refit variability at 27B, and why it matters more than at 0.8B

Three refits, same 512-block evaluation, varying one calibration draw and one
codebook seed:

| Refit | VQ8 vs scalar | 95% CI | |
|---|---:|:--|:--|
| draw0, seed0 | $-0.006056$ | $[-0.008697,-0.003446]$ | excludes 0 |
| draw1, seed0 | $-0.004014$ | $[-0.006576,-0.001267]$ | excludes 0 |
| draw0, seed1 | $-0.001029$ | $[-0.003777,+0.001767]$ | **includes 0** |

The point estimate varies six-fold across three refits, and **one of the three
does not distinguish the vector code from scalar ternary at all.**

The cross-scale comparison is the part worth carrying into the paper. Changing
only the codebook seed moves the contrast by $0.005057$ at 0.8B and $0.005026$
at 27B — the same absolute amount to three decimal places. But the effect being
measured is roughly six times smaller at 27B ($\approx 0.006$ against
$\approx 0.036$). **Refit variability does not shrink with scale; the effect
does.** At 0.8B that variability is a modest fraction of the result; at 27B, on
this 9.36% target subset, it is comparable to the whole of it.

### Consequence for the 27B claim

§11's frozen full-stream margin of $-0.007121$ $[-0.008513,-0.005725]$ is
correct as measured, and its interval is genuinely tight because the evaluation
is large. But it is **one refit**, and should be read as one draw from a spread
that these three refits bound at $-0.001029$ to $-0.006056$ on a smaller
evaluation. The defensible 27B statement is therefore:

> On the 48 DeltaNet QKV projections of Qwen3.8-27B, the eight-dimensional code
> reduced damage relative to learned scalar ternary in every configuration
> measured — two corpora, two context lengths, three refits — but the magnitude
> varies several-fold with the fitted quantizer, and in one refit of three the
> advantage was not separable from zero at this evaluation size.

Three refits bound the effect rather than characterize it; the eight-refit
protocol of §12 at 27B would cost roughly twenty hours and was not run.

## 16. Eight refits at 27B: the effect is smaller than the refit spread (2026-09-23)

§15 bounded 27B refit variability with three refits and said so. Eight refits —
five calibration draws at seed 0, plus three further seeds at draw 0 —
characterize it, and the characterization is unfavourable.

| Refit | VQ8 vs scalar | 95% CI | individually significant |
|---|---:|:--|:--|
| draw0, seed0 | $-0.006281$ | $[-0.008923,-0.003681]$ | yes |
| draw1, seed0 | $-0.004014$ | $[-0.006576,-0.001267]$ | yes |
| draw2, seed0 | $-0.001554$ | $[-0.004194,+0.001195]$ | no |
| draw3, seed0 | $-0.007049$ | $[-0.009609,-0.004351]$ | yes |
| draw4, seed0 | $-0.004677$ | $[-0.007409,-0.001916]$ | yes |
| draw0, seed1 | $-0.001029$ | $[-0.003777,+0.001767]$ | no |
| draw0, seed2 | $-0.002339$ | $[-0.005052,+0.000417]$ | no |
| draw0, seed3 | $\mathbf{+0.000185}$ | $[-0.002643,+0.003030]$ | no |

Seven of eight favour the vector code, four of eight reach significance
individually, and **one refit favours scalar ternary**.

### The comparison that matters

| | 0.8B | 27B |
|---|---:|---:|
| mean over 8 refits | $-0.034095$ | $-0.003345$ |
| range over refits | $0.006748$ | $0.007234$ |
| **effect / refit range** | **5.05** | **0.46** |
| sign consistent | 8/8 | 7/8 |

The absolute refit spread is essentially the same at both scales — $0.0067$
and $0.0072$ — while the effect is an order of magnitude smaller at 27B. At
0.8B the advantage is five times the spread it sits in; at 27B it is less than
half of it. **On this 9.36% target subset the 27B effect is smaller than the
variability induced by refitting the quantizer.**

Reported by axis, since the design is unbalanced (the five draws share seed 0
and the four seeds share draw 0, so the eight cells are not independent):

| axis | n | mean | sd | negative |
|---|---:|---:|---:|---:|
| calibration draw, seed 0 | 5 | $-0.004715$ | $0.002144$ | 5/5 |
| codebook seed, draw 0 | 4 | $-0.002366$ | $0.002806$ | 3/4 |

The draw axis is the better-behaved one: all five negative, and a one-sample
$t$ over those five gives $t=-4.92$. The seed axis is where the sign flips. We
do not pool the eight into a single test, because draw 0 appears four times and
the cells are not independent; a balanced factorial would be the right design
and was not run.

### What this does to the 27B claim

§11's frozen full-stream measurement of $-0.007121$ is correct for the
quantizer it measured, and its interval is tight because the evaluation is
large. But that quantizer is `draw0, seed0`, and the refit distribution places
it **near the favourable end** of what refitting produces. A practitioner
fitting once, with a different seed, could observe $+0.000185$ and conclude
there is no advantage at all.

The defensible 27B statement is therefore weaker than any previous draft's:

> On the 48 DeltaNet QKV projections of Qwen3.8-27B, averaging over calibration
> draws, the eight-dimensional code reduces damage relative to learned scalar
> ternary by roughly $0.003$ to $0.005$ NLL. The effect is smaller than the
> spread induced by refitting the quantizer: across eight refits it ranged from
> $-0.007049$ to $+0.000185$, and only half were individually significant. Any
> single-refit 27B number, including the frozen one reported here, should be
> read as one draw from that spread.

This does not touch the 0.8B result, where the same eight-refit protocol gives
8/8 consistency at five times the spread. It does mean the 27B evidence
supports *direction on average*, not a reliable per-conversion benefit, and the
paper should not lean on it.

## 17. A whole model, and a verified compressed matmul (2026-09-23)

Two gaps have been listed as unmeasured since the beginning: no full-coverage
conversion, and no packed inference path. Both are now measured, and both
results are worse for the deployment story than the subset experiments implied.

This section has been corrected twice since it was first written. The
subsections below carry the corrected numbers; "Defects found in this section"
at the end records what was wrong and in which direction, because two of the
corrections helped the project and two hurt it.

### Full coverage costs far more than the subsets suggested

Every 2-D weight except the embedding: 186 tensors, 497,614,848 of 752,393,024
parameters (66.1%), evaluated on the complete validation stream. That is *every*
matrix outside the embedding, not a selection from them.

What "the model" means here, since the first writeup of this run got it wrong.
The checkpoint holds 873,438,784 parameters in three pieces — the language
model (752,393,024), a vision tower (100,592,896) and a multi-token-prediction
head (20,452,864) — and `AutoModelForCausalLM` instantiates only the first, so
the other two are never loaded and enter no byte total below. What stays BF16
inside the loaded model is the embedding (254,279,680, which
`tie_word_embeddings` also makes the output head), the depthwise causal
convolutions (442,368, which are 3-D and outside this codec's shape
assumption), and the norms (56,128).

| Arm | covered bpw | model MB | effective bpw | shrink | ΔNLL | perplexity |
|---|---:|---:|---:|---:|---:|---:|
| dim-4 K=81 | 1.7250 | 616.9 | 6.559 | 2.44x | $+1.000022$ | $+171.8\%$ |
| dim-4 K=255 | 2.1250 | 641.7 | 6.823 | 2.34x | $+0.470715$ | $+60.1\%$ |
| dim-4 K=1625 | 2.7919 | 683.2 | 7.264 | 2.20x | $+0.171490$ | $+18.7\%$ |

Ternary-rate full coverage costs **a full nat** — perplexity nearly triples.

**A correction to how this was first reported.** The original comparison put
full-model dim-4 damage against the 18-projection *dim-8* figure ($+0.077061$)
and concluded that damage is "markedly super-linear in coverage". Both halves
were wrong: the like-for-like dim-4 subset number is $+0.088512$, and adding
tensor families changes composition as well as coverage, so nothing about
super-linearity follows from two points. This project has enough runs at
identical rate, recipe, BF16 baseline ($3.436710$) and evaluation stream
($261{,}284$ targets) to test additivity directly.

| target set | params | coverage | ΔNLL at dim-4 K=81 | source |
|---|---:|---:|---:|---|
| `in_proj_qkv` (18) | 113,246,208 | 15.1% | $+0.088512$ | `ternary_task_v4` |
| MLP (72) | 264,241,152 | 35.1% | $+0.552558$ | `mlp_task_v1` |
| both (90) | 377,487,360 | 50.2% | $+0.637829$ | `mixed_alloc_v1` |
| all non-embedding (186) | 497,614,848 | 66.1% | $+1.000022$ | `fullmodel_v1` |

The two disjoint families give an apparent additive reference: $0.088512 +
0.552558 = 0.641069$ against a measured $0.637829$ for converting both
together, agreeing to within $0.5\%$.

**That agreement is real but it is not a measurement of additivity**, and a
review was right to say so. Those three numbers come from three separate runs,
each fitting its shared codebook on its own tensor pool, so expanding coverage
changes the perturbation applied to the *previously covered* tensors as well.
The comparison confounds an interaction between tensor families with a change
in the codec each family receives. See below for the test that removes it.

What the jump to 66.1% shows instead is composition. Damage per covered
parameter:

| target set | ΔNLL per 100M params |
|---|---:|
| `in_proj_qkv` | $0.078$ |
| MLP | $0.209$ |
| the remaining 120.1M (attention, `in_proj_z`, `out_proj`) | $0.302$ |

The tensors added last are the most damaging per parameter in the model, so the
extra $+0.362$ from $50.2\%$ to $66.1\%$ is largely what those particular
tensors cost rather than a penalty for breadth as such. But the same confound
applies, and more strongly: **assigning that entire difference to the added
120.1M parameters is not justified**, because the full-model run also re-fits
the codebook over a pool four times larger, changing what the already-covered
tensors receive. The defensible statement is the weak one: converting every
non-embedding matrix costs far more than any subset here, and composition is
clearly part of why.

### The interaction test, with the codec held fixed

The review proposed the clean design: quantize once, freeze the artifacts, and
install subsets. One Hessian capture, one codebook fitted over all 90 tensors,
one quantization pass — then the dequantized weights installed as QKV only, MLP
only, and both, so **every tensor carries bit-identical weights in every arm
that includes it** (`results/interaction_v1`).

| arm | tensors | params | ΔNLL | 95% CI |
|---|---:|---:|---:|:--|
| A — `in_proj_qkv` | 18 | 113,246,208 | $+0.085350$ | $[+0.082522,+0.088216]$ |
| B — MLP | 72 | 264,241,152 | $+0.548660$ | $[+0.540311,+0.557142]$ |
| AB — both | 90 | 377,487,360 | $+0.637829$ | $[+0.628446,+0.647323]$ |

$$I = L_{AB} - L_A - L_B + L_{\text{BF16}} = +0.003818 \quad [+0.000769,\, +0.006835]$$

paired over the same resampled evaluation blocks. **The interaction excludes
zero: damage is slightly super-additive, not additive.**

Two things worth noting. First, the effect is small — $+0.003818$ against a
total of $+0.637829$, about $0.6\%$ — so "approximately additive" remains a
good working description even though strict additivity is rejected. Second, the
confound was real and measurable: with the codebook fitted over all 90 tensors
instead of each subset alone, QKV damage falls from $+0.088512$ to $+0.085350$
and MLP damage from $+0.552558$ to $+0.548660$. The earlier $0.5\%$ agreement
was partly those two shifts cancelling the interaction, which is exactly why a
frozen-artifact design was needed to see either.

So the record on this question now reads: the original "markedly super-linear"
claim was wrong in magnitude by orders of magnitude; the correction to
"additive" was wrong in sign; and the measured answer is a small, resolvable,
positive interaction.

### Embeddings dominate the compressed model

The byte accounting is the more useful surprise:

| Arm | converted tensors | untouched BF16 | untouched share |
|---|---:|---:|---:|
| dim-4 K=81 | 107.3 MB | 509.6 MB | **82.6%** |
| dim-4 K=1625 | 173.7 MB | 509.6 MB | 74.6% |

Of that 509.6 MB, 508.6 MB is the embedding alone; the convolutions and norms
together are 1.0 MB and do not matter.

The weight matrices compress 9.28x at ternary rate — 995 MB to 107 MB, which
is what the codec work bought. The *model* shrinks only 2.44x, because the
254.3M-parameter embedding stays at BF16 and then accounts for 83% of what
remains. Effective rate over the whole model is 6.559 bits/weight even with the
covered tensors at 1.725. The embedding is tied, so it is already charged once
and used twice; there is no second copy to remove.

**At low rates the binding constraint is embedding precision, not the weight
codec.** Halving the embedding to FP8 would save more bytes than moving the
covered tensors from 2.667 bits to ternary, and would cost an unmeasured but
probably far smaller amount of quality. That comparison was never run here
because the project's attention was on the weight codec throughout; on this
evidence that emphasis was misplaced for deployment purposes, though not for
the codec question the paper actually answers.

### A compressed matmul, on a projection the loss numbers actually used

The first version of this subsection rested on `bench_packed_matmul.py`, which
quantizes **Gaussian** weights at the 27B QKV shape with no rotation, no GPTQ
and no real activations, checks correctness against an FP32 decoded product,
and compares latency against BF16. A reviewer was right that this establishes a
decoder and a GEMV at a realistic shape and nothing about running a converted
model; the heading "a kernel that runs it" was not earned. That file is now
labelled a synthetic microbenchmark, and the claim below rests on a new run
(`results/real_projection_v1`) that uses the real recipe.

**Storage layout is not runtime layout.** The storage packing puts five
base-6561 codes in a uint64 for 1.600 bits of index, which is optimal;
$6561^5 = 1.216\times10^{19}$ exceeds $2^{63}$, so the words set the sign bit
and must be treated as unsigned. A runtime layout of one code per int16 costs
2.000 bits of index instead. Quoting 1.725 bits/weight as a resident footprint,
as earlier sections did implicitly, is wrong.

*A factual correction.* An earlier draft justified the wider runtime layout by
saying "neither PyTorch nor Triton has unsigned 64-bit arithmetic". That is
wrong about Triton, which has `tl.uint64`. Checked on this machine (torch 2.11,
triton 3.6): PyTorch will hold a `torch.uint64` tensor but raises
`NotImplementedError: "add_stub" not implemented for 'UInt64'` on arithmetic,
so the host side cannot do the digit extraction; inside a kernel the extraction
is possible and costs five integer divisions per word on the critical path. The
reason for the wider layout is cost and host-side support, not a missing type.

**And the runtime activation is not the stored one.** The codes represent
`rotate(W) = W A^T` while the quality runs install `unrotate(dq) = dq A`, so
$y = x (dq A)^\top = \mathrm{rotate}(x)\, dq^\top$: a compressed path must
apply the same blockwise Hadamard to its *input*. The synthetic benchmark
omitted that step entirely, and it turns out to dominate the cost.

#### Does the compressed path compute what the loss numbers measured?

`in_proj_qkv` of Qwen3.5-0.8B at layer 12 ($6144\times1024$), quantized with the
`vq8_rot_gptq` recipe — rotation, a shared 6561-point dimension-8 codebook
pooled over all 18 projections, GPTQ against a Hessian from 65,536 calibration
tokens — and evaluated on activations captured from the model itself.

All paths scored on the same 256 activation rows with the same denominator. An
earlier version of this table took the BF16 maximum over 1024 rows and the
Triton maximum over 16, so the ratio between them compared samples of different
size; that defect is fixed and the numbers below are the corrected ones.

| comparison | relative error |
|---|---:|
| installed BF16 vs exact FP32 | $3.55\times10^{-3}$ |
| streamed path vs exact FP32 | $1.08\times10^{-7}$ |
| **Triton kernel vs exact FP32** | $\mathbf{2.71\times10^{-7}}$ |
| Triton kernel vs installed BF16 | $3.54\times10^{-3}$ |

"Exact" is the FP32 product with the dequantized weights; "installed" is the
BF16 product with `unrotate(dq)`, which is the operation behind every ΔNLL in
this document. The kernel reproduces the quantized FP32 operator to four orders
of magnitude better than the BF16 weights do.

**That is operator fidelity, and an earlier draft drew more from it than it
supports.** It said "every quality number here is therefore conservative with
respect to compressed execution". That does not follow. A smaller local
numerical error does not imply lower end-to-end loss: BF16 rounding can
reinforce *or partially cancel* the quantization error, and the nonlinear
layers downstream need not respond monotonically to a local perturbation. The
claim is withdrawn. What is established: **the compressed kernel closely
reproduces the quantized FP32 reference on the tested activations; its
end-to-end quality relative to decoded BF16 execution remains unmeasured.**

#### Memory: the claim holds, after an accounting fix

The resident figure was also wrong, in the project's favour. The benchmark
charged two bytes an element for the codebook and the scales because FP16 was
what the design intended, while both tensors were FP32 at runtime — the Triton
wrapper converted them. The honest number for that code was $7.01\times$, not
$7.47\times$. Rather than report the smaller number, the tables are now stored
in FP16 and read in FP16 by the kernel, which promotes them in-register; bytes
are counted from each tensor's own `element_size`.

| shape | BF16 | compressed | ratio |
|---|---:|---:|---:|
| real `in_proj_qkv`, $6144\times1024$ | 12.583 MB | 1.776 MB | $7.08\times$ |
| synthetic 27B QKV, $10240\times5120$ | 104.858 MB | 14.031 MB | $7.47\times$ |

The difference between them is codebook amortization: the same 6561-point book
is charged against a tensor six times smaller.

Peak allocation during one product, at the synthetic shape, each arm measured
holding only its own state: dense BF16 113.5 MB, Triton 22.7 MB
($\mathbf{5.00\times}$), streamed 90.8 MB. The streamed path's figure is a
tile-size choice — 39.7 MB at `tile_cols=128` against 295.3 MB at 2048, with
latency flat to within $9\%$ — not a property of the format.

#### Latency: worse than first reported, and for a different reason

| path | ms, batch 1 |
|---|---:|
| installed BF16 (cuBLAS) | 0.0187 |
| Hadamard rotation, PyTorch prototype | 0.4760 |
| **Hadamard rotation, fused Triton** | **0.0182** |
| Triton kernel + prototype rotation | 0.5233 |
| **Triton kernel + fused rotation** | **0.0555** |
| streamed path + prototype rotation | 0.8415 |

The first version of this subsection reported only the prototype row and
concluded the compressed path is $28\times$ slower than cuBLAS, "and about 90%
of it is the activation rotation". A review objected that this diagnoses the
implementation rather than the transform, and that no optimized figure should
be inferred by subtracting separately timed components. Both objections were
right, and the fused transform settles it by measurement.

**`fwht` was the problem, not the Hadamard.** The prototype runs ten
Python-controlled butterfly stages over 1024 columns, each cloning two
half-tiles and writing two slice assignments back — eleven passes over the data
for one activation vector, almost entirely launch and allocation overhead. The
radix-2 FWHT in natural order is exactly a Kronecker product, so a
$1024$-point block viewed as a $32\times32$ tile $X$ satisfies
$\mathrm{FWHT}(x) = \mathrm{vec}(H_{32} X H_{32})/32$: two small matrix products
in one kernel, with the sign flip folded in. Verified against `fwht` at three
block sizes and three batch sizes (agreement $2\times10^{-7}$; on the benchmark
vector, **bit-identical**).

| | prototype | fused | |
|---|---:|---:|---:|
| rotation alone | 0.4760 ms | 0.0182 ms | **26.1x faster** |
| whole compressed path | 0.5233 ms | 0.0555 ms | **9.4x faster** |
| against cuBLAS | 28.0x slower | **3.0x slower** | |

So the format's latency cost at this shape is **$3.0\times$ cuBLAS, not
$28\times$**. That figure is also consistent with the synthetic benchmark's
$3.1\times$ GEMV-only comparison at the 27B shape, where no transform is in the
loop — two independent measurements agreeing, where the earlier subtraction
argument would have been guesswork.

What remains true: the compressed path is still slower than a tuned BF16 GEMM
at batch one, and neither kernel is tuned beyond correctness. What is withdrawn:
the claim that the rotation intrinsically dominates compressed inference.

At the larger synthetic shape the GEMV comparison is $0.132$ ms against
$0.404$ ms, $3.1\times$. So the kernel gap narrows with size, as a
memory-bound argument predicts, but the rotation is a fixed additional cost
that no amount of tensor size amortizes away and that a deployed system would
have to fuse into the preceding operation. That fusion is not implemented or
measured here.

**Where this leaves the runtime claim.** The format is decodable at speed,
reproduces the quantized operator four orders of magnitude more closely than
the BF16 weights the quality results used, and peaks at $5\times$ less memory.
It is $3.0\times$ slower than cuBLAS at batch one once the transform is fused —
a real cost, but a tractable one, and no longer dominated by an artifact of the
offline tooling. "A kernel that runs it" overstated the first version of this
section; "a verified compressed path with a real memory win and a $3\times$
latency cost" is what is established now.

### Defects found in this section, and what they cost

Each was a defect in measurement or description rather than in the codec. Two
corrections moved in the project's favour and two against it, which is the only
reason this list is worth keeping.

**The peak-allocation comparison was contaminated, and hid a real result.**
`reset_peak_memory_stats` rebases the peak counter to whatever is allocated at
the moment it is called, and the first benchmark held the 210 MB FP32 reference
through both arms. Dense 811.7 MB against streamed 775.1 MB — a 1.05x ratio —
was a property of the harness. Measured with each arm holding only its own
state (`results/packed_matmul_v2`):

| path | baseline | peak | transient | peak vs dense |
|---|---:|---:|---:|---:|
| dense BF16 | 113.4 MB | 113.5 MB | 0.0 MB | — |
| streamed, `tile_cols=512` | 22.6 MB | 90.8 MB | 68.2 MB | 1.25x |
| Triton | 22.6 MB | 22.7 MB | 0.1 MB | **5.00x** |

The streamed path looks worse only because it materializes a decoded FP32
column tile, which is a tile-size choice rather than a property of the format —
39.7 MB at `tile_cols=128` against 295.3 MB at 2048, with latency flat to
within 9%. (A common ~8.5 MB sits in every baseline: the cuBLAS workspace, a
cost of running any matmul rather than of either format.)

**The resident-byte count charged FP16 for FP32 tensors.** The codebook and the
scales were built as FP16 and immediately promoted, and the Triton wrapper
converted them again at every call, while the accounting charged two bytes an
element throughout. The reduction for that code was $7.01\times$, not the
$7.47\times$ reported. The tables are now genuinely FP16, read as FP16 by the
kernel and promoted in-register, and bytes come from `element_size` so the
claim cannot drift from the object again. Keeping FP16 through the *arithmetic*
is not free, though: the streamed path's error against the reference decoder
went from $3\times10^{-7}$ to $2\times10^{-4}$ until its tile construction was
put back in FP32.

**The kernel was never validated on the operation the quality results use.**
Gaussian weights, no rotation, no GPTQ, correctness against an FP32 decoded
product. That is a microbenchmark, and calling §17 "a kernel that runs it" was
not supported by it. The real-projection run above replaces the claim, and it
cost the latency story: including the activation rotation the compressed path
is $28\times$ slower than cuBLAS on a real projection, not $3.5\times$.

**The streamed kernel accepted tile sizes it could not handle.** It asserted
only `tile_cols % dim == 0` while slicing scales by whole groups, so
`tile_cols=64` at `group=128` selected an empty scale slice and crashed inside
the loop. Reproduced, and the assertion now requires group alignment. The tile
sweep above used 128, 512 and 2048, all group-aligned, so no reported number
was affected.

**The vision tower is absent, not included.** The module docstring claimed it
was counted in the byte total. It was not, and the arithmetic is now checked
rather than asserted: 752,393,024 (language model) + 100,592,896 (vision tower)
+ 20,452,864 (MTP head) = 873,438,784, and `AutoModelForCausalLM` instantiates
only the first. The earlier note that "the missing 121.0M is the vision tower"
was itself imprecise — 100.6M of it is, and the other 20.5M is the
multi-token-prediction head. Coverage is 66.1% of the loaded text model and
every byte figure above is for that model. The runner now computes the
BF16 leftovers by category and asserts they sum to the untouched total, so this
particular description cannot drift from the run again.

## 18. Calibration mismatch: an identity, and a mixture that acts on it (2026-09-23)

A review made the sharpest observation in this project so far: **changing the
calibration corpus changes damage substantially while changing storage not at
all.** Every other lever here — dimension, rate, coverage — is paid for in
bytes. This one is free. §14A established that mismatch drives the domain
effect; this section says why it must, and tests the intervention that follows.

### Why mismatch has an effect at all

Write $E = \widehat{W} - W$ for the error a codec leaves in a weight matrix.
For a fixed input distribution $D$ the layer reconstruction objective is
quadratic,

$$L_D(E) = \mathbb{E}_D \lVert E x \rVert_2^2 = \operatorname{tr}\!\left(E H_D E^{\top}\right), \qquad H_D = \mathbb{E}_D[x x^{\top}],$$

so evaluating on one domain a quantizer fitted on another costs exactly

$$L_{\text{test}}(E) - L_{\text{cal}}(E) = \operatorname{tr}\!\left[E\left(H_{\text{test}} - H_{\text{cal}}\right)E^{\top}\right],$$

bounded elementarily by
$\lVert H_{\text{test}} - H_{\text{cal}}\rVert_2 \lVert E \rVert_F^2$.

**This is an identity and an elementary bound. It is not a theorem, and it says
nothing about NLL.** What it does say is that the penalty depends on how $E$ is
*oriented* relative to the difference of second moments: two quantizers with
identical $\lVert E \rVert_F$ can differ arbitrarily in that trace. Plain weight
MSE is $\lVert E \rVert_F^2$ up to a constant, so it is blind to the orientation
by construction — the same blindness that makes it invert against loss in §14C
and in the compensation results. Hessian-weighted objectives of this form
underpin GPTQ and GPTVQ; nothing here is new, but it is the right frame for
what §14A found empirically.

### Why a mixture should help at all

A review supplied the mechanism, and it is worth stating because it is more
specific than "more diverse data is better". The codebook here is fitted from
weights with a fixed seed, so changing the calibration corpus changes almost
nothing about the alphabet — what it changes is the **Hessian used for GPTQ
assignment and compensation**. A pure-domain Hessian protects the activation
directions that domain uses and leaves the rest comparatively unprotected, even
if another domain leans on them heavily. Since the second-moment matrices are
positive semidefinite, a direction is unpenalized under
$H_{\text{mix}} = \tfrac12 H_{\text{WT}} + \tfrac12 H_{\text{TS}}$ only if
*both* domains leave it unpenalized. Broader calibration buys coverage of
directions, not merely more samples.

The results below are consistent with that reading: half the wikitext samples
retain most of the wikitext benefit while the TinyStories samples remove most
of the mismatch penalty. But **it is a plausible explanation, not a
demonstrated one** — no directions were inspected, no conditioning measured,
and the test that would separate direction coverage from sample count is the
half-budget control in §19 rather than anything in this section.

### The mixture, and what it buys

The identity suggests a fixed-budget intervention. For the quadratic objective,
$H_{\text{mix}} = \tfrac12 H_{\text{WT}} + \tfrac12 H_{\text{TS}}$ represents
*exactly* the equally weighted average of the two domains' reconstruction
losses. Whether that improves **worst-domain NLL** does not follow — the
objective is a local proxy and the mixture halves the data seen from each
domain — so it is an experimental question.

Protocol: the split is prespecified at 50/50 rather than tuned; the total
calibration budget is held at 65,536 tokens, the same as every other condition;
each domain's calibration text is disjoint from the text either is evaluated
on; codebook fitting, seed, recipe and evaluated text are unchanged. Because
the crossed results were inspected before this was designed, it is an
**exploratory intervention**, not a confirmatory one.

ΔNLL against BF16 (`results/mechanism_v2`):

| calibration | code | wikitext-2 test | TinyStories | **worst domain** |
|---|---|---:|---:|---:|
| wikitext | scalar | 0.121115 | 0.183486 | 0.183486 |
| wikitext | VQ8 | 0.082309 | 0.140296 | 0.140296 |
| TinyStories | scalar | 0.288304 | 0.036675 | 0.288304 |
| TinyStories | VQ8 | 0.199492 | 0.029201 | 0.199492 |
| **mixed 50/50** | scalar | 0.144351 | 0.051790 | **0.144351** |
| **mixed 50/50** | VQ8 | 0.089604 | 0.042378 | **0.089604** |

**The mixture's worst-domain damage is lower than either pure condition's, for
both codes**, at identical storage and identical calibration budget: $0.089604$
against $0.140296$ and $0.199492$ for VQ8 — a $36\%$ reduction against the
better pure condition — and $0.144351$ against $0.183486$ and $0.288304$ for
scalar, a $21\%$ reduction.

> These are **single-draw** figures. §19 repeats all of it over three disjoint
> calibration draws and reports means ($0.095211$ against $0.147013$ and
> $0.191296$ for VQ8, a $35\%$ reduction), with a control that separates domain
> coverage from sample count. Where the two sections differ numerically, §19 is
> the one to quote.

The price is on the matched diagonal, and it is small: VQ8 on wikitext goes
from $0.082309$ matched to $0.089604$ mixed ($+0.0073$), and on TinyStories
from $0.029201$ to $0.042378$ ($+0.0132$). Half the calibration data per domain
costs far less than being calibrated on the wrong domain entirely.

VQ8 beats scalar in all six cells, so the codec conclusion is unchanged by the
mixture.

**Scope, stated plainly.** Two corpora, one model, one recipe, one split, one
calibration draw, one seed. Note also that the mixture averages two
**32,768-token** Hessian estimates while each pure condition uses a single
**65,536-token** estimate, so "mixing helps" and "two smaller estimates of
different things beat one larger estimate of one thing" are not separated here.
§19 adds disjoint draws, direct paired contrasts and a half-budget control that
addresses exactly that. This does not establish that mixing is optimal, that 50/50 is the right
weighting, or that the result survives more domains — all of which the identity
is silent on, since it describes a local quadratic proxy and the measurement is
end-to-end loss. What it does show is that the free lever is real and
actionable: a practitioner who does not know the deployment domain pays less by
mixing calibration than by guessing.

### The g64 2x2, with the interval that was missing

Reran with all six pairwise contrasts. The four cells reproduce §14C exactly
(0.121115, 0.121852, 0.135133, 0.115284), which is a useful reproducibility
check in its own right.

| contrast | holds fixed | ΔNLL | 95% CI |
|---|---|---:|:--|
| grouping, g128 book | codebook | $+0.014019$ | $[+0.012133,+0.015908]$ |
| grouping, g64 book | codebook | $+0.006568$ | $[+0.004410,+0.008588]$ |
| codebook, g128 grouping | grouping | $-0.005831$ | $[-0.007809,-0.003792]$ |
| codebook, g64 grouping | grouping | $-0.013281$ | $[-0.015222,-0.011297]$ |
| *diagonal (both change)* | nothing | $+0.000737$ | $[-0.001213,+0.002710]$ |

The contrast §14C never computed now has an interval, and it **excludes zero**:
moving to g64 with the g64-fitted codebook held fixed costs $+0.006568$
$[+0.004410,+0.008588]$. So all four single-factor contrasts exclude zero and
only the diagonal does not. Main effects $+0.010293$ (grouping) and $-0.009556$
(codebook), interaction $+0.007451$.

There is no reading of this in which group size "moves nothing". The correct
one: **finer grouping hurts at a fixed alphabet, refitting the alphabet in the
finer normalized space helps by about as much, and the two nearly cancel.** The
original conclusion — that the instability is a codebook story and not a
grouping story — was drawn from the one contrast in the table that confounds
them. The cancellation is close to exact along one path through the factorial:
$+0.014019$ (change grouping, book fixed) $-0.013281$ (adapt book, grouping
fixed) $= +0.000738$, against the measured diagonal of $+0.000737$.

**What this still does not show.** Cancellation explains why the end-to-end
contrast is *small*; it does not establish that these opposing effects are what
make it *change sign* across refits.

> **§20 ran that test and the hypothesis failed.** Repeated over six refits, three
> of the four single-factor contrasts change sign themselves, and the one that
> varies most varies more than the diagonal it was meant to explain. §20 also
> found that this factorial was measured on wikitext-2 *test* while the sign
> instability it was invoked to explain was measured on *validation* — the two
> were never on the same stream. Read §20 instead of this paragraph.

### Position-resolved damage, with the intervals that change the answer

§14B read a trend off five bucket means with no measure of block-to-block
spread. With all 127 blocks' per-position losses retained and eight buckets,
the picture is weaker than §14B reported.

| positions | scalar damage | VQ8 damage | VQ8 advantage |
|---|---:|---:|---:|
| 0–256 | $+0.117398$ | $+0.079553$ | $-0.037845$ |
| 256–512 | $+0.115513$ | $+0.082541$ | $-0.032973$ |
| 512–768 | $+0.114274$ | $+0.082985$ | $-0.031289$ |
| 768–1024 | $+0.117822$ | $+0.082576$ | $-0.035247$ |
| 1024–1280 | $+0.112215$ | $+0.083584$ | $-0.028631$ |
| 1280–1536 | $+0.115263$ | $+0.085315$ | $-0.029948$ |
| 1536–1792 | $+0.117893$ | $+0.087258$ | $-0.030636$ |
| 1792–2048 | $+0.118514$ | $+0.090364$ | $-0.028150$ |

Paired over blocks, last bucket against first. A review pointed out that at
5,000 replicates the seed can flip a threshold this close to zero, so these are
100,000 replicates and the interval is reported as its range over five seeds:

| quantity | Δ | lower bound | upper bound | |
|---|---:|:--|:--|:--|
| VQ8 damage | $+0.010811$ | $[-0.000223, -0.000124]$ | $[+0.021601,+0.021675]$ | **includes 0, all seeds** |
| scalar damage | $+0.001116$ | $[-0.010856,-0.010721]$ | $[+0.013004,+0.013096]$ | includes 0, all seeds |
| VQ8 advantage eroded | $+0.009695$ | $[+0.000052,+0.000126]$ | $[+0.019264,+0.019330]$ | excludes 0, all seeds — *barely* |

The classifications are stable across seeds at this replicate count, but the
erosion's lower bound sits between $5\times10^{-5}$ and $1.3\times10^{-4}$: it
excludes zero by a margin two orders of magnitude smaller than the effect
itself.

And as a rank trend over all eight buckets, computed per block and bootstrapped
over blocks:

| quantity | mean Spearman ρ | 95% CI | blocks with ρ > 0 |
|---|---:|:--|---:|
| VQ8 damage | $+0.0934$ | $[+0.0244,+0.1631]$ | 77/127 |
| scalar damage | $+0.0075$ | $[-0.0637,+0.0784]$ | 57/127 |
| VQ8 advantage | $+0.0688$ | $[+0.0067,+0.1307]$ | 75/127 |

**§14B's headline is withdrawn.** "VQ8 damage rises monotonically, about 12%"
was read off means. With intervals, the endpoint contrast for VQ8 damage does
not exclude zero; the rank trend over all eight buckets does, but weakly — mean
ρ of 0.093, positive in 77 of 127 blocks, so the effect is real in aggregate
and absent in nearly 40% of individual blocks. It is also not monotone: bucket
4 sits below bucket 3.

What survives, and is now properly supported:

* scalar damage shows **no** position trend by either test;
* VQ8's **advantage over scalar erodes** across the block — $-0.037845$ in the
  first bucket against $-0.028150$ in the last, excluding zero on both the
  endpoint contrast and the rank trend, though only just;
* the ordering is preserved in every bucket, so the headline codec claim is
  untouched.

**How much weight this should carry: not much.** The erosion is the one
surviving positive finding here, and it survives by a margin of about
$10^{-4}$. It is also a post-hoc analysis — the buckets were inspected in §14B
before this rerun was designed — so it has none of the protection a
prespecified test would have. It belongs in the paper as weak evidence of
position-dependent erosion and should not be promoted above the calibration
result, which is larger, better identified, and actionable.

Position and token content remain confounded: later positions hold different
tokens as well as more accumulated recurrent state. **This is
position-dependent damage, not identified drift**, and separating them needs
the same tokens evaluated with and without a state reset, which is not run
here.

## 19. The mixture with draws, contrasts, and a sample-count control (2026-09-23)

§18 reported the mixture from one calibration draw, gave worst-domain damage
without an interval, and discarded per-block losses so no direct mixed-versus-
pure contrast could be formed. A review asked for all three, and named a
confound worth removing: **the 50/50 mixture averages two 32,768-token Hessian
estimates while each pure condition uses one 65,536-token estimate**, so
"mixing helps" and "two smaller estimates of different things beat one larger
estimate of one thing" were not separated.

Five conditions, three **disjoint** calibration draws each, per-block losses
retained (`results/calib_mix_v2`). The `_half` arms spend the mixture's
per-domain budget on a single domain, which is the control.

Worst-domain ΔNLL, by draw:

| condition | tokens | code | by draw | mean | range |
|---|---:|---|---|---:|---:|
| wikitext | 65,536 | VQ8 | 0.140296, 0.149136, 0.151607 | 0.147013 | 0.011311 |
| tinystories | 65,536 | VQ8 | 0.199492, 0.182104, 0.192293 | 0.191296 | 0.017389 |
| **mixed 50/50** | 32,768 + 32,768 | VQ8 | 0.089604, 0.094118, 0.101910 | **0.095211** | 0.012307 |
| wikitext half | 32,768 | VQ8 | 0.142857, 0.160200, 0.164745 | 0.155934 | 0.021888 |
| tinystories half | 32,768 | VQ8 | 0.213931, 0.190821, 0.214691 | 0.206481 | 0.023870 |

### The control separates coverage from sample count, decisively

Two differences, both means over three draws, for the vector code:

* **halving the wikitext budget costs $+0.008921$** (65,536 → 32,768 tokens of
  one domain);
* **spending that same halved budget on a second domain instead gains
  $-0.060724$** (wikitext-half → mixed).

So at this budget domain coverage is worth about **seven times** what sample
count is worth, and in the opposite direction. The mixture does not win because
two half-size estimates beat one full-size estimate; it wins despite each
estimate being half-size. The scalar code gives the same answer ($+0.006021$
against $-0.052864$).

This is consistent with the mechanism §18 sketched — the mixture's Hessian
leaves a direction unprotected only if *both* domains leave it unprotected —
without demonstrating it, since no directions were inspected.

### Direct contrasts, paired within each draw

All sixteen mixed-versus-pure contrasts keep their sign across all three draws.
The shape of the trade, for VQ8:

| contrast | evaluated on | mean Δ | range over draws |
|---|---|---:|---:|
| mixed vs wikitext-half | wikitext | $+0.007311$ | 0.010504 |
| mixed vs wikitext-half | TinyStories | $-0.113104$ | 0.019177 |
| mixed vs tinystories-half | TinyStories | $+0.017898$ | 0.013396 |
| mixed vs tinystories-half | wikitext | $-0.111270$ | 0.027624 |

Against the matched half-budget condition the mixture gives up $0.007$ on the
home domain and gains $0.113$ on the other — a sixteen-to-one trade. That
asymmetry, not a small average improvement, is why worst-domain damage falls.

### Measured against this project's own yardstick

§16 judged the 27B codec result by effect ÷ refit range and found $0.46$,
which is why the paper does not lean on it. The same yardstick here: the
mixture's gain over the best pure condition is $0.051803$ against a refit range
of $0.012307$, a ratio of **4.2** for VQ8 and **16.8** for scalar. That is in
the range of the 0.8B codec result ($5.05$), not the 27B one. The mixture is
also the most *stable* condition for the scalar code, with a refit range of
$0.002791$ against $0.013840$ for pure wikitext.

### What this still does not establish

One model, one recipe, two corpora, one split. It does not show that 50/50 is
the right weighting, that mixing is optimal, that the result survives more
domains, or that the direction-coverage mechanism is the operative one. The
50/50 split remained prespecified rather than tuned, but the line of work is
exploratory: the crossed cells were inspected before any of it was designed.

What it does establish, on three disjoint draws with a control that separates
the obvious alternative: **at fixed storage and fixed calibration budget, a
practitioner who does not know the deployment domain pays substantially less by
mixing calibration than by choosing one domain and hoping.** That is the only
result in this project that costs nothing to adopt.

## 20. The cancellation story, tested across refits — and falsified (2026-09-23)

§18 explained the unstable g64 contrast as a cancellation: finer grouping hurts,
refitting the alphabet in the finer space helps by about as much, and the
end-to-end difference is what is left. A review accepted that for the single
fit and pointed out precisely what it does not show — that cancellation
explains *smallness*, not *sign instability*. This run tests it.

The prediction was explicit and falsifiable: **if the diagonal is a small
difference between two larger stable effects, the single-factor contrasts
should hold their signs across refits while their difference wanders.**

### Two design points that make this sharper than §18's version

**Both codebooks are fitted from weights by a deterministic Lloyd-Max
procedure**, so neither the calibration draw nor the seed can move them. Across
refits the *only* thing that changes is the Hessian used for GPTQ assignment
and compensation. That also explains why §12's seed axis moved this contrast by
exactly zero — a fact previously recorded as a harness check rather than
understood.

**And this run evaluates on the corpus the instability was measured on.** §14C
and §18 ran the 2x2 on wikitext-2 *test* (BF16 3.394295) while the refit
instability it was invoked to explain comes from §12, on wikitext-2
*validation* (BF16 3.436710). The two were never on the same stream, which
nobody noticed. This run is on validation and reproduces §12's diagonal exactly
at the shared cell: $-0.001815$ against §12's $-0.0018154856816332554$.

### Result: every contrast is unstable, not just their difference

Six refits — three disjoint calibration draws at 65,536 tokens, three
calibration sizes at draw 0.

| refit | tokens | grouping\|g128 book | grouping\|g64 book | codebook\|g128 | codebook\|g64 | diagonal |
|---|---:|---:|---:|---:|---:|---:|
| draw0_65k | 65,536 | $+0.008219$ | $+0.001926$ | $-0.003741$ | $-0.010034$ | $-0.001815$ |
| draw1_65k | 65,536 | $+0.002475$ | $-0.001646$ | $-0.003525$ | $-0.007646$ | $-0.005171$ |
| draw2_65k | 65,536 | $+0.005464$ | $-0.006581$ | $+0.001943$ | $-0.010102$ | $-0.004638$ |
| draw0_16k | 16,384 | $+0.003979$ | $+0.005722$ | $-0.000687$ | $+0.001056$ | $+0.005035$ |
| draw0_32k | 32,768 | $+0.007709$ | $-0.001560$ | $-0.000847$ | $-0.010115$ | $-0.002406$ |
| draw0_131k | 131,072 | $+0.005366$ | $+0.008598$ | $-0.005929$ | $-0.002697$ | $+0.002669$ |

| contrast | mean | range | same sign | excludes 0 |
|---|---:|---:|:--|:--|
| grouping at fixed g128 book | $+0.005535$ | 0.005744 | yes | 6/6 |
| grouping at fixed g64 book | $+0.001077$ | 0.015179 | **no** | 3/6 |
| codebook at fixed g128 grouping | $-0.002131$ | 0.007872 | **no** | 3/6 |
| codebook at fixed g64 grouping | $-0.006590$ | 0.011171 | **no** | 5/6 |
| confounded diagonal | $-0.001055$ | 0.010206 | **no** | 5/6 |
| interaction | $+0.004459$ | 0.015278 | **no** | 0/6 |

**The prediction fails.** Exactly one of the four single-factor contrasts holds
its sign across refits — grouping at a fixed g128 codebook, $+0.005535$, which
excludes zero in all six. The other three all change sign, and the one that
wanders most (grouping at a fixed g64 codebook, range $0.015179$) wanders
*more* than the diagonal it was supposed to explain (range $0.010206$). The
interaction excludes zero in **none** of the six refits, against the
$+0.007451$ that looked resolved on a single fit in §18.

So §18's explanation is withdrawn. Cancellation is a correct description of one
fit and not a mechanism for the instability.

### What replaces it

Since the codebooks are provably fixed across refits here, **every effect in
the 2x2 is a function of the Hessian alone**, and the measurement says all of
them are sensitive to it. The instability is not located in the grouping, nor
in the codebook, nor in their cancellation: it is that GPTQ's compensation
statistics, estimated from 16k–131k tokens, are themselves noisy enough to move
every contrast in this family by $0.005$ to $0.015$ — comparable to or larger
than the contrasts themselves.

That is a more useful conclusion than the one it replaces, and it connects to
§18's identity: the Hessian is what the mismatch term is built from, and a
quantity estimated from a finite calibration sample carries sampling error into
every downstream contrast. It also predicts that the g64 family should stabilize
with more calibration tokens, which the size axis here does not obviously show
(the 131k cell is not the tightest) and which is not tested properly by three
nested sizes on one draw.

**Scope.** One model, one tensor family, six refits, one corpus. The
falsification is clean because the prediction was specific; the replacement is
an interpretation of the same six numbers and has not been tested against
anything.

## 21. Where the bytes should go: the embedding, not the codec (2026-09-23)

§17 ended with a conjecture it explicitly marked unmeasured — that halving the
embedding to FP8 would save more bytes than moving the covered tensors from
2.667 bits to ternary, and cost far less quality. It is now measured, and the
answer is more extreme than the conjecture.

The two axes crossed: the §17 weight ladder over all 186 non-embedding matrices,
against four embedding precisions. The embedding is **tied** to the output head,
so quantizing it perturbs the input lookup *and* every logit. Its quantizer is
deliberately naive — per-row scaling for FP8, per-group symmetric integers
otherwise, no rotation, no compensation — which makes the comparison
conservative against the weight codec, which has both.

| weights | embedding | model MB | eff. bpw | shrink | ΔNLL |
|---|---|---:|---:|---:|---:|
| BF16 | BF16 | 1504.8 | 16.000 | 1.00x | $0$ |
| BF16 | INT8-g128 | 1254.5 | 13.339 | 1.20x | $+0.000244$ |
| BF16 | FP8-e4m3 | 1251.0 | 13.302 | 1.20x | $+0.001434$ |
| BF16 | INT4-g128 | 1127.3 | 11.987 | 1.33x | $+0.032871$ |
| 2.792 bpw | BF16 | 683.2 | 7.264 | 2.20x | $+0.171490$ |
| 2.792 bpw | INT8-g128 | 432.9 | 4.603 | 3.48x | $+0.171893$ |
| 2.792 bpw | INT4-g128 | 305.8 | 3.251 | 4.92x | $+0.208926$ |
| 2.125 bpw | BF16 | 641.7 | 6.823 | 2.34x | $+0.470715$ |
| 1.725 bpw | BF16 | 616.9 | 6.559 | 2.44x | $+1.000022$ |
| 1.725 bpw | INT4-g128 | 239.4 | 2.546 | 6.29x | $+1.053793$ |

### The exchange rate is not close

Marginal cost of each move, in ΔNLL per 100 MB saved:

| move | MB saved | ΔNLL | per 100 MB |
|---|---:|---:|---:|
| embedding BF16 → INT8-g128 | 250.3 | $+0.000244$ | $0.000098$ |
| embedding BF16 → FP8-e4m3 | 253.8 | $+0.001434$ | $0.000565$ |
| embedding BF16 → INT4-g128 | 377.4 | $+0.032871$ | $0.008709$ |
| weights BF16 → 2.792 bpw | 821.6 | $+0.171490$ | $0.020873$ |
| weights 2.792 → 2.125 bpw | 41.5 | $+0.299226$ | $0.721380$ |
| weights 2.125 → 1.725 bpw | 24.9 | $+0.529306$ | $2.127315$ |

**Quantizing the embedding to INT8 is 213 times cheaper per byte than the first
step of the weight codec, and 21,700 times cheaper than its last step.** §17's
conjecture is confirmed with room to spare: FP8 on the embedding saves
$253.8$ MB against $66.4$ MB for the 2.792→1.725 weight move, and costs
$+0.001434$ against $+0.828532$ — $3.8\times$ the bytes at $578\times$ less
damage.

### The sub-two-bit regime is off the frontier

The sharpest way to put it is a direct dominance:

| configuration | model MB | ΔNLL |
|---|---:|---:|
| ternary weights + BF16 embedding | 616.9 | $+1.000022$ |
| **2.792-bit weights + INT4 embedding** | **305.8** | **$+0.208926$** |

The second model is **half the size and takes a fifth of the damage**. Every
sub-2-bit configuration measured here is dominated in both axes at once by a
configuration that spends more bits on weights and fewer on the embedding.
**On this model, at whole-model byte budgets, the entire low-rate weight regime
this project studies is off the efficient frontier.**

That is an uncomfortable finding and it is the correct one to report. It does
not invalidate the codec results: dim-8 VQ still beats scalar ternary at matched
bytes, GPTQ still removes 68–69% of the damage, and those comparisons are
between weight codecs at fixed coverage. What it invalidates is the implicit
premise that driving the weight rate down is the way to make *this model*
smaller.

### The qualification that decides how far this travels

**This conclusion is a function of the embedding's parameter share, and that
share is a property of small models with large vocabularies.** Here the
embedding is $254{,}279{,}680$ of $752{,}393{,}024$ parameters — $33.8\%$ — on a
$248{,}320$-token vocabulary. The 27B checkpoint has the same vocabulary at
hidden size $5120$ and does *not* tie its head, so its embedding and head are
about $2\times1.27$B of $27.78$B parameters, roughly $9\%$, and the arithmetic
inverts: there the weight codec governs the byte budget and the embedding cannot
pay for it.

So the honest reading is not "embedding precision beats weight quantization" but
"**the tensor worth attacking is whichever one holds the bytes, and at 0.8B that
is not the one this project attacked.**" The result explains §17 — full coverage
shrank the model only $2.44\times$ because $83\%$ of what remained was one
tensor — and it says the 27B direction was the right one for the codec question
even though the 27B evidence is weak.

Scope: one model, one corpus, one naive embedding quantizer, no attempt to
apply rotation or compensation to the embedding, and no measurement at 27B where
the share argument predicts the opposite conclusion.

## 22. Testing §20's replacement explanation: two predictions hold, the sharpest fails (2026-09-23)

§20 falsified the cancellation account of the unstable g64 contrast and replaced
it with an interpretation: since both codebooks there are deterministic fits
over weights, the Hessian is the only thing varying across refits, so GPTQ's
compensation statistics must themselves be noisy enough to move the contrast.
That was an interpretation of six numbers. This project has withdrawn enough
interpretations to know better than to leave it there, so it was given three
falsifiable predictions and tested against a reference Hessian estimated from
524,288 tokens of separate text, with four disjoint draws at each of four
calibration sizes.

| tokens | mean ‖ΔH‖/‖H‖ | mean ‖ΔU‖/‖U‖ | contrast range | contrast sd |
|---:|---:|---:|---:|---:|
| 16,384 | 0.30675 | 0.39711 | 0.009269 | 0.003906 |
| 32,768 | 0.20851 | 0.30326 | 0.014760 | 0.006139 |
| 65,536 | 0.14364 | 0.23676 | 0.003991 | 0.001996 |
| 131,072 | 0.11774 | 0.19541 | 0.004366 | 0.001813 |

**Prediction 1 — the Hessian's deviation should fall as $N^{-1/2}$. Holds.**
Log-log slope $-0.468$ against the $-0.5$ of finite-sample noise. The
compensation operator $\mathrm{chol}^{-1}(H)$ falls more slowly, $-0.343$,
which is what inversion does to error in the small eigenvalues, and is a reason
to expect GPTQ to be noisier than its input statistics.

**Prediction 2 — the contrast's spread should fall with $N$ too. Holds, and
tightly.** Log-log slope of the within-size standard deviation is $-0.494$,
almost exactly $-1/2$ and almost exactly the Hessian's own slope. If the
contrast's instability were driven by something other than calibration sampling
there is no reason for those two exponents to agree.

**Prediction 3 — draws with a more deviant Hessian should give a more deviant
contrast. Fails.** The within-size correlation between compensation-operator
deviation and $|$contrast deviation$|$ is $-0.142$ over 16 draws: no
relationship, and if anything the wrong sign.

### What the failure means, and why it is not a retraction

Predictions 1 and 2 are about magnitude scaling and they hold. Prediction 3 asks
whether a *scalar* measure of Hessian error predicts the contrast, and it does
not — which §18's own identity explains. The mismatch penalty is
$\operatorname{tr}[E(H_{\text{test}}-H_{\text{cal}})E^{\top}]$: it depends on how
the Hessian error is **oriented** relative to the weight error, not on how large
it is. Two draws with identical $\lVert \Delta H \rVert_F$ can land anywhere.
Prediction 3 measured the one quantity the identity says should not predict
anything, and got the answer the identity implies.

So the replacement explanation survives in a narrowed form, stated as what was
tested rather than as what is plausible: **the g64 contrast's refit-to-refit
spread scales with calibration size exactly as finite-sample Hessian noise
does, and no scalar measure of Hessian error predicts which way an individual
refit will land.** The sharper test — whether
$\operatorname{tr}[E(H_d - H_{\text{ref}})E^{\top}]$ predicts the per-draw
contrast, using the actual weight errors — needs $E$ retained per draw and was
not run.

### One thing the design surfaced that was not predicted

Calibration size does not only add noise; it moves the answer. All four draws at
65,536 tokens give a negative contrast, all four at 131,072 give a positive one,
and the 524,288-token reference gives $-0.002714$. A quantity whose *sign*
depends on calibration size in this way is not converging to a stable value over
this range, which is a stronger statement than §12's "calibration size changes
the experimental condition" and independently supports the paper's refusal to
quote this contrast as a recipe recommendation.

## 23. A model that actually runs on compressed weights (2026-09-23)

Every quality number in this project was produced by decoding packed bytes into
BF16 weights, installing those, and measuring loss. That is the right way to
measure a codec, and it is why "no runtime claim" has been in the paper's
boundary list from the beginning: at inference the model still occupied its
dense footprint, and "the packed weights would fit" was arithmetic about bytes
on disk.

All 186 non-embedding matrices are now **replaced** by modules that hold only
int16 codes, an FP16 codebook and FP16 group scales, with the Hadamard rotation
fused into the single Triton kernel of §17. The dense weights are dropped.
Dimension 4, $K=1625$, 2.792 covered bits/weight.

### Does the served model compute what the paper measured?

| model | NLL over 32,768 targets |
|---|---:|
| decoded BF16 — what every earlier number used | 3.398479 |
| compressed, served | 3.398463 |

Difference $-1.59\times10^{-5}$, 95% CI $[-4.58\times10^{-4}, +4.28\times10^{-4}]$.
**The served model is the model the paper measured**, to well inside the
evaluation's own noise. This is the check that could most easily have been
faked by reporting "close enough"; it is a paired comparison on identical
blocks against the exact weights the loss numbers came from.

### What it weighs

| model | resident | |
|---|---:|---:|
| BF16 | 1504.8 MB | |
| compressed | 783.2 MB | **1.92x** |

Measured with `torch.cuda.memory_allocated`, so this is the entire model —
including the BF16 embedding, which is the majority of what remains — and not a
payload count. It is **lower than §17's derived $2.20\times$ at the same rate**,
and the gap is exactly the storage-versus-runtime distinction §17 insisted on:
the served layout spends 2.000 bits of index per weight where the packed file
spends 1.600. A figure derived from payload bytes would have overstated the
served model by 15%.

> An earlier attempt at this table reported $0.85\times$ — the compressed model
> measuring *larger* than the dense one — because the target list still held
> references to the original `nn.Linear` modules, so their weights were never
> freed. The same class of harness contamination as §17's peak-allocation
> defect, caught the same way: by measuring the allocator instead of trusting
> the arithmetic.

### What a token costs

| model | ms per decode step |
|---|---:|
| BF16 | 85.638 |
| compressed | 99.364 |

A factor of $1.16$. Both figures are dominated by this model's Python-level
recurrent fallback — 86 ms per token for a 0.8B model is not a serving number,
and the installed `transformers` reports that the fused linear-attention path is
unavailable — so this measures the *increment* the compressed weights add to an
unoptimized decode loop, not a serving comparison. Taken with §17's isolated
projection measurement ($3.0\times$ on the GEMV once the rotation is fused), the
consistent reading is that the format costs a small multiple on the projections
and that multiple is diluted by everything else in the step.

### Scope

The embedding, the norms, the KV cache and the recurrent state stay BF16. Both
kernels are tuned only far enough to be correct. This is a model whose covered
weights are served compressed, on one GPU, at batch one, and nothing here is a
serving stack.

What it does settle: the project can no longer be accused of reporting byte
counts for a model that never ran. It ran, it weighed what the allocator says it
weighed, and it computed the same loss.
