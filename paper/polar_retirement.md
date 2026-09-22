# Retiring the polar weight code, and where the headroom actually is

**Date:** 2026-09-20. **Supersedes:** the polar-versus-real-4 conclusions in
`small_screen_findings.md`, `model_selection_and_failure_analysis.md`, and
`phase_weights.md`.

## Summary

The reported advantage of the 3-bit-magnitude + 5-bit-phase pair code over an
equal-byte real 4-bit control was an artifact of the control. Three
measurements establish this and point to a replacement.

## 1. The control was not matched in strength

Every earlier comparison used a real-4 control whose codebook is a **uniform**
16-level grid on `[-1, 1]`. The polar code's effective codebook is not uniform:
a radius grid crossed with a phase grid produces annular sectors, so the polar
code was implicitly getting a shape adaptation the control was denied.

We added `quantize_real4_fitted`: the same scalar code with a per-tensor
Lloyd-Max codebook and the identical per-row FP32 scale. The extra storage is
16 FP32 per tensor — under `1e-4` bits per weight for these matrices.

On the full WikiText-2 validation stream (261,248 targets, all 18 Qwen3.5-0.8B
DeltaNet input projections quantized simultaneously, ΔNLL vs BF16, lower better):

| Method | ΔNLL | 95% CI |
|---|---:|:--|
| real-4, uniform codebook *(earlier control)* | +0.022571 | [+0.021615, +0.023511] |
| polar 3+5, decoupled rounding *(earlier method)* | +0.019776 | [+0.018821, +0.020732] |
| polar 3+5, exact NN + fitted radii | +0.016016 | [+0.015169, +0.016830] |
| **real-4, Lloyd codebook** *(matched control)* | **+0.014778** | [+0.013951, +0.015582] |
| paired 2-D codebook (256 points) | +0.009789 | [+0.009125, +0.010448] |

Polar beats the uniform control, exactly as reported before. It loses to the
matched control. Raw data: `results/scq_qwen08b_v1/results.json`.

**Caveat on comparability.** This run selects each layer's pairing by
activation-weighted error, while the earlier draft selected by calibration NLL,
which is a stronger rule. The ladder above is internally consistent — every
codec gets the same selection rule — but the earlier draft's figure (+0.009333)
should not be read off against it directly.

### 1b. The same conclusion under the earlier draft's own selection rule

We therefore re-ran with the earlier draft's protocol: per-layer candidate
selection by calibration NLL with all other weights at BF16, then cumulative
application. Every codec gets that protocol.

| Method (NLL-selected, 16 windows) | ΔNLL | 95% CI |
|---|---:|:--|
| real-4, uniform codebook | +0.024164 | [+0.023088, +0.025239] |
| polar 3+5 *(earlier draft's method)* | +0.015902 | [+0.015006, +0.016825] |
| real-4, Lloyd codebook | +0.008539 | [+0.007662, +0.009403] |
| paired 2-D codebook | +0.009145 | [+0.008325, +0.009950] |
| **paired 2-D, activation-weighted** | **+0.007693** | [+0.007029, +0.008398] |

Paired head-to-head differences under this one protocol:

| Comparison | ΔNLL | 95% CI | |
|---|---:|:--|:--|
| polar − real-4 Lloyd | +0.007364 | [+0.006332, +0.008376] | excludes 0 |
| polar − paired 2-D (weighted) | +0.008210 | [+0.007205, +0.009186] | excludes 0 |
| paired 2-D − same, weighted | +0.001452 | [+0.000556, +0.002327] | excludes 0 |
| real-4 Lloyd − paired 2-D (weighted) | +0.000846 | [−0.000169, +0.001822] | **includes 0** |

So polar is significantly behind both alternatives, activation weighting
significantly improves the 2-D codec, and the 2-D codec's advantage over a
well-tuned scalar code is **not** significant. The defensible claim is
asymmetric: 2-D reaches the scalar code's regime and may exceed it; polar does
not reach either.

The ranking is unchanged, and the gap widens: under its own selection rule the
polar code loses to the matched scalar control by +0.007364 NLL. The
Lloyd-fitted scalar control at +0.008539 is also **better than the earlier
draft's published polar figure of +0.009333**, so the headline comparison
reverses even taken at face value.

Selection rule matters for both codecs — polar improves from +0.019776 to
+0.015902 when moved from weighted-error to NLL selection — but it does not
change which codec wins. Raw data: `results/selection_fairness_v1/summary.md`.

### 1c. Under the earlier draft's *exact* calibration setup, nothing is decidable

The earlier draft used **4** calibration windows with seed 62016. Reproducing
that exact setup and granting it to every codec:

| Method (NLL-selected, 4 windows, seed 62016) | ΔNLL |
|---|---:|
| real-4, uniform codebook | +0.025256 |
| polar 3+5 *(earlier draft's method)* | +0.011358 |
| paired 2-D codebook | +0.010689 |
| **real-4, Lloyd codebook** | **+0.007609** |

On that seed the matched control wins comfortably. But repeating with two more
calibration seeds shows the 4-window protocol does not actually discriminate:

| Seed | polar 3+5 | real-4 Lloyd | Winner |
|---|---:|---:|:--|
| 62016 *(earlier draft's)* | +0.011358 | +0.007609 | Lloyd |
| 1 | +0.010515 | +0.009857 | Lloyd |
| 2 | +0.010960 | +0.014196 | **polar** |
| **mean ± sd** | **0.010945 ± 0.000422** | **0.010554 ± 0.003349** | tied |

Two things follow, and the second is a caution against over-reading this
protocol.

1. **The published figure was a fortunate draw.** All three of our polar seeds
   land in [+0.010515, +0.011358]; the published +0.009333 is better than every
   one of them. With 512 calibration tokens the per-layer selector is a
   high-variance procedure and that number sits outside the spread we could
   reproduce.
2. **At this calibration size the two codecs are statistically tied.** The
   matched control has an eight-times larger seed-to-seed spread, because its
   clip-factor search is a finer decision than polar's four-way pairing choice
   and 512 tokens cannot resolve it. On seed 2 it loses outright.

So this protocol does **not** support the conclusion on its own. The conclusion
rests on the 16-window comparisons, where the statistics are adequately
estimated and the control wins decisively under both selection rules:

| Protocol | polar | real-4 Lloyd | Discriminates? |
|---|---:|---:|:--|
| weighted-error selection, 16 windows | +0.019776 | +0.014717 | yes, Lloyd |
| NLL selection, 16 windows | +0.015902 | +0.008539 | yes, Lloyd |
| NLL selection, 4 windows (3 seeds) | 0.010945 ± 0.000422 | 0.010554 ± 0.003349 | no |

Raw data: `results/selection_fairness_v1/`, `results/selection_seed*_w4/`.

## 2. Repairing polar's assignment rule changes almost nothing

The original `polar35` rounds magnitude and phase independently:

```
p = round(angle(z) * 32 / 2pi)      # nearest phase
m = round(|z| * 7 / s)              # nearest magnitude of |z|
```

That is not nearest-neighbour assignment in the 8x32 codebook. For a decoded
phase ray `theta_p`,

```
|z - g_m e^{i theta_p}|^2 = r^2 sin^2(d) + (g_m - r cos(d))^2,   d = theta - theta_p
```

so the optimal radius level is nearest the **projection** `r cos(d)`, not
nearest `r`. We implemented exact nearest-neighbour search over all 256
codepoints. It reduced weight MSE by **0.035%** (0.032-0.036% across the
three tensors).

The reason is computable, and it retrospectively explains a long run of null
results in this project. With five phase bits, `|d| <= pi/32`, hence
`1 - cos(d) <= 4.8e-3`. The projection correction is at most half a percent of
the radius, while one step of the three-bit radius grid is about 14% of the row
scale — so the correction essentially never changes which radius level wins.

Every phase-side refinement attempted in this project — learned global phase
offsets, activation-weighted phase choice, output-error phase selection — was
tuning a quantity that is bounded to be small. That is why they all produced
sub-noise, sign-unstable effects across layers and corpora. **The binding
constraint is the radius grid, not the phase grid.** Consistent with this, a
sweep over bit splits at fixed 8 bits/pair finds `8x32` optimal on every tensor,
with `4x64` roughly 3x worse and `2x128` roughly 14x worse.

Raw data: `results/polar_codec_ablation_v1/summary.md`.

## 3. The polar family is structurally capped; pairing is not

We fitted an **unconstrained 256-point 2-D codebook** by k-means to the weight
pairs. Any scheme that pairs two real weights and spends 8 index bits on the
pair — polar, Cartesian, rotated — is a constrained special case, so this
lower-bounds the whole family at this budget.

Weight MSE relative to the Lloyd-fitted real-4 control (below 1.000 beats it):

| Checkpoint | Tensor | real4-Lloyd MSE | real4-unif. | polar-legacy | polar-fitted | vq2d |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | DeltaNet QKV L0 | 3.252e-06 | 1.308 | 1.194 | 1.065 | **0.759** |
| Qwen3.5-0.8B | DeltaNet QKV L8 | 3.135e-06 | 1.214 | 1.156 | 1.064 | **0.793** |
| Qwen3.5-0.8B | DeltaNet QKV L16 | 5.039e-06 | 1.240 | 1.165 | 1.059 | **0.795** |
| Qwen3.8-27B | DeltaNet QKV L0 | 2.563e-06 | 1.186 | 1.162 | 1.073 | **0.818** |
| Qwen3.8-27B | Attention Q L3 | 3.317e-06 | 1.217 | 1.154 | 1.060 | **0.791** |
| Qwen3.8-27B | MLP up L3 | 1.182e-06 | 1.169 | 1.163 | 1.081 | **0.817** |

Fully optimized polar stays 6–8% **above** the matched scalar control; a free
2-D codebook sits 18–24% **below** it. The pairing idea was right; the polar
parameterization was wrong.

Raw data: `results/codebook_ceiling_v1/summary.md`.

## Practical consequence

Replacing the polar grid with a learned 2-D codebook also removes the obstacle
that defeated the packed kernels. Decoding a polar index needs a sine and a
cosine per pair, which is why packed decode-plus-GEMM ran 3–4.5x slower than
resident BF16 and the fused variant was worse still at batch. Decoding a 2-D
codebook index is a 256-entry lookup (2 KB) that fits in shared memory. That
kernel has **not** been written or benchmarked — the claim here is that the
known blocker is removed, not that a speedup exists.

## What carries forward

- Joint quantization of weight pairs: **keep**.
- Magnitude/phase parameterization: **drop**.
- Per-layer pairing selection: keep, but apply the same rule to every codec.
- The quantum framing: drop. A complex multiplication by `rho e^{i phi}` is the
  real 2x2 rotation block `rho [[cos,-sin],[sin,cos]]`. No quantum hardware was
  used and no quantum advantage is claimed.
- The self-consistency gap identified in `scq.py` is **independent of codec** and
  is the direction with the most remaining headroom; see `architect.md`.
