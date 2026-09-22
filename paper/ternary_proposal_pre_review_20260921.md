# Proposal: Rate-Resolved Joint Coding, and a Ternary Qwen3.8-27B

**Date:** 2026-09-21
**Status:** Step 0 complete (§4b, mechanism revised); **Step 1 PASSED (§5b)** — Step 3 justified
**Supersedes:** `scq_proposal.md` (that direction is closed)

---

## 1. Why the previous work failed to discriminate

Two hypotheses failed, and the post-mortem points at the same cause.

At four bits per weight, on 18 of 64 layers, total quantization damage is
**+0.008393 NLL against a 3.436670 baseline — 0.24%**. Every effect we spent
months chasing was a fraction of that:

| Effect we measured | Size | Fraction of total damage |
|---|---:|---:|
| Total 4-bit damage (2-D codec) | +0.008393 | 100% |
| Best codec difference (2-D vs Lloyd scalar) | 0.000846 | 10% |
| Self-consistent calibration (best case) | 0.000585 | 7% |
| Sequential calibration | ±0.0006 | 7%, sign-unstable |
| Fisher layer protection, over uniform bits | ~0.0005 | 6%, inside noise |

Four bits is simply an easy regime for a 750M–27B model. Nothing we could have
built would have produced a large effect there, because there was no large
effect available. **The regime, not the ideas, is what made every result
marginal.**

This also explains the parity result specifically. After a decorrelating
transform, weight distributions are close to isotropic Gaussian, and that is
precisely the regime where scalar quantization is near-optimal: the
vector-over-scalar granular gain is bounded near 0.25 bit (~1.53 dB) in the
limit of infinite dimension, and about 0.17 dB for dimension 2. At four bits
that ceiling is a few percent of MSE, which is what we measured and what we
then failed to convert into held-out loss.

## 2. What Bonsai 2 27B establishes

Verified from the model repository, not the press coverage:

| Property | Value |
|---|---|
| Base model | **Qwen3.8-27B** (the checkpoint whose shards we already hold and verified) |
| Weight format | ternary {−1, 0, +1}, **g128** FP16 group scales |
| Transform | **blockwise Hadamard rotation**, folded into weights offline; matching activation transform at runtime |
| `PTQ1_0` (dense trits) | 5.95 GB, 1.75 bits/weight |
| `PQ2_0` (2-bit slots) | 7.21 GB, 2.13 bits/weight |
| F16 reference | 53.8 GB |
| Vision tower | Q8_0, 0.63 GB, loaded only for images |
| Claimed retention | 98.2% of FP16, 84.78 avg over 14 thinking-mode benchmarks (self-reported) |

Our bit accounting reproduces theirs, which is a useful check that we
understand the format. Five trits pack into one byte ($3^5=243\le256$) giving
1.600 bits/weight; one FP16 scale per 128 weights adds 0.125; total **1.725**,
matching the stated 1.75. The 2-bit-slot variant gives $2+0.125=2.125$,
matching 2.13.

Two consequences:

1. **5.95 GB means a 27B model fits on the RTX 3090 with room for KV cache.**
   That is a concrete, achievable deliverable on hardware we have.
2. Bonsai is a published reference point at the rate we care about, on our exact
   base model. That is unusually good experimental luck.

The press coverage is inconsistent (5.9 GB vs 8.6 GB footprint; 1.71 vs 1.76
bits). The repository resolves it: different variants, plus the vision tower.
We should cite the repository, not the coverage.

## 3. The hypothesis, restated so it is falsifiable

The surviving kernel of the original idea is not magnitude--phase. It is:

> **Code weights jointly in groups rather than one at a time.**

Standard vector quantization. Its value is **rate-dependent**, and that
dependence is the hypothesis:

> **H1.** The advantage of joint coding over scalar coding, at matched bits per
> weight, grows as the rate falls, and is material at ternary rates
> (1.585–2.0 bits/weight) even though it is negligible at 4 bits.

The mechanism: at 4 bits a scalar quantizer has 16 levels and can track the
marginal density closely, leaving only the small granular gain. At 1.585 bits it
has **three** levels, and a $3\times3$ product grid is a poor use of nine
codepoints in the plane. At low rate the *shape* term dominates and joint coding
has room that it does not have at high rate.

### 3.1 The question that could subsume the whole line

Before testing H1 we must answer a prior question, because a negative answer
kills the direction outright:

> **H0.** Is joint coding just a poor man's rotation?

Our pairing experiments found that *which* weights are paired matters (adjacent
vs split-half vs reverse-half changed results). That means we were exploiting
residual correlation structure. A Hadamard rotation removes exactly that
structure. If rotation plus scalar ternary captures everything our pairing
captured, then Bonsai's design already subsumes our codec and there is nothing
to add.

**This is the first thing to measure, and it is cheap.** It is the ceiling-first
discipline that this project learned the hard way: measure what is available
before building something to capture it.

## 4. Step 0 — Rate-resolved ceiling sweep (weight reconstruction only)

**No model forward pass. Reuses `run_codebook_ceiling.py` machinery.**

### Design

For each tensor in a fixed set (Qwen3.8-27B layer-0 DeltaNet QKV, layer-3
attention Q, layer-3 MLP up; plus three Qwen3.5-0.8B tensors for cross-checking):

* **Rotation:** {none, blockwise randomized Hadamard, 1024-element blocks along
  the input dimension}.
* **Rate:** bits/weight ∈ {1.585, 2.0, 2.5, 3.0, 4.0}, all with g128 FP16 scales
  so the scale overhead (0.125 bits/weight) is identical across arms.
* **Codec dimension at each rate:**
  | dim | codebook size at 1.585 bpw | comparison |
  |---|---|---|
  | 1 (scalar) | 3 levels | ternary, the Bonsai baseline |
  | 2 | 9 points | vs the $3\times3$ product grid |
  | 4 | 81 points | vs the 4-fold product |
  | 8 | 6561 points | upper-dimension probe |
  Codebooks are fitted by Lloyd/$k$-means; the product grid is the scalar
  quantizer applied independently.

### Metric and its validity

Weight MSE. Rotation is orthogonal, so
$\lVert W-\hat W' H^{\!\top}\rVert_F=\lVert WH-\hat W'\rVert_F$: error measured
in the rotated basis **is** the error in the original basis. The comparison is
therefore exactly matched, and rotation can help only by making the source
easier to quantize, not by changing the metric. This should be asserted in the
code as a test.

### Predeclared outcomes

* **Subsumption (H0 fails → stop).** If rotation alone closes ≥80% of the gap
  between unrotated scalar and unrotated dim-2 VQ, then our pairing gains were
  substantially a worse version of what rotation does. We write that up as a
  short note and stop the codec line.
* **H1 fails → stop.** If, post-rotation at 1.585 bpw, no VQ dimension beats the
  product grid by ≥15% weight MSE, or if that margin is not at least 2× the
  post-rotation margin at 4.0 bpw, the rate-dependence claim is false and the
  direction ends.
* **Proceed** only if both clear.

I want to be explicit that Step 0 is designed to kill this, not to confirm it.
Given the record, that is the right default.

### Budget

Weight-only, chunked on GPU. Estimated **60–90 minutes** including
implementation of the Hadamard transform and the dim-4/dim-8 codebook fitting.
Hard wall-clock cap in the script, as in `run_scf_phase0.py`.

## 4b. Step 0 results (2026-09-21) — gate returns STOP on the stated mechanism

Raw: `results/rate_sweep_v1/`. Six tensors across both checkpoints, 2 rotations,
4 rates, dims 1/2/4/8, matched to the bit. A self-test asserts the Hadamard is
orthonormal (error 0.0) and norm-preserving, so MSE in the rotated basis is the
error in the original basis.

### An implementation bug worth recording

The first run had every free codebook scoring *worse* than the product grid it
contains, which is impossible. Cause: the VQ arm was fitting its codebook on
un-normalized weights while the scalar arm received g128 normalization, so one
codebook had to span group scales differing by orders of magnitude. A free code
scoring worse than a constrained subset of itself is a reliable signal that the
fit failed, not a result. Fixed by fitting in normalized space and adding
empty-cluster respawn to k-means.

### Best achievable MSE, relative to the unrotated scalar baseline at each rate

| bits/weight | rotation only | VQ only | rotation + VQ | best dim |
|---:|---:|---:|---:|---:|
| 1.585 (ternary) | 0.966 | **0.757** | 0.749 | 8 |
| 2.000 | 0.962 | 0.798 | 0.784 | 4 |
| 3.000 | 0.963 | 0.744 | 0.732 | 4 |
| 4.000 | 0.967 | 0.871 | 0.853 | 2 |

### H0 — rotation does not subsume joint coding, but the two are largely redundant

Fraction of the VQ gap that rotation alone closes, at ternary:

| | dim 2 | dim 4 | dim 8 |
|---|---:|---:|---:|
| closed by rotation | 62.5% | 21.6% | 13.2% |

Rotation is worth only ~3.4% MSE on its own. It captures most of what *dim-2*
joint coding captures and little of what dim-8 captures. But combining them adds
almost nothing over VQ alone (0.757 → 0.749): they exploit overlapping
structure, and high-dimensional VQ strictly dominates. **H0 verdict: not
subsumed — but rotation becomes nearly redundant once the codec is
high-dimensional.**

### H1 — falsified as stated

The hypothesis was that the joint-coding advantage grows as rate falls. At fixed
dimension it does the opposite (post-rotation, mean over tensors):

| bits/weight | dim 2 | dim 4 | dim 8 |
|---:|---:|---:|---:|
| 1.585 | 4.9% | 13.8% | 22.5% |
| 2.000 | 8.6% | 18.5% | — |
| 3.000 | 13.3% | 23.9% | — |
| 4.000 | 11.8% | — | — |

At dim 2 the margin falls from 11.8% at 4 bits to 4.9% at ternary; at dim 4 from
23.9% at 3 bits to 13.8%. **The information-theoretic mechanism I proposed is
wrong.** The gate therefore returns STOP.

### The effect that is real, and why I am not claiming it as a pass

Gain is driven by codebook *dimension*, not by rate. What is rate-dependent is
which dimensions are affordable:

| | dim-8 codebook size | feasible? |
|---|---|---|
| ternary (L=3) | $3^8 = 6{,}561$ | yes |
| 4-bit (L=16) | $16^8 = 4.3\times10^9$ | no |

So the best *achievable* gain is 24.3% at ternary against 12.9% at 4 bits —
about the 2x the gate asked for, but arising from computational feasibility
rather than from source statistics. That is arguably a better reason, since it
is an engineering fact rather than a subtle distributional claim.

I am recording this as a **failed gate with a surviving alternative mechanism**,
not as a pass. Reinterpreting a predeclared criterion after seeing the data is
precisely the move this project has been burned by, and the decision to continue
on the revised mechanism should be taken explicitly rather than absorbed into a
verdict line.

### A cost the sweep did not charge

Codebook storage was excluded from the rate matching. Charging it:

| tensor | dim-4 (K=81) | dim-8 (K=6561) |
|---|---:|---:|
| Qwen3.5-0.8B, 6144x1024 | +0.1% | **+8.4%** on top of ternary |
| Qwen3.8-27B, 10240x5120 | +0.0% | **+1.0%** on top of ternary |

Dim-8 is viable at 27B scale and marginal at 0.8B scale unless the codebook is
shared across layers. Any Step 1 comparison must charge this, or it repeats the
control-strength error the last paper documents.

## 5. Step 1 — Does it convert to task loss at ternary rate?

Conditional on Step 0.

The central lesson of Part I of the current paper is that weight-MSE rankings
did **not** predict held-out loss at 4 bits. So a Step 0 pass is necessary, not
sufficient.

* Model: Qwen3.5-0.8B (fast loop), all 18 DeltaNet input projections.
* Arms, all at matched bits/weight including scale overhead: rotated scalar
  ternary (Bonsai-style baseline), rotated dim-2 VQ, rotated dim-4 VQ,
  unrotated versions of each, BF16 reference.
* Evaluation: **complete WikiText-2 validation stream, 261,248 targets.** No
  screening slices for anything that will be reported — this session produced
  three reversals from reduced slices, and the rule is now full-stream or it
  does not go in a document.
* Paired block bootstraps against BF16 and head-to-head.

**Gate:** rotated VQ must beat rotated scalar ternary with a paired interval
excluding zero. Point estimates do not count.

**Budget:** ~3 hours.

## 5b. Step 1 results (2026-09-21) — PASSES, decisively

Raw: `results/ternary_task_v1/`. Qwen3.5-0.8B, all 18 DeltaNet input
projections, **complete WikiText-2 validation stream (261,248 targets)**, no
activation weighting on any arm, codebook storage charged in bits/weight. The
rotation round-trip is asserted (relative error $2\times10^{-7}$).

| Arm | bits/wt | ΔNLL vs BF16 | ppl | 95% CI |
|---|---:|---:|---:|:--|
| scalar ternary g128 | 1.710 | +0.441267 | 48.32 | [+0.435534, +0.446983] |
| scalar ternary g128 + rotation *(Bonsai-style)* | 1.710 | +0.358803 | 44.50 | [+0.354169, +0.363503] |
| scalar ternary g64 + rotation | 1.835 | +0.353497 | 44.26 | [+0.348844, +0.358319] |
| dim-4 VQ g128 + rotation | 1.711 | +0.277004 | 41.00 | [+0.272871, +0.281136] |
| **dim-8 VQ g128 + rotation, shared codebook** | **1.717** | **+0.221516** | **38.79** | [+0.217679, +0.225362] |
| dim-8 VQ g128 + rotation, per-tensor codebook | 1.843 | +0.209203 | 38.32 | [+0.205581, +0.212775] |
| dim-8 VQ g128, no rotation, shared codebook | 1.717 | +0.246314 | 39.76 | [+0.242474, +0.250071] |

**Headline, byte-matched to 0.4%:** dim-8 vector quantization cuts ternary PTQ
damage by **38.3%** against rotated scalar ternary, for +0.007 bits/weight.
Paired difference $-0.137287$, CI $[-0.141188,-0.133457]$ — far outside noise.
This is the first result in the project that is both large and survives a
matched-strength control.

### Four secondary findings

1. **Rotation is worth much more on task loss than weight MSE suggested.** Step 0
   put it at ~3.4% MSE; here it is $-0.082464$ NLL on scalar ternary (an 18.7%
   damage reduction) and still $-0.024798$ on dim-8 VQ. Weight MSE *understated*
   it, and rotation and VQ are **not** redundant on the metric that matters —
   the opposite of what Step 0's MSE analysis implied. Another instance of
   weight error mispredicting task damage, this time in the helpful direction.
2. **Codebook dimension is ~200x more byte-efficient than scale granularity.**
   Spending 0.006 bits/weight to go from dim-4 to dim-8 buys 0.0555 NLL
   (8.4 NLL per bit/weight). Spending 0.125 bits/weight to halve the group size
   buys 0.0053 NLL (0.042 NLL per bit/weight). For anyone tuning a ternary
   format, this is the actionable result.
3. **A shared codebook is nearly as good as per-tensor and much cheaper.**
   Per-tensor wins by 0.0123 NLL but costs 0.126 bits/weight — the same poor
   return as finer scales. One codebook for all 18 tensors is the right default.
4. **Absolute damage is still large.** +0.2215 NLL is perplexity 31.08 → 38.79,
   a 25% degradation. Our pipeline is deliberately simple: no activation
   weighting, no GPTQ-style error compensation, no per-channel treatment. This
   is a *codec* comparison under an identical simple pipeline, **not** a
   competitive ternary system, and it is nowhere near Bonsai's claimed 98.2%
   retention. We compare codecs to each other, not to their product.

### Gate

The predeclared gate was a paired interval excluding zero against rotated scalar
ternary. It excludes zero by roughly 35 standard errors. **Step 3 is justified.**

## 6. Step 2 — An external baseline we did not produce

Bonsai publishes GGUF artifacts and a benchmark table. Running our own
evaluation on their artifact is worth doing regardless of our codec, because it
gives us an external anchor and tells us whether the format is as good as
claimed.

* Measure WikiText-2 perplexity of `PTQ1_0` and `PQ2_0` against an F16 or
  Q8_0 reference of the same base model, under our own protocol.
* **Caveat to state loudly:** their 98.2% figure is over 14 thinking-mode
  benchmarks. Perplexity is a *different axis*. We would not be reproducing
  their number, and must not present it as such.
* **Dependency risk:** `PTQ1_0` and `PQ2_0` are custom quant types. Stock
  llama.cpp may not read them; their runtime also applies the Hadamard
  activation transform. If it does not load with available tooling, we drop this
  step rather than spending days on it.

**Budget:** half a day, abandoned quickly if the artifact does not load.

## 7. Step 3 — Our own ternary Qwen3.8-27B

The deliverable you asked for. Conditional on Steps 0–1.

* 55.56 GB BF16 language model → ~6 GB at 1.725 bits/weight, which **fits the
  3090 resident** with room for KV cache.
* Quantization must be **streamed shard by shard**: 55.56 GB does not fit in
  60 GB RAM alongside working copies. One tensor at a time, write packed output
  incrementally, never hold the full model.
* Calibration activations require a forward pass on the CPU-offloaded BF16
  model. Our existing gate measured ~23 min per validation pass at batch 8 with
  layers 0–20 on GPU; a calibration pass over 64 windows is far cheaper.
* Evaluation: WikiText-2 perplexity full stream, against the BF16 reference we
  can already run offloaded.

**What we are not doing:** we will not out-engineer Bonsai's runtime. They have
packed ternary kernels and a fused Hadamard activation transform. Our artifact
would be a *research* quantization evaluated by perplexity, decoded to BF16 for
inference unless Step 4 happens. This must be stated in any writeup.

**Budget:** 2–3 days including debugging.

## 8. Step 4 — Packed kernel (optional, only if Step 3 produces something)

The decode path for a dim-2 VQ at ternary rate is a 9-entry lookup table —
trivially small, unlike the 256-entry table at 4 bits and unlike the per-pair
sine/cosine that made our polar kernels 3–4.5× slower than BF16. This is the
first time in the project that a fused kernel looks straightforward. It is still
optional and last.

## 9. What makes this publishable even if H1 fails

This is the part I would push hardest. Our infrastructure is now well suited to
a measurement the field does not have in one place:

> **A rate-resolved map of how much joint coding buys over scalar coding for LLM
> weights, before and after rotation, from 4 bits down to ternary, with the
> unconstrained VQ ceiling at each rate.**

That is a clean contribution that does not require beating anyone. It answers
"is rotation or vector quantization the better use of engineering effort at low
rate?", which is a live question given that QuaRot/QuIP#-style rotation and
VQ-style codebooks are usually presented as alternatives rather than compared at
matched rate. A null result — "rotation subsumes joint coding at every rate" —
is a useful, citable finding, and Step 0 produces the curve either way.

Combined with the two negative results already written up, the honest framing of
the whole project becomes: *what actually determines quantization damage, and
what does not.*

## 10. Risks and things I expect to go wrong

1. **Ternary PTQ without training may simply not work for us.** Bonsai reports
   98.2% retention from what the coverage describes as post-training
   quantization. If they are also doing distillation or partial retraining and
   not saying so, the bar is unreachable by PTQ alone and our 27B artifact will
   look bad by comparison. We should evaluate our result against *our own* BF16
   reference, not against their headline.
2. **Rotation may capture everything.** That is H0, and it is a real
   possibility. Step 0 answers it in an afternoon.
3. **Weight MSE may again fail to convert.** It did at 4 bits. Step 1 exists
   precisely because Step 0 is not sufficient.
4. **Custom GGUF types may be unreadable** with available tooling (Step 2).
5. **Memory.** 55.56 GB against 60 GB RAM in Step 3 is the tightest constraint
   in the plan and needs streaming from the first line of code, not as a fix.

## 11. Summary of the decision points

| Step | Question | Cost | Kill criterion |
|---|---|---|---|
| 0 | Is joint coding just a poor man's rotation? Does its value grow as rate falls? | 60–90 min | rotation closes ≥80% of the gap, **or** post-rotation VQ margin at 1.585 bpw <15% or <2× the 4-bit margin |
| 1 | Does the weight-MSE gap convert to held-out loss at ternary rate? | ~3 h | paired interval vs rotated scalar ternary includes zero |
| 2 | Is Bonsai's format as good as claimed, on our axis? | ½ day | artifact does not load — drop |
| 3 | Can we build a ternary Qwen3.8-27B that fits the 3090? | 2–3 days | perplexity degradation unacceptable vs our own BF16 reference |
| 4 | Fused 9-entry LUT kernel | open | optional |

**Recommended order: 0 → (1, 2 in parallel) → 3 → 4.** Step 0 first, alone, and
we look at the curve before committing to anything else.
