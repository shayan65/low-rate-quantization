# Proposal: Self-Consistent Quantization, reformulated

**Date:** 2026-09-21
**Status:** Phase 0 passed (§3b); **Phase 1 failed (§4b) — recommend stopping this direction**

## 1. What changed: the forward-only version is prior art

Before proposing further work I checked whether the gap we identified is already
solved. It is.

* **GPTAQ** (arXiv [2504.02692](https://arxiv.org/abs/2504.02692), 2025),
  "asymmetric calibration", explicitly targets accumulated error from prior
  layer quantization and always matches the quantized layer's output to the
  *full-precision* model's output. Closed-form, **one pass**, ~20 lines on top
  of GPTQ.
* **CoreQ** (arXiv [2602.05902](https://arxiv.org/abs/2602.05902), 2026),
  "mismatch correction", states the problem in our words — "errors from earlier
  quantized layers alter the inputs received by later layers, causing the
  activations to deviate from those of the full-precision model" — and solves it
  with a closed-form coefficient. Also **one pass**, learning-free.

There is also a structural reason one pass suffices, which we should have
derived before building an iteration:

> In a feedforward stack, the input statistics of layer $\ell$,
> $H^{xx}_\ell = \mathbb{E}[x_\ell x_\ell^\top]$, depend only on layers
> $1..\ell-1$. Quantizing in topological order while propagating the quantized
> activations forward therefore reaches the fixed point in **one sweep**. The
> dependency graph is acyclic, so there is nothing for an iteration to do.

Our v2 result is consistent with this: the gains were real but small
(−0.000585 NLL for the 2-D codec), which is about what you would expect from
refitting codebooks on slightly better statistics rather than from solving a
genuinely coupled problem.

**Conclusion: the forward-only SCF loop should not be the contribution.** It is
a more expensive route to something a one-pass method already gets, and two
published methods already get it. This also means the current paper's SCF
motivation needs correcting — see §6.

## 2. The reformulation: make the coupling bidirectional

The fixed point becomes non-trivial exactly when the per-layer objective depends
on layers **downstream** as well as upstream.

Expand the end-to-end calibration loss to second order in the output
perturbation $\Delta y_\ell = \Delta W_\ell x_\ell$ of each layer. With the
Gauss-Newton/Fisher approximation to the output Hessian,

$$
\mathcal{L}(\hat W) - \mathcal{L}(W) \;\approx\; \tfrac12 \sum_\ell \sum_j
S_{\ell,j}\; \mathbb{E}\big[(\Delta w_{\ell,j}^\top x_\ell)^2\big],
\qquad
S_{\ell,j} = \mathbb{E}\!\left[\left(\frac{\partial \mathcal{L}}{\partial y_{\ell,j}}\right)^{\!2}\right].
$$

Per layer this factorizes into a **forward** term and a **backward** term:

$$
\text{cost}_\ell \;=\; \sum_j S_{\ell,j}\; \Delta w_{\ell,j}^\top H^{xx}_\ell\, \Delta w_{\ell,j}
$$

* $H^{xx}_\ell$ depends on layers $1..\ell-1$ — **upstream**.
* $S_{\ell,j}$ depends on layers $\ell+1..L$ — **downstream**, because the
  gradient at layer $\ell$ is backpropagated through every later layer, and
  those are quantized too.

The dependency graph is now **cyclic**: $Q_1$ depends on $Q_2 \ldots Q_L$ through
$S$, and $Q_L$ depends on $Q_1 \ldots Q_{L-1}$ through $H^{xx}$. No topological
order exists, so no single sweep in either direction satisfies both conditions.
**Iteration is not an implementation choice here; it is forced.**

This is the Kohn-Sham situation proper. The reason DFT must iterate is that the
effective potential depends on the density *everywhere*, not only upstream.
GPTAQ and CoreQ correct the upstream half in closed form; nothing we found
addresses the downstream half, because a layer-local output-matching target
assumes all output errors matter equally, which is exactly what $S_\ell$ denies.

### Where $S$ actually bites

An important caveat we should state up front rather than discover later. If a
codec decides each output row independently (per-row scale, per-row assignment),
then scaling row $j$'s objective by a constant $S_j$ **does not change that
row's optimal code**. $S$ only changes decisions that are *shared* across rows
or layers:

1. **Per-tensor codebook fitting** — our Lloyd radius grid and 256-point 2-D
   codebook are fitted tensor-globally, so $S$ reweights which rows dominate the
   fit. Moderate lever.
2. **Cross-layer bit allocation / mixed precision** — which layers get 3, 4, or
   5 bits. This is where the coupling is unavoidable and the payoff is largest.
3. **Which layers to quantize at all.**

So the headline application is **self-consistent mixed-precision allocation**,
not per-row rounding. That is a real, practical problem, and the one where a
one-pass method provably cannot be correct.

### Bonus: it attacks this project's oldest unexplained result

Every phase of this project found that layer-local weight MSE fails to predict
task NLL. The missing factor in the layer-local objective is precisely
$S_\ell$ — downstream sensitivity. If the reformulation is right, $S$-weighted
error should predict task NLL substantially better than weight MSE does. That
is independently checkable and is Phase 1's main diagnostic.

## 3. Phase 0: a 20-minute falsification gate

Two measurements, both cheap, both able to kill the program. Run **before** any
further building. Hard wall-clock cap enforced in the script.

### Test A — is the forward-only loop redundant? *(predicted: yes)*

Quantize the same 18 DeltaNet projections four ways and evaluate on a reduced
validation slice (256 contiguous blocks = 32,768 targets, ~1/8 the cost of the
full stream):

| Arm | Statistics source |
|---|---|
| `one_shot` | BF16 model (current practice) |
| `sequential` | quantize in layer order, recompute statistics with earlier layers already quantized — one pass |
| `scf_forward` | our v2 iterative loop |
| `bf16` | reference |

**Read:** if `sequential` $\approx$ `scf_forward`, the forward-only loop is
confirmed redundant and we drop it, as §1 predicts. If `scf_forward` is
meaningfully better, §1's acyclicity argument is wrong somewhere and we stop to
find out why — that would itself be worth knowing.

### Test B — is there a downstream residual to solve at all? *(the real gate)*

No quantization decisions needed; pure measurement.

1. Measure $S_\ell = \mathbb{E}[(\partial\mathcal{L}/\partial y_\ell)^2]$ on the
   BF16 model (one forward+backward over 64 calibration windows).
2. Measure $S_\ell$ again with all 18 projections quantized.
3. Report the relative residual $\|S^{q}_\ell - S^{fp}_\ell\| / \|S^{fp}_\ell\|$
   per layer, alongside the forward residual
   $\|H^{q}_\ell - H^{fp}_\ell\| / \|H^{fp}_\ell\|$ we already measured at
   0.5–2.2%.

**Kill criterion:** if the backward residual is the same order as the forward
one (≲2%), the downstream coupling is negligible, iteration buys nothing, and
**we abandon the SCF direction entirely** and write up §1 as a short negative
note. If it is substantially larger (say >10%), there is a genuine coupled fixed
point and Phase 1 is justified.

I want to be explicit that Test B is designed to kill the idea, not to confirm
it. Given how this project has gone, that is the right default.

### Budget

| Step | Est. |
|---|---|
| model load + tokenize | ~2 min |
| Test A: 4 arms × (quantize + 256-block eval) | ~8 min |
| Test B: 2 forward+backward sweeps over 64 windows | ~2 min |
| slack | ~8 min |
| **cap** | **20 min, enforced** |

## 3b. Phase 0 results (2026-09-21, 277s wall clock of the 1200s cap)

Raw: `results/scf_phase0_v1/`. Qwen3.5-0.8B, 2-D codec, all 18 DeltaNet input
projections, 32,768 WikiText-2 validation targets, 32 fit + 32 guard windows.

### Test A: the forward-only loop is not merely redundant -- it is worse

| Arm | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| BF16 | 3.233953 | — | — |
| `one_shot` | 3.245136 | +0.011184 | [+0.009132, +0.013299] |
| **`sequential`** | **3.242358** | **+0.008406** | [+0.006635, +0.010215] |
| `scf_forward` | 3.246586 | +0.012633 | [+0.010697, +0.014610] |

The one-pass `sequential` arm beats `one_shot` by 0.002778 and beats our
iterative loop by **0.004228** NLL. The prediction in §1 was that iteration
would be redundant; the measurement is stronger than that, and the mechanism is
clear in hindsight:

* `sequential` uses, for each layer, exactly the statistics that layer will see
  in the deployed model. Because the dependency graph is acyclic, those
  statistics are *correct*, not approximate.
* Our loop mixes a **global** statistics vector toward `H[Q]` with `alpha < 1`.
  The mixed statistic corresponds to **no actual model state** — it is an
  interpolation between two different networks — and all 18 layers are refit
  against it simultaneously. Damping, which is what makes SCF stable in DFT, is
  precisely what makes it wrong here, because the exact answer is available in
  one sweep and damping moves away from it.

This retires the forward-only SCF loop. It also means `sequential` propagation
is a free improvement we were not using anywhere in the existing results.

### Test B: the downstream residual is real

| Quantity | Pooled relative residual |
|---|---:|
| forward, `E[x^2]` | 0.8719% |
| backward, `E[(dL/dy)^2]` | **7.3318%** |
| **ratio** | **8.41x** |

Predeclared threshold was 2x. **Verdict: PROCEED.**

Per layer:

| Layer | backward `S` | forward `H` | ratio |
|---|---:|---:|---:|
| L0 | 6.95% | 0.000% | — |
| L1 | 30.56% | 0.336% | 90.8x |
| L2 | 13.51% | 0.549% | 24.6x |
| L4 | 12.83% | 0.567% | 22.6x |
| L5 | 8.13% | 0.610% | 13.3x |
| L6 | 7.92% | 0.423% | 18.7x |
| L8 | 18.83% | 0.737% | 25.6x |
| L9 | 8.03% | 0.983% | 8.2x |
| L10 | 5.44% | 0.517% | 10.5x |
| L12 | 7.93% | 0.579% | 13.7x |
| L13 | 6.57% | 0.832% | 7.9x |
| L14 | 4.05% | 0.947% | 4.3x |
| L16 | 12.18% | 0.825% | 14.8x |
| L17 | 17.87% | 0.923% | 19.4x |
| L18 | 28.05% | 1.072% | 26.2x |
| L20 | 6.02% | 1.082% | 5.6x |
| L21 | 21.48% | 1.491% | 14.4x |
| L22 | 8.12% | 1.582% | 5.1x |

Two structural features are worth keeping. The forward residual grows roughly
monotonically with depth (0.336% at L1 to 1.582% at L22) — it is accumulated
upstream error, exactly as GPTAQ and CoreQ describe. Layer 0's forward residual
is **identically zero**, because its input is upstream of every quantized layer;
its backward residual is 6.95%, because everything downstream of it is
quantized. That single row is the asymmetry in miniature: the first layer has no
upstream coupling at all and substantial downstream coupling, and a
forward-only method has nothing to say about it.

The backward residual is also not monotone in depth — it peaks at L1 (30.6%),
L18 (28.1%) and L21 (21.5%) — so it carries information that no depth-ordered
heuristic reproduces. Quantizing these 18
projections perturbs the downstream loss sensitivity roughly an order of
magnitude more than it perturbs the upstream activation statistics — and the
upstream half is the half GPTAQ and CoreQ already solve.

### What Phase 0 changes about the algorithm

The two results together specify the right design, which is *not* what we built:

> Solve the forward half **exactly, by sequential propagation, one sweep**.
> Iterate **only** over the backward half, which is the genuinely cyclic part.

So the loop becomes: given current `S`, do one sequential forward sweep
quantizing every layer in topological order under the metric `S_j * dw^T H^xx dw`;
then recompute `S` on the resulting quantized model; mix and repeat. Damping now
applies only to the quantity that actually needs it.

## 4. Phase 1 (justified by Test B)

1. **Diagnostic first, no new quantizer.** Does `S`-weighted output error rank
   layers by actual task-NLL damage better than plain weight MSE does? This is a
   correlation study over the 18 layers using data we can already produce, and
   it is the cheapest possible test of the central premise. It also speaks
   directly to this project's oldest unexplained result.
2. **Sequential-inside-SCF**, as specified in §3b: exact forward sweep, iterated
   backward statistics, damped mixing on `S` only, existing energy guard on a
   disjoint set.
3. **Baselines that matter now.** `sequential` (which beat everything in Test A)
   and a local GPTAQ-style asymmetric-calibration implementation. The `one_shot`
   arm is no longer an interesting comparison.

## 4b. Phase 1 results (2026-09-21) — the reformulation also fails

Raw: `results/fisher_diagnostic_v1/`, `results/protect_layer_v2/`.

### 1.1 Fisher-weighted error does not rank layers by task damage

One layer quantized at a time, all others BF16, 65,536 validation targets.
10 of 18 layers have a ΔNLL interval excluding zero.

| Predictor | Spearman | Pearson | Pearson w/o L0 |
|---|---:|---:|---:|
| weight MSE *(what earlier phases used)* | −0.218 | −0.202 | +0.072 |
| diagonal activation-weighted | +0.020 | −0.061 | +0.146 |
| exact output error (forward only) | +0.005 | −0.057 | +0.145 |
| Fisher: forward × backward | +0.015 | **+0.898** | **+0.181** |

The headline Pearson of 0.898 is an artifact of a single leverage point.
Excluding L0 it collapses to 0.181, and Spearman goes to −0.169. **No predictor,
including Fisher, has rank skill across the other 17 layers.** The central
premise of §2 — that $S$ is the missing factor that makes local error predict
task damage — is not supported.

Two honest caveats on the test's power. Eight of 18 layers have ΔNLL intervals
covering zero, so the comparison among the small-damage layers is weak. And
three layers (L4, L22, and marginally L12) are *significantly negative* —
quantizing them improves validation NLL — which no unsigned error surrogate can
ever predict. That caps the achievable correlation regardless of the metric.

### The one real finding: the dominant layer is the one weight MSE calls safest

| | rank of L0 among 18 |
|---|---|
| actual ΔNLL damage | **1st (most damaging)** |
| Fisher | **1st (highest)** |
| weight MSE | **1st from the bottom (lowest)** |

L0 alone accounts for **58.2%** of the summed single-layer damage. Fisher
identifies it; weight MSE ranks it as the safest tensor in the model. This is a
crisp mechanism for the project's oldest observation, even though it did not
generalize into a usable ranking.

### 1.2 But exploiting it does not beat simply spending the bytes

If one tensor may stay BF16, which one? Costs 1.167x the all-18 payload, so the
honest controls are a larger codebook applied uniformly, bracketing that cost.

| Arm | ΔNLL | 95% CI | recovered | payload |
|---|---:|:--|---:|---:|
| all_18 | +0.008883 | [+0.007492, +0.010293] | — | 1.000x |
| protect L0 (Fisher's pick) | +0.003759 | [+0.002719, +0.004817] | 57.7% | 1.167x |
| protect L20 (weight MSE's pick) | +0.008609 | [+0.007244, +0.009991] | 3.1% | 1.167x |
| protect L10 (arbitrary) | +0.009002 | [+0.007689, +0.010343] | −1.3% | 1.167x |
| **all 18 at k=512 (4.5 bit)** | +0.004881 | [+0.003912, +0.005861] | 45.1% | 1.125x |
| **all 18 at k=1024 (5 bit)** | **+0.002877** | [+0.002027, +0.003713] | **67.6%** | 1.250x |

Fisher's pick beats weight MSE's pick enormously (57.7% vs 3.1%) — but that is a
comparison against a bad alternative. Against the right one, the uniform-bits
frontier interpolated to the same 1.167x payload recovers **52.6%**, so
protecting L0 is worth **+5.1 percentage points**, against a CI half-width of
**11.8 points**. It is inside the noise.

**Protecting the Fisher-identified layer is not better than spending the same
bytes uniformly.** This is the same trap the polar work fell into — beating a
weak control — caught this time by adding the matched-byte arm before drawing a
conclusion.

### Recommendation: stop the SCF/Fisher direction

Phase 0 killed the forward-only loop. Phase 1 killed the bidirectional
reformulation: the metric has no ranking skill, and its one correct call yields
no advantage over uniform bit allocation at matched bytes. Phase 2 as written
depended on graded cross-layer allocation, which 1.1 shows there is no signal
for. I recommend not running it.

### What should survive from this arc

1. **`sequential` propagation is a free win** and we were not using it: +0.008406
   vs +0.011184 for one-shot at identical bytes (Phase 0 Test A). Adopt it as the
   default everywhere in the repository.
2. **Simple beats clever, again**: uniform 5-bit at 1.25x payload recovers 67.6%
   of the damage, more than any layer-selective scheme we tested.
3. **The negative results are worth writing up** as a short methodological note:
   forward-only self-consistency is redundant by acyclicity; Fisher weighting
   does not rank layers; outlier-protection does not beat uniform bits. Together
   with the L0 observation these are a coherent contribution about what does and
   does not explain quantization damage.

## 5. Phase 2 (not recommended — Phase 1 failed)

Self-consistent mixed-precision allocation across all linear layers (attention,
MLP, DeltaNet), where the coupling is unavoidable, against a one-pass allocation
baseline at matched total bytes. Second checkpoint, second corpus.

## 6. Correction owed to the current paper

The paper currently motivates SCQ with the claim that activation-aware PTQ
calibrates on full-precision activations. That is too strong: GPTAQ and CoreQ
explicitly correct the upstream mismatch, and sequential propagation is standard
in GPTQ implementations. Regardless of Phase 0's outcome, the paper needs:

* related-work entries for GPTAQ and CoreQ,
* the acyclicity argument of §1 stated as the reason one pass suffices for
  forward-only statistics,
* the SCF section rescoped to whatever survives Phase 0.

This is a correction to work already written, and it should land whether or not
we continue the direction.
