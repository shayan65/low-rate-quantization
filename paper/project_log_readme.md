# Paired weight codes and quantization calibration: two negative results

Post-training quantization research on frozen Qwen checkpoints. The project
started from a quantum-inspired polar (magnitude/phase) weight code; that line
produced a **negative result** and was replaced. See
[`architect.md`](architect.md) for the current design and
[`paper/main.tex`](paper/main.tex) for the write-up.

No quantum hardware is used and no quantum advantage is claimed.

## Current state (September 2026)

**Current implementation: measured-storage low-rate VQ.** The revised
[`paper/ternary_proposal.md`](paper/ternary_proposal.md) specifies genuine
scaled ternary, learned three-level scalar, and shared eight-dimensional VQ
controls on Qwen3.5-0.8B-Base. The new codec serializes its actual indices,
FP16 scales, codebooks and rotation metadata; evaluation reloads those bytes.
This is classical VQ and produces compressed projection artifacts for BF16
evaluation, not a packed inference engine or a fully quantized 27B model.

The legacy low-rate run in `results/ternary_task_v1` is preliminary: its
storage figures were estimates, its quantizer metadata was FP32, and it
omitted the final 36 validation targets. The original rate hypothesis remains
failed. A corrected implementation must pass a training-only smoke before
the complete small-model validation run. Results from different protocols
must not be merged. Historical proposal claims are preserved in
`paper/ternary_proposal_pre_review_20260921.md` solely for provenance.

**Negative result — the polar code was retired.** An earlier draft reported that
a 3-bit-magnitude + 5-bit-phase pair code beat an equal-byte real-4 control on
Qwen3.5-0.8B. That advantage came from the control: it used a *uniform* 16-level
codebook, while a radius grid crossed with a phase grid is not a square lattice.
Giving the scalar control the same per-tensor Lloyd-Max fitting reverses the
ranking on every tensor measured and on the complete WikiText-2 validation
stream. Repairing polar's assignment rule to exact nearest-neighbour search over
all 256 codepoints changes weight MSE by 0.035%, because with five phase
bits the projection correction is bounded by `1 - cos(pi/32) = 4.8e-3` of the
radius while one radius-grid step is ~14% of the row scale.

**Positive result — keep the pairing, drop the polar grid.** An unconstrained
256-point 2-D codebook over the same weight pairs sits 18-24% *below* the fitted
scalar code at an identical index budget, while the best possible polar
arrangement stays 6-8% above it. In held-out loss under one matched selection
protocol, the activation-weighted 2-D codec gives the best point estimate of any
codec tested (+0.007693 vs the scalar control's +0.008539), but that paired
difference has interval [-0.000169, +0.001822] and so is **not** significant.
It *is* significantly ahead of polar (+0.008210 [+0.007205, +0.009186]). Decoding is also a table lookup rather than a
sine/cosine, which removes the cost that made the packed polar kernels 3-4.5x
slower than resident BF16.

**Self-consistent quantization (SCQ) — retired.** We formulated calibration as
a fixed point `Q* = Quantize(H[Q*])` and solved it with a guarded SCF loop by
analogy with Kohn-Sham DFT. The direction is closed:

* The forward-only problem is **acyclic** — layer `l`'s statistics depend only on
  earlier layers — so one sequential pass is exact and iteration is the wrong
  tool. Two published methods already do this in closed form (GPTAQ
  [2504.02692](https://arxiv.org/abs/2504.02692), CoreQ
  [2602.05902](https://arxiv.org/abs/2602.05902)). On a 32,768-target screen the
  one-pass arm beat our iteration by 0.004228 NLL.
* On the **full** validation stream, solving it correctly barely matters:
  sequential calibration is +0.000022 for the Lloyd scalar code, bit-identical
  for decoupled polar, -0.000596 for repaired polar, and **+0.000607 (worse)**
  for the 2-D codec. The forward statistics move only 0.87% when the input projections in 18 of 24 layers
  are quantized. The screening-slice advantage did not replicate.
* A **bidirectional** reformulation (Fisher: forward x backward) identifies a
  genuinely cyclic coupling 8.41x larger than the one the literature corrects,
  but the metric has no rank skill across layers, and its one correct call does
  not beat spending the same bytes uniformly.

**The finding worth keeping:** layer 0 is the most damaging layer to quantize,
the layer Fisher ranks highest, and the layer **weight MSE ranks as the safest
of all 18** — 58.2% of summed single-layer damage. That is a concrete mechanism
for why weight MSE never predicted task loss in this project.

```sh
# weight-reconstruction gates (fast, no model forward)
python experiments/run_polar_codec_ablation.py --models-root ../models --out ../results/polar_codec_ablation_v1
python experiments/run_codebook_ceiling.py

# full codec ladder + self-consistent calibration on the complete validation stream
python experiments/run_scq_qwen.py --out ../results/scq_qwen08b_v1 --cal-windows 16 --scf-iters 6

# regenerate the paper tables from raw JSON
python experiments/make_paper_tables.py results/scq_qwen08b_v1/results.json paper
```

Core modules: [`experiments/polar_codec.py`](experiments/polar_codec.py) (polar
codec and its repair), [`experiments/vq_codec.py`](experiments/vq_codec.py)
(paired 2-D codebook), [`experiments/scq.py`](experiments/scq.py) (the SCF loop).

**Not measured:** inference latency and end-to-end memory. Task-quality
evaluation decodes to BF16 and uses ordinary BF16 matrix multiplies, so the
byte reductions are storage results only. Only the 18 DeltaNet input projections
are quantized; all other weights stay BF16.

---

# Historical record

Everything below documents earlier phases of the project and is retained for
provenance. Its conclusions about polar quantization are superseded by the
negative result above.


## Current resource policy and magnitude/phase screen

Long sweeps are paused. New representations must first pass small real-data checks. `experiments/run_polar_screen.py` compares real FP32, real 8-bit, per-weight continuous polar parameters, and 4-bit magnitude + 4-bit phase using roughly 0.6M-parameter hybrids, three seeds, and recorded original WikiText-103 blocks. It selects 262,144 training targets and 16,384 validation targets. The test split is untouched. These are sample-validation results, not full-dataset results.

```sh
python experiments/run_polar_lm.py --verify
python experiments/run_polar_screen.py --out results/new_polar_screen --steps 2000 --seconds 300
```

The total screen has a five-minute default cap. A completed screen must pass exact coefficient/logit deployment checks. Its exploratory promotion gate requires the 4+4 representation's mean paired validation NLL to be at most 0.05 above real 8-bit and continuous polar controls, and requires the real 8-bit mean NLL below 6.0 to reject an undertrained screen. Passing warrants another limited validation stage, not an automatic full run. Larger-run scripts are available as proposed machinery but have not been launched for magnitude/phase quantization.

`run_polar_lm.py` supports 1+1, 2+2, 2+6 and 4+4 magnitude/phase index budgets, includes zero magnitude, and packs the joint code. Scales and unchanged FP32 components are counted. Its canonical STE forward values avoid subtract/add cancellation, and training/export equality is checked without relaxing tolerances. The earlier shared-radius runs remain separate and unchanged.

## Byte-level hybrid language-model smoke test

`experiments/run_lm_smoke.py` compares continuous phases, signed 1–90-degree phases (180 directions), full-circle 180 directions, and full-circle 256 directions. It uses a tiny hybrid causal-attention/complex-recurrence model and UTF-8 bytes from WikiText-2. This verifies the training path; it is not a Qwen experiment or a standard word/subword perplexity benchmark. Only phase projection layers are fake-quantized, while saved checkpoints and execution remain floating point.

```sh
python experiments/run_lm_smoke.py --steps 500 --out results/new_lm_smoke
```

Requires PyTorch, NumPy and PyArrow. Evaluation uses a fixed validation subset; test evaluation is deferred. See [the downstream protocol](paper/downstream_protocol.md) for the larger study and matched controls.

A reproducible classical weight-only QAT pilot and a paper experiment plan. The pilot does not run on IBM Quantum hardware or establish quantum advantage.

## Full real-data language-model study

The full WikiText-103 study uses a train-only 8K BPE tokenizer and all source rows. Each of three training epochs covers all next-token targets; validation and test cover entire splits, including final partial blocks. The 51-run single-GPU grid includes three seeds, hybrid real/polar/complex controls and Transformer controls. Runs resume from saved optimizer and coverage state; completed runs are skipped.

```sh
python experiments/run_full_lm.py --verify
python experiments/run_full_lm.py --prepare
python experiments/audit_full_data.py --out results/data_audit.json
python experiments/run_full_sweep.py
python experiments/run_full_sweep.py --collect
```

Requires PyTorch, NumPy, PyArrow, tokenizers and huggingface_hub. On the workstation set `PYTHONPATH=/home/shayan/quantum_quantization/.lm_deps` to use isolated dependencies. The launched sweep lives at `/home/shayan/quantum_quantization/results/wikitext103_full_v1`; its outer log is `results/wikitext103_full_v1_sweep.log`. See [launch status](results/wikitext103_full_v1/launch_status.md). Local progress files are snapshots, not live workstation logs.

The training runner records dataset hashes, source snapshots, actual token coverage, complete-split losses, parameter counts, elapsed time and GPU allocation. It exports packed quantized projection indices with FP32 scales, retains other parameters in FP32 and reloads the export for test evaluation. Execution and optimizer states remain floating point. Token perplexity is specific to the shared tokenizer. This is a custom model study; SAWB, learned-codebook, temporal calibration and Qwen validation are not included in this launched grid.

Read [the related-work table and complete experiment plan](paper/research_plan.md).

## Run

Requires compatible PyTorch and torchvision, with CUDA optional:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_pilot.py --dataset mnist --epochs 3 --seeds 0 --out results/new_mnist_pilot
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_pilot.py --dataset fashion --epochs 30 --seeds 0 1 2 3 4 --out results/new_fashion_study
```

Use a new output directory for each run. The pilot downloads MNIST/Fashion-MNIST into `data/`, holds out 5,000 training examples for validation, and records JSON metadata, split indices, checkpoints, and CSV metrics. Quantized layers still execute floating-point GPU operations; checkpoints contain FP32 latent weights.

Workstation: `shayan@192.168.1.26`, project `/home/shayan/quantum_quantization`, Python `/home/shayan/miniconda3/bin/python`. See `results/run_status.md` for the actual launch and outcome.

Remote workstation address is read from `$QQ_WORKSTATION`. SSH was verified through this address using the existing trusted workstation host key. The result sync helper now uses Tailscale. Tailscale must be connected on the client; a separate travel computer must join the same account. The workstation's tailscaled service is enabled at boot.

## Complex phase weights

[Polar weight definitions, comparison table, and limitations](paper/phase_weights.md) describe the `q exp(i phi)` experiment.

```sh
python experiments/run_phase_pilot.py --verify
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_phase_pilot.py --epochs 3 --seeds 0 --out results/new_phase_pilot
```

The four methods compare continuous, 4-phase and 8-phase complex weights with a real-projection control. Magnitudes are shared per layer and learned. The quantized variants still execute floating-point operations and save FP32 checkpoints.

## Recurrent memory quantization study

The current mechanism study fits quantizers on sampled linear recurrent systems. It does not train a Qwen checkpoint or a downstream task model. See [the complete results](results/memory_quantization_summary.md) and [paper source](paper/main.tex).

```sh
python experiments/run_memory_quantization.py --verify
python experiments/run_memory_quantization.py --verify --index-bits 8
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_memory_quantization.py --seeds 0 1 2 --index-bits 4 --out results/new_memory_b4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_memory_quantization.py --seeds 0 1 2 --index-bits 8 --out results/new_memory_b8
```

Requires PyTorch and NumPy. The workstation's existing Conda base has both. Each output directory must be new. `--index-bits` is bits per complex mode; radii stay FP32. All compressed families use identical actual operator bytes at each budget. The binary decoder reconstructs the operator for evaluation; frozen projections are counted separately.

The completed version 2 runs contain source snapshots, source hashes, exported binary operators, sampled systems, calibration histories and raw CSV. Version 1 is retained as a numerical diagnostic; only version 2 results appear in the paper. A zero-output reference has relative output MSE 1 and is included in the plot to expose poor reconstruction.

To regenerate the report, install Matplotlib in an isolated reporting environment, then run `python experiments/report_memory.py`. It checks run completion, row uniqueness and recorded source hashes. Compile `paper/main.tex` with Tectonic to `output/pdf/` after generating the report.
