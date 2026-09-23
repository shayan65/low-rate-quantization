# Polar phase-weight experiment

> **Superseded in part (2026-09-20).** The polar-versus-real-4 comparisons in
> this document used a real-4 control with a *uniform* codebook. Giving the
> scalar control the same per-tensor Lloyd-Max fitting reverses the ranking, and
> the polar family is structurally capped above it. See
> [`polar_retirement.md`](polar_retirement.md) for the measurements that
> supersede these conclusions. The protocol notes, negative findings, and
> mechanism hypotheses below remain valid.



## Working interpretation

The user's “yes” is treated as approval to try w = q exp(i phi), with q a learned magnitude shared by all weights of one layer. A separate real-valued phase phi is learned per connection. This is classical complex-valued learning on the RTX 3090. The magnitude is positive, q=exp(log_q), with no upper bound; it is not a normalized quantum amplitude.

## Implemented methods

| Method | Effective weight | Nominal stored phase bits/connection | Magnitudes | Activations / readout |
|---|---|---:|---|---|
| complex_continuous | q exp(i phi) | 32 | Four learned FP32 scalars | Complex hidden activations, real-part final logits |
| complex_phase4 | q exp(i Q4(phi)) | 2 | Same | Same |
| complex_phase8 | q exp(i Q8(phi)) | 3 | Same | Same |
| real_phase4 | q cos(Q4(phi)) | 2 nominal | Same | Real hidden activations and logits |

QK(phi)=2pi/K * round(K phi/(2pi)), periodic modulo 2pi. An identity STE propagates gradients through phase rounding. The complex layers use (Wr Xr - Wi Xi) + i(Wr Xi + Wi Xr). Hidden nonlinearities apply ReLU to each component. Spatial pooling selects the largest squared magnitude in each 2x2 window and gathers both components from that location. The real control uses ordinary ReLU/maxpool. Last-layer logits use the real output component. The final imaginary bias is consequently functionally unused; the initial implementation still allocates and marks it trainable in complex models, and nominal trainable scalar counts reflect this allocation.

Four phases {0,pi/2,pi,3pi/2} yield weights {q,iq,-q,-iq}, up to floating-point trigonometric error. For the real projection, these collapse to {q,0,-q,0}; the control is therefore effectively ternary and the four phase labels contain redundancy. Its nominal 2-bit estimate is not a minimum compressed representation.

## Fairness and limits

- Same connection topology as the previous pilot: 1→32→64 convolutional channels, FC128, FC10; 421,408 connections. Same MNIST 55k/5k/10k split, split seed 2026, Adam lr=0.001, batch 256, seed 0, three epochs.
- All four polar methods share initial continuous phases, initial per-layer magnitudes, zero biases, and data order. Quantized effective initial weights differ because phases are rounded.
- The previous real FP32/SAWB/uniform models used different initialization, real activations, and conventional PyTorch biases. Their accuracies are contextual references, not an isolated matched experiment proving the effect of complex weights.
- Complex hidden states double the number of real activation components. Generic implementation uses four real linear/convolution operations for each complex layer. Equal phase-index bits do not imply equal arithmetic cost or equal expressivity to a real network.
- Two phase bits identify four complex directions. This is still a two-bit classical codebook quantizer and does not store continuously variable phase in two bits. The continuous case stores one FP32 phase per connection plus shared magnitudes; that differs from a general complex weight with two independently learned FP32 Cartesian components.
- Theoretical payload includes all nominal phase indices, shared FP32 magnitudes, and FP32 real/imaginary biases. The pilot includes even the unused final imaginary bias in complex payload accounting, so it is conservative by 320 bits. Container overhead and optimizer state are excluded. Actual checkpoints contain FP32 phase parameters/log-magnitudes, not packed indices.
- No IBM Quantum access, physical qubits, quantum circuit, inference speedup, or quantum advantage is involved.
- One seed and three epochs are a feasibility pilot. Full comparisons need matched tuning budgets, multiple seeds, an unconstrained complex baseline, shared-magnitude real/binary controls, and either matched compute or separately reported compute.

## Verification

`python experiments/run_phase_pilot.py --verify` checks periodic phase-grid cardinality, identity STE gradients, agreement of real-pair linear/convolution arithmetic with native PyTorch complex arithmetic in float64, finite logits, and finite gradients to phase and magnitude for all four methods. This verification passed on the workstation before launch.

## Related work

These sources make a novelty claim based solely on phase encoding inappropriate:

- Artificial neural networks using complex numbers and phase encoded weights (2010): https://opg.optica.org/ao/abstract.cfm?uri=ao-49-10-b71
- Deep Complex Networks (2017 preprint): https://arxiv.org/abs/1705.09792
- Ultra-efficient physical field computing by complex-valued network quantization (2026): https://doi.org/10.1038/s41467-026-70319-0

A possible paper question is whether shared-magnitude, low-bit phase codebooks improve accuracy per stored bit at a measured and explicitly stated inference cost, compared with matched real and complex baselines. The present pilot tests feasibility, not novelty or that hypothesis conclusively.

## Run

```sh
cd /home/shayan/quantum_quantization
CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/shayan/miniconda3/bin/python -u experiments/run_phase_pilot.py --epochs 3 --seeds 0 --out results/phase_mnist_pilot_20260912
```

The output directory must be new. Original pilot files are preserved. Checkpoints are selected by validation accuracy, with earliest epoch retained on ties; the selected model is evaluated once on the test set. Artifacts include code hash, versions, arguments, exact split indices, checkpoints, CSV and log.
