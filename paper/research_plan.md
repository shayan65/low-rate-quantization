# Qubit-inspired low-bit weight quantization: research and experiment draft

Date: 2026-09-12. Working interpretation: “qbit” means a qubit-inspired classical quantizer. User confirmation of the intended encoding and IBM reference is pending. This document is a research plan, not evidence of novelty or a completed paper.

## Related work table

| Work | Core idea | Relationship to this project | Important difference / comparison |
|---|---|---|---|
| Choi et al., Accurate and Efficient 2-bit Quantized Neural Networks, MLSys/SysML 2019 | SAWB weight scales with PACT activation clipping | Likely intended IBM reference; essential baseline | Original weight-and-activation method and architectures must be reproduced before claiming an IBM result |
| Esser et al., Learned Step Size Quantization, ICLR 2020 | Learn quantization step sizes using scaled gradient estimates | Strong additional low-bit baseline, also IBM affiliated | Our pilot does not implement LSQ; required in the expanded study |
| Courbariaux et al., Binarized Neural Networks, 2016 | Binary weights and activations during forward computation | Context for one-bit discretization and STE training | Pilot binary baseline uses weights only and is not a reproduction of the complete method |
| Hubara et al., Quantized Neural Networks, 2016 | Train low-precision weights and activations | Establishes classical low-bit training context | Quantum terminology alone does not distinguish a new method |
| Liu et al., Quantum-Train, 2024; journal version 2025 | Quantum states and a classical mapping model generate classical NN weights | Closely related qubit-based model compression direction | Parameter generation/compression differs from independently assigning low-bit indices to weights |
| Li et al., Quantum-Inspired Weight-Constrained Neural Network, 2024 preprint; Physical Review Research 2026 | Classical weight constraints derived from quantum-network mathematics | Closely related quantum-inspired compression | Structured constraints differ from the scalar quantizers in the pilot |
| Liu et al., Quantum-Train with Tensor Network Mapping Model and Distributed Circuit Ansatz, 2024 | Tensor-network mapping and smaller distributed circuit ansatz | Relevant if extending toward shared circuit-generated weight blocks | Requires additional parameter, reconstruction, and measurement accounting |

Primary sources:

- IBM overview: https://research.ibm.com/blog/2-bit-precision-ai
- IBM paper: https://proceedings.mlsys.org/paper_files/paper/2019/file/c443e9d9fc984cda1c5cc447fe2c724d-Paper.pdf
- LSQ: https://arxiv.org/abs/1902.08153
- BNN: https://arxiv.org/abs/1602.02830
- QNN: https://arxiv.org/abs/1609.07061
- Quantum-Train: https://arxiv.org/abs/2405.11304
- Quantum-Train authors' toolkit: https://github.com/Hon-Hai-Quantum-Computing/QuantumTrain
- Weight-constrained network: https://arxiv.org/abs/2412.19355 (latest version v2, 2026)
- Tensor-network Quantum-Train: https://arxiv.org/abs/2409.06992

Search scope: targeted web searches for IBM 2-bit quantization, qubit weight quantization, quantum-inspired compression, and Quantum-Train. This is not an exhaustive systematic review. No claim that an identical quantizer is absent from prior work is justified by this search.

## Proposed mathematical definition

For a real-amplitude qubit |psi(theta)> = cos(theta/2)|0> + sin(theta/2)|1>, the Pauli-Z expectation is cos(theta). Let w_hat = a cos(theta), with a positive per-layer scale. On a classical GPU this is a real-valued formula; no quantum processor is involved.

For b=2, choose theta_k=k*pi/3 for k=0,1,2,3. The four normalized reconstructed levels are {1, 1/2, -1/2, -1}. Each weight therefore still needs **two classical index bits**, plus shared scale metadata. A continuous theta stored as FP32 is not a two-bit weight. One measured qubit produces one binary outcome; estimating its expectation takes repeated preparations and measurements. For independent shots, the Z sample-mean variance is sin(theta)^2/S for S shots. This sampling formula does not include hardware noise or state-preparation cost.

For x=clip(w/a,-1,1), angle rounding sets theta_q=round(3*acos(x)/pi)*pi/3. Weight-space thresholds are {-sqrt(3)/2,0,sqrt(3)/2}. Nearest reconstruction-value rounding on the same four levels instead has thresholds {-3/4,0,3/4}. The latter minimizes individual scalar squared error for a fixed scale and codebook; include it as a mandatory control. Angle rounding needs task-level evidence to justify it.

Pilot scale: a=max(2.587*sqrt(mean(w^2))-1.693*mean(abs(w)),epsilon), from the IBM paper's Eq. 7 coefficients for four bins. SAWB, Bloch-angle, and Bloch-nearest use the same detached scale. A separate uniform baseline uses a=2.5*RMS(w). Uniform and SAWB use levels {-1,-1/3,1/3,1}. All quantizers use an identity straight-through estimator and FP32 latent training weights. Biases and activations remain FP32, including the first and last layers' activations; all convolution and linear weight matrices are quantized.

The Bloch construction above is mathematically a classical nonuniform scalar quantizer. Reparameterizing four levels does not establish novelty, a quantum advantage, or greater information capacity. A stronger research direction would test shared circuit-generated block codebooks under a fixed **total stored-bit** budget against learned classical codebooks and tensor factorizations, explicitly including angle precision and reconstruction cost.

## Full experiment matrix

| Stage | Dataset / split | Model | Methods | Seeds / budget | Main purpose | Status |
|---|---|---|---|---|---|---|
| P0 | MNIST: 55k train / 5k validation / official 10k test | CNN 1→32→64, FC128, FC10 | FP32, binary1, uniform2, SAWB2, Bloch-angle2, Bloch-nearest2 | Seed 0; 3 epochs/method | Check pipeline, dataset, gradients, artifacts | Completed; see results/mnist_pilot_20260912/summary.md |
| P1 | MNIST, identical split | Same CNN | Same six methods | Seeds 0–4; 20 epochs | Repeated small-data evidence | Planned |
| P2 | Fashion-MNIST: same split sizes | Same CNN | Same six methods | Seeds 0–4; 30 epochs | Harder grayscale task | Planned; runner supports dataset |
| P3 | CIFAR-10: 45k / 5k / official 10k | CIFAR ResNet-20 | FP32, binary, uniform2, SAWB2, LSQ2, Bloch-angle2, Bloch-nearest2, learned 4-level codebook | Seeds 0–4; 200 epochs | Main classification experiment | Planned; model/data adapter needed |
| P4 | CIFAR-100: 45k / 5k / official 10k | CIFAR ResNet-32 | Same main methods | Seeds 0–4; 200 epochs | Generalization across task difficulty | Planned |
| P5 | CIFAR-10 / CIFAR-100 | ResNet-20 / ResNet-32 | SAWB+PACT vs Bloch+same PACT; include LSQ W2A2 | Seeds 0–4; matched original schedules | Fair weight-and-activation comparison | Planned; PACT/LSQ implementations required |
| P6 | ImageNet, licensed/downloaded by user; held-out validation for tuning | ResNet-18 | Best candidates + SAWB/LSQ/reference baselines | Three seeds if feasible; schedule fixed before test | Larger-scale confirmation | Optional; storage and runtime assessment needed |
| A1 | CIFAR-10 | ResNet-20 | Angle vs nearest-value rounding; fixed vs learned scale; learned symmetric vs free codebooks | Five seeds; same compute/tuning budget | Separate geometry from ordinary codebook learning | Planned |
| A2 | CIFAR-10 | ResNet-20 | 1,2,3,4 classical index bits; per-layer vs per-channel scales | Five seeds | Rate–accuracy curve incl. metadata | Planned |
| A3 | Small fixed weight block only initially | Circuit-generated weights / codebooks | Exact expectation vs 128,512,2048,8192 shots + classical stochastic control | Repeated measurement seeds | Measurement variance and reconstruction cost | Planned; not part of GPU pilot |
| H1 | Fixed calibration/inference batches | Selected trained models | Packed storage + supported integer/codebook kernels | Warm-up; synchronized repeated batches | Actual file size, throughput, latency, power if measurable | Planned; fake-QAT timings cannot establish integer speedup |

## Paper tables to populate

Main accuracy table (P0 is a smoke/pilot result, not the main table):

| Dataset/model | FP32 | Binary W1A32 | Uniform W2A32 | SAWB W2A32 | LSQ W2A32 | Bloch-angle W2A32 | Bloch-nearest W2A32 | Learned codebook W2A32 |
|---|---|---|---|---|---|---|---|---|
| MNIST / small CNN | TBD | TBD | TBD | TBD | Not implemented | TBD | TBD | Not implemented |
| Fashion-MNIST / small CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-10 / ResNet-20 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CIFAR-100 / ResNet-32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Report test top-1 accuracy mean ± sample SD, all seed values, paired per-seed differences, and uncertainty of differences. Select checkpoints and hyperparameters using validation only; do not tune on repeatedly observed test outcomes. Report top-5 on ImageNet. Do not mix published numbers with locally reproduced numbers without explicit labeling.

| Method | Weight-index bits | Scale/codebook bits | Whole-model effective bits/parameter | Actual packed bytes | Top-1 mean±SD | Weight MSE | Peak training VRAM | Inference latency | Throughput |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 32 | 0 | 32 | TBD | TBD | 0 | TBD | TBD | TBD |
| Binary | 1 | 32/layer scale | Compute incl. FP32 biases | TBD | TBD | TBD | TBD | TBD | TBD |
| Uniform2 / SAWB2 | 2 | 32/layer scale | Compute incl. FP32 biases | TBD | TBD | TBD | TBD | TBD | TBD |
| Bloch-angle2 / nearest2 | 2 | 32/layer scale; fixed shared levels | Compute incl. FP32 biases | TBD | TBD | TBD | TBD | TBD | TBD |
| Learned codebook | 2 | All stored learned levels/scales | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The current runner records theoretical bits/parameter, FP32 checkpoint bytes, weight reconstruction MSE, validation/test metrics, elapsed training+evaluation time, and peak allocated CUDA memory. It does not produce packed inference artifacts or measure inference throughput. Its theoretical storage excludes serialization/container overhead and assumes the fixed codebook is shared by the decoder. Never label FP32 state_dict bytes as compressed storage.

## Reproducibility and next decisions

Pilot: Adam lr=0.001, batch=256, three epochs, no augmentation or normalization beyond ToTensor; all methods share initial weights and data order per seed. Fixed validation split seed=2026. Deterministic PyTorch algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8. Checkpoints selected by maximum validation accuracy; ties retain the first. Test evaluated once per selected model. No FP32 pretraining in this pilot. The dataset class downloads the official archive and checks torchvision's archive integrity; torchvision version should be captured with the environment lock.

Before main experiments: obtain user's actual qubit encoding and IBM reference; freeze architecture/schedules and equal validation-tuning budgets; implement and verify LSQ and PACT; decide whether the claim concerns training behavior, stored classical bits, or actual quantum reconstruction. Reject a novelty claim based solely on the Bloch cosine mapping. If the proposal uses physical qubits, specify circuits, connectivity, shots, error mitigation, readout/state-preparation overhead, and how classical inference consumes reconstructed weights.

## Extension: complex polar weights

Following the user's phase-weight suggestion, the project now also implements w=q exp(i phi), with a learned shared magnitude per layer and independently learned phases. See [the phase-weight experiment definition](phase_weights.md) for the four-method matrix, architecture differences, storage accounting, and prior work. These results should be presented separately from the original real-valued Bloch pilot.
