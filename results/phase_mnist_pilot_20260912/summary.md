# Completed polar-weight MNIST pilot

Weight representation: q exp(i phi), with a learned positive magnitude q shared per layer. RTX 3090; seed 0; three epochs per method; 55,000 training / 5,000 validation / official 10,000 test samples. Validation-selected checkpoints.

| Method | Phase bits/connection | Validation accuracy | Test accuracy | Theoretical payload (bytes) | Peak allocated VRAM (MiB) | Elapsed train + eval (s) |
|---|---:|---:|---:|---:|---:|---:|
| Complex continuous phase | 32 | 94.02% | 95.08% | 1,687,520 | 423.5 | 14.61 |
| Complex 4-phase | 2 | 93.16% | 93.99% | 107,240 | 423.5 | 14.17 |
| Complex 8-phase | 3 | 93.60% | 94.42% | 159,916 | 423.5 | 14.16 |
| Real projection, 4-phase | 2 | 89.96% | 91.31% | 106,304 | 208.9 | 11.29 |

The continuous-phase method leads this group. Complex 4-phase is 1.09 percentage points lower than continuous phase; 8-phase is 0.66 points lower. No uncertainty or superiority conclusion is justified from one seed. All validation accuracies were still improving at epoch 3, suggesting a longer fixed training schedule should be tested before drawing conclusions.

For context, the earlier separately initialized real FP32, uniform 2-bit, and SAWB 2-bit pilots reached 98.29%, 98.21%, and 98.08%, respectively. The polar pilot does not demonstrate an advantage over those references. Differences include shared magnitude, initial weights, complex hidden states, and pooling. The real-projection control reached 91.31%; the difference from complex 4-phase does not isolate interference because the activation/pooling rules and effective codebooks also differ.

Actual checkpoints are approximately 1.69 MB and contain FP32 latent phases/log-magnitudes and biases. The payload column is a theoretical decoder representation, including shared magnitudes and biases, not a measured packed file size. The real-projection labels are redundant (two phases map to zero), so its two-bit estimate is nominal. Complex models include an unused final imaginary bias in the recorded storage and nominal scalar counts.

Verification passed before training: phase-grid cardinality and STE, float64 agreement with native complex linear/convolution, finite outputs and gradients. Local and remote code hashes agree. Checkpoints remain on the workstation under /home/shayan/quantum_quantization/results/phase_mnist_pilot_20260912/.

See [the method and fairness discussion](../../paper/phase_weights.md).
