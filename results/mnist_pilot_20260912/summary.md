# Completed MNIST pilot

Six methods, seed 0, three epochs each; 421,642 parameters. Validation selects the checkpoint. All activations and biases are FP32. This is a classical fake-quantization experiment.

| Method | Validation accuracy | Test accuracy | Weight MSE | Theoretical bits/parameter | Training + evaluation (s) |
|---|---:|---:|---:|---:|---:|
| FP32 | 97.70% | 98.29% | 0 | 32.0000 | 11.42 |
| Uniform 2-bit | 97.86% | 98.21% | 0.00018 | 2.0170 | 11.50 |
| SAWB 2-bit | 97.70% | 98.08% | 0.000184 | 2.0170 | 11.34 |
| Bloch-angle 2-bit | 97.48% | 98.00% | 0.000262 | 2.0170 | 11.33 |
| Bloch-nearest 2-bit | 97.56% | 98.06% | 0.00023 | 2.0170 | 11.59 |
| Binary 1-bit | 96.72% | 97.27% | 0.000669 | 1.0175 | 11.29 |

Bloch-angle trails SAWB by 0.08 percentage points (8 of 10,000 test examples) and the separate uniform baseline by 0.21 points. One seed and three epochs cannot establish superiority, equivalence, or statistical significance. Weight MSE compares each trained model with its own latent weights; it is not a common-checkpoint PTQ comparison.

Theoretical bits include FP32 biases and one FP32 scale per quantized layer. Actual checkpoints remain about 1.69 MB and store FP32 latent weights. No compressed inference kernel, IBM QPU, or speed/energy advantage was tested.
