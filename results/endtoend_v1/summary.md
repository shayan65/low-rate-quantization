# A served compressed model (all coverage)

186 tensors, 497,614,848 of 752,393,024 parameters, dimension 4, K=1625. Each target `nn.Linear` is replaced by a module holding int16 codes, an FP16 codebook and FP16 group scales; the dense weights are dropped and the Hadamard rotation is fused into one Triton kernel.

## Does the served model compute what the paper measured?

| model | NLL over 32,768 targets |
|---|---:|
| decoded BF16 (what every earlier number used) | 3.398479 |
| compressed, served | 3.398463 |

Difference -1.59e-05, 95% CI [-4.58e-04, +4.28e-04].

## What it weighs

| model | resident bytes | |
|---|---:|---:|
| BF16 | 1504.8 MB | |
| compressed | 783.2 MB | **1.92x** |

Measured with `torch.cuda.memory_allocated`, so this is the whole model including the BF16 embedding and norms, not a payload count.

## Single-token decode

| model | ms/step |
|---|---:|
| BF16 | 85.638 |
| compressed | 99.364 |

A factor of 1.2. Decode takes the GEMV kernel; prefill and the evaluation above take the streamed path, which is the one checked against the reference.

Scope: the embedding, norms, KV cache and recurrent state stay BF16, and both kernels are tuned only far enough to be correct.
