# The compressed path on a real projection (layer 12)

`in_proj_qkv` of Qwen3.5-0.8B, 6144x1024, quantized with the same recipe as the `vq8_rot_gptq` arm: blockwise randomized Hadamard rotation, a shared 6561-point dimension-8 codebook pooled over all 18 projections, GPTQ against the Hessian from 65,536 calibration tokens. Activations are captured from the model on validation text (1,024 rows).

## Fidelity to the weights the loss numbers were measured with

The compressed path rotates its input, since the stored codes are in the rotated basis: `y = rotate(x) @ dq^T`. `exact` is the FP32 product with the dequantized weights; `installed` is the BF16 product with `unrotate(dq)`, which is the operation behind every reported ΔNLL.

| comparison | relative error |
|---|---:|
| installed bf16 vs exact | 3.41e-03 |
| streamed vs exact | 7.38e-07 |
| triton vs exact | 1.37e-07 |
| triton vs installed | 1.94e-03 |

**kernel at or below the BF16 install error.**

## Resident bytes

| tensor | dtype | MB |
|---|---|---:|
| codes | torch.int16 | 1.573 |
| codebook | torch.float16 | 0.105 |
| scales | torch.float16 | 0.098 |
| **total** | | **1.776** |
| BF16 dense | torch.bfloat16 | 12.583 |

Reduction: **7.08x**.

## Latency, batch one

| path | ms |
|---|---:|
| installed bf16 | 0.0188 |
| rotate only | 0.4796 |
| streamed with rotation | 0.8477 |
| triton with rotation | 0.5322 |

The rotation is a required part of the compressed path and is inside the timed region; it is also reported alone so its share is visible.
