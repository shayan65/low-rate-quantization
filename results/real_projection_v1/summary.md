# The compressed path on a real projection (layer 12)

`in_proj_qkv` of Qwen3.5-0.8B, 6144x1024, quantized with the same recipe as the `vq8_rot_gptq` arm: blockwise randomized Hadamard rotation, a shared 6561-point dimension-8 codebook pooled over all 18 projections, GPTQ against the Hessian from 65,536 calibration tokens. Activations are captured from the model on validation text (1,024 rows).

## Fidelity to the weights the loss numbers were measured with

The compressed path rotates its input, since the stored codes are in the rotated basis: `y = rotate(x) @ dq^T`. `exact` is the FP32 product with the dequantized weights; `installed` is the BF16 product with `unrotate(dq)`, which is the operation behind every reported ΔNLL.

| comparison | relative error |
|---|---:|
| installed bf16 vs exact | 3.55e-03 |
| streamed vs exact | 1.08e-07 |
| triton vs exact | 2.71e-07 |
| triton vs installed | 3.54e-03 |

All paths scored on the same 256 activation rows with the same denominator.

This is **operator fidelity** to the quantized FP32 reference. It does not establish that the kernel's end-to-end language-model loss is lower than decoded BF16 execution's: BF16 rounding can reinforce or partially cancel quantization error, and the downstream layers need not respond monotonically to local numerical error. That comparison is unmeasured.

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
| installed bf16 | 0.0187 |
| rotate prototype | 0.4760 |
| streamed with prototype rotation | 0.8415 |
| triton with prototype rotation | 0.5233 |
| rotate fused | 0.0182 |
| triton with fused rotation | 0.0555 |

The rotation is a required part of the compressed path and is inside the timed region; both the ten-stage PyTorch prototype and the fused Triton transform are reported, alone and in the full path, so the format's cost is separated from the prototype's.

Fusing the rotation makes it **26.1x** faster and brings the whole compressed path to **3.0x** cuBLAS, from 28.0x with the prototype. Fused rotation agrees with the reference to 0.00e+00.
