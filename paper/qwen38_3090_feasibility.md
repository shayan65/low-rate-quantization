# Qwen3.8-27B RTX 3090 feasibility gate (2026-09-20)

The first official BF16 shard was downloaded; the other 17 shards were not. The official index gives 55,562,855,904 weight bytes across 18 shards. The workstation has 24 GiB GPU memory, 60 GiB RAM, and 380 GB free disk. Full BF16 GPU residency is impossible; CPU/GPU offload is required for task-level evaluation.

## Measured short probe

On the complete 10,240×5,120 layer-0 DeltaNet QKV matrix:

| Operation | Measured time |
|---|---:|
| CPU-to-GPU transfer, FP32 tensor (209.7 MB) | 0.0235 s |
| Optimized real4 quantization | 0.168 s |
| Polar 3+5 quantization | 0.0735 s |
| Pack magnitude and phase indices | 0.582 s |
| Resident BF16 projection, 1/16/128 tokens | 0.145 / 0.142 / 0.194 ms |

Peak PyTorch GPU allocation was 1.05 GB for real4 and 1.76 GB for polar. Both exports occupy 26,255,361 bytes including row scales and one method/pairing byte. Polar MSE is 2.977745e-6 versus real4 3.041003e-6. These are actual measured values; the projection timing does not include the remaining 27B model.

The measured transfer is about 8.93 GB/s for this one tensor. If 18–22 GiB of the full weight set can remain on GPU, roughly 31.9–36.2 GB must be offloaded. For the *existing batch-one protocol*, that implies a **2.0–2.3 hour transfer lower bound** across 2,042 independent blocks, before model computation and offload-framework overhead. A batch-one planning range is **2–4 hours per full NLL pass** and **6–12 hours for BF16 plus two quantized methods**. Batching independent blocks could amortize the transfers substantially; batch size 8 would reduce the transfer-only floor to roughly 15–17 minutes, but its compute and memory costs are not yet measured. These are extrapolations, not measured full-model runtimes. The remaining checkpoint download is about 51.6 GB and should be budgeted separately.

## Promotion gate before GPU hours

1. Download the full checkpoint only after retaining at least 100 GB free disk space. Do not run all-method validation automatically.
2. Load the BF16 text model with explicit CPU/GPU memory caps and verify one 129-token forward pass, recording RSS, GPU peak, and correctness. Abort on swap growth or unstable memory.
3. Time 16 fixed 129-token WikiText-2 blocks at batch sizes 1, 4, 8, and 16 when memory permits. Extrapolate from measured total throughput, not only per-block latency. Promote only if one full pass is projected below two hours and the model can run without swap. Otherwise stop at layer-level activation/output diagnostics.
4. If promoted, run BF16 and one polar projection replacement on at most 2,048 validation targets first; compare with equal-byte real4 on the identical targets. Full validation requires a predeclared improvement gate and a wall-time cap.

The existing full-tensor weight geometry result is in [`results/qwen38_27b_weight_geometry_full_v2/summary.md`](../results/qwen38_27b_weight_geometry_full_v2/summary.md); the hardware probe is in [`results/qwen38_3090_probe_v1/results.json`](../results/qwen38_3090_probe_v1/results.json). Neither is a full-model task-accuracy result.
