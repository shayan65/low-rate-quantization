# Qwen3.8-27B RTX 3090 forward gate

The official checkpoint download completed with all 18 index-listed shards present (55,563,006,776 file bytes; 55,562,855,904 tensor bytes in the index). The BF16 text model loaded in 8.83 seconds with a 20 GiB GPU / 40 GiB CPU placement cap. Layers 0–20 were on GPU, and layers 21–63 plus the head were on CPU.

The first 16 contiguous 128-target WikiText-2 validation windows (2,048 real targets) were evaluated with context reset at each window. These are timing measurements of the unquantized reference only, not polar results.

| Batch | 16-window time | Extrapolated 2,042-window time | Mean NLL | Peak PyTorch GPU allocation | Swap growth |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 70.29 s | 149.5 min | 2.21774 | 21.15 GB | 28.7 KB |
| 4 | 18.78 s | 40.0 min | 2.21814 | 21.34 GB | 28.7 KB |
| 8 | 10.72 s | 22.8 min | 2.21883 | 21.60 GB | 28.7 KB |
| 16 | 6.68 s | 14.2 min | 2.21737 | 22.61 GB | 28.7 KB |

The under-two-hour timing and no-swap-growth gate passed with batching. The extrapolation excludes loading, codec construction, and quantized offloading overhead; a short codec pilot must measure those before committing to the full comparison. The four NLL values differ slightly across batches and require a fixed-batch baseline for paired quality comparisons.
