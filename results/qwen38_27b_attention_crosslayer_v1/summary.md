# Qwen3.8-27B cross-layer attention Q screen

Frozen official model, three attention Q matrices (layers 7, 11, 15), each quantized **individually**. All other weights remain BF16. The same 4,096 real WikiText-2 validation targets (blocks 96–127, disjoint from prior screens) were used for BF16, optimized row-wise real-4, and 3-magnitude/5-phase polar with adjacent pairing. Equal payload for either quantizer was 31,506,433 bytes per matrix, including row scales and one metadata byte. Lower NLL is better.

| Attention layer | Real-4 NLL | Polar NLL | Polar − real-4 | Winner |
| ---: | ---: | ---: | ---: | --- |
| 7 | 2.160781 | 2.160941 | +0.000161 | Real-4 |
| 11 | 2.159714 | 2.159294 | -0.000420 | Polar |
| 15 | 2.160889 | 2.160406 | -0.000484 | Polar |

The BF16 reference NLL was 2.160729. All three polar tensors had lower weight MSE than their real-4 counterparts, but layer 7 had worse task NLL. The run took 162.9 seconds; each 4,096-target forward evaluation took about 22.5 seconds at batch 8, with 21.60 GB peak PyTorch GPU allocation. No full-dataset or simultaneously quantized model was evaluated.

This supports a **selective**, rather than universal, attention-polar hypothesis. The small NLL gaps on one validation slice are insufficient for a general accuracy or speed claim. Next, choose codecs on a separate calibration split and confirm on disjoint held-out blocks; compare against an activation-aware real-4 control before scaling up.
