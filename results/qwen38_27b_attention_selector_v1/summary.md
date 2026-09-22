# Qwen3.8-27B attention codec selection gate

Each attention Q projection (layers 7, 11, 15) was quantized individually at an equal four-index-bits-per-real-weight budget. Four fixed WikiText-2 **training** windows (512 targets) selected among optimized row-wise real-4 scale factors 0.90/1.00/1.10 and polar 3+5 pairings adjacent/split-half/reverse-half/stride257. The chosen candidate for each codec was then evaluated on the **same, disjoint** WikiText-2 validation blocks 128–159 (4,096 targets); other model weights remained BF16. Lower NLL is better.

| Layer | Selected real-4 factor | Selected polar pairing | Real-4 NLL | Polar NLL | Polar − real-4 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 7 | 1.00 | adjacent | 2.672991 | 2.672647 | -0.000344 |
| 11 | 1.00 | adjacent | 2.672602 | 2.672640 | +0.000038 |
| 15 | 1.00 | stride257 | 2.673544 | 2.674229 | +0.000684 |

The run completed in 241.5 seconds. Polar won **one of three** held-out comparisons. Its chosen pairing for layer 15 was worse than calibrated real-4 on held-out validation, illustrating that four training windows can overfit candidate selection. The signs also differ from the prior 4,096-target cross-layer screen on validation blocks 96–127. These small, inconsistent differences do not justify a cumulative or full-dataset polar run. This experiment uses task-calibrated selection; it is **not** a GPTQ or AWQ comparison and demonstrates no inference-speed advantage.
