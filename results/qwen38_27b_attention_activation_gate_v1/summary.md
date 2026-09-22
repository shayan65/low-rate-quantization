# Qwen3.8-27B activation-aware attention gate

The official frozen text model was evaluated on 8,192 real WikiText-2 validation targets (blocks 160–223), disjoint from earlier Qwen3.8 screens. Sixteen fixed WikiText-2 **training** windows supplied activation statistics. For each of three attention Q projections, only that one matrix was replaced; the rest of the model stayed BF16. Polar 3+5 pairing and a local AWQ-style real-4 channel-scaling exponent were selected by diagonal activation-weighted weight error on training activations. Optimized row-wise real-4 is another control. This AWQ-style implementation is local, not an official AutoAWQ run.

| Layer | Real-4 NLL | Polar NLL | AWQ-style real-4 NLL | Best | Polar pairing | AWQ alpha |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 7 | 2.881376 | 2.880972 | **2.880003** | AWQ-style | reverse-half | 0.25 |
| 11 | 2.880775 | **2.880561** | 2.880815 | Polar | reverse-half | 0.25 |
| 15 | 2.880772 | 2.881555 | **2.880266** | AWQ-style | adjacent | 0.25 |

The common BF16 NLL was 2.880854. Polar beat plain real-4 on two layers, but beat AWQ-style real-4 on only one. Each polar and plain real-4 tensor used 31,506,433 bytes including index, row scales, and method ID; AWQ-style used 31,526,913 bytes because its 5,120 FP32 channel scales require another 20,480 bytes (0.065% larger). The 12 held-out forward evaluations took about 45 seconds each; the whole gate took 567.5 seconds. Peak PyTorch GPU allocation and individual timings are in `results.json`. The workstation GPU was idle after completion.

As an exploratory variability check, paired differences were averaged in eight consecutive groups of eight validation blocks and bootstrapped over those eight groups (10,000 draws). The 95% intervals for **polar minus AWQ-style real-4** were layer 7: [+0.000100,+0.001841], layer 11: [-0.001451,+0.000990], and layer 15: [+0.000260,+0.002382]. These are descriptive intervals on one contiguous validation slice, not a generalization guarantee. In particular, the layer-11 polar advantage is too small to distinguish confidently from zero with this check.

The gate does **not** support a general polar accuracy advantage over an activation-aware real-4 control. These task-quality runs decode candidates into BF16 and therefore demonstrate no memory or inference-speed advantage. Further work should focus on a predeclared subset of projections or a redesigned activation-aware polar quantizer, with stronger classical controls and held-out confirmation before full-model validation.
