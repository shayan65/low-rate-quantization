# Actual-output polar selection: cross-layer and dataset gates

All runs used the official frozen Qwen3.8-27B text model. A candidate was selected from four polar 3+5 codes by minimizing actual Q-projection output error on 16 WikiText-2 **training** windows. Each evaluation replaced one attention Q tensor; all other model weights remained BF16. Optimized real-4 and a local AWQ-style real-4 method were controls. The three WikiText-2 tests used 8,192-target held-out validation slices, and the TinyStories transfer test used the first 8,192 targets from its validation stream. TinyStories was evaluated without changing the WikiText-derived candidate selection. Lower NLL is better.

| Layer / held-out data | Legacy polar | Output-selected polar | Real-4 | AWQ-style real-4 | Output vs legacy | Output vs AWQ-style |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 / WikiText-2 blocks 288–351 | 2.819199 | **2.819068** | 2.819683 | 2.819500 | -0.000130 | -0.000432 |
| 7 / WikiText-2 blocks 352–415 | 2.468694 | **2.467267** | 2.468943 | 2.467449 | -0.001428 | -0.000183 |
| 15 / WikiText-2 blocks 352–415 | 2.468220 | 2.468420 | 2.468376 | **2.467835** | +0.000199 | +0.000584 |
| 11 / TinyStories blocks 0–63 | **1.611913** | 1.612826 | 1.612173 | 1.613034 | +0.000913 | -0.000208 |

For layer 11 WikiText-2, an exploratory paired bootstrap over eight contiguous groups of eight windows gave an interval of [-0.001321,+0.001204] for output-selected minus legacy polar. For layer 7 WikiText-2, the corresponding interval was [-0.002923,+0.000090]. Both include zero. Layer 15's output-selected minus AWQ-style interval was [+0.000172,+0.000957], favoring AWQ-style on that slice. These descriptive intervals are based on few contiguous groups and should not be treated as broad generalization guarantees.

The **promotion gate failed**: output-selected polar was not a consistent improvement over legacy polar or AWQ-style real-4 across layers, and it lost to legacy polar and plain real-4 on TinyStories. Full-dataset, cumulative-model, and packed-runtime Qwen3.8 experiments were not launched. Current candidate weights were decoded to BF16 for inference, so no memory or speed advantage was demonstrated. The preceding packed-kernel prototypes were slower than resident BF16 on isolated projections. A new method would need a predeclared, stable layer/task selector and a viable packed kernel before full-scale testing is economical.
