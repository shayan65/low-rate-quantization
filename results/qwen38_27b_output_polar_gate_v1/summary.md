# Actual-output polar selection gate, Qwen3.8-27B layer 11

Four equal-index-bit polar candidates were built from 16 fixed WikiText-2 training windows. The redesigned quantizer used activation-weighted magnitude decisions and row scales, then candidate pairing and global phase offset were selected by **actual layer-output** reconstruction error, `mean((X(Q(W)-W)^T)^2)`. That selected reverse-half pairing and a half-phase-bin offset; the diagonal activation-error proxy selected adjacent pairing with the same offset. Each candidate retained a 3-bit magnitude and 5-bit phase index per two real weights. A separate, untouched 8,192-target WikiText-2 validation slice (blocks 288–351) supplied task NLL.

| Method | Validation NLL |
| --- | ---: |
| BF16 | 2.819302 |
| Optimized real-4 | 2.819683 |
| Legacy polar 3+5 | 2.819199 |
| Diagonal-proxy polar 3+5 | 2.819873 |
| **Actual-output-selected polar 3+5** | **2.819068** |
| Local AWQ-style real-4 | 2.819500 |

The output-selected polar candidate had the lowest NLL in this single-layer screen. Its advantage over legacy polar was only 0.000130 NLL. An exploratory bootstrap over eight contiguous groups of eight validation blocks gave a 95% interval of [-0.001321,+0.001204] for output-selected minus legacy polar, and [-0.001205,+0.000313] versus AWQ-style real-4. Both include zero, so this is a **directional pilot, not a confirmed improvement**. The run took 293.5 seconds. These are decoded BF16, one-projection replacements; no cumulative accuracy, packed-runtime, memory, or quantum-computing advantage follows.
