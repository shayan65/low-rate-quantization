# Hybrid byte-language-model smoke results

Completed on RTX 3090, seed 0, 500 updates per method, 215,298 parameters. Only PhaseLinear layers use shared-scale polar weights. All methods execute floating-point operations; checkpoints are FP32.

| Method | Validation bits/byte (lower better) | Seconds |
|---|---:|---:|
| continuous | 3.7904 | 25.7 |
| restricted180 | 3.7866 | 26.0 |
| full180 | 3.7840 | 26.1 |
| full256 | 3.7939 | 26.2 |

Evaluation: first 16 disjoint 128-byte validation windows (2,048 bytes) from WikiText-2. Test set untouched. Differences are tiny; this single-seed smoke run establishes no superiority, compression speedup or novelty. The earlier v1 run had poor embedding initialization and is excluded.

The architecture alternates complex diagonal recurrence and real causal attention; its phase-projection layers have shared radii and contribute to real nonlinear outputs. This is a proposed tiny hybrid, not Qwen/Gated DeltaNet. Larger matched Transformer controls, ordinary quantizers, three seeds, complete held-out evaluation, memory tasks and packed export remain in the downstream protocol.

Checkpoints remain on the workstation at /home/shayan/quantum_quantization/results/lm_smoke_phase_v2/.
