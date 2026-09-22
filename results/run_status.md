# Workstation run status

Completed: all six MNIST pilot methods, seed 0, three epochs each. No pilot process needs to remain running.

- Host: shayan@192.168.1.26 (`quark`), NVIDIA RTX 3090 24 GiB.
- Remote project: /home/shayan/quantum_quantization
- Python: /home/shayan/miniconda3/bin/python, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128.
- Remote checkpoints, split, manifest, CSV: /home/shayan/quantum_quantization/results/mnist_pilot_20260912/
- Remote log: /home/shayan/quantum_quantization/results/mnist_pilot_20260912.log
- Local copies: CSV, manifest, log and environment freeze; model checkpoints remain on the workstation.
- Verification passed: all six method runs completed; quantizer codebook cardinality, finite zero-input behavior, identity STE gradients, and nearest-value MSE control checked on workstation.

Command:

```sh
cd /home/shayan/quantum_quantization
CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/shayan/miniconda3/bin/python -u experiments/run_pilot.py --epochs 3 --seeds 0 --out results/mnist_pilot_20260912
```

The system Python lacks PyTorch and python3-venv support. Its attempted virtual environment created /home/shayan/quantum_quantization_env but was not used. The existing Conda base had all dependencies; no packages were installed or upgraded.

See mnist_pilot_20260912/summary.md for the complete results table. Expanded study stages are planned, not running.

## Polar phase-weight pilot

Completed four additional methods: continuous complex phase, complex 4-phase, complex 8-phase, and real-projected 4-phase. All used seed 0 and three epochs. Remote artifacts: /home/shayan/quantum_quantization/results/phase_mnist_pilot_20260912/. Local metrics and report: phase_mnist_pilot_20260912/summary.md. No additional study is currently scheduled.

## Recurrent memory study

Completed version 2: three sampled systems × eight controls × four evaluation lengths × two index budgets = 192 metric rows. These are trained quantizers on frozen random linear systems, not trained language models. Both runs include actual packed operator exports, common FP32 projections, decoded evaluation, code hashes and source snapshots. See memory_quantization_summary.md.

- Remote runs: /home/shayan/quantum_quantization/results/memory_quantization_v2_b4 and memory_quantization_v2_b8.
- Local copies include all run artifacts and logs.
- Verification passed for 4- and 8-bit formats: impulse/recurrence equivalence, phase drift identity, all controls, odd-mode packing, round trip, radius preservation, and memory-calibration improvement checks.
- Version 1 is preserved as a diagnostic. Its unconstrained fitted Cartesian control can be unstable and large-error coordinate comparisons are numerically unreliable. The reported version 2 stabilizes fitted Cartesian controls. Phase results at 4 bits are unchanged.
- No Qwen weights were downloaded. No quantum hardware or inference speedup was tested.
- Reporting required Matplotlib; it was installed in local tmp/pdfs/python_deps, without modifying workstation packages. The report source and plot are retained.
- Paper: paper/main.tex; compiled output: output/pdf/main.pdf.
