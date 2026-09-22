# Quantum-inspired weight quantization research

## Byte-level hybrid language-model smoke test

`experiments/run_lm_smoke.py` compares continuous phases, signed 1–90-degree phases (180 directions), full-circle 180 directions, and full-circle 256 directions. It uses a tiny hybrid causal-attention/complex-recurrence model and UTF-8 bytes from WikiText-2. This verifies the training path; it is not a Qwen experiment or a standard word/subword perplexity benchmark. Only phase projection layers are fake-quantized, while saved checkpoints and execution remain floating point.

```sh
python experiments/run_lm_smoke.py --steps 500 --out results/new_lm_smoke
```

Requires PyTorch, NumPy and PyArrow. Evaluation uses a fixed validation subset; test evaluation is deferred. See [the downstream protocol](paper/downstream_protocol.md) for the larger study and matched controls.

A reproducible classical weight-only QAT pilot and a paper experiment plan. The pilot does not run on IBM Quantum hardware or establish quantum advantage.

## Full real-data language-model study

The full WikiText-103 study uses a train-only 8K BPE tokenizer and all source rows. Each of three training epochs covers all next-token targets; validation and test cover entire splits, including final partial blocks. The 51-run single-GPU grid includes three seeds, hybrid real/polar/complex controls and Transformer controls. Runs resume from saved optimizer and coverage state; completed runs are skipped.

```sh
python experiments/run_full_lm.py --verify
python experiments/run_full_lm.py --prepare
python experiments/audit_full_data.py --out results/data_audit.json
python experiments/run_full_sweep.py
python experiments/run_full_sweep.py --collect
```

Requires PyTorch, NumPy, PyArrow, tokenizers and huggingface_hub. On the workstation set `PYTHONPATH=/home/shayan/quantum_quantization/.lm_deps` to use isolated dependencies. The launched sweep lives at `/home/shayan/quantum_quantization/results/wikitext103_full_v1`; its outer log is `results/wikitext103_full_v1_sweep.log`. See [launch status](results/wikitext103_full_v1/launch_status.md). Local progress files are snapshots, not live workstation logs.

The training runner records dataset hashes, source snapshots, actual token coverage, complete-split losses, parameter counts, elapsed time and GPU allocation. It exports packed quantized projection indices with FP32 scales, retains other parameters in FP32 and reloads the export for test evaluation. Execution and optimizer states remain floating point. Token perplexity is specific to the shared tokenizer. This is a custom model study; SAWB, learned-codebook, temporal calibration and Qwen validation are not included in this launched grid.

Read [the related-work table and complete experiment plan](paper/research_plan.md).

## Run

Requires compatible PyTorch and torchvision, with CUDA optional:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_pilot.py --dataset mnist --epochs 3 --seeds 0 --out results/new_mnist_pilot
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_pilot.py --dataset fashion --epochs 30 --seeds 0 1 2 3 4 --out results/new_fashion_study
```

Use a new output directory for each run. The pilot downloads MNIST/Fashion-MNIST into `data/`, holds out 5,000 training examples for validation, and records JSON metadata, split indices, checkpoints, and CSV metrics. Quantized layers still execute floating-point GPU operations; checkpoints contain FP32 latent weights.

Workstation: `shayan@192.168.1.26`, project `/home/shayan/quantum_quantization`, Python `/home/shayan/miniconda3/bin/python`. See `results/run_status.md` for the actual launch and outcome.

## Complex phase weights

[Polar weight definitions, comparison table, and limitations](paper/phase_weights.md) describe the `q exp(i phi)` experiment.

```sh
python experiments/run_phase_pilot.py --verify
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_phase_pilot.py --epochs 3 --seeds 0 --out results/new_phase_pilot
```

The four methods compare continuous, 4-phase and 8-phase complex weights with a real-projection control. Magnitudes are shared per layer and learned. The quantized variants still execute floating-point operations and save FP32 checkpoints.

## Recurrent memory quantization study

The current mechanism study fits quantizers on sampled linear recurrent systems. It does not train a Qwen checkpoint or a downstream task model. See [the complete results](results/memory_quantization_summary.md) and [paper source](paper/main.tex).

```sh
python experiments/run_memory_quantization.py --verify
python experiments/run_memory_quantization.py --verify --index-bits 8
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_memory_quantization.py --seeds 0 1 2 --index-bits 4 --out results/new_memory_b4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/run_memory_quantization.py --seeds 0 1 2 --index-bits 8 --out results/new_memory_b8
```

Requires PyTorch and NumPy. The workstation's existing Conda base has both. Each output directory must be new. `--index-bits` is bits per complex mode; radii stay FP32. All compressed families use identical actual operator bytes at each budget. The binary decoder reconstructs the operator for evaluation; frozen projections are counted separately.

The completed version 2 runs contain source snapshots, source hashes, exported binary operators, sampled systems, calibration histories and raw CSV. Version 1 is retained as a numerical diagnostic; only version 2 results appear in the paper. A zero-output reference has relative output MSE 1 and is included in the plot to expose poor reconstruction.

To regenerate the report, install Matplotlib in an isolated reporting environment, then run `python experiments/report_memory.py`. It checks run completion, row uniqueness and recorded source hashes. Compile `paper/main.tex` with Tectonic to `output/pdf/` after generating the report.
