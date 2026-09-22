# Downstream experiment protocol

Status: WikiText-103 full-data language-model sweep launched on the RTX 3090. Results remain pending. See `results/wikitext103_full_v1/launch_status.md` and the workstation logs for progress.

## First model and data

Train causal language models from scratch: a pure Transformer and a hybrid Transformer/complex diagonal recurrence. The launched configuration uses six blocks, width 384, six attention heads, a fourfold-expanded feed-forward projection, and context 512. Hybrid polar variants have 13,086,738 parameters; the unconstrained complex baseline has 21,049,344. Count complex parameters as two real scalars and report actual counts. Compare different architectures separately from matched quantizer variants; widths are fixed rather than claiming exact parameter matching between architectures. Use the same train-only 8K ByteLevel BPE tokenizer, training token order per seed, updates and held-out splits. This is a proposed architecture, not an implementation of Qwen or Gated DeltaNet.

Use Salesforce/wikitext, wikitext-2-raw-v1, for the smoke test. Use wikitext-103-raw-v1 for the substantive training experiment. Official train/validation/test splits remain separate; only train tokens fit scales/codebooks and train weights. Select checkpoints on validation loss and evaluate test once. Record dataset revision and tokenizer revision. WikiText is a limited-domain benchmark and may overlap pretrained Qwen data.

The full sweep pins dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Train has 1,801,350 rows / 135,350,275 BPE tokens; validation has 3,760 rows / 282,592 tokens; test has 4,358 rows / 323,000 tokens. All source rows are retained and followed by a newline. Each epoch predicts all N-1 next-token targets exactly once, including the final partial block. Validation and test cover all N-1 targets, resetting context at contiguous 512-token block boundaries. No subsampling is used. Token perplexity depends on this tokenizer and evaluation convention.

Launched grid: 12 hybrid representations (real FP32, uniform real 2/4/8-bit QAT, unconstrained complex FP32, continuous shared-radius phases, restricted 180 directions, full-circle 180/256 directions, and full-circle 2/3/4-bit phases), plus five Transformer controls (real FP32, continuous phase, restricted 180, full 180, full 256). Seeds 0/1/2, three complete train epochs per run: 51 runs. Best epoch is selected by complete validation loss, followed by one complete test pass on a decoded deployment export. This grid does not yet include unconstrained learned codebooks, SAWB, temporal calibration, Qwen, or memory-task experiments; it cannot complete those claims by itself.

Dataset preparation writes pinned source-file hashes, normalized text hashes, tokenizer hash, token-file hashes and row/token counts. `audit_full_data.py` independently re-encodes every source row and checks every row decodes without text loss. Coverage assertions and source snapshots are retained per training run. The sweep writes metrics only for completed, audited full runs; pending cells have no invented numbers.

Synthetic selective-copy and associative-recall tasks independently test memory at train lengths 128/256 and unseen lengths 512/1024. Fix and publish task generators, distractor distributions and seeds before training.

## Representations

1. Full-precision real baseline.
2. Continuous complex baseline, with the same complex architecture as phase variants.
3. Uniform real 2/4/8-bit QAT on the real architecture.
4. Full-circle 2/3/4/8-bit phase codes with learned nonnegative group scales.
5. User proposal: sign times exp(i*pi*k/180), k=1,...,90; 180 states, 8 fixed bits excluding scales.
6. Full-circle 180-state codebook: same number of directions and same fixed bits as the user proposal, to isolate angular coverage.
7. Full-circle 256-state codebook: same 8 fixed bits, to measure the opportunity cost of unused codes.
8. Weight-reconstruction versus temporal-response calibration for identical code families.

For complex recurrent eigenvalues retain separately accounted nonnegative decay radii. Do not replace decay with a signed magnitude. Keep embeddings, normalization and output head precision identical across matched runs. Specify which layers are quantized. Do not call FP32 fake quantization a packed low-bit training implementation.

Use three seeds after smoke verification. Report perplexity, task accuracy, length generalization, actual exported bytes including all scale/radius overhead, peak CUDA memory, elapsed time, and measured throughput. Report real and complex architecture comparisons separately from within-architecture quantizer comparisons. Include an unconstrained learned codebook control before attributing gains to a quantum-inspired geometry.

## Pretrained validation

Use Qwen/Qwen3.5-0.8B-Base, text only, after the small model study establishes a viable representation. It has 24 layers in a 3:1 Gated DeltaNet/full-attention hybrid layout. Start with a frozen-block reconstruction experiment and measure CUDA memory before broader adaptation. Conversion from real to complex operators must preserve the unquantized function and be independently verified; a real-part-only projection cannot test phase expressivity. This checkpoint is not a diagonal SSM and phase eigenvalue quantization is not a drop-in replacement for its recurrence.

Do not start Qwen3.8-Flash-Next training on the single RTX 3090. Do not claim general weight compression from quantizing only adapters or selected blocks.

## Related work to integrate before submission

- iFairy: https://arxiv.org/abs/2508.05571 (fourth-root-of-unity complex LLM weights).
- Fairy2i: https://arxiv.org/abs/2512.02901 (conversion of pretrained real LLMs and complex phase quantization).
- Qwen model: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- Data: https://huggingface.co/datasets/Salesforce/wikitext

Neither polar weights nor restricting phases establishes novelty. The temporal calibration objective needs comparison with these methods and general learned-codebook calibration.
