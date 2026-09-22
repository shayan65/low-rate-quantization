# Small real-data screen, 2026-09-15

> **Superseded in part (2026-09-20).** The polar-versus-real-4 comparisons in
> this document used a real-4 control with a *uniform* codebook. Giving the
> scalar control the same per-tensor Lloyd-Max fitting reverses the ranking, and
> the polar family is structurally capped above it. See
> [`polar_retirement.md`](polar_retirement.md) for the measurements that
> supersede these conclusions. The protocol notes, negative findings, and
> mechanism hypotheses below remain valid.



These are workstation measurements from real WikiText-103 BPE tokens, not synthetic figures or full-corpus language-model results. The coordinate run used 2,048 fixed 128-token training blocks (262,144 unique selected target positions), 128 fixed validation blocks (16,384 target positions), three seeds and 1,200 updates per seed. Its nine jobs completed in 58.6 seconds of measured invocation time. The test split was untouched. Source, selection indices, checkpoint hashes, decoded exports and exact metrics are in `results/coordinate_diagnostic_v1/`.

| Trainable small hybrid | Mean selected-validation perplexity ± seed SD | Exported tensor bytes |
|---|---:|---:|
| Real FP32 | 693.153 ± 6.260 | 2,493,184 |
| Cartesian complex FP32 | 672.488 ± 2.885 | 2,788,352 |
| Polar complex FP32 | 770.840 ± 3.375 | 2,788,352 |

The paired polar minus Cartesian complex validation NLL gap was 0.13650, above the 0.05 screen gate. Cartesian complex slightly improves on real FP32 on this particular selected sample, while polar optimization is worse. These variants do not have identical real-vs-complex capacity or deployment size. This run cannot substantiate a language-model phase-weight claim. No larger training sweep was launched.

The follow-on froze each of the three trained Cartesian complex checkpoints and changed only the first recurrent transition poles. Four original training-text windows per seed selected optional shared offsets; eight held-out validation-text windows per seed at each length evaluated the decoded exports. All five controls stored 64 four-bit indices, 64 FP32 decay factors and one FP32 shared offset: exactly 292 bytes per transition record, verified by binary round trip. The Cartesian control used all 16 distinct 2+2-bit grid points for a complex pole, quantizing radius and angle; the phase controls retained the trained FP32 decay factor exactly. The 512-token probe cycles the learned 128-position embedding and reports isolated layer output relative MSE, not sequence-model perplexity.

| Frozen transition code | Relative output MSE, 128, mean ± seed SD | Relative output MSE, 512, mean ± seed SD |
|---|---:|---:|
| 16-angle nearest phase | 0.2365 ± 0.111 | 0.2680 ± 0.129 |
| 16-angle coefficient-fitted offset | 0.2185 ± 0.115 | 0.2395 ± 0.125 |
| 16-angle train-response-fitted offset | 0.1975 ± 0.0877 | 0.2194 ± 0.103 |
| 2+2-bit Cartesian nearest complex pole | 0.3279 ± 0.171 | 0.3425 ± 0.174 |
| 2+2-bit Cartesian train-response-fitted pole | 0.2830 ± 0.106 | 0.3007 ± 0.116 |

Phase response fitting beat the fitted Cartesian grid in each seed at length 512, but this compares different *codebook geometries*. A 16-point Cartesian circular codebook is algebraically equivalent to the same phase codebook and would erase any representation-only claim. Our potential distinct hypothesis is **training-calibrated recurrent memory preservation under an explicitly stored 4-bit transition budget**, not quantum hardware, phase notation alone, or a universal advantage of polar codes. Establishing that hypothesis requires a meaningful learned task model, equal-budget strong circular/Cartesian controls, task metrics, and a separate validation/test protocol. Existing quantum-inspired diagonal SSM work includes [iFairy](https://arxiv.org/abs/2508.05571), [Fairy2i](https://arxiv.org/abs/2512.02901), and [QS4D](https://arxiv.org/abs/2507.06079); novelty must be checked against their actual mechanisms.

Promotion gate: the initial coordinate-optimization gate **failed**. The frozen operator diagnostic passed its limited phase-vs-grid comparison, which authorizes another small task-level test, not full-data training or a publication conclusion. The next low-cost experiment should quantize a frozen trained Cartesian complex model's transition only, measure selected held-out next-token NLL and long-history response, add circular codebook and radius-aware Cartesian controls, and require reproducible paired gains before any GPU-intensive sweep.

## Frozen language-model next-token screen, 2026-09-16

The proposed follow-up completed on all three trained Cartesian-complex checkpoints. The model is the same custom 697,088-parameter hybrid: 8K train-only BPE vocabulary, width 64, two layers, context 128, with a complex diagonal recurrent first layer and causal-attention second layer. Only the first layer's 64 transition angles were changed. Evaluation covered the same fixed 16,384 original WikiText-103 validation targets; test data remained untouched. Each quantized transition export contains 64 four-bit indices, 64 unchanged FP32 decay values, and one FP32 offset, totaling 292 bytes.

| Frozen LM transition | Mean validation NLL | Mean perplexity | Mean NLL increase from FP32 |
|---|---:|---:|---:|
| FP32 transition | 6.510978 | 672.488 | 0 |
| Nearest 16-angle phase | 6.514951 | 675.168 | 0.003974 |
| Coefficient-fitted 16-angle phase | 6.514543 | 674.894 | 0.003566 |
| Response-fitted 16-angle phase | **6.513567** | **674.233** | **0.002589** |
| Response-fitted 16-point circular Cartesian control | **6.513567** | **674.233** | **0.002589** |
| Response-fitted square-grid directions | 6.516995 | 676.554 | 0.006017 |

The response-fitted circular quantizer preserved next-token NLL better than the square-grid direction control and stayed below the predeclared 0.01 NLL degradation gate. The phase and circular Cartesian controls were bit-for-bit equivalent after decoding and produced identical metrics. Consequently, this result supports **response-calibrated circular transition quantization**, but it provides no evidence that a qubit interpretation or polar notation is responsible. A credible next method must add something beyond the equivalent circular codebook, such as jointly optimized nonuniform phase states, mode-sensitive bit allocation, or a hardware-motivated constraint, and compare it with learned classical circular quantizers.

## Equal-budget novelty probes, 2026-09-16

Two additions were screened at the same 292-byte transition budget. The warped codebook used 16 states `2πk/16 + α sin(2πk/16)` and spent the single FP32 metadata value on a training-response-calibrated warp. The adaptive code deterministically assigned five phase bits to the 32 modes with the largest already-stored decay and three bits to the other 32 modes, retaining 256 total index bits and calibrating one shared offset. Deterministic decay ranking required no stored group mask.

| Frozen LM transition | Mean validation NLL | Mean perplexity | Mean NLL increase from FP32 |
|---|---:|---:|---:|
| Uniform response-calibrated 4-bit circle | **6.513691** | **674.317** | **0.002713** |
| One-parameter warped 4-bit circle | 6.513952 | 674.493 | 0.002975 |
| Decay-ranked adaptive 5/3-bit phase | 6.516453 | 676.181 | 0.005475 |

Neither proposed addition beat the uniform response-calibrated circle across all three seeds. They should not be promoted to a larger model in their present forms. The negative result narrows the next method: estimate per-mode sensitivity from the task loss rather than decay alone, and learn a circular codebook with a training-only objective while charging all codebook metadata. A Qwen experiment should begin as a frozen, subset-only feasibility study rather than quantization-aware training.

## Qwen3.5-0.8B frozen projection feasibility, 2026-09-16

The official Qwen3.5-0.8B-Base checkpoint was loaded in BF16 on the RTX 3090. It has 752,393,024 parameters and a hybrid text stack with Gated DeltaNet and full-attention layers. Inspection showed that its recurrent quantities (`A_log` and `dt_bias`) are real-valued rather than complex transition phases. The feasibility test therefore quantized only layer 0's frozen `linear_attn.in_proj_qkv.weight` tensor of shape 6144×1024. Adjacent real coefficients were paired as complex coordinates. This pairing is arbitrary and basis-dependent; the experiment tests the encoding, not a native quantum or phase structure in Qwen.

Each paired-weight method spent eight index bits per two real values. Row-wise real4 stored two four-bit scalar indices; polar variants divided the same eight bits between magnitude and phase. Every quantized payload was 3,170,304 bytes including one FP32 scale per row, compared with 12,582,912 BF16 tensor bytes. The model was frozen. Three independent fixed samples of 2,048 real WikiText-2 validation targets were evaluated; no training or additional-layer quantization occurred.

| One-projection encoding | Mean NLL increase from BF16 across three samples |
|---|---:|
| Row-wise real4 | +0.009888 |
| Polar magnitude4 + phase4 | +0.003952 |
| **Polar magnitude3 + phase5** | **+0.001260** |
| Polar magnitude5 + phase3 | +0.019343 |

At equal payload size, a polar variant beat real4 on each confirmation sample. Magnitude3+phase5 had the best mean preservation, while the sample-wise winner alternated between magnitude3+phase5 and magnitude4+phase4. This is promising screening evidence, but 6,144 targets, one tensor, one checkpoint, and an arbitrary pairing do not establish general language-model superiority. The next gate should predeclare the complex pairing, evaluate several projection types and layers independently, include GPTQ/AWQ-style or Hessian-aware four-bit baselines, and use a larger held-out sample before any quantization-aware training.

## Qwen cross-layer pairing and clipping control, 2026-09-16

A predeclared follow-up independently quantized `linear_attn.in_proj_qkv.weight` in DeltaNet layers 0, 8, and 16. It used the same 2,048 fixed real WikiText-2 validation targets for every comparison. The equal-byte real4 control selected one of ten clipping ratios per output row using weight reconstruction MSE. Polar magnitude3+phase5 used the same clipping search and compared adjacent pairing `(w[2j],w[2j+1])` with split-half pairing `(w[j],w[j+d/2])`. Each method retained the same 3,170,304-byte payload. Tensors were changed one at a time; errors did not accumulate.

| Layer | Optimized real4 ΔNLL | Polar 3+5 adjacent ΔNLL | Polar 3+5 split-half ΔNLL |
|---:|---:|---:|---:|
| 0 | +0.005817 | **+0.004158** | +0.008692 |
| 8 | **+0.000052** | +0.001323 | +0.000511 |
| 16 | +0.001907 | **+0.001562** | +0.002187 |

Adjacent polar pairing beat optimized real4 in two of three layers; split-half pairing beat it in zero of three. Polar reconstruction MSE was lower in every layer, but task NLL did not follow reconstruction MSE consistently. The all-layers promotion requirement failed. This establishes pairing sensitivity and prevents a general polar-weight claim. No cumulative quantization or training run was launched. A next method would need task-aware pairing or rotation learned solely on calibration data and must be compared with an equally task-aware classical baseline.

## Task-aware Qwen pairing gate, 2026-09-16

The next screen separated 512 calibration targets from 2,048 evaluation targets. For each of layers 0, 8, and 16, polar magnitude3+phase5 selected one of four deterministic pairing rules by calibration NLL. Real4 selected one of four global clipping multipliers by the identical calibration metric. Each export added one candidate-ID byte to the same weights-and-scales payload, for 3,170,305 bytes. Layers were evaluated independently in the otherwise frozen BF16 model.

| Layer | Selected real4 clip, evaluation ΔNLL | Selected polar pairing, evaluation ΔNLL |
|---:|---:|---:|
| 0 | 0.95, +0.008660 | reverse-half, **+0.004083** |
| 8 | 0.85, +0.001316 | reverse-half, **-0.000176** |
| 16 | 0.85, +0.004793 | split-half, **-0.000703** |

Calibration-selected polar pairing beat calibration-selected real4 clipping in all three held-out layer evaluations, satisfying the predeclared promotion gate. This result is stronger than fixed adjacent pairing because pairing was selected without evaluation-label access and its metadata was charged. It still does not equal a GPTQ/AWQ comparison: the classical control adapts clipping, while polar adapts a structural pairing choice, and calibration contains only 512 targets. The warranted next experiment is a small cumulative quantization of these three tensors using frozen calibration choices, evaluated on a larger disjoint sample alongside a cumulative task-calibrated real4 control. Full-model quantization and training remain premature.

## Full cumulative Qwen DeltaNet projection experiment, 2026-09-16

The cumulative experiment quantized `linear_attn.in_proj_qkv.weight` simultaneously in all 18 Gated DeltaNet layers of Qwen3.5-0.8B-Base. Four fixed 129-token windows from the WikiText-2 training split calibrated each layer independently. Final evaluation covered every one of the 261,284 next-token targets in the Qwen-tokenized WikiText-2 validation stream, using contiguous 128-target blocks with recurrent state reset at block boundaries. The test split was untouched. No parameters were trained.

Every quantized method exported 57,065,490 bytes for the 18 tensors, including per-row FP32 scales, eight index bits per real pair, and one pairing/clipping candidate byte per tensor. The corresponding BF16 tensors occupy 226,492,416 bytes, so this portion of the model is compressed by 3.968×. This ratio applies only to the selected DeltaNet input projections, not the complete checkpoint.

| Cumulative method across 18 projections | Validation NLL | Perplexity | ΔNLL from BF16 |
|---|---:|---:|---:|
| BF16 | 3.436621 | 31.082 | 0 |
| Task-calibrated real4 | 3.458566 | 31.771 | +0.021945 |
| Fixed-adjacent polar magnitude3+phase5 | 3.452048 | 31.565 | +0.015427 |
| **Task-selected polar magnitude3+phase5** | **3.445954** | **31.373** | **+0.009333** |

Task-selected polar pairing reduced the NLL degradation by 57.5% relative to the equal-byte task-calibrated real4 control. It also improved on fixed adjacent polar by 0.006094 NLL, showing that calibration-selected pairing matters after errors accumulate across layers. This is the strongest result in the project so far because it covers all relevant Qwen DeltaNet input projections and the complete validation stream with actual packed exports. The remaining central limitation is the baseline: task-calibrated clipping is stronger than round-to-nearest real4, but GPTQ, AWQ, and other Hessian/activation-aware four-bit baselines have not yet been run. The current evidence supports a claim about **calibration-selected complex pairing plus magnitude/phase bit allocation for hybrid-model projections**, not quantum computation or native complex Qwen states.

## Activation-aware and block-GPTQ controls, 2026-09-16

Two stronger real4 controls were applied cumulatively to the same 18 tensors and evaluated on the identical complete validation stream. Available production packages did not expose Qwen3.5's custom DeltaNet projections, so these are transparent local implementations rather than official AutoAWQ or GPTQModel outputs. The AWQ-style control searches activation-RMS channel scaling exponents and explicitly stores channel scales. Block-GPTQ uses calibration activation Hessians in 128-column groups, one-percent damping, and sequential within-group error propagation. Its eight row/group FP32 scales per tensor make its export larger than polar.

| Method | Validation NLL | Perplexity | ΔNLL | Export bytes |
|---|---:|---:|---:|---:|
| **Task-selected polar magnitude3+phase5** | **3.445954** | **31.373** | **+0.009333** | **57,065,490** |
| Block-GPTQ real4 | 3.449311 | 31.479 | +0.012689 | 60,162,048 |
| Task-calibrated real4 | 3.458566 | 31.771 | +0.021945 | 57,065,490 |
| AWQ-style real4 | 3.465755 | 32.001 | +0.029134 | 57,139,200 |

Polar remains better than block-GPTQ by 0.003356 NLL while using 3,096,558 fewer bytes (5.1% less than block-GPTQ). AWQ-style scaling performs poorly on these projections and often selects the zero-scaling exponent. These controls materially strengthen the result, but an official backend comparison, a learned vector-codebook baseline, additional checkpoints/datasets, and packed-kernel latency remain necessary.

## Learned Cartesian product-codebook control, 2026-09-16

A learned classical codebook baseline reused each layer's training-selected pairing, normalized complex pairs by a per-row magnitude scale, and learned separate 16-level real and imaginary codebooks by Lloyd updates. It spends four bits on each Cartesian coordinate, the same eight index bits per real pair as polar. All codebook values, pairing IDs, and row scales are included in the 57,067,794-byte export. Evaluation again covers the complete 261,284-target WikiText-2 validation stream.

| Method | Validation NLL | Perplexity | ΔNLL | Export bytes |
|---|---:|---:|---:|---:|
| **Task-selected polar magnitude3+phase5** | **3.445954** | **31.373** | **+0.009333** | **57,065,490** |
| Learned Cartesian product codebook | 3.456411 | 31.703 | +0.019790 | 57,067,794 |

The learned product codebook does not beat polar and nearly doubles the NLL degradation. This closes the first scheduled stage without invalidating the claim. It is a product codebook rather than unrestricted 256-centroid vector quantization, so a fully general vector codebook remains a possible stronger control.

## TinyStories cross-dataset replication, 2026-09-16

The second-dataset stage used eight fixed calibration windows from the TinyStories training split and evaluated last-token NLL for every one of the 21,990 nonempty examples in the official validation parquet, using up to 128 preceding tokens. Pairing and Hessian calibration used training text only; the test split was untouched. All 18 DeltaNet input projections were quantized cumulatively.

| Method | Validation NLL | Perplexity | ΔNLL | Export bytes |
|---|---:|---:|---:|---:|
| BF16 | 1.717422 | 5.570 | 0 | 0 |
| Block-GPTQ real4 | **1.719900** | **5.584** | **+0.002479** | 60,162,048 |
| Task-selected polar magnitude3+phase5 | 1.722666 | 5.599 | +0.005245 | 57,065,490 |

Polar did not reproduce its WikiText-2 advantage over block-GPTQ. Although polar uses 5.1% fewer bytes, its NLL degradation is more than twice GPTQ's on this dataset and evaluation protocol. This invalidates a dataset-general superiority claim and triggers the predeclared stop rule. The current defensible conclusion is that calibration-selected polar pairing is competitive and can outperform strong real quantizers on WikiText-2, but its advantage does not transfer uniformly. Further work requires a revised method or a narrower hypothesis rather than additional confirmatory scaling.

### Matched all-token TinyStories protocol

The prior TinyStories result used one last-token target per story, unlike WikiText-2's contiguous all-token evaluation. A follow-up removes this confound by evaluating exactly 261,284 contiguous TinyStories validation targets with the same 128-target reset policy and target count as WikiText-2. Pairing, clipping, and Hessian calibration again use TinyStories training text only.

| Method | Validation NLL | Perplexity | ΔNLL | Export bytes |
|---|---:|---:|---:|---:|
| BF16 | 2.314341 | 10.118 | 0 | 0 |
| **Task-selected polar magnitude3+phase5** | **2.319747** | **10.173** | **+0.005406** | **57,065,490** |
| Task-calibrated real4 | 2.322563 | 10.202 | +0.008222 | 57,065,490 |
| Block-GPTQ real4 | 2.323978 | 10.216 | +0.009637 | 60,162,048 |

Under the matched all-token protocol, polar again wins and uses fewer bytes than block-GPTQ. The conflict with the complete last-token-per-story result is scientifically important: codec ranking depends on the prediction objective and context sampling policy. The evidence now supports cross-corpus all-token preservation under matched resets, but not universal superiority across evaluation tasks. The earlier stop decision was appropriate for the original general claim; future work should predeclare the target task and analyze why last-token prediction favors GPTQ.

## Layer-adaptive mixed codec and uncertainty, 2026-09-17

For each of the 18 DeltaNet input projections, a training-only calibration screen compared the already selected polar pairing with block-GPTQ while replacing that layer alone in the otherwise BF16 model. The resulting fixed plan chose polar for 13 layers and GPTQ for five. One codec/pairing byte per tensor is charged. We then applied all choices cumulatively and reevaluated the complete matched TinyStories stream. The 95% intervals below use 5,000 paired bootstrap replicates over contiguous 128-target blocks and compare each method with BF16.

| Method | Validation NLL | Perplexity | ΔNLL (95% paired CI) | Export bytes |
|---|---:|---:|---:|---:|
| BF16 | 2.314341 | 10.118 | 0 | 0 |
| Task-real4 | 2.322563 | 10.202 | +0.008222 [0.007428, 0.009030] | 57,065,490 |
| Polar magnitude3+phase5 | 2.319747 | 10.173 | +0.005406 [0.004666, 0.006148] | 57,065,490 |
| Block-GPTQ real4 | 2.323978 | 10.216 | +0.009637 [0.009123, 0.010181] | 60,162,066 |
| **Mixed polar/GPTQ** | **2.315292** | **10.128** | **+0.000951 [0.000243, 0.001658]** | **57,925,650** |

The mixed codec cuts the polar degradation by 82.4% and the GPTQ degradation by 90.1%. Its export is 3.7% smaller than all-GPTQ. The mixed interval excludes zero, so BF16 remains measurably better, while its upper endpoint is below the lower endpoints of the uniform-codec intervals. Since the calibration decisions were made per layer rather than by optimizing the cumulative mixture, this large improvement may partly reflect favorable cancellation or interaction among quantization errors.

## Frozen mixed-codec replication on WikiText-2, 2026-09-17

We froze the TinyStories selection rule and reran it with the original four WikiText-2 training calibration windows and all 261,284 WikiText-2 validation targets. Calibration again chose 13 polar and five GPTQ layers, but the GPTQ sets differed: TinyStories selected layers 2, 8, 13, 20, and 22; WikiText-2 selected 0, 1, 8, 9, and 22. Only layers 8 and 22 overlap.

| Method | Validation NLL | Perplexity | ΔNLL vs BF16 (95% paired CI) | Export bytes |
|---|---:|---:|---:|---:|
| BF16 | 3.436621 | 31.082 | 0 | 0 |
| **Polar magnitude3+phase5** | **3.445954** | **31.373** | **+0.009333 [0.008380, 0.010303]** | **57,065,490** |
| Mixed polar/GPTQ | 3.447188 | 31.412 | +0.010567 [0.009605, 0.011490] | 57,925,650 |
| Block-GPTQ real4 | 3.450313 | 31.510 | +0.013691 [0.012866, 0.014511] | 60,162,066 |

The direct mixed-minus-polar difference is +0.001234 with 95% CI [0.000368, 0.002087], so mixed is significantly worse. The direct mixed-minus-GPTQ difference is -0.003124 [-0.004023, -0.002243], so mixed is significantly better than GPTQ. The TinyStories mixed-codec gain therefore fails corpus-level replication. This negative result narrows the method: uniform calibration-selected polar remains the more stable all-token codec across the two corpora, while layer-wise codec selection is corpus-sensitive and needs a stronger cumulative calibration objective before further scaling.

## Greedy cumulative-selection gate, 2026-09-17

An exploratory follow-up selected codecs in ascending layer order while retaining every earlier choice, so each decision minimized cumulative rather than isolated-layer NLL. Selection used the original four WikiText-2 training windows. Promotion required the finished mixture to beat uniform polar on four distinct, fixed training windows before any validation evaluation.

The selector chose 12 polar and six GPTQ layers. On the disjoint train-only gate, its NLL was 3.517684 versus 3.502284 for uniform polar, a degradation of 0.015400. It therefore failed the predeclared promotion gate. The script stopped without evaluating validation data. This result rejects the simple greedy correction and avoids another full GPU sweep; future selectors need a stronger global objective rather than additional layer-wise heuristics.

## Qwen3.5-2B checkpoint gate, 2026-09-17

The next-smallest official base hybrid checkpoint, Qwen3.5-2B-Base, has 1,881,825,088 parameters and the same 18 DeltaNet projection layers, with each tested matrix enlarged to 6,144 by 2,048. A bounded screen changed layers 0, 8, and 16 independently. Four WikiText-2 training windows selected polar pairing and real4 clipping; 16 fixed validation windows supplied 2,048 targets. Each polar or real4 tensor export used 6,316,032 bytes, while block-GPTQ used 6,684,672 bytes.

| Layer | Real4 ΔNLL | Polar 3+5 ΔNLL | Block-GPTQ ΔNLL | Winner |
|---:|---:|---:|---:|---|
| 0 | +0.002222 | +0.002452 | **+0.000643** | Block-GPTQ |
| 8 | **-0.000158** | +0.000817 | +0.000953 | Real4 |
| 16 | **-0.000006** | +0.000614 | +0.000640 | Real4 |

Polar won no layer and failed the predeclared promotion gate, so no cumulative or complete-validation 2B run was launched. The 0.8B polar advantage is therefore checkpoint-sensitive under this small screen.

## Packed Triton decoder benchmark, 2026-09-17

A Triton prototype reads the actual packed three-bit magnitude and five-bit phase streams for the 6,144 by 1,024 layer-0 projection, reconstructs the selected reverse-half pairing on GPU, and then calls BF16 GEMM. Reconstruction agrees with the reference quantizer to 1.49e-8 maximum absolute error and 7.72e-10 RMSE. The packed tensor occupies 3,170,304 bytes versus 12,582,912 BF16 bytes, a 3.969x reduction. Median packed decode time is 0.0382 ms.

| Input tokens | Resident BF16 GEMM | Packed decode + GEMM | Slowdown |
|---:|---:|---:|---:|
| 1 | 0.0229 ms | 0.1040 ms | 4.53x |
| 16 | 0.0232 ms | 0.1048 ms | 4.51x |
| 128 | 0.0417 ms | 0.1243 ms | 2.98x |

This prototype establishes correct execution from the packed representation but not an inference speedup. Materializing the full BF16 matrix and launching cuBLAS dominates small batches. A fused packed matrix multiplication is required before making latency or memory-bandwidth claims.

## Learned phase-lattice rotation gates, 2026-09-17

To move beyond four fixed pairings, an activation-aware variant searches 16 global offsets within one phase-quantization bin for every pairing. Activation-weighted reconstruction error selects the offset, and training-only NLL selects the pairing. The export adds one offset byte per tensor. Evaluation uses 2,048 disjoint WikiText-2 validation targets at layers 0, 8, and 16.

| Checkpoint | Layer | Rotated ΔNLL | Standard polar ΔNLL | Rotated minus standard |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0 | +0.004985 | +0.004457 | +0.000528 |
| Qwen3.5-0.8B | 8 | +0.001095 | +0.002607 | **-0.001512** |
| Qwen3.5-0.8B | 16 | +0.002291 | +0.001038 | +0.001252 |
| Qwen3.5-2B | 0 | +0.000948 | -0.001999 | +0.002947 |
| Qwen3.5-2B | 8 | +0.001248 | +0.001356 | **-0.000108** |
| Qwen3.5-2B | 16 | +0.001997 | +0.001282 | +0.000715 |

The learned rotation improves only layer 8 in each checkpoint and fails both promotion gates. Consequently, the conditional Qwen3.5-9B experiment is not launched. A single global lattice offset is too weak; future learned geometry would need per-group rotations or a jointly optimized codebook, with its metadata charged.

## Fused packed polar matrix multiplication, 2026-09-17

A second Triton kernel decodes magnitude and phase inside the matrix-multiplication reduction and never materializes the full weight matrix. Against a resident BF16 cuBLAS baseline on the same 6,144 by 1,024 projection:

| Input tokens | Fused packed | Resident BF16 | Slowdown | Output RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.0449 ms | 0.0232 ms | 1.93x | 0.00110 |
| 16 | 0.2699 ms | 0.0234 ms | 11.52x | 0.00126 |
| 128 | 2.2096 ms | 0.0424 ms | 52.15x | 0.00125 |

Fusion substantially reduces single-token overhead relative to decode-then-GEMM, but the naive kernel scales poorly with batch size. It evaluates trigonometric functions inside the reduction and does not use tensor cores. A useful production kernel needs phase lookup tables, better tiling, and tensor-core-friendly accumulation.

## Qwen3.8-27B official-weight geometry, 2026-09-20

We chose the latest dense Qwen3.8-27B checkpoint for the next architecture screen because it retains both Gated DeltaNet and full-attention blocks. Its official BF16 language-model index totals 55.56 GB, so the first bounded analysis downloads only shard 1 and evaluates actual complete tensors without loading the full model. Equal-byte real4 and polar 3+5 use ten row-scale candidates; polar selects among four pairings by weight MSE. All rows are processed, and scale and pairing metadata are charged.

| Tensor | Shape | Real4 MSE | Best polar MSE | Polar MSE reduction |
|---|---:|---:|---:|---:|
| DeltaNet input QKV | 10,240×5,120 | 3.0410e-6 | 2.9777e-6 | 2.1% |
| Attention Q | 12,288×5,120 | 4.0363e-6 | 3.8263e-6 | 5.2% |
| MLP up | 17,408×5,120 | 1.3820e-6 | 1.3753e-6 | 0.5% |

The largest-magnitude quartile accounts for 43–48% of polar squared error. This is a genuine weight-reconstruction result on current Qwen weights, including a full-attention projection, but it says nothing yet about language-model NLL. The 27B checkpoint is post-trained, and a full BF16 reference nearly fills the workstation's 60 GiB RAM; activation- and task-level validation require a separate feasible execution plan. See [the full result](../results/qwen38_27b_weight_geometry_full_v2/summary.md).
