# Generalization: fixed quantizer, varied evaluation

Qwen3.5-0.8B, 18 DeltaNet QKV projections. The quantizer is fitted
once (65,536 wikitext-2 train tokens, one seed); only the
evaluation set changes. `wt2_val_128` is the corpus and protocol every other
result used, so it is a harness check rather than new evidence.

| Evaluation | corpus | context | blocks | targets | BF16 NLL |
|---|---|---:|---:|---:|---:|
| wt2_val_128 | wikitext-2 val | 128 | 2,042 | 261,284 | 3.436710 |
| wt2_test_128 | wikitext-2 test | 128 | 2,048 | 262,143 | 3.394295 |
| tiny_128 | TinyStories | 128 | 2,048 | 262,143 | 2.314432 |
| wt2_test_512 | wikitext-2 test | 512 | 512 | 262,143 | 2.870202 |
| wt2_test_2048 | wikitext-2 test | 2048 | 128 | 262,143 | 2.567222 |

## vq8 vs scalar3

| Evaluation | ΔNLL | 95% CI | |
|---|---:|:--|:--|
| wt2_val_128 | -0.036118 | [-0.038414, -0.033768] | excludes 0 |
| wt2_test_128 | -0.038806 | [-0.041398, -0.036269] | excludes 0 |
| tiny_128 | -0.043775 | [-0.045718, -0.041850] | excludes 0 |
| wt2_test_512 | -0.035788 | [-0.038293, -0.033366] | excludes 0 |
| wt2_test_2048 | -0.031832 | [-0.034372, -0.029361] | excludes 0 |

## vq8 vs scalar3_g64

| Evaluation | ΔNLL | 95% CI | |
|---|---:|:--|:--|
| wt2_val_128 | -0.034303 | [-0.036660, -0.031971] | excludes 0 |
| wt2_test_128 | -0.039543 | [-0.042026, -0.036941] | excludes 0 |
| tiny_128 | -0.054717 | [-0.056691, -0.052628] | excludes 0 |
| wt2_test_512 | -0.035518 | [-0.037961, -0.033168] | excludes 0 |
| wt2_test_2048 | -0.031779 | [-0.034007, -0.029615] | excludes 0 |

## vq8 vs vq4

| Evaluation | ΔNLL | 95% CI | |
|---|---:|:--|:--|
| wt2_val_128 | -0.011451 | [-0.013743, -0.009221] | excludes 0 |
| wt2_test_128 | -0.011840 | [-0.014202, -0.009422] | excludes 0 |
| tiny_128 | -0.015677 | [-0.017468, -0.013876] | excludes 0 |
| wt2_test_512 | -0.015220 | [-0.017536, -0.012998] | excludes 0 |
| wt2_test_2048 | -0.012270 | [-0.014385, -0.010092] | excludes 0 |

## scalar3_g64 vs scalar3

| Evaluation | ΔNLL | 95% CI | |
|---|---:|:--|:--|
| wt2_val_128 | -0.001815 | [-0.003794, +0.000172] | **includes 0** |
| wt2_test_128 | +0.000737 | [-0.001213, +0.002710] | **includes 0** |
| tiny_128 | +0.010942 | [+0.009363, +0.012543] | excludes 0 |
| wt2_test_512 | -0.000270 | [-0.002243, +0.001744] | **includes 0** |
| wt2_test_2048 | -0.000053 | [-0.001964, +0.001852] | **includes 0** |
