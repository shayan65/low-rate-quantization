# Refit variability: do the reported contrasts survive refitting?

Qwen3.5-0.8B, 18 DeltaNet QKV projections, 261,284 validation targets, BF16 NLL 3.436710.
Recipe fixed throughout; only the calibration draw, codebook seed and
calibration size vary. `half-width` is the median paired-bootstrap
half-width within a single refit, which is what the paper's intervals
report.

## vq8 vs scalar3

Median within-refit bootstrap half-width: **0.002416**

| factor varied | range of the contrast | spread / half-width | values |
|---|---:|---:|:--|
| calibration draw (3 disjoint, 65k) | 0.004255 | 1.76x | -0.036118, -0.031863, -0.031933 |
| codebook seed (3, fixed calibration) | 0.005057 | 2.09x | -0.036118, -0.034865, -0.031061 |
| calibration size (16k-131k, nested) | 0.004474 | 1.85x | -0.033335, -0.035775, -0.036118, -0.037809 |

## vq8 vs scalar3_g64

Median within-refit bootstrap half-width: **0.002416**

| factor varied | range of the contrast | spread / half-width | values |
|---|---:|---:|:--|
| calibration draw (3 disjoint, 65k) | 0.007078 | 2.93x | -0.034303, -0.027224, -0.032842 |
| codebook seed (3, fixed calibration) | 0.005057 | 2.09x | -0.034303, -0.033050, -0.029246 |
| calibration size (16k-131k, nested) | 0.007110 | 2.94x | -0.038370, -0.033368, -0.034303, -0.040478 |

## vq8 vs vq4

Median within-refit bootstrap half-width: **0.002324**

| factor varied | range of the contrast | spread / half-width | values |
|---|---:|---:|:--|
| calibration draw (3 disjoint, 65k) | 0.003893 | 1.68x | -0.011451, -0.007558, -0.011318 |
| codebook seed (3, fixed calibration) | 0.006350 | 2.73x | -0.011451, -0.005101, -0.007008 |
| calibration size (16k-131k, nested) | 0.011189 | 4.82x | -0.007359, -0.014820, -0.011451, -0.018548 |

## scalar3_g64 vs scalar3

Median within-refit bootstrap half-width: **0.001983**

| factor varied | range of the contrast | spread / half-width | values |
|---|---:|---:|:--|
| calibration draw (3 disjoint, 65k) | 0.005548 | 2.80x | -0.001815, -0.004638, +0.000910 |
| codebook seed (3, fixed calibration) | 0.000000 | 0.00x | -0.001815, -0.001815, -0.001815 |
| calibration size (16k-131k, nested) | 0.007441 | 3.75x | +0.005035, -0.002406, -0.001815, +0.002669 |

