# Selection-fairness check: the earlier draft's protocol, applied to every codec

Per-layer candidate selection by calibration NLL (4 WikiText-2 training windows), then cumulative application; 261,248 validation targets.
BF16 NLL 3.436670.

| Method | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| polar_legacy_nllsel | 3.448029 | +0.011358 | [+0.010435, +0.012303] |
| real4_uniform_nllsel | 3.461926 | +0.025256 | [+0.024180, +0.026343] |
| real4_lloyd_nllsel | 3.444279 | +0.007609 | [+0.006736, +0.008488] |
| vq2d_nllsel | 3.447359 | +0.010689 | [+0.009945, +0.011438] |
| vq2d_h_nllsel | 3.447507 | +0.010837 | [+0.010139, +0.011549] |

## Paired head-to-head differences

| Comparison | ΔNLL | 95% CI |
|---|---:|:--|
| polar_legacy_nllsel − real4_uniform_nllsel | -0.013898 | [-0.015253, -0.012523] |
| polar_legacy_nllsel − real4_lloyd_nllsel | +0.003750 | [+0.002680, +0.004813] |
| polar_legacy_nllsel − vq2d_nllsel | +0.000670 | [-0.000366, +0.001746] |
| polar_legacy_nllsel − vq2d_h_nllsel | +0.000522 | [-0.000532, +0.001566] |
| real4_uniform_nllsel − real4_lloyd_nllsel | +0.017648 | [+0.016451, +0.018835] |
| real4_uniform_nllsel − vq2d_nllsel | +0.014567 | [+0.013351, +0.015790] |
| real4_uniform_nllsel − vq2d_h_nllsel | +0.014419 | [+0.013156, +0.015651] |
| real4_lloyd_nllsel − vq2d_nllsel | -0.003080 | [-0.004084, -0.002056] |
| real4_lloyd_nllsel − vq2d_h_nllsel | -0.003228 | [-0.004227, -0.002271] |
| vq2d_nllsel − vq2d_h_nllsel | -0.000148 | [-0.000975, +0.000660] |
