# Calibration mixture, with draws and a sample-count control

3 disjoint calibration draws per condition. `mixed50` spends half its budget on each domain; the `_half` conditions spend the same reduced budget on a single domain, which separates domain coverage from sample count.

| condition | code | worst-domain by draw | mean | range |
|---|---|---|---:|---:|
| wikitext | scalar3 | 0.183486, 0.190162, 0.197326 | 0.190325 | 0.013840 |
| wikitext | vq8 | 0.140296, 0.149136, 0.151607 | 0.147013 | 0.011311 |
| tinystories | scalar3 | 0.288304, 0.273917, 0.285724 | 0.282648 | 0.014387 |
| tinystories | vq8 | 0.199492, 0.182104, 0.192293 | 0.191296 | 0.017389 |
| mixed50 | scalar3 | 0.144351, 0.144442, 0.141652 | 0.143482 | 0.002791 |
| mixed50 | vq8 | 0.089604, 0.094118, 0.101910 | 0.095211 | 0.012307 |
| wikitext_half | scalar3 | 0.191117, 0.182358, 0.215563 | 0.196346 | 0.033205 |
| wikitext_half | vq8 | 0.142857, 0.160200, 0.164745 | 0.155934 | 0.021888 |
| tinystories_half | scalar3 | 0.296023, 0.287733, 0.281602 | 0.288453 | 0.014421 |
| tinystories_half | vq8 | 0.213931, 0.190821, 0.214691 | 0.206481 | 0.023870 |

## Mixed against each pure condition, paired per draw

| contrast | eval | mean Δ | range over draws | same sign |
|---|---|---:|---:|:--|
| mixed50_vs_wikitext (scalar3) | wt2_test | +0.022350 | 0.001569 | yes |
| mixed50_vs_tinystories (scalar3) | wt2_test | -0.139167 | 0.014598 | yes |
| mixed50_vs_wikitext_half (scalar3) | wt2_test | +0.017774 | 0.013067 | yes |
| mixed50_vs_tinystories_half (scalar3) | wt2_test | -0.144971 | 0.011721 | yes |
| mixed50_vs_wikitext (scalar3) | tinystories | -0.133362 | 0.004396 | yes |
| mixed50_vs_tinystories (scalar3) | tinystories | +0.022804 | 0.012321 | yes |
| mixed50_vs_wikitext_half (scalar3) | tinystories | -0.139384 | 0.029833 | yes |
| mixed50_vs_tinystories_half (scalar3) | tinystories | +0.027619 | 0.011760 | yes |
| mixed50_vs_wikitext (vq8) | wt2_test | +0.011011 | 0.008184 | yes |
| mixed50_vs_tinystories (vq8) | wt2_test | -0.096086 | 0.021903 | yes |
| mixed50_vs_wikitext_half (vq8) | wt2_test | +0.007311 | 0.010504 | yes |
| mixed50_vs_tinystories_half (vq8) | wt2_test | -0.111270 | 0.027624 | yes |
| mixed50_vs_wikitext (vq8) | tinystories | -0.104183 | 0.010674 | yes |
| mixed50_vs_tinystories (vq8) | tinystories | +0.016595 | 0.006003 | yes |
| mixed50_vs_wikitext_half (vq8) | tinystories | -0.113104 | 0.019177 | yes |
| mixed50_vs_tinystories_half (vq8) | tinystories | +0.017898 | 0.013396 | yes |
