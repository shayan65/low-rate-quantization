# Step 0: rate-resolved ceiling sweep

`vq/product` is free-codebook MSE divided by the matched-rate scalar product
grid MSE. Below 1.000 favours joint coding. Every arm carries the same g128
FP16 scale overhead, so it cancels.

## 0.linear_attn.in_proj_qkv

| rotation | bits/wt | dim=2 | dim=4 | dim=8 |
|---|---:|---:|---:|---:|
| none | 1.585 | 0.956 | 0.857 | 0.785 |
| none | 2.000 | 0.907 | 0.808 | — |
| none | 3.000 | 0.867 | 0.758 | — |
| none | 4.000 | 0.869 | — | — |
| hadamard | 1.585 | 0.958 | 0.861 | 0.792 |
| hadamard | 2.000 | 0.911 | 0.815 | — |
| hadamard | 3.000 | 0.870 | 0.767 | — |
| hadamard | 4.000 | 0.884 | — | — |

## 16.linear_attn.in_proj_qkv

| rotation | bits/wt | dim=2 | dim=4 | dim=8 |
|---|---:|---:|---:|---:|
| none | 1.585 | 0.946 | 0.837 | 0.715 |
| none | 2.000 | 0.899 | 0.784 | — |
| none | 3.000 | 0.846 | 0.723 | — |
| none | 4.000 | 0.858 | — | — |
| hadamard | 1.585 | 0.947 | 0.864 | 0.748 |
| hadamard | 2.000 | 0.926 | 0.815 | — |
| hadamard | 3.000 | 0.868 | 0.752 | — |
| hadamard | 4.000 | 0.875 | — | — |

## 3.mlp.up_proj

| rotation | bits/wt | dim=2 | dim=4 | dim=8 |
|---|---:|---:|---:|---:|
| none | 1.585 | 0.943 | 0.858 | 0.790 |
| none | 2.000 | 0.913 | 0.810 | — |
| none | 3.000 | 0.860 | 0.762 | — |
| none | 4.000 | 0.874 | — | — |
| hadamard | 1.585 | 0.958 | 0.862 | 0.795 |
| hadamard | 2.000 | 0.912 | 0.815 | — |
| hadamard | 3.000 | 0.864 | 0.769 | — |
| hadamard | 4.000 | 0.887 | — | — |

## 3.self_attn.q_proj

| rotation | bits/wt | dim=2 | dim=4 | dim=8 |
|---|---:|---:|---:|---:|
| none | 1.585 | 0.953 | 0.847 | 0.774 |
| none | 2.000 | 0.903 | 0.798 | — |
| none | 3.000 | 0.853 | 0.747 | — |
| none | 4.000 | 0.874 | — | — |
| hadamard | 1.585 | 0.946 | 0.860 | 0.793 |
| hadamard | 2.000 | 0.912 | 0.815 | — |
| hadamard | 3.000 | 0.866 | 0.767 | — |
| hadamard | 4.000 | 0.883 | — | — |

## 8.linear_attn.in_proj_qkv

| rotation | bits/wt | dim=2 | dim=4 | dim=8 |
|---|---:|---:|---:|---:|
| none | 1.585 | 0.932 | 0.840 | 0.720 |
| none | 2.000 | 0.906 | 0.792 | — |
| none | 3.000 | 0.856 | 0.731 | — |
| none | 4.000 | 0.879 | — | — |
| hadamard | 1.585 | 0.946 | 0.861 | 0.746 |
| hadamard | 2.000 | 0.909 | 0.814 | — |
| hadamard | 3.000 | 0.865 | 0.749 | — |
| hadamard | 4.000 | 0.883 | — | — |

## Predeclared gates

- **H0** (rotation subsumes joint coding, stop if true): `False`
- **H1** (margin >=15% at ternary and >=2x the 4-bit margin): `False`

**Verdict: STOP: joint-coding advantage does not grow as rate falls**
