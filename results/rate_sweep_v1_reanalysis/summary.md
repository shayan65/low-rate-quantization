# Corrected historical rate sweep

These are measured weight-reconstruction errors, not language-model quality results.
Ratios compare observed VQ MSE with the scalar product baseline at equal ideal
index entropy. Historical FP32 scales/codebooks, packing, and metadata overhead
were not measured as a deployable format. These fits are not optimality bounds.

## Qwen3.5-0.8B-Base / 0.linear_attn.in_proj_qkv

| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |
|---|---:|---:|---:|---:|
| none | 1.584963 | 0.927923 | 0.826763 | 0.692776 |
| none | 2.000000 | 0.886318 | 0.767692 | — |
| none | 3.000000 | 0.834261 | 0.705350 | — |
| none | 4.000000 | 0.852211 | — | — |
| hadamard | 1.584963 | 0.955797 | 0.857640 | 0.733943 |
| hadamard | 2.000000 | 0.917143 | 0.810907 | — |
| hadamard | 3.000000 | 0.863809 | 0.746627 | — |
| hadamard | 4.000000 | 0.886737 | — | — |

## Qwen3.5-0.8B-Base / 16.linear_attn.in_proj_qkv

| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |
|---|---:|---:|---:|---:|
| none | 1.584963 | 0.945564 | 0.836798 | 0.715307 |
| none | 2.000000 | 0.898765 | 0.783813 | — |
| none | 3.000000 | 0.846403 | 0.722546 | — |
| none | 4.000000 | 0.857528 | — | — |
| hadamard | 1.584963 | 0.947020 | 0.863627 | 0.747992 |
| hadamard | 2.000000 | 0.926414 | 0.814857 | — |
| hadamard | 3.000000 | 0.867989 | 0.752378 | — |
| hadamard | 4.000000 | 0.875216 | — | — |

## Qwen3.5-0.8B-Base / 8.linear_attn.in_proj_qkv

| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |
|---|---:|---:|---:|---:|
| none | 1.584963 | 0.931692 | 0.840400 | 0.719978 |
| none | 2.000000 | 0.906164 | 0.791987 | — |
| none | 3.000000 | 0.856314 | 0.730761 | — |
| none | 4.000000 | 0.878972 | — | — |
| hadamard | 1.584963 | 0.945964 | 0.860630 | 0.746142 |
| hadamard | 2.000000 | 0.909386 | 0.814319 | — |
| hadamard | 3.000000 | 0.864727 | 0.748909 | — |
| hadamard | 4.000000 | 0.883056 | — | — |

## Qwen3.8-27B-metadata / 0.linear_attn.in_proj_qkv

| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |
|---|---:|---:|---:|---:|
| none | 1.584963 | 0.955886 | 0.857286 | 0.784915 |
| none | 2.000000 | 0.906613 | 0.807667 | — |
| none | 3.000000 | 0.866818 | 0.758483 | — |
| none | 4.000000 | 0.868790 | — | — |
| hadamard | 1.584963 | 0.958357 | 0.861365 | 0.792463 |
| hadamard | 2.000000 | 0.910636 | 0.815461 | — |
| hadamard | 3.000000 | 0.870076 | 0.766597 | — |
| hadamard | 4.000000 | 0.883950 | — | — |

## Qwen3.8-27B-metadata / 3.mlp.up_proj

| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |
|---|---:|---:|---:|---:|
| none | 1.584963 | 0.943437 | 0.857876 | 0.790041 |
| none | 2.000000 | 0.912837 | 0.810469 | — |
| none | 3.000000 | 0.859591 | 0.762374 | — |
| none | 4.000000 | 0.873988 | — | — |
| hadamard | 1.584963 | 0.958182 | 0.861908 | 0.794908 |
| hadamard | 2.000000 | 0.911533 | 0.814880 | — |
| hadamard | 3.000000 | 0.864299 | 0.768683 | — |
| hadamard | 4.000000 | 0.886710 | — | — |

## Qwen3.8-27B-metadata / 3.self_attn.q_proj

| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |
|---|---:|---:|---:|---:|
| none | 1.584963 | 0.953389 | 0.846501 | 0.774093 |
| none | 2.000000 | 0.902506 | 0.797549 | — |
| none | 3.000000 | 0.853369 | 0.747001 | — |
| none | 4.000000 | 0.873869 | — | — |
| hadamard | 1.584963 | 0.946423 | 0.860341 | 0.793232 |
| hadamard | 2.000000 | 0.911749 | 0.815185 | — |
| hadamard | 3.000000 | 0.866394 | 0.767223 | — |
| hadamard | 4.000000 | 0.882869 | — | — |

## Equal-tensor post-rotation mean MSE reductions

Models are separated below; ALL_MODELS gives each of the six tensors equal weight.

| Model | Levels | Dim | Tensors | MSE reduction |
|---|---:|---:|---:|---:|
| ALL_MODELS | 3 | 2 | 6 | 4.8043% |
| ALL_MODELS | 3 | 4 | 6 | 13.9082% |
| ALL_MODELS | 3 | 8 | 6 | 23.1887% |
| ALL_MODELS | 4 | 2 | 6 | 8.5523% |
| ALL_MODELS | 4 | 4 | 6 | 18.5732% |
| ALL_MODELS | 8 | 2 | 6 | 13.3784% |
| ALL_MODELS | 8 | 4 | 6 | 24.1597% |
| ALL_MODELS | 16 | 2 | 6 | 11.6910% |
| Qwen3.5-0.8B-Base | 3 | 2 | 3 | 5.0406% |
| Qwen3.5-0.8B-Base | 3 | 4 | 3 | 13.9368% |
| Qwen3.5-0.8B-Base | 3 | 8 | 3 | 25.7308% |
| Qwen3.5-0.8B-Base | 4 | 2 | 3 | 8.2352% |
| Qwen3.5-0.8B-Base | 4 | 4 | 3 | 18.6639% |
| Qwen3.5-0.8B-Base | 8 | 2 | 3 | 13.4492% |
| Qwen3.5-0.8B-Base | 8 | 4 | 3 | 25.0695% |
| Qwen3.5-0.8B-Base | 16 | 2 | 3 | 11.8330% |
| Qwen3.8-27B-metadata | 3 | 2 | 3 | 4.5679% |
| Qwen3.8-27B-metadata | 3 | 4 | 3 | 13.8795% |
| Qwen3.8-27B-metadata | 3 | 8 | 3 | 20.6466% |
| Qwen3.8-27B-metadata | 4 | 2 | 3 | 8.8694% |
| Qwen3.8-27B-metadata | 4 | 4 | 3 | 18.4824% |
| Qwen3.8-27B-metadata | 8 | 2 | 3 | 13.3077% |
| Qwen3.8-27B-metadata | 8 | 4 | 3 | 23.2499% |
| Qwen3.8-27B-metadata | 16 | 2 | 3 | 11.5490% |

## Historical decision gate

The H1 rule selects the best observed permitted dimension at each rate. It is not
a fixed-dimension causal test and cannot falsify a general coding mechanism.

- Coverage complete: `True`
- H0 rotation-closure threshold met on every tensor: `False`
- H1 margin ≥15% and ≥2× four-bit margin on every tensor: `False`

**STOP: the tested configurations failed the predeclared H1 gate**

Source: `results/rate_sweep_v1/results.json`

Source SHA-256: `75b5d2226c91b27dbae4b3d2e78629b17c0a1e456b25f216147fa6b1c33419a7`
