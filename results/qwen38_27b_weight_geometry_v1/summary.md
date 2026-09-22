# Qwen3.8-27B weight geometry screen

Real official checkpoint weights. Weight-only reconstruction; no task accuracy measured.

| Tensor | Rows | Real4 MSE | Best polar MSE | Best pairing | Polar/real MSE |
|---|---:|---:|---:|---|---:|
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | 256 | 3.0848046e-06 | 3.0329797e-06 | split_half | 0.983 |
| model.language_model.layers.3.self_attn.q_proj.weight | 256 | 4.1512235e-06 | 3.9000834e-06 | split_half | 0.940 |
| model.language_model.layers.3.mlp.up_proj.weight | 256 | 1.3691077e-06 | 1.3691055e-06 | adjacent | 1.000 |
