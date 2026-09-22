# Qwen3.8-27B weight geometry screen

Real official checkpoint weights. Weight-only reconstruction; no task accuracy measured.

| Tensor | Rows | Real4 MSE | Best polar MSE | Best pairing | Polar/real MSE |
|---|---:|---:|---:|---|---:|
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | 10240 | 3.0410026e-06 | 2.977745e-06 | split_half | 0.979 |
| model.language_model.layers.3.self_attn.q_proj.weight | 12288 | 4.0362902e-06 | 3.8263138e-06 | adjacent | 0.948 |
| model.language_model.layers.3.mlp.up_proj.weight | 17408 | 1.3820182e-06 | 1.3753266e-06 | split_half | 0.995 |
