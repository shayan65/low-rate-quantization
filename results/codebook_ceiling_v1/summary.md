# Codebook ceiling at 8 index bits per weight pair

All columns are weight MSE relative to the **Lloyd-fitted real-4** control
(< 1.000 beats it). `vq2d` is an unconstrained 256-point 2-D codebook and
upper-bounds every pairing-based scheme at this budget.

| Model | Tensor | real4-Lloyd MSE | real4-uniform | polar-legacy | polar-fitted | best split | vq2d ceiling |
|---|---|---:|---:|---:|---:|:--:|---:|
| Qwen3.5-0.8B-Base | 0.linear_attn.in_proj_qkv | 3.2517e-06 | 1.308 | 1.194 | 1.065 | 8x32 | 0.759 |
| Qwen3.5-0.8B-Base | 8.linear_attn.in_proj_qkv | 3.1353e-06 | 1.214 | 1.156 | 1.064 | 8x32 | 0.793 |
| Qwen3.5-0.8B-Base | 16.linear_attn.in_proj_qkv | 5.0393e-06 | 1.240 | 1.165 | 1.059 | 8x32 | 0.795 |
| Qwen3.8-27B-metadata | 0.linear_attn.in_proj_qkv | 2.5631e-06 | 1.186 | 1.162 | 1.073 | 8x32 | 0.818 |
| Qwen3.8-27B-metadata | 3.self_attn.q_proj | 3.3165e-06 | 1.217 | 1.154 | 1.060 | 8x32 | 0.791 |
| Qwen3.8-27B-metadata | 3.mlp.up_proj | 1.1824e-06 | 1.169 | 1.163 | 1.081 | 8x32 | 0.817 |
