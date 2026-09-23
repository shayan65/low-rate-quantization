# Full-coverage conversion of Qwen3.5-0.8B

Every 2-D weight except the embedding: 186 tensors, 497,614,848 of 752,393,024 parameters (**66.1%**).
BF16 model is 1505 MB; BF16 NLL 3.436710 over 261,284 validation targets.

`../models/Qwen3.5-0.8B-Base` holds three pieces; `AutoModelForCausalLM` instantiates only the language model, so the checkpoint's vision tower and multi-token-prediction head are **not loaded** and appear in no byte total here. Every figure below is against the text model.

What stays BF16 (254,778,176 params):

  * embedding: 254,279,680
  * non_matrix_weights: 442,368
  * norms_and_biases_1d: 56,128

The embedding is tied to the output head, so it is stored once and used twice -- and it alone is 33.8% of the model.

| Arm | covered bpw | model MB | effective bpw | shrink | ΔNLL | 95% CI |
|---|---:|---:|---:|---:|---:|:--|
| r1600_k81 | 1.7250 | 616.9 | 6.559 | 2.44x | +1.000022 | [+0.986485, +1.013520] |
| r2000_k255 | 2.1250 | 641.7 | 6.823 | 2.34x | +0.470715 | [+0.462434, +0.478752] |
| r2667_k1625 | 2.7919 | 683.2 | 7.264 | 2.20x | +0.171490 | [+0.167185, +0.175860] |

`effective bpw` is the whole model's bytes divided by its parameter count, so it includes the BF16 embedding; `covered bpw` is the converted tensors alone. The gap between them is what keeping the embedding at BF16 costs.
