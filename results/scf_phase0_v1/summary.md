# SCF Phase 0: falsification gate

Qwen3.5-0.8B, vq2d codec, all 18 DeltaNet input projections, 32,768 WikiText-2 validation targets. Wall clock 277.4s.

## Test A: is the one-pass sequential arm as good as the iterative loop?

BF16 NLL 3.233953.

| Arm | NLL | ΔNLL vs BF16 | 95% CI |
|---|---:|---:|:--|
| one_shot | 3.245136 | +0.011184 | [+0.009132, +0.013299] |
| sequential | 3.242358 | +0.008406 | [+0.006635, +0.010215] |
| scf_forward | 3.246586 | +0.012633 | [+0.010697, +0.014610] |

`scf_forward` minus `sequential`: **+0.004228** NLL (negative means the iteration still adds something).

## Test B: is there a downstream residual to solve?

| Quantity | Pooled relative residual |
|---|---:|
| forward, E[x^2] | 0.8719% |
| backward, E[(dL/dy)^2] | 7.3318% |
| **ratio** | **8.41x** |

Predeclared kill criterion: abandon if ratio <= 2. **Verdict: PROCEED.**
