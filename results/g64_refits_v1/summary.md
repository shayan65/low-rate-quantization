# Does cancellation explain the g64 sign flips?

The grouping/codebook 2x2 repeated across 6 refits. Both codebooks are deterministic Lloyd fits over weights, so no draw or seed can move them: across refits the **only** thing that changes is the Hessian used for GPTQ assignment and compensation.

BF16 NLL 3.436710 over 261,284 validation targets.

| contrast | mean | min | max | range | same sign | excl. 0 |
|---|---:|---:|---:|---:|:--|:--|
| grouping at fixed g128 book | +0.005535 | +0.002475 | +0.008219 | 0.005744 | yes | 6/6 |
| grouping at fixed g64 book | +0.001077 | -0.006581 | +0.008598 | 0.015179 | **no** | 3/6 |
| codebook at fixed g128 grouping | -0.002131 | -0.005929 | +0.001943 | 0.007872 | **no** | 3/6 |
| codebook at fixed g64 grouping | -0.006590 | -0.010115 | +0.001056 | 0.011171 | **no** | 5/6 |
| confounded diagonal | -0.001055 | -0.005171 | +0.005035 | 0.010206 | **no** | 5/6 |
| interaction | +0.004459 | -0.003232 | +0.012045 | 0.015278 | **no** | 0/6 |
