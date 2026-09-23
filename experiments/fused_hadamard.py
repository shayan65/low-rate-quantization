"""A fused randomized Hadamard transform, because the prototype dominated latency.

`run_rate_sweep.fwht` runs ten Python-controlled butterfly stages over 1024
columns, each stage cloning two half-tiles and writing two slice assignments
back. That is fine for quantizing weights offline, where it runs once per
tensor. It is not fine on an inference critical path: measured on one 1024-wide
activation vector it took 0.471 ms against 0.019 ms for the whole cuBLAS
projection, and so accounted for about 90% of the compressed path's time. A
review correctly objected that this diagnoses the implementation rather than
the transform, and that no optimized speedup should be inferred by subtracting
separately timed components. This module removes the objection by building the
fused version and measuring it.

**The factorization.** The radix-2 FWHT in natural order is exactly the
Kronecker product of smaller Hadamard matrices, so for a block of
`R*R` elements viewed as an `R x R` tile `X`,

    FWHT_{R*R}(x)  =  vec( H_R @ X @ H_R ) / sqrt(R*R),

with `H_R` the unnormalized +/-1 Hadamard matrix. At `R = 32` that turns ten
dependent butterfly stages into two 32x32 matrix products, which is one Triton
program per block with the whole tile resident in registers. The sign flip is
folded into the same kernel, so the randomized rotation costs one pass over the
data instead of eleven.

Correctness is checked against the existing `fwht` rather than against theory,
because the transform's *ordering convention* is what matters: a Hadamard
transform in sequency order would be equally valid mathematically and would
silently invalidate every stored code.

Precision: `tl.dot` defaults to TF32 on Ampere, which would quietly cost this
transform about three decimal digits. It is pinned to IEEE FP32 here, and the
check against the reference would catch a regression.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


def hadamard_matrix(r: int, device="cuda", dtype=torch.float32) -> torch.Tensor:
    """Unnormalized +/-1 Hadamard matrix of order r, natural (Sylvester) order."""
    assert r & (r - 1) == 0 and r >= 2, f"order {r} is not a power of two"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < r:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


if HAVE_TRITON:

    @triton.jit
    def _rht_kernel(x_ptr, s_ptr, y_ptr, h_ptr, NBLK, R: tl.constexpr,
                    SCALE: tl.constexpr):
        """One program per (row, Hadamard block): sign flip then H @ X @ H."""
        pid = tl.program_id(0)
        row = pid // NBLK
        blk = pid % NBLK
        i = tl.arange(0, R)
        idx = i[:, None] * R + i[None, :]
        base = row * NBLK * R * R + blk * R * R
        x = tl.load(x_ptr + base + idx)
        s = tl.load(s_ptr + blk * R * R + idx)       # signs indexed by column
        h = tl.load(h_ptr + idx)
        y = tl.dot(h, x * s, input_precision="ieee")
        y = tl.dot(y, h, input_precision="ieee")
        tl.store(y_ptr + base + idx, y * SCALE)


@torch.no_grad()
def rotate_fused(x: torch.Tensor, signs: torch.Tensor, blk: int,
                 hmat: torch.Tensor | None = None) -> torch.Tensor:
    """Randomized Hadamard rotation of the last axis, in one kernel.

    `x` is (rows, cols) with `cols` a multiple of `blk`; `signs` is (cols,).
    Equivalent to `run_ternary_task_v4.rotate(x, signs, blk)`.
    """
    assert HAVE_TRITON, "triton unavailable"
    rows, cols = x.shape
    assert cols % blk == 0, f"cols={cols} is not a multiple of blk={blk}"
    r = int(round(blk ** 0.5))
    assert r * r == blk and (r & (r - 1)) == 0, (
        f"blk={blk} must be a square power of two (256, 1024, 4096); "
        "other block sizes need a rectangular factorization")
    if hmat is None:
        hmat = hadamard_matrix(r, x.device, torch.float32)
    x = x.contiguous().float()
    y = torch.empty_like(x)
    _rht_kernel[(rows * (cols // blk),)](
        x, signs.contiguous().float(), y, hmat, cols // blk,
        R=r, SCALE=1.0 / blk ** 0.5)
    return y
