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
    def _rht_kernel(x_ptr, s_ptr, y_ptr, h1_ptr, h2_ptr, NBLK,
                    R1: tl.constexpr, R2: tl.constexpr, SCALE: tl.constexpr):
        """One program per (row, block): sign flip then H1 @ X @ H2.

        A block of R1*R2 elements is viewed as an R1 x R2 tile. The radix-2
        FWHT in natural order is a Kronecker product, so the square case
        (R1 == R2) and the rectangular one are the same formula; 512-wide
        blocks factor as 32 x 16, which is why this is not fixed at 32 x 32.
        """
        pid = tl.program_id(0)
        row = pid // NBLK
        blk = pid % NBLK
        i = tl.arange(0, R1)
        j = tl.arange(0, R2)
        idx = i[:, None] * R2 + j[None, :]
        base = row * NBLK * R1 * R2 + blk * R1 * R2
        x = tl.load(x_ptr + base + idx)
        s = tl.load(s_ptr + blk * R1 * R2 + idx)     # signs indexed by column
        h1 = tl.load(h1_ptr + i[:, None] * R1 + i[None, :])
        h2 = tl.load(h2_ptr + j[:, None] * R2 + j[None, :])
        y = tl.dot(h1, x * s, input_precision="ieee")
        y = tl.dot(y, h2, input_precision="ieee")
        tl.store(y_ptr + base + idx, y * SCALE)


def factor_block(blk: int) -> tuple[int, int]:
    """Split a power-of-two block into R1 x R2 with both at least 16.

    `tl.dot` needs each dimension to be at least 16, so 256 -> 16x16,
    512 -> 32x16, 1024 -> 32x32, 2048 -> 64x32, 4096 -> 64x64.
    """
    assert blk & (blk - 1) == 0, f"blk={blk} is not a power of two"
    assert blk >= 256, f"blk={blk} is too small to factor with both sides >= 16"
    r2 = 1 << ((blk.bit_length() - 1) // 2)
    return blk // r2, r2


class FusedRotation:
    """Cached Hadamard factors for one block size, so a forward pass allocates none."""

    def __init__(self, blk: int, device="cuda"):
        self.blk = blk
        self.r1, self.r2 = factor_block(blk)
        self.h1 = hadamard_matrix(self.r1, device)
        self.h2 = hadamard_matrix(self.r2, device)
        self.scale = 1.0 / blk ** 0.5

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        assert cols % self.blk == 0, f"cols={cols} not a multiple of {self.blk}"
        x = x.contiguous().float()
        y = torch.empty_like(x)
        _rht_kernel[(rows * (cols // self.blk),)](
            x, signs.contiguous().float(), y, self.h1, self.h2,
            cols // self.blk, R1=self.r1, R2=self.r2, SCALE=self.scale)
        return y


@torch.no_grad()
def rotate_fused(x: torch.Tensor, signs: torch.Tensor, blk: int,
                 rot: "FusedRotation | None" = None) -> torch.Tensor:
    """Randomized Hadamard rotation of the last axis, in one kernel.

    `x` is (rows, cols) with `cols` a multiple of `blk`; `signs` is (cols,).
    Equivalent to `run_ternary_task_v4.rotate(x, signs, blk)`.
    """
    assert HAVE_TRITON, "triton unavailable"
    if rot is None or rot.blk != blk:
        rot = FusedRotation(blk, x.device)
    return rot(x, signs)
