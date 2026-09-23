"""Matrix products that keep the weights compressed, and an honest benchmark.

Every quality result in this project installs decoded BF16 weights and measures
loss. That is right for measuring quality, but it means no storage claim here
has been connected to a running system: at inference the model still occupies
its dense footprint.

**Storage format and runtime format are not the same thing, and this is where
that becomes concrete.** The storage packing puts five base-6561 codes in a
uint64 for exactly 1.600 bits of index per weight, which is optimal. It is also
not directly indexable on a GPU: 6561^5 = 1.216e19 exceeds 2^63, so the words
set the sign bit, and PyTorch and Triton have no unsigned 64-bit arithmetic.
Peeling base-6561 digits at every access would also cost five integer
divisions per word on the critical path.

So the runtime layout differs from the storage layout. A dim-8 code needs 13
bits and fits in an int16, which is directly indexable:

| layout | bits/weight of index | note |
|---|---|---|
| storage (5 codes / uint64) | 1.600 | optimal, not directly indexable |
| runtime (1 code / int16) | 2.000 | 25% larger, one load per code |

Both are far below BF16's 16. Converting between them is a one-off at load
time. Reporting 1.600 as though it were also the resident footprint would be
wrong, and this module exists partly to make that distinction measurable.

Two paths, for two questions. `matmul_streamed` answers the memory question: it
decodes one column tile at a time so the dense matrix is never resident, in
plain PyTorch that is easy to check against the reference decoder.
`matmul_triton` attempts the latency question. At batch one a projection is
memory-bound on weights and the compressed form moves ~8x less, but whether a
kernel realizes that against cuBLAS is empirical, and a slower result is a real
finding about the format's runtime cost rather than something to bury.

Neither path makes the 27B model run end to end: that needs every other tensor
handled, a KV-cache and recurrent-state budget, and integration with the
forward pass. Those stay unmeasured and are named as such.
"""

from __future__ import annotations

import numpy as np
import torch

from packed_lowrate import unpack_codes

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


def storage_to_runtime(index_bytes: bytes, n_codes: int, k: int,
                       device="cuda") -> torch.Tensor:
    """Storage packing -> directly indexable int16 codes, once at load time."""
    assert k <= 32767, f"K={k} does not fit a signed 16-bit code"
    codes = unpack_codes(index_bytes, n_codes, k)
    return torch.from_numpy(codes.astype(np.int16)).to(device)


def runtime_bits_per_weight(rows: int, cols: int, dim: int, group: int,
                            k: int) -> dict:
    """Resident bits per weight for the runtime layout, counted not derived."""
    n = rows * cols
    idx_bits = (n // dim) * 16
    scale_bits = (rows * (cols // group)) * 16          # FP16 group scales
    book_bits = k * dim * 16                            # shared, charged once
    return {"index": idx_bits / n, "scales": scale_bits / n,
            "codebook_amortized": book_bits / n,
            "total": (idx_bits + scale_bits + book_bits) / n}


# ---------------------------------------------------------------------------
# memory path: decode a tile at a time, never hold the dense matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def matmul_streamed(codes: torch.Tensor, book: torch.Tensor, scale: torch.Tensor,
                    x: torch.Tensor, rows: int, cols: int, dim: int, group: int,
                    tile_cols: int = 512) -> torch.Tensor:
    """y = x @ W^T, reconstructing W in column tiles and discarding each.

    x is (batch, cols) and the result is (batch, rows). Only (rows, tile_cols)
    of weight is resident at any moment, so peak allocation is the compressed
    payload plus one tile rather than the dense matrix.

    The tile must be group-aligned, not merely code-aligned: scales are sliced
    by whole groups below, so a tile narrower than one group selects an empty
    slice and the multiply fails on a shape mismatch. Checking only `dim`
    divisibility, as this did originally, accepted `tile_cols=64` at
    `group=128` and crashed inside the loop.
    """
    assert tile_cols % group == 0, (
        f"tile_cols={tile_cols} must be a multiple of group={group}")
    assert group % dim == 0, f"group={group} must be a multiple of dim={dim}"
    ng = cols // group
    codes = codes.view(rows, cols // dim)
    y = torch.zeros(x.shape[0], rows, device=x.device, dtype=torch.float32)
    sc = scale.view(rows, ng)
    for c0 in range(0, cols, tile_cols):
        c1 = min(c0 + tile_cols, cols)
        blk = codes[:, c0 // dim : c1 // dim].reshape(-1).long()
        # Tables are resident in FP16 but the product is formed in FP32, the
        # same policy the Triton kernel follows. Multiplying in FP16 costs
        # about three decimal digits: it took the streamed path's error against
        # the reference decoder from 3e-7 to 2e-4.
        tile = book[blk].view(rows, c1 - c0).float()
        g0, g1 = c0 // group, c1 // group
        tile = tile * sc[:, g0:g1].float().repeat_interleave(group, dim=1)[:, : c1 - c0]
        y += x[:, c0:c1].float() @ tile.T.float()
        del blk, tile
    return y


# ---------------------------------------------------------------------------
# latency path
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _gemv_kernel(codes_ptr, book_ptr, scale_ptr, x_ptr, y_ptr,
                     COLS, NG, DIM: tl.constexpr, GROUP: tl.constexpr,
                     BLOCK: tl.constexpr):
        """One program per output row, striding the row's int16 codes.

        Each code expands to DIM contiguous weights, so a BLOCK of codes covers
        BLOCK*DIM columns. The codebook is small enough (K*DIM floats) to live
        in L2 across the whole launch.
        """
        row = tl.program_id(0)
        n_codes = COLS // DIM
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c0 in range(0, n_codes, BLOCK):
            off = c0 + tl.arange(0, BLOCK)
            m = off < n_codes
            code = tl.load(codes_ptr + row * n_codes + off, mask=m, other=0).to(tl.int32)
            for d in tl.static_range(DIM):
                col = off * DIM + d
                mm = m & (col < COLS)
                # Read the tables in whatever precision they are stored and
                # promote here. Converting them in the wrapper instead, as this
                # did originally, silently doubled their resident cost while
                # the byte accounting still charged two bytes an element.
                wv = tl.load(book_ptr + code * DIM + d, mask=mm,
                             other=0.0).to(tl.float32)
                sv = tl.load(scale_ptr + row * NG + col // GROUP, mask=mm,
                             other=0.0).to(tl.float32)
                xv = tl.load(x_ptr + col, mask=mm, other=0.0)
                acc += tl.where(mm, wv * sv * xv, 0.0)
        tl.store(y_ptr + row, tl.sum(acc))


@torch.no_grad()
def matmul_triton(codes: torch.Tensor, book: torch.Tensor, scale: torch.Tensor,
                  x: torch.Tensor, rows: int, cols: int, dim: int, group: int,
                  block: int = 64) -> torch.Tensor:
    """Single-vector compressed product; x is (cols,), returns (rows,)."""
    assert HAVE_TRITON, "triton unavailable"
    y = torch.empty(rows, device=x.device, dtype=torch.float32)
    _gemv_kernel[(rows,)](
        codes.contiguous(), book.contiguous(), scale.contiguous(),
        x.contiguous().float(), y, cols, cols // group,
        DIM=dim, GROUP=group, BLOCK=block,
    )
    return y


def resident_bytes(codes: torch.Tensor, book: torch.Tensor,
                   scale: torch.Tensor) -> dict:
    """Bytes actually occupied, from each tensor's own element size.

    The first version of this benchmark charged two bytes an element for the
    codebook and the scales because that is what the design intended, while
    both tensors were in fact FP32 at runtime. Element size is read from the
    tensor so the claim cannot drift from the object again.
    """
    parts = {"codes": codes.numel() * codes.element_size(),
             "codebook": book.numel() * book.element_size(),
             "scales": scale.numel() * scale.element_size()}
    return {**parts, "total": sum(parts.values()),
            "dtypes": {"codes": str(codes.dtype), "codebook": str(book.dtype),
                       "scales": str(scale.dtype)}}
