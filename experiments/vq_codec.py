"""Paired 2-D vector codec with an optional activation metric.

The codebook-ceiling screen showed that the polar (magnitude/phase) grid is a
*structurally* poor 2-D codebook for LLM weight pairs: fully optimized it still
loses to a Lloyd-fitted scalar 4-bit code, while an unconstrained 256-point 2-D
codebook beats that scalar code by 18-24% weight MSE at the same index budget.

So this module keeps the part of the original idea that works -- pairing two
real weights and quantizing them jointly -- and drops the polar
parameterization.  A tensor stores:

  * 8 index bits per pair  (= 4 bits per real weight, identical to real-4)
  * one FP32 scale per row
  * a 256 x 2 FP32 codebook header per tensor (2 KB, amortized to ~0 bits/weight)

Decoding is a 256-entry lookup, not a sine/cosine evaluation, which also
removes the trigonometric decode that made the packed polar kernels 3-4.5x
slower than resident BF16.

``h`` (per-input-column activation second moment) switches both assignment and
the centroid update to the diagonal metric ``ha dx^2 + hb dy^2``.  That is the
hook the self-consistent loop in :mod:`scq` drives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from polar_codec import PAIRINGS, pairing_indices  # noqa: F401  (re-exported)

CODEBOOK_SIZE = 256


@dataclass
class VQCode:
    weight: torch.Tensor
    index: torch.Tensor  # uint8, (rows, pairs)
    row_scale: torch.Tensor  # float32, (rows, 1)
    codebook: torch.Tensor  # float32, (256, 2)
    pairing: str
    weighted_error: float = 0.0
    weight_mse: float = 0.0
    meta: dict = field(default_factory=dict)

    def payload_bytes(self, rows: int, cols: int) -> int:
        import math as _m
        bits = _m.ceil(_m.log2(self.codebook.shape[0]))
        return (rows * cols // 2) * bits // 8 + rows * 4 + self.codebook.shape[0] * 2 * 4 + 1


@torch.no_grad()
def _assign(
    px: torch.Tensor,
    py: torch.Tensor,
    cb: torch.Tensor,
    wa: torch.Tensor | None = None,
    wb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest centroid under the (optionally weighted) metric.

    Shapes: ``px``/``py`` are (rows, pairs); ``wa``/``wb`` broadcast over rows.
    Returns ``(index, per_pair_error)``.
    """
    best_i = torch.zeros(px.shape, dtype=torch.long, device=px.device)
    best_e = torch.full(px.shape, float("inf"), device=px.device)
    for c in range(cb.shape[0]):
        dx = cb[c, 0] - px
        dy = cb[c, 1] - py
        e = dx.square() + dy.square() if wa is None else wa * dx.square() + wb * dy.square()
        take = e < best_e
        best_e = torch.where(take, e, best_e)
        best_i = torch.where(take, torch.full_like(best_i, c), best_i)
        del dx, dy, e, take
    return best_i, best_e


@torch.no_grad()
def fit_codebook(
    px: torch.Tensor,
    py: torch.Tensor,
    wa: torch.Tensor | None = None,
    wb: torch.Tensor | None = None,
    k: int = CODEBOOK_SIZE,
    iters: int = 25,
    init: torch.Tensor | None = None,
    seed: int = 0,
    max_rows: int = 2048,
) -> torch.Tensor:
    """Weighted Lloyd (k-means) fit of a shared 2-D codebook.

    Fitted on at most ``max_rows`` rows: the pair distribution is stable across
    rows and this keeps each self-consistent iteration cheap.  ``init`` warm
    starts from the previous outer iteration's codebook.
    """
    rows = px.shape[0]
    step = max(1, rows // max_rows)
    sx, sy = px[::step], py[::step]
    swa = None if wa is None else wa
    swb = None if wb is None else wb

    if init is not None:
        cb = init.clone()
    else:
        g = torch.Generator(device=px.device).manual_seed(seed)
        flat = torch.stack([sx.flatten(), sy.flatten()], 1)
        pick = torch.randperm(flat.shape[0], generator=g, device=px.device)[:k]
        cb = flat[pick].clone()
        del flat

    for _ in range(iters):
        idx, _ = _assign(sx, sy, cb, swa, swb)
        flat_i = idx.flatten()
        wt = (
            torch.ones_like(sx)
            if swa is None
            else (swa.expand_as(sx) + swb.expand_as(sy)) * 0.5
        ).flatten().double()
        num = torch.zeros(k, 2, device=px.device, dtype=torch.float64)
        den = torch.zeros(k, device=px.device, dtype=torch.float64)
        num.index_add_(0, flat_i, torch.stack([sx.flatten(), sy.flatten()], 1).double() * wt.unsqueeze(1))
        den.index_add_(0, flat_i, wt)
        alive = den > 0
        new = cb.double().clone()
        new[alive] = num[alive] / den[alive].unsqueeze(1)
        new = new.float()
        if torch.allclose(new, cb, atol=1e-7):
            cb = new
            break
        cb = new
        del idx, flat_i, wt, num, den, alive, new
    return cb


@torch.no_grad()
def quantize_vq2d(
    w: torch.Tensor,
    pairing: str = "adjacent",
    h: torch.Tensor | None = None,
    codebook: torch.Tensor | None = None,
    refit: bool = True,
    fit_iters: int = 25,
    row_chunk: int = 1024,
    k: int = CODEBOOK_SIZE,
) -> VQCode:
    """Quantize ``w`` with a shared per-tensor 256-point 2-D codebook."""
    rows, cols = w.shape
    dev = w.device
    ia, ib = pairing_indices(cols, pairing, dev)
    x = w[:, ia].float()
    y = w[:, ib].float()

    wa = wb = None
    if h is not None:
        wa = h[ia].float().clamp_min(1e-12).view(1, -1)
        wb = h[ib].float().clamp_min(1e-12).view(1, -1)

    # Per-row scale keeps the shared codebook on a common footing across rows
    # whose dynamic ranges differ by orders of magnitude.
    base = torch.maximum(x.abs().amax(1, keepdim=True), y.abs().amax(1, keepdim=True)).clamp_min(1e-8)
    px, py = x / base, y / base

    if codebook is None or refit:
        codebook = fit_codebook(px, py, wa, wb, k=k, iters=fit_iters, init=codebook)

    qx = torch.empty_like(px)
    qy = torch.empty_like(py)
    idx = torch.empty(px.shape, dtype=torch.uint8 if k <= 256 else torch.int32, device=dev)
    err_sum = 0.0
    for s0 in range(0, rows, row_chunk):
        s1 = min(s0 + row_chunk, rows)
        i, e = _assign(px[s0:s1], py[s0:s1], codebook, wa, wb)
        qx[s0:s1] = codebook[i, 0]
        qy[s0:s1] = codebook[i, 1]
        idx[s0:s1] = i.to(idx.dtype)
        err_sum += (e * base[s0:s1].square()).double().sum().item()
        del i, e

    out = torch.empty_like(w, dtype=torch.float32)
    out[:, ia] = qx * base
    out[:, ib] = qy * base
    return VQCode(
        weight=out,
        index=idx,
        row_scale=base,
        codebook=codebook,
        pairing=pairing,
        weighted_error=err_sum / (rows * cols / 2),
        weight_mse=(out - w).square().mean().item(),
    )
