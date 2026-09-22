"""Improved polar (magnitude-phase) vector codec for weight pairs.

Background
----------
The legacy ``polar35`` quantizer in this repository rounds magnitude and phase
*independently*:

    p = round(angle(z) * 32 / 2pi)          # nearest phase
    m = round(|z| * 7 / s)                  # nearest magnitude of |z|

That is **not** nearest-neighbour assignment in the 8x32 = 256 point polar
codebook.  Given a decoded phase ray ``theta_p``, the squared error is

    |z - g_m e^{i theta_p}|^2 = r^2 sin^2(d) + (g_m - r cos(d))^2 ,  d = theta - theta_p

so the optimal magnitude level is the one nearest to the **projection**
``r cos(d)``, not the one nearest to ``r``.  Independent rounding therefore
pays an avoidable bias of ``r (1 - cos d)`` on every pair.  In addition the
legacy code only ever considers the single nearest phase, while a slightly
worse phase with a much better-fitting radius can win once the radius grid is
coarse (3 bits = 8 levels).

This module implements

* exact nearest-neighbour assignment over all 256 codepoints,
* a fitted (Lloyd-Max) non-uniform radius grid shared per tensor, which costs
  8 FP32 per tensor and matches the Rayleigh-like magnitude distribution far
  better than the uniform ``k/7`` grid,
* an optional diagonal activation weighting ``h`` (per input column), which
  turns the isotropic circle metric into an ellipse metric and is what the
  self-consistent loop in ``scq.py`` drives.

Byte budget is unchanged: 8 index bits per pair (= 4 bits per real weight),
one FP32 row scale, plus a per-tensor header of 8 grid floats + 1 offset float.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

PAIRINGS = ("adjacent", "split_half", "reverse_half", "stride257")
N_PHASE = 32
N_MAG = 8


def pairing_indices(d: int, name: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two column-index vectors that form the real/imaginary halves."""
    if name == "adjacent":
        return torch.arange(0, d, 2, device=device), torch.arange(1, d, 2, device=device)
    a = torch.arange(d // 2, device=device)
    if name == "split_half":
        return a, a + d // 2
    if name == "reverse_half":
        return a, d - 1 - a
    if name == "stride257":
        perm = (torch.arange(d, device=device) * 257) % d
        return perm[: d // 2], perm[d // 2 :]
    raise ValueError(name)


@dataclass
class PolarCode:
    """A decoded polar quantization of one weight matrix."""

    weight: torch.Tensor  # dequantized weight, same shape/dtype layout as input
    mag_index: torch.Tensor  # uint8, (rows, pairs), values 0..7
    phase_index: torch.Tensor  # uint8, (rows, pairs), values 0..31
    row_scale: torch.Tensor  # float32, (rows, 1)
    grid: torch.Tensor  # float32, (8,) normalized radius levels in [0, 1]
    phase_offset: float
    pairing: str
    weighted_error: float = 0.0
    weight_mse: float = 0.0
    meta: dict = field(default_factory=dict)

    def payload_bytes(self, rows: int, cols: int) -> int:
        """Index bits + row scales + per-tensor grid header. Matches real-4 budget."""
        return rows * cols // 2 + rows * 4 + N_MAG * 4 + 4 + 1


def _bucketize(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Index of the nearest level in a sorted 1-D ``grid`` for every entry."""
    mid = (grid[1:] + grid[:-1]) * 0.5
    return torch.bucketize(values.contiguous(), mid)


@torch.no_grad()
def fit_radius_grid(
    u: torch.Tensor,
    weights: torch.Tensor | None = None,
    iters: int = 25,
    levels: int | None = None,
) -> torch.Tensor:
    """Lloyd-Max fit of ``levels`` non-negative radius levels to samples ``u``.

    ``u`` holds normalized magnitudes in [0, 1]; ``weights`` optionally gives a
    per-sample importance (used for activation-weighted fitting).  Level 0 is
    pinned to exactly zero so that near-zero pairs stay exactly representable.
    """
    levels = N_MAG if levels is None else levels
    u = u.flatten().float()
    w = None if weights is None else weights.flatten().float()
    # Initialize on quantiles so empty cells are unlikely.
    qs = torch.linspace(0, 1, levels, device=u.device)
    grid = torch.quantile(u[:: max(1, u.numel() // 1_000_000)], qs).clamp_min(0)
    grid[0] = 0.0
    for _ in range(iters):
        idx = _bucketize(u, grid)
        num = torch.zeros(levels, device=u.device, dtype=torch.float64)
        den = torch.zeros(levels, device=u.device, dtype=torch.float64)
        src = u.double() if w is None else (u * w).double()
        cnt = torch.ones_like(u, dtype=torch.float64) if w is None else w.double()
        num.scatter_add_(0, idx, src)
        den.scatter_add_(0, idx, cnt)
        new = torch.where(den > 0, num / den.clamp_min(1e-30), grid.double())
        new[0] = 0.0
        new = new.float().clamp_min(0)
        new, _ = torch.sort(new)
        if torch.allclose(new, grid, atol=1e-7):
            grid = new
            break
        grid = new
    grid[0] = 0.0
    return grid


@torch.no_grad()
def assign_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    grid: torch.Tensor,
    offset: float,
    ha: torch.Tensor | None = None,
    hb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact nearest-neighbour assignment over all 8x32 polar codepoints.

    Returns ``(qx, qy, mag_index, phase_index, error)`` where ``error`` is the
    per-pair (optionally activation-weighted) squared reconstruction error.

    With ``ha``/``hb`` supplied the metric is the diagonal-weighted
    ``ha (qx - x)^2 + hb (qy - y)^2``; otherwise it is plain Euclidean.
    """
    wa = None if ha is None else ha.view(1, -1)
    wb = None if hb is None else hb.view(1, -1)

    best_err = torch.full(x.shape, float("inf"), device=x.device, dtype=torch.float32)
    best_m = torch.zeros(x.shape, device=x.device, dtype=torch.int64)
    best_p = torch.zeros(x.shape, device=x.device, dtype=torch.int64)

    # Sweep the 32 phase rays and keep a running minimum.  Materializing all
    # rays at once costs 32x the tensor and overflows a 24 GB card on the wide
    # 27B projections, so the loop trades a little speed for a flat footprint.
    for p in range(N_PHASE):
        ang = p * (2 * math.pi / N_PHASE) + offset
        c = math.cos(ang)
        s = math.sin(ang)
        if wa is None:
            proj = x * c + y * s
        else:
            proj = (wa * x * c + wb * y * s) / (wa * c * c + wb * s * s).clamp_min(1e-20)
        u = (proj / scale).clamp(0, 1)
        m = _bucketize(u, grid)
        r = scale * grid[m]
        dx = r * c - x
        dy = r * s - y
        err = dx.square() + dy.square() if wa is None else wa * dx.square() + wb * dy.square()
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_m = torch.where(take, m, best_m)
        best_p = torch.where(take, torch.full_like(best_p, p), best_p)
        del proj, u, m, r, dx, dy, err, take

    r_best = scale * grid[best_m]
    ang = best_p.float() * (2 * math.pi / N_PHASE) + offset
    return (
        r_best * ang.cos(),
        r_best * ang.sin(),
        best_m.to(torch.uint8),
        best_p.to(torch.uint8),
        best_err,
    )


@torch.no_grad()
def quantize_polar(
    w: torch.Tensor,
    pairing: str = "adjacent",
    h: torch.Tensor | None = None,
    offset: float = 0.0,
    grid: torch.Tensor | None = None,
    scale_ratios=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0),
    grid_iters: int = 4,
    row_chunk: int = 512,
) -> PolarCode:
    """Quantize ``w`` (rows x cols, float32, on device) with the improved codec.

    ``h`` is an optional per-input-column importance vector (diagonal activation
    second moment).  When supplied, both the radius grid and the codepoint
    assignment minimize the activation-weighted output error, which is the
    quantity the self-consistent loop iterates on.
    """
    rows, cols = w.shape
    dev = w.device
    ia, ib = pairing_indices(cols, pairing, dev)
    x = w[:, ia].float()
    y = w[:, ib].float()
    ha = hb = None
    if h is not None:
        ha = h[ia].float().clamp_min(1e-12)
        hb = h[ib].float().clamp_min(1e-12)

    r = torch.hypot(x, y)
    base = r.amax(1, keepdim=True).clamp_min(1e-8)

    if grid is None:
        # Fit the grid on a bounded row subsample: the magnitude distribution is
        # stable across rows and this keeps the Lloyd iterations cheap.
        sub = slice(None) if rows <= 1024 else slice(0, rows, max(1, rows // 1024))
        xs_, ys_, bs_ = x[sub], y[sub], base[sub]
        grid = fit_radius_grid(r[sub] / bs_)
        # Alternate grid fit and assignment so the grid sees the projected radii
        # that assignment actually uses (Lloyd on the true code, not on |z|).
        for _ in range(grid_iters):
            _, _, _, p_i, _ = assign_exact(xs_, ys_, bs_, grid, offset, ha, hb)
            ang = p_i.float() * (2 * math.pi / N_PHASE) + offset
            proj = (xs_ * ang.cos() + ys_ * ang.sin()) / bs_
            grid = fit_radius_grid(proj.clamp(0, 1))

    # The row scale is chosen per row, exactly as the legacy control does: a
    # single tensor-wide ratio lets a few wide-dynamic-range rows dictate the
    # clip point for every other row and measurably loses to per-row selection.
    qx = torch.empty_like(x)
    qy = torch.empty_like(y)
    mi = torch.zeros(x.shape, dtype=torch.uint8, device=dev)
    pi = torch.zeros(x.shape, dtype=torch.uint8, device=dev)
    best_scale = torch.empty_like(base)
    best_row_err = torch.full((rows, 1), float("inf"), device=dev)

    for ratio in scale_ratios:
        scale = base * float(ratio)
        for s0 in range(0, rows, row_chunk):
            s1 = min(s0 + row_chunk, rows)
            cx, cy, cm, cp, ce = assign_exact(
                x[s0:s1], y[s0:s1], scale[s0:s1], grid, offset, ha, hb
            )
            row_err = ce.mean(1, keepdim=True)
            take = row_err < best_row_err[s0:s1]
            t = take.expand_as(cx)
            qx[s0:s1] = torch.where(t, cx, qx[s0:s1])
            qy[s0:s1] = torch.where(t, cy, qy[s0:s1])
            mi[s0:s1] = torch.where(t, cm, mi[s0:s1])
            pi[s0:s1] = torch.where(t, cp, pi[s0:s1])
            best_scale[s0:s1] = torch.where(take, scale[s0:s1], best_scale[s0:s1])
            best_row_err[s0:s1] = torch.where(take, row_err, best_row_err[s0:s1])
            del cx, cy, cm, cp, ce, row_err, take, t

    out = torch.empty_like(w, dtype=torch.float32)
    out[:, ia] = qx
    out[:, ib] = qy
    return PolarCode(
        weight=out,
        mag_index=mi,
        phase_index=pi,
        row_scale=best_scale,
        grid=grid.clone(),
        phase_offset=offset,
        pairing=pairing,
        weighted_error=best_row_err.mean().item(),
        weight_mse=(out - w).square().mean().item(),
        meta={"scale_ratios": list(scale_ratios)},
    )


@torch.no_grad()
def quantize_polar_legacy(w: torch.Tensor, pairing: str = "adjacent") -> PolarCode:
    """The repository's original decoupled rounding rule, kept as a control."""
    rows, cols = w.shape
    ia, ib = pairing_indices(cols, pairing, w.device)
    z = torch.complex(w[:, ia].float(), w[:, ib].float())
    mag = z.abs()
    phase = torch.angle(z)
    pi = ((phase % (2 * math.pi)) / (2 * math.pi) * N_PHASE).round().remainder(N_PHASE)
    angle = pi.float() * (2 * math.pi / N_PHASE)
    base = mag.amax(1, keepdim=True).clamp_min(1e-8)
    best = best_s = best_mi = None
    best_e = torch.full((rows,), float("inf"), device=w.device)
    for ratio in torch.linspace(0.55, 1.0, 10, device=w.device):
        s = base * ratio
        mi = (mag / s * 7).round().clamp(0, 7)
        q = torch.polar(s * (mi / 7), angle)
        e = (q - z).abs().square().mean(1)
        take = e < best_e
        if best is None:
            best, best_s, best_mi = q, s, mi
        else:
            best[take], best_s[take], best_mi[take] = q[take], s[take], mi[take]
        best_e = torch.minimum(best_e, e)
    out = torch.empty_like(w, dtype=torch.float32)
    out[:, ia] = best.real
    out[:, ib] = best.imag
    return PolarCode(
        weight=out,
        mag_index=best_mi.to(torch.uint8),
        phase_index=pi.to(torch.uint8),
        row_scale=best_s,
        grid=torch.linspace(0, 1, N_MAG, device=w.device),
        phase_offset=0.0,
        pairing=pairing,
        weight_mse=(out - w).square().mean().item(),
    )


@torch.no_grad()
def quantize_real4(w: torch.Tensor, h: torch.Tensor | None = None, ratios=None):
    """Equal-byte per-row symmetric real 4-bit control (16 levels per weight)."""
    ratios = ratios if ratios is not None else torch.linspace(0.55, 1.0, 10, device=w.device)
    base = w.abs().amax(1, keepdim=True).clamp_min(1e-8)
    best_q = None
    best_e = torch.full((w.shape[0],), float("inf"), device=w.device)
    best_s = None
    for ratio in ratios:
        s = base * float(ratio)
        q = s * ((w / s).clamp(-1, 1).add(1).mul(7.5).round().div(7.5).sub(1))
        d = (q - w).square()
        e = (d if h is None else d * h.view(1, -1)).mean(1)
        take = e < best_e
        if best_q is None:
            best_q, best_s = q.clone(), s.clone()
        else:
            best_q[take], best_s[take] = q[take], s[take]
        best_e = torch.minimum(best_e, e)
    return best_q, best_s, best_e


@torch.no_grad()
def fit_scalar_grid(v: torch.Tensor, levels: int = 16, iters: int = 30) -> torch.Tensor:
    """Lloyd-Max fit of a symmetric ``levels``-point scalar codebook on [-1, 1].

    This is the *fair* counterpart to :func:`fit_radius_grid`: it gives the
    real-4 control the same per-tensor shape adaptation that the polar codec
    gets, so the comparison isolates codebook geometry rather than tuning
    effort.  Storage cost is identical (16 FP32 per tensor).
    """
    v = v.flatten().float()
    v = v[:: max(1, v.numel() // 4_000_000)]
    qs = torch.linspace(0, 1, levels, device=v.device)
    grid = torch.quantile(v, qs)
    for _ in range(iters):
        idx = _bucketize(v, grid)
        num = torch.zeros(levels, device=v.device, dtype=torch.float64)
        den = torch.zeros(levels, device=v.device, dtype=torch.float64)
        num.scatter_add_(0, idx, v.double())
        den.scatter_add_(0, idx, torch.ones_like(v, dtype=torch.float64))
        new = torch.where(den > 0, num / den.clamp_min(1e-30), grid.double()).float()
        new, _ = torch.sort(new)
        if torch.allclose(new, grid, atol=1e-7):
            grid = new
            break
        grid = new
    return grid


@torch.no_grad()
def quantize_real4_fitted(
    w: torch.Tensor,
    h: torch.Tensor | None = None,
    grid: torch.Tensor | None = None,
    scale_ratios=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0),
    levels: int = 16,
):
    """Real 4-bit control with a per-tensor Lloyd-Max codebook and per-row scale."""
    rows = w.shape[0]
    base = w.abs().amax(1, keepdim=True).clamp_min(1e-8)
    if grid is None:
        grid = fit_scalar_grid((w / base).clamp(-1, 1), levels)
    best_q = torch.empty_like(w)
    best_s = torch.empty_like(base)
    best_e = torch.full((rows, 1), float("inf"), device=w.device)
    for ratio in scale_ratios:
        s = base * float(ratio)
        u = (w / s).clamp(grid[0].item(), grid[-1].item())
        q = s * grid[_bucketize(u, grid)]
        d = (q - w).square()
        e = (d if h is None else d * h.view(1, -1)).mean(1, keepdim=True)
        take = e < best_e
        best_q = torch.where(take.expand_as(q), q, best_q)
        best_s = torch.where(take, s, best_s)
        best_e = torch.where(take, e, best_e)
    return best_q, best_s, best_e, grid
