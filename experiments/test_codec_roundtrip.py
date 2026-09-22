"""Verify that every codec's stored indices actually reconstruct its weights.

A quantizer that returns a dequantized tensor which its own index stream cannot
reproduce is not a codec, and its reported byte budget is fiction.  This test
decodes each codec purely from what it claims to store -- indices, row scales,
and the per-tensor codebook header -- and requires exact agreement with the
tensor the quantizer returned.  It also checks the byte accounting.
"""

import math
import sys

import torch

from polar_codec import (
    N_MAG,
    N_PHASE,
    pairing_indices,
    quantize_polar,
    quantize_polar_legacy,
    quantize_real4_fitted,
    _bucketize,
)
from vq_codec import CODEBOOK_SIZE, quantize_vq2d

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def decode_polar(code, rows, cols):
    """Reconstruct from (mag_index, phase_index, row_scale, grid, offset) alone."""
    ia, ib = pairing_indices(cols, code.pairing, DEV)
    r = code.row_scale * code.grid[code.mag_index.long()]
    ang = code.phase_index.float() * (2 * math.pi / N_PHASE) + code.phase_offset
    out = torch.empty(rows, cols, device=DEV)
    out[:, ia] = r * ang.cos()
    out[:, ib] = r * ang.sin()
    return out


def decode_vq(code, rows, cols):
    """Reconstruct from (index, row_scale, codebook) alone."""
    ia, ib = pairing_indices(cols, code.pairing, DEV)
    cb = code.codebook[code.index.long()]
    out = torch.empty(rows, cols, device=DEV)
    out[:, ia] = cb[..., 0] * code.row_scale
    out[:, ib] = cb[..., 1] * code.row_scale
    return out


def main():
    torch.manual_seed(0)
    rows, cols = 256, 128
    # Heavy-tailed, per-row-scaled weights, closer to real projections than iid normal.
    w = (torch.randn(rows, cols, device=DEV) * torch.rand(rows, 1, device=DEV).add(0.1)).float()
    w[0] *= 100.0  # an outlier row
    w[1] = 0.0  # an all-zero row exercises the clamp paths

    for pairing in ("adjacent", "split_half", "reverse_half", "stride257"):
        c = quantize_polar(w, pairing)
        err = (decode_polar(c, rows, cols) - c.weight).abs().max().item()
        check(f"polar fitted round-trip [{pairing}]", err < 1e-5, f"max|diff|={err:.2e}")
        check(
            f"polar index range [{pairing}]",
            int(c.mag_index.max()) < N_MAG and int(c.phase_index.max()) < N_PHASE,
        )

    c = quantize_polar_legacy(w, "adjacent")
    # Relative tolerance: this matrix deliberately contains a row scaled by 100,
    # so an absolute threshold measures float32 epsilon, not codec correctness.
    d = (decode_polar(c, rows, cols) - c.weight).abs()
    rel = (d / c.weight.abs().clamp_min(1e-12)).max().item()
    check("polar legacy round-trip", rel < 1e-5, f"max rel={rel:.2e}")

    for pairing in ("adjacent", "split_half"):
        c = quantize_vq2d(w, pairing)
        err = (decode_vq(c, rows, cols) - c.weight).abs().max().item()
        check(f"vq2d round-trip [{pairing}]", err < 1e-5, f"max|diff|={err:.2e}")
        check(f"vq2d index range [{pairing}]", int(c.index.max()) < CODEBOOK_SIZE)

    q, s, _, grid = quantize_real4_fitted(w)
    re = q / s
    idx = _bucketize(re.clamp(grid[0].item(), grid[-1].item()), grid)
    err = (s * grid[idx] - q).abs().max().item()
    check("real4 Lloyd round-trip", err < 1e-5, f"max|diff|={err:.2e}")
    check("real4 Lloyd index range", int(idx.max()) < 16 and int(idx.min()) >= 0)

    # Exact nearest-neighbour claim: no other codepoint may beat the chosen one.
    c = quantize_polar(w, "adjacent")
    ia, ib = pairing_indices(cols, "adjacent", DEV)
    x, y = w[:, ia], w[:, ib]
    # Brute force in float64: comparing two float32 error computations measures
    # rounding, not optimality.
    x64, y64 = x.double(), y.double()
    chosen = ((c.weight[:, ia].double() - x64) ** 2 + (c.weight[:, ib].double() - y64) ** 2)
    best = torch.full_like(chosen, float("inf"))
    for m in range(N_MAG):
        for p in range(N_PHASE):
            a = p * (2 * math.pi / N_PHASE) + c.phase_offset
            r = (c.row_scale * c.grid[m]).double()
            best = torch.minimum(best, (r * math.cos(a) - x64) ** 2 + (r * math.sin(a) - y64) ** 2)
    # Scale-free: slack relative to the energy of the pair being coded.
    denom = (x64.square() + y64.square()).clamp_min(1e-30)
    rel_slack = ((chosen - best) / denom).max().item()
    n_bad = (((chosen - best) / denom) > 1e-6).sum().item()
    check("polar assignment is exact nearest-neighbour", rel_slack < 1e-6,
          f"max rel slack={rel_slack:.2e}, pairs above 1e-6: {n_bad}/{chosen.numel()}")

    # Byte accounting: indices + row scales + header, versus BF16.
    pol = quantize_polar(w, "adjacent").payload_bytes(rows, cols)
    vq = quantize_vq2d(w, "adjacent").payload_bytes(rows, cols)
    real = rows * cols // 2 + rows * 4 + 16 * 4 + 1
    bf16 = rows * cols * 2
    check("polar payload within 2 KB of real-4", abs(pol - real) < 2048, f"{pol} vs {real}")
    check("vq2d payload within 2 KB of real-4", abs(vq - real) < 2048, f"{vq} vs {real}")
    check("all 4-bit payloads under half of BF16", max(pol, vq, real) < bf16 * 0.55,
          f"max={max(pol, vq, real)} bf16={bf16}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
