"""Checks on the v4 additions, run before any GPU time is spent.

The load-bearing claim is that the block form
    E = (W_J - Q_J) inv(U_JJ);  W_rest -= E U_J,rest
generalizes GPTQ to a vector codec. At d=1 it must reduce *exactly* to the
textbook scalar update, so that is checked against an independent reference
implementation rather than against itself.
"""

import sys

import torch

from packed_lowrate import decode, encode_scalar3, encode_vq, gnorm_fp16
from run_ternary_task_v4 import (cholesky_inv_upper, quantize_tensor, rotate,
                                 rotate_hessian, rot_signs, wassign)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def ref_scalar_gptq(W, levels, group, hinv):
    """Textbook GPTQ, written straight from the paper, no shared code."""
    rows, cols = W.shape
    W = W.clone()
    mid = (levels[1:] + levels[:-1]) * 0.5
    Q = torch.empty_like(W)
    for g0 in range(0, cols, group):
        s = W[:, g0 : g0 + group].abs().amax(1, keepdim=True).to(torch.float16).float()
        s = s.clamp_min(float(torch.finfo(torch.float16).tiny))
        for j in range(g0, g0 + group):
            w = W[:, j]
            c = torch.bucketize((w / s.squeeze(1)).contiguous(), mid)
            q = levels[c] * s.squeeze(1)
            Q[:, j] = q
            err = (w - q) / hinv[j, j]
            if j + 1 < cols:
                W[:, j + 1 :] -= err.unsqueeze(1) * hinv[j, j + 1 :].unsqueeze(0)
            W[:, j] = q
    return Q


def main() -> int:
    torch.manual_seed(0)
    rows, cols, group = 64, 128, 32

    # --- a Hessian with real structure, not a scaled identity ---
    X = torch.randn(4000, cols, device=DEV) @ (torch.randn(cols, cols, device=DEV) * 0.2)
    H = (X.T @ X / X.shape[0]).float()
    H = (H + H.T) * 0.5
    hinv = cholesky_inv_upper(H)
    check("hinv is upper triangular", torch.allclose(hinv, torch.triu(hinv)))

    # --- rotation of the Hessian is the same as rotating the activations ---
    s = rot_signs(cols, DEV)
    Hr = rotate_hessian(H, s, cols)
    Xr = rotate(X, s, cols)
    Hr_direct = (Xr.T @ Xr / X.shape[0]).float()
    rel = (Hr - Hr_direct).abs().max() / Hr_direct.abs().max()
    check("rotated Hessian == Hessian of rotated activations", rel < 1e-4, f"rel={rel:.2e}")

    # --- d=1 block form reproduces textbook scalar GPTQ exactly ---
    W = torch.randn(rows, cols, device=DEV) * 0.05
    levels = torch.tensor([-0.7, 0.0, 0.7], device=DEV)
    idx, sc, Q = quantize_tensor(W, levels, 1, group, hinv=hinv)
    Qref = ref_scalar_gptq(W, levels, group, hinv)
    check("d=1 vector-GPTQ == textbook scalar GPTQ", torch.equal(Q, Qref),
          f"max|diff|={(Q - Qref).abs().max().item():.3e}")

    # --- compensation must actually reduce activation-space error ---
    idx0, sc0, Q0 = quantize_tensor(W, levels, 1, group)
    e0 = (((W - Q0) @ H) * (W - Q0)).sum().item()
    e1 = (((W - Q) @ H) * (W - Q)).sum().item()
    check("GPTQ lowers activation-space error", e1 < e0, f"{e0:.5f} -> {e1:.5f}")
    check("GPTQ raises plain weight MSE (it trades it away)",
          (W - Q).square().mean() > (W - Q0).square().mean(),
          f"{(W - Q0).square().mean():.3e} -> {(W - Q).square().mean():.3e}")

    # --- uncompensated path matches the v3 recipe exactly ---
    u, scale = gnorm_fp16(W, group)
    mid = (levels[1:] + levels[:-1]) * 0.5
    ci = torch.bucketize(u.contiguous(), mid)
    Qv3 = (levels[ci].view(rows, cols // group, group) * scale).view(rows, cols)
    check("no-metric path reproduces the v3 quantizer", torch.equal(Q0, Qv3),
          f"max|diff|={(Q0 - Qv3).abs().max().item():.3e}")

    # --- metric assignment: brute force vs the transformed-search shortcut ---
    dim, k = 4, 81
    cb = torch.randn(k, dim, device=DEV) * 0.3
    pts = torch.randn(512, dim, device=DEV) * 0.3
    M = torch.linalg.solve_triangular(hinv[:dim, :dim], torch.eye(dim, device=DEV), upper=True)
    brute = torch.stack([(((pts[i] - cb) @ M).square().sum(1)).argmin() for i in range(512)])
    fast = ((pts @ M).unsqueeze(1) - (cb @ M).unsqueeze(0)).square().sum(2).argmin(1)
    check("transformed search == brute-force metric NN", torch.equal(brute, fast))

    # --- weighted assignment matches its own brute force ---
    w = torch.rand(512, dim, device=DEV) + 0.1
    bw = torch.stack([((w[i] * (pts[i] - cb).square()).sum(1)).argmin() for i in range(512)])
    check("wassign == brute-force weighted NN", torch.equal(wassign(pts, w, cb), bw))

    # --- decode(encode(.)) stays bit-exact under compensation ---
    for dim, k in ((1, 3), (4, 81), (8, 6561)):
        lv = levels if dim == 1 else torch.randn(k, dim, device=DEV).to(torch.float16).float() * 0.3
        W2 = torch.randn(rows, 256, device=DEV) * 0.05
        hin = cholesky_inv_upper((lambda Z: (Z.T @ Z / 2000).float())(
            torch.randn(2000, 256, device=DEV)))
        i2, s2, Q2 = quantize_tensor(W2, lv, dim, 64, hinv=hin)
        p = encode_scalar3(i2, s2, W2.shape, 64) if dim == 1 else \
            encode_vq(i2, s2, W2.shape, 64, dim, k)
        d = (decode(p, lv, DEV) - Q2).abs().max().item()
        check(f"dim={dim} decode(encode(w)) bit-exact under GPTQ", d == 0.0, f"max|diff|={d:.3e}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
