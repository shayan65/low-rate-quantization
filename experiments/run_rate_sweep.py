"""Step 0: historical rate-resolved weight-reconstruction sweep.

Two predeclared questions (see paper/ternary_proposal.md):

  H0  Does a blockwise Hadamard rotation capture what joint coding captures?
      Stop if rotation closes >= 80% of the scalar-to-2-D gap on every tensor.
      This tests observed fits; it is not a universal subsumption claim.

  H1  Does the joint-coding advantage grow as the rate falls?
      Post-rotation at the ternary rate it must beat the product grid by >= 15%
      weight MSE, and by >= 2x the post-rotation margin at 4 bits.

Design. At each rate the d-dimensional product grid and fitted VQ both have
L^d entries. log2(L) is ideal index entropy, not measured packed bits/weight.
This historical implementation uses FP32 scales and reconstruction codebooks;
it does not serialize indices, account for metadata/padding, or measure kernels.
The configuration gate chooses the best allowed dimension at each rate and
therefore does not isolate a causal rate effect at fixed dimension.

The scalar baseline is a sampled Lloyd-fitted L-level quantizer. At L=3 it
has three arbitrary levels, not literal scaled {-1,0,+1} weights. Sampled,
locally fitted codebooks do not establish globally optimal reconstruction.

Metric validity.  The rotation is orthogonal, so
||W - Q(WH)H^T||_F = ||WH - Q(WH)||_F: error measured in the rotated basis *is*
the error in the original basis.  Rotation can only help by making the source
easier to quantize.  This is asserted as a self-test.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from run_scf_phase0 import Budget
from rate_sweep_analysis import (DEFAULT_TENSORS, DIMS, LEVELS, MAX_K,
                                 aggregate_rows, evaluate_gates, render)

GROUP = 128  # historical fitting executes with FP32 scales
BLOCK = 1024  # Hadamard block size


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last axis."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, "Hadamard block size must be a power of two"
    orig = x.shape
    x = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.reshape(-1, n)
        h *= 2
    return (x / math.sqrt(n)).reshape(orig)


def rotate(w: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Blockwise randomized Hadamard along the input (column) dimension."""
    rows, cols = w.shape
    assert cols % BLOCK == 0, f"cols={cols} not divisible by block {BLOCK}"
    g = torch.Generator(device=w.device).manual_seed(seed)
    signs = (torch.randint(0, 2, (cols,), generator=g, device=w.device) * 2 - 1).float()
    x = (w * signs).view(rows, cols // BLOCK, BLOCK)
    return fwht(x).view(rows, cols)


def group_normalize(w: torch.Tensor):
    """g128 scaling: one scale per 128 consecutive weights. Returns (u, scale)."""
    rows, cols = w.shape
    assert cols % GROUP == 0
    g = w.view(rows, cols // GROUP, GROUP)
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return (g / scale).view(rows, cols), scale


@torch.no_grad()
def lloyd_scalar(v: torch.Tensor, levels: int, iters: int = 40) -> torch.Tensor:
    """Lloyd-Max fit of `levels` scalar reconstruction points."""
    v = v.flatten()
    v = v[:: max(1, v.numel() // 4_000_000)].float()
    q = torch.linspace(0, 1, levels, device=v.device)
    c = torch.quantile(v, q)
    for _ in range(iters):
        mid = (c[1:] + c[:-1]) * 0.5
        idx = torch.bucketize(v.contiguous(), mid)
        num = torch.zeros(levels, device=v.device, dtype=torch.float64)
        den = torch.zeros(levels, device=v.device, dtype=torch.float64)
        num.scatter_add_(0, idx, v.double())
        den.scatter_add_(0, idx, torch.ones_like(v, dtype=torch.float64))
        new = torch.where(den > 0, num / den.clamp_min(1e-30), c.double()).float()
        new, _ = torch.sort(new)
        if torch.allclose(new, c, atol=1e-8):
            c = new
            break
        c = new
    return c


@torch.no_grad()
def scalar_mse(u: torch.Tensor, scale: torch.Tensor, c: torch.Tensor) -> float:
    """MSE of the scalar quantizer -- also the d-dimensional PRODUCT grid,
    since applying a scalar code independently per coordinate is exactly that."""
    rows, cols = u.shape
    mid = (c[1:] + c[:-1]) * 0.5
    total = 0.0
    for i in range(0, rows, 512):
        blk = u[i : i + 512]
        q = c[torch.bucketize(blk.contiguous(), mid)]
        s = scale[i : i + 512]
        d = ((q - blk).view(blk.shape[0], -1, GROUP) * s).square()
        total += d.double().sum().item()
    return total / (rows * cols)


@torch.no_grad()
def assign(points: torch.Tensor, cent: torch.Tensor, chunk: int = 200_000,
           max_buffer_bytes: float = 6e9):
    """Nearest centroid by chunked matmul; returns (index, sum of sq distance).

    The distance buffer is chunk x K floats, so a fixed chunk does not survive a
    large codebook: at K=65535 the default would ask for 52 GB, which is why the
    rate ladder previously stopped at K=7131. The chunk is reduced only when the
    buffer would exceed `max_buffer_bytes`, which leaves every previously
    reported configuration on exactly the path it ran (K=7131 needs 5.7 GB and
    is untouched). Chunking partitions points, never centroids, so each point is
    still compared against the whole codebook.
    """
    k = cent.shape[0]
    if chunk * k * 4 > max_buffer_bytes:
        chunk = max(1024, int(max_buffer_bytes // (k * 4)))
    cn = cent.square().sum(1)
    idx = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    sq = 0.0
    for i in range(0, points.shape[0], chunk):
        p = points[i : i + chunk]
        d = cn.unsqueeze(0) - 2.0 * (p @ cent.T)
        m, j = d.min(1)
        idx[i : i + chunk] = j
        sq += (m + p.square().sum(1)).double().sum().item()
        del d, m, j
    return idx, sq


@torch.no_grad()
def kmeans(points: torch.Tensor, k: int, iters: int = 50, seed: int = 0, sub: int = 500_000):
    """Sampled local Lloyd fit with empty-cluster respawn.

    Respawn improves initialization failures but guarantees neither the global
    optimum nor dominance over the product grid. The fit minimizes normalized
    error; reported original-weight MSE also weights errors by group scale².
    """
    g = torch.Generator(device=points.device).manual_seed(seed)
    n = points.shape[0]
    s = points[torch.randperm(n, generator=g, device=points.device)[: min(n, sub)]]
    cent = s[torch.randperm(s.shape[0], generator=g, device=points.device)[:k]].clone()
    d = points.shape[1]
    for it in range(iters):
        idx, _ = assign(s, cent)
        num = torch.zeros(k, d, device=points.device, dtype=torch.float64)
        den = torch.zeros(k, device=points.device, dtype=torch.float64)
        num.index_add_(0, idx, s.double())
        den.index_add_(0, idx, torch.ones(s.shape[0], device=points.device, dtype=torch.float64))
        alive = den > 0
        new = cent.double().clone()
        new[alive] = num[alive] / den[alive].unsqueeze(1)
        cent = new.float()
        dead = (~alive).nonzero(as_tuple=True)[0]
        if dead.numel() and it < iters - 1:
            err = (s - cent[idx]).square().sum(1)
            worst = err.topk(min(dead.numel(), s.shape[0])).indices
            cent[dead[: worst.numel()]] = s[worst]
    return cent


@torch.no_grad()
def vq_mse(u: torch.Tensor, scale: torch.Tensor, dim: int, k: int, seed: int = 0) -> float:
    """MSE of a free k-point codebook over dim-dimensional groups of weights.

    The codebook is fitted in the *g128-normalized* space, exactly like the
    scalar Lloyd baseline, and the resulting error is then rescaled by each
    group's FP32 scale.  Fitting on un-normalized weights instead would force a
    single codebook to span group scales that differ by orders of magnitude and
    would hand the scalar arm an unfair advantage.  `dim` divides GROUP, so each
    d-dimensional vector lies entirely inside one group and carries one scale.
    """
    rows, cols = u.shape
    assert GROUP % dim == 0
    ng = cols // GROUP
    pts = u.reshape(-1, dim)
    cent = kmeans(pts, k, seed=seed)
    per_vec = scale.view(rows, ng, 1).expand(rows, ng, GROUP // dim).reshape(-1)
    total = 0.0
    chunk = 2_000_000
    for i in range(0, pts.shape[0], chunk):
        p = pts[i : i + chunk]
        idx, _ = assign(p, cent)
        e = (p - cent[idx]).square().sum(1) * per_vec[i : i + chunk].square()
        total += e.double().sum().item()
        del idx, e
    return total / (rows * cols)


def load_tensor(model_dir: Path, key: str) -> torch.Tensor:
    idx = model_dir / "model.safetensors.index.json"
    shard = model_dir / (json.loads(idx.read_text())["weight_map"][key] if idx.exists()
                         else "model.safetensors")
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


def selftest(device="cuda"):
    """Rotation must be orthonormal and Frobenius-norm preserving."""
    x = torch.randn(64, BLOCK, device=device)
    y = fwht(x)
    assert abs(y.square().sum().item() - x.square().sum().item()) / x.square().sum().item() < 1e-4
    e = torch.zeros(BLOCK, BLOCK, device=device)
    e.fill_diagonal_(1.0)
    h = fwht(e)
    err = (h @ h.T - e).abs().max().item()
    assert err < 1e-3, f"Hadamard not orthonormal: {err}"
    w = torch.randn(256, BLOCK * 2, device=device)
    assert abs(rotate(w).square().sum().item() - w.square().sum().item()) / w.square().sum().item() < 1e-4
    print(json.dumps({"selftest": "ok", "orthonormality_err": err}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="../models")
    ap.add_argument("--out", default="../results/rate_sweep_v1")
    ap.add_argument("--seconds", type=float, default=5400)
    ap.add_argument("--tensors", nargs="*", default=[f"{model}:{tensor}" for model, tensor in DEFAULT_TENSORS])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)
    selftest()

    rows_out = []
    for spec in a.tensors:
        model_name, key = spec.split(":", 1)
        mdir = Path(a.models_root) / model_name
        if not mdir.exists():
            continue
        w0 = load_tensor(mdir, key).cuda()
        for rot in ("none", "hadamard"):
            if not B.check(f"{key} {rot}", 120):
                break
            w = w0 if rot == "none" else rotate(w0)
            u, scale = group_normalize(w)
            for L in LEVELS:
                c = lloyd_scalar(u, L)
                s_mse = scalar_mse(u, scale, c)
                for d in DIMS:
                    k = L ** d
                    if k > MAX_K:
                        continue
                    if not B.check(f"{key} {rot} L{L} d{d}", 30):
                        break
                    t0 = time.monotonic()
                    v = s_mse if d == 1 else vq_mse(u, scale, d, k)
                    rec = {
                        "model": model_name, "tensor": key, "rotation": rot,
                        "levels": L, "bits_per_weight": round(math.log2(L), 4),
                        "dim": d, "codebook": k,
                        "product_grid_mse": s_mse, "free_vq_mse": v,
                        "vq_over_product": v / s_mse, "seconds": time.monotonic() - t0,
                    }
                    rows_out.append(rec)
                    print(json.dumps(rec), flush=True)
                    (out / "results.json").write_text(json.dumps({"rows": rows_out}, indent=2) + "\n")
            del u, scale
            if rot != "none":
                del w
            torch.cuda.empty_cache()
        del w0
        torch.cuda.empty_cache()

    expected = [tuple(spec.split(":", 1)) for spec in a.tensors]
    verdict = evaluate_gates(rows_out, expected)
    aggregates = aggregate_rows(rows_out, expected)
    (out / "results.json").write_text(
        json.dumps({"rows": rows_out, "gates": verdict, "aggregates": aggregates,
                    "seconds": B.stamp()}, indent=2, allow_nan=False) + "\n")
    (out / "summary.md").write_text(render(rows_out, verdict, aggregates))
    print(render(rows_out, verdict, aggregates))


if __name__ == "__main__":
    main()
