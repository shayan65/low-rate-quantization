"""How much headroom does *any* paired 8-bit weight code actually have?

Every codec compared here spends exactly 8 index bits per weight pair (= 4 bits
per real weight), one FP32 scale per row, and a small per-tensor codebook
header.  The ladder is:

  real4-uniform   the repository's original control (uniform 16-level grid)
  real4-Lloyd     the same scalar code with a per-tensor Lloyd-Max codebook
  polar-legacy    decoupled magnitude/phase rounding, uniform radii (3+5)
  polar-fitted    exact nearest-neighbour polar assignment, Lloyd radii,
                  swept over every magnitude/phase bit split
  vq2d-kmeans     an unconstrained 256-point 2-D codebook fitted by k-means

``vq2d-kmeans`` is the ceiling: no pairing-based 8-bit-per-pair scheme,
polar or otherwise, can beat a free 2-D codebook on this source.  Measuring it
tells us whether the polar direction has room to win or is structurally capped.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from polar_codec import (
    PAIRINGS,
    fit_radius_grid,
    pairing_indices,
    quantize_polar,
    quantize_polar_legacy,
    quantize_real4,
    quantize_real4_fitted,
)

SPLITS = [(2**m, p) for m, p in [(1, 128), (2, 64), (3, 32), (4, 16), (5, 8)]]


def load_tensor(model_dir: Path, key: str) -> torch.Tensor:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shard = model_dir / json.loads(index.read_text())["weight_map"][key]
    else:
        shard = model_dir / "model.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


@torch.no_grad()
def kmeans_2d(points: torch.Tensor, k: int = 256, iters: int = 40, seed: int = 0):
    """Fit ``k`` centroids to 2-D ``points`` (n, 2) with k-means++ style init."""
    g = torch.Generator(device=points.device).manual_seed(seed)
    n = points.shape[0]
    sub = points[torch.randperm(n, generator=g, device=points.device)[: min(n, 2_000_000)]]
    # Initialize on a random sample of distinct points.
    cent = sub[torch.randperm(sub.shape[0], generator=g, device=points.device)[:k]].clone()
    for _ in range(iters):
        idx = assign_2d(sub, cent)
        num = torch.zeros(k, 2, device=points.device, dtype=torch.float64)
        den = torch.zeros(k, device=points.device, dtype=torch.float64)
        num.index_add_(0, idx, sub.double())
        den.index_add_(0, idx, torch.ones(sub.shape[0], device=points.device, dtype=torch.float64))
        alive = den > 0
        new = cent.double().clone()
        new[alive] = num[alive] / den[alive].unsqueeze(1)
        cent = new.float()
    return cent


@torch.no_grad()
def assign_2d(points: torch.Tensor, cent: torch.Tensor, chunk: int = 1_000_000):
    out = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    cn = cent.square().sum(1)
    for i in range(0, points.shape[0], chunk):
        p = points[i : i + chunk]
        d = cn.unsqueeze(0) - 2 * (p @ cent.T)  # ||p||^2 is constant per row
        out[i : i + chunk] = d.argmin(1)
        del d
    return out


@torch.no_grad()
def vq2d_mse(w: torch.Tensor, pairing: str, k: int = 256) -> float:
    rows, cols = w.shape
    ia, ib = pairing_indices(cols, pairing, w.device)
    x, y = w[:, ia].float(), w[:, ib].float()
    base = torch.hypot(x, y).amax(1, keepdim=True).clamp_min(1e-8)
    pts = torch.stack([(x / base).flatten(), (y / base).flatten()], 1)
    cent = kmeans_2d(pts, k)
    total = 0.0
    for i in range(0, rows, 512):
        s = slice(i, min(i + 512, rows))
        p = torch.stack([(x[s] / base[s]).flatten(), (y[s] / base[s]).flatten()], 1)
        q = cent[assign_2d(p, cent)]
        d = (q - p).square().sum(1).view(x[s].shape) * base[s].square()
        total += d.double().sum().item()
        del p, q, d
    return total / (rows * cols)


@torch.no_grad()
def polar_split_mse(w: torch.Tensor, pairing: str, n_mag: int, n_phase: int) -> float:
    """Polar quantization at an arbitrary (magnitude levels, phase levels) split."""
    import polar_codec as pc

    old_m, old_p = pc.N_MAG, pc.N_PHASE
    pc.N_MAG, pc.N_PHASE = n_mag, n_phase
    try:
        return quantize_polar(w, pairing).weight_mse
    finally:
        pc.N_MAG, pc.N_PHASE = old_m, old_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="../models")
    ap.add_argument("--out", default="../results/codebook_ceiling_v1")
    ap.add_argument(
        "--tensors",
        nargs="*",
        default=[
            "Qwen3.5-0.8B-Base:model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "Qwen3.5-0.8B-Base:model.language_model.layers.8.linear_attn.in_proj_qkv.weight",
            "Qwen3.5-0.8B-Base:model.language_model.layers.16.linear_attn.in_proj_qkv.weight",
            "Qwen3.8-27B-metadata:model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "Qwen3.8-27B-metadata:model.language_model.layers.3.self_attn.q_proj.weight",
            "Qwen3.8-27B-metadata:model.language_model.layers.3.mlp.up_proj.weight",
        ],
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    rows_out = []
    for spec in a.tensors:
        model_name, key = spec.split(":", 1)
        model_dir = Path(a.models_root) / model_name
        if not model_dir.exists():
            continue
        w = load_tensor(model_dir, key).cuda()
        rec: dict = {"model": model_name, "tensor": key, "shape": list(w.shape)}

        qr, _, _ = quantize_real4(w)
        rec["real4_uniform"] = (qr - w).square().mean().item()
        del qr
        qf, _, _, _ = quantize_real4_fitted(w)
        rec["real4_lloyd"] = (qf - w).square().mean().item()
        del qf

        rec["polar_legacy"] = min(quantize_polar_legacy(w, p).weight_mse for p in PAIRINGS)

        splits = {}
        for n_mag, n_phase in SPLITS:
            best = min(polar_split_mse(w, p, n_mag, n_phase) for p in PAIRINGS)
            splits[f"{n_mag}x{n_phase}"] = best
        rec["polar_fitted_by_split"] = splits
        rec["polar_fitted_best"] = min(splits.values())
        rec["polar_fitted_best_split"] = min(splits, key=splits.get)

        rec["vq2d_ceiling"] = min(vq2d_mse(w, p) for p in PAIRINGS)

        ref = rec["real4_lloyd"]
        for k in ("real4_uniform", "polar_legacy", "polar_fitted_best", "vq2d_ceiling"):
            rec[f"{k}_vs_lloyd"] = rec[k] / ref
        rows_out.append(rec)
        print(json.dumps(rec), flush=True)
        del w
        torch.cuda.empty_cache()

    (out / "results.json").write_text(
        json.dumps({"rows": rows_out, "seconds": time.monotonic() - started}, indent=2) + "\n"
    )

    lines = [
        "# Codebook ceiling at 8 index bits per weight pair",
        "",
        "All columns are weight MSE relative to the **Lloyd-fitted real-4** control",
        "(< 1.000 beats it). `vq2d` is an unconstrained 256-point 2-D codebook and",
        "upper-bounds every pairing-based scheme at this budget.",
        "",
        "| Model | Tensor | real4-Lloyd MSE | real4-uniform | polar-legacy | polar-fitted | best split | vq2d ceiling |",
        "|---|---|---:|---:|---:|---:|:--:|---:|",
    ]
    for r in rows_out:
        short = r["tensor"].split("layers.")[1].replace(".weight", "")
        lines.append(
            f"| {r['model']} | {short} | {r['real4_lloyd']:.4e} "
            f"| {r['real4_uniform_vs_lloyd']:.3f} | {r['polar_legacy_vs_lloyd']:.3f} "
            f"| {r['polar_fitted_best_vs_lloyd']:.3f} | {r['polar_fitted_best_split']} "
            f"| {r['vq2d_ceiling_vs_lloyd']:.3f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
