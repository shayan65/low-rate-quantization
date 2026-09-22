"""Weight-reconstruction ablation of the improved polar codec.

Isolates two defects of the legacy polar quantizer at an identical byte budget:
  A. decoupled magnitude/phase rounding instead of nearest-neighbour assignment
     over the 8x32 codebook;
  B. a uniform radius grid ``k/7`` applied to Rayleigh-like pair magnitudes.

No task loss here -- this is the cheap gate that decides whether the codec is
worth a GPU forward pass.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from polar_codec import (
    PAIRINGS,
    N_MAG,
    quantize_polar,
    quantize_polar_legacy,
    quantize_real4,
    quantize_real4_fitted,
)

TENSORS = {
    "Qwen3.5-0.8B-Base": [
        "model.layers.{L}.linear_attn.in_proj_qkv.weight",
        "model.layers.{L}.mlp.up_proj.weight",
    ],
    "Qwen3.8-27B-metadata": [
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.mlp.up_proj.weight",
    ],
}


def load_tensor(model_dir: Path, key: str) -> torch.Tensor:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        shard = model_dir / wm[key]
    else:
        shard = model_dir / "model.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


def available_keys(model_dir: Path) -> set[str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        return set(f.keys())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--out", default="results/polar_codec_ablation_v1")
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 8, 16])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    rows_out = []
    for model_name, patterns in TENSORS.items():
        model_dir = Path(a.models_root) / model_name
        if not model_dir.exists():
            print(json.dumps({"skip": model_name}), flush=True)
            continue
        keys = available_keys(model_dir)
        wanted = []
        for pat in patterns:
            if "{L}" in pat:
                wanted += [pat.format(L=L) for L in a.layers]
            else:
                wanted.append(pat)
        for key in wanted:
            if key not in keys:
                continue
            w_cpu = load_tensor(model_dir, key)
            w = w_cpu.cuda()
            rec: dict = {"model": model_name, "tensor": key, "shape": list(w.shape)}

            _, _, e = quantize_real4(w)
            qr, _, _ = quantize_real4(w)
            rec["real4_mse"] = (qr - w).square().mean().item()
            del qr

            # Legacy polar: best pairing by weight MSE (the repo's own protocol).
            legacy_best = None
            for p in PAIRINGS:
                c = quantize_polar_legacy(w, p)
                if legacy_best is None or c.weight_mse < legacy_best.weight_mse:
                    legacy_best = c
            rec["legacy_polar_mse"] = legacy_best.weight_mse
            rec["legacy_pairing"] = legacy_best.pairing

            # Fix A only: exact NN assignment, still the uniform k/7 grid.
            uniform = torch.linspace(0, 1, N_MAG, device=w.device)
            nn_best = None
            for p in PAIRINGS:
                c = quantize_polar(w, p, grid=uniform)
                if nn_best is None or c.weight_mse < nn_best.weight_mse:
                    nn_best = c
            rec["nn_uniform_mse"] = nn_best.weight_mse
            rec["nn_uniform_pairing"] = nn_best.pairing

            # Fix A + B: exact NN assignment with a fitted Lloyd-Max radius grid.
            full_best = None
            for p in PAIRINGS:
                c = quantize_polar(w, p)
                if full_best is None or c.weight_mse < full_best.weight_mse:
                    full_best = c
            rec["nn_fitted_mse"] = full_best.weight_mse
            rec["nn_fitted_pairing"] = full_best.pairing
            rec["nn_fitted_grid"] = full_best.grid.tolist()

            # Fair control: give real-4 the SAME per-tensor Lloyd-Max shape
            # adaptation.  Without this we would be crediting polar geometry
            # for what is really just codebook fitting.
            qrf, _, _, sgrid = quantize_real4_fitted(w)
            rec["real4_fitted_mse"] = (qrf - w).square().mean().item()
            rec["real4_fitted_grid"] = sgrid.tolist()
            del qrf
            

            for k in ("legacy_polar", "nn_uniform", "nn_fitted", "real4_fitted"):
                rec[f"{k}_vs_real4"] = rec[f"{k}_mse"] / rec["real4_mse"]
            rec["nn_fitted_vs_real4_fitted"] = rec["nn_fitted_mse"] / rec["real4_fitted_mse"]
            rec["payload_bytes_polar"] = full_best.payload_bytes(*w.shape)
            rec["payload_bytes_real4"] = w.shape[0] * w.shape[1] // 2 + w.shape[0] * 4 + 1
            rows_out.append(rec)
            print(json.dumps(rec), flush=True)
            del w, legacy_best, nn_best, full_best
            torch.cuda.empty_cache()

    (out / "results.json").write_text(
        json.dumps({"rows": rows_out, "seconds": time.monotonic() - started}, indent=2) + "\n"
    )

    lines = [
        "# Polar codec ablation (weight reconstruction only)",
        "",
        "Ratio columns are MSE relative to the equal-byte real-4 control; < 1.000 favours polar.",
        "",
        "| Model | Tensor | real4 MSE | legacy polar | +exact NN | +fitted grid | real4+Lloyd | polar/real4-Lloyd |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        lines.append(
            f"| {r['model']} | {r['tensor'].split('.')[-2]} L{r['tensor'].split('layers.')[1].split('.')[0]} "
            f"| {r['real4_mse']:.4e} | {r['legacy_polar_vs_real4']:.3f} "
            f"| {r['nn_uniform_vs_real4']:.3f} | {r['nn_fitted_vs_real4']:.3f} "
            f"| {r['real4_fitted_vs_real4']:.3f} | {r['nn_fitted_vs_real4_fitted']:.3f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
