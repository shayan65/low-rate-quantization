"""Qwen3.5-0.8B: codec ladder plus self-consistent calibration, full validation.

All 18 DeltaNet ``in_proj_qkv`` projections are replaced simultaneously -- the
same setting as the repository's headline result -- and every codec spends the
same 4 index bits per real weight plus one FP32 row scale.

Codec ladder
  real4_uniform  the original control (uniform 16-level scalar grid)
  real4_lloyd    the same code with a per-tensor Lloyd-Max codebook (fair control)
  polar_legacy   decoupled magnitude/phase rounding (the paper's method)
  polar_fitted   exact nearest-neighbour polar assignment + Lloyd radii
  vq2d           unconstrained 256-point 2-D codebook over weight pairs

Calibration protocol
  one_shot       activations measured on the BF16 model (what everyone does)
  scf            guarded self-consistent field iteration (:mod:`scq`)

Pairing is chosen per layer by activation-weighted output error on the
calibration windows -- identically for every paired codec, so no codec gets a
stronger selection rule than another.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from polar_codec import PAIRINGS, pairing_indices, quantize_polar, quantize_polar_legacy
from polar_codec import quantize_real4, quantize_real4_fitted
from scq import SCFConfig, run_scf, scf_summary
from vq_codec import quantize_vq2d

KEY = "model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight"


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def find_layers(model) -> list[int]:
    out = []
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "in_proj_qkv"):
            out.append(i)
    return out


def get_module(model, idx: int):
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return lm.layers[idx].linear_attn.in_proj_qkv


@torch.no_grad()
def evaluate(model, blocks, batch: int = 8, device: str = "cuda") -> dict:
    """Mean per-block NLL over contiguous 128-target blocks (context reset per block)."""
    per_block = []
    t0 = time.monotonic()
    for i in range(0, len(blocks), batch):
        x = torch.tensor(blocks[i : i + batch], device=device)
        logits = model(x[:, :-1], use_cache=False).logits.float()
        y = x[:, 1:]
        tok = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
        ).reshape(x.shape[0], -1)
        per_block.extend(tok.mean(1).cpu().tolist())
        del x, y, logits, tok
    return {
        "nll": float(np.mean(per_block)),
        "ppl": float(np.exp(np.mean(per_block))),
        "blocks": len(per_block),
        "target_tokens": 128 * len(per_block),
        "block_nll": per_block,
        "seconds": time.monotonic() - t0,
    }


def paired_bootstrap(a: list[float], b: list[float], reps: int = 5000, seed: int = 7):
    """Percentile CI for mean(a) - mean(b) over paired validation blocks."""
    rng = np.random.default_rng(seed)
    x = np.asarray(a) - np.asarray(b)
    n = len(x)
    draws = x[rng.integers(0, n, size=(reps, n))].mean(1)
    return float(x.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------
# Codec adapters: all share the signature (name, w, h, state) -> (q, state, idx)
# --------------------------------------------------------------------------


def _pick_pairing(w, h, fn):
    best = None
    for p in PAIRINGS:
        c = fn(w, p, h)
        score = c.weighted_error if h is not None else c.weight_mse
        if best is None or score < best[0]:
            best = (score, c)
    return best[1]


def make_codec(kind: str):
    if kind == "real4_uniform":
        def f(name, w, h, state):
            q, _, _ = quantize_real4(w, h)
            ix = ((q / q.abs().amax(1, keepdim=True).clamp_min(1e-9)) * 7.5).round().to(torch.int16)
            return q, state, ix
        return f
    if kind == "real4_lloyd":
        def f(name, w, h, state):
            q, _, _, grid = quantize_real4_fitted(w, h, grid=None)
            ix = torch.bucketize(
                (q / q.abs().amax(1, keepdim=True).clamp_min(1e-9)).contiguous(),
                (grid[1:] + grid[:-1]) * 0.5,
            ).to(torch.int16)
            state["grid"] = grid
            return q, state, ix
        return f
    if kind == "polar_legacy":
        def f(name, w, h, state):
            c = _pick_pairing(w, None, lambda ww, p, hh: quantize_polar_legacy(ww, p))
            state["pairing"] = c.pairing
            return c.weight, state, c.mag_index.to(torch.int16) * 32 + c.phase_index.to(torch.int16)
        return f
    if kind == "polar_fitted":
        def f(name, w, h, state):
            # Pairing is fixed on the first call and reused: re-selecting it each
            # SCF iteration costs 4x the work and would let the pairing drift
            # with the statistics, which is not the fixed point we want.
            pairing = state.get("pairing")
            if pairing is None:
                c = _pick_pairing(w, h, lambda ww, p, hh: quantize_polar(ww, p, hh))
                pairing = c.pairing
            else:
                c = quantize_polar(w, pairing, h)
            state["pairing"] = pairing
            state["grid"] = c.grid
            return c.weight, state, c.mag_index.to(torch.int16) * 32 + c.phase_index.to(torch.int16)
        return f
    if kind == "vq2d":
        def f(name, w, h, state):
            init = state.get("codebook")
            pairing = state.get("pairing")
            if pairing is None:
                c = _pick_pairing(w, h, lambda ww, p, hh: quantize_vq2d(ww, p, hh, fit_iters=8))
                pairing = c.pairing
            c = quantize_vq2d(w, pairing, h, codebook=init, refit=True)
            state["pairing"] = pairing
            state["codebook"] = c.codebook
            return c.weight, state, c.index.to(torch.int16)
        return f
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/scq_qwen08b_v1")
    ap.add_argument("--eval-blocks", type=int, default=0, help="0 = full validation stream")
    ap.add_argument("--cal-windows", type=int, default=16)
    ap.add_argument("--guard-windows", type=int, default=0,
                    help="disjoint training windows used only for the SCF energy guard; "
                         "0 reuses the calibration windows (overfits, see paper)")
    ap.add_argument("--scf-iters", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--anderson", type=int, default=0)
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    tok = AutoTokenizer.from_pretrained(a.model)
    train_ids = tok(
        "\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()), add_special_tokens=False
    ).input_ids
    val_ids = tok(
        "\n".join(pd.read_parquet(f"{a.data}/validation.parquet").text.tolist()),
        add_special_tokens=False,
    ).input_ids

    rng = np.random.default_rng(20260921)
    n_draw = a.cal_windows + a.guard_windows
    starts = np.sort(rng.choice(len(train_ids) - 129, n_draw, replace=False)).tolist()
    cal = [train_ids[s : s + 129] for s in starts[: a.cal_windows]]
    guard = [train_ids[s : s + 129] for s in starts[a.cal_windows :]] or None
    n_blocks = (len(val_ids) - 1) // 128
    if a.eval_blocks:
        n_blocks = min(n_blocks, a.eval_blocks)
    blocks = [val_ids[i * 128 : i * 128 + 129] for i in range(n_blocks)]
    blocks = [b for b in blocks if len(b) == 129]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    modules = {f"L{i}": get_module(model, i) for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}

    payload = sum(p.numel() // 2 + p.shape[0] * 4 for p in params.values())
    bf16_bytes = sum(p.numel() * 2 for p in params.values())
    plan = {
        "model": a.model,
        "layers": layers,
        "tensor": "linear_attn.in_proj_qkv.weight",
        "eval": f"{len(blocks)} contiguous 128-target WikiText-2 validation blocks "
        f"({128 * len(blocks)} targets), context reset per block",
        "calibration": f"{a.cal_windows} fixed WikiText-2 training windows (seed 20260921)",
        "guard_windows": a.guard_windows,
        "cal_starts": starts,
        "budget": "4 index bits per real weight + one FP32 row scale per row; "
        "per-tensor codebook headers (<=2 KB) counted separately",
        "index_payload_bytes": payload,
        "bf16_bytes": bf16_bytes,
        "scf": {"alpha": a.alpha, "max_iters": a.scf_iters, "anderson_m": a.anderson,
                "guard_set": "disjoint training windows" if guard else "same as calibration",
                "guard": "reject any iteration that raises calibration NLL; halve alpha"},
    }
    atomic_json(out / "plan.json", plan)
    print(json.dumps({"layers": layers, "eval_blocks": len(blocks)}), flush=True)

    results: dict = {"plan": plan, "methods": {}}

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    results["methods"]["bf16"] = bf
    print(json.dumps({"method": "bf16", "nll": bf["nll"], "ppl": bf["ppl"]}), flush=True)
    atomic_json(out / "results.json", results)

    ladder = a.methods or [
        "real4_uniform", "real4_lloyd", "real4_lloyd+seq",
        "polar_legacy", "polar_legacy+seq", "polar_fitted", "polar_fitted+seq",
        "vq2d", "vq2d+seq",
    ]
    for method in ladder:
        kind = method.replace("+scf", "").replace("+seq", "")
        use_scf = method.endswith("+scf")
        use_seq = method.endswith("+seq")
        codec = make_codec(kind)
        restore()
        t0 = time.monotonic()
        if use_scf:
            cfg = SCFConfig(alpha=a.alpha, max_iters=a.scf_iters, anderson_m=a.anderson)
            state, hist = run_scf(
                model, params, modules,
                {k: v for k, v in originals.items()}, codec, cal, cfg,
                guard_windows=guard,
            )
            extra = {"scf": scf_summary(hist), "scf_history": hist.iterations}
        elif use_seq:
            # Sequential propagation: quantize in topological order, recomputing
            # each layer's statistics with the earlier layers already replaced.
            # Because layer L's input statistics depend only on layers < L, this
            # one pass reaches the calibration fixed point exactly -- no
            # iteration required (see results/scf_phase0_v1).
            from scq import capture_inputs
            with torch.no_grad():
                for k in params:
                    h = capture_inputs(model, {k: modules[k]}, cal)
                    q, _, _ = codec(k, originals[k].cuda().float(), h.get(k), {})
                    params[k].copy_(q.to(params[k].dtype))
            extra = {"calibration": "sequential propagation, one pass"}
        else:
            from scq import capture_inputs
            h = capture_inputs(model, modules, cal)
            with torch.no_grad():
                for k in params:
                    q, _, _ = codec(k, originals[k].cuda().float(), h.get(k), {})
                    params[k].copy_(q.to(params[k].dtype))
            extra = {"calibration": "one-shot, BF16 activations"}
        cal_t = time.monotonic() - t0
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev["block_nll"], bf["block_nll"])
        rec = {k: v for k, v in ev.items() if k != "block_nll"}
        rec.update(
            delta_nll=ev["nll"] - bf["nll"],
            delta_ci=[lo, hi],
            calibration_seconds=cal_t,
            **extra,
        )
        results["methods"][method] = rec
        results["methods"][method]["block_nll"] = ev["block_nll"]
        print(json.dumps({"method": method, "nll": ev["nll"], "delta": rec["delta_nll"],
                          "ci": [lo, hi], "cal_s": round(cal_t, 1)}), flush=True)
        atomic_json(out / "results.json", results)

    # Pairwise comparisons against the fair scalar control and against polar.
    comps = {}
    base_names = [m for m in results["methods"] if m != "bf16"]
    for m in base_names:
        for ref in ("real4_lloyd", "polar_legacy"):
            if m == ref or ref not in results["methods"]:
                continue
            d, lo, hi = paired_bootstrap(
                results["methods"][m]["block_nll"], results["methods"][ref]["block_nll"]
            )
            comps[f"{m}_minus_{ref}"] = {"delta": d, "ci": [lo, hi]}
    results["comparisons"] = comps
    for v in results["methods"].values():
        v.pop("block_nll", None)
    results["seconds"] = time.monotonic() - started
    atomic_json(out / "results.json", results)

    lines = [
        "# Qwen3.5-0.8B: codec ladder and self-consistent calibration",
        "",
        f"All {len(layers)} DeltaNet `in_proj_qkv` projections replaced simultaneously; "
        f"{128 * len(blocks):,} WikiText-2 validation targets; equal 4-bit index budget.",
        f"BF16 reference NLL {bf['nll']:.6f} (ppl {bf['ppl']:.3f}).",
        "",
        "| Method | NLL | ppl | ΔNLL vs BF16 | 95% CI |",
        "|---|---:|---:|---:|:--|",
    ]
    for m in ladder:
        if m not in results["methods"]:
            continue
        r = results["methods"][m]
        lines.append(
            f"| {m} | {r['nll']:.6f} | {r['ppl']:.3f} | {r['delta_nll']:+.6f} "
            f"| [{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
