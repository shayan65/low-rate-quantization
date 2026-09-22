"""Does the earlier draft's headline survive a matched-strength control?

The earlier draft selected each layer's codec candidate by *calibration NLL*:
four pairings for polar, four clipping multipliers for real-4.  The codec ladder
in ``run_scq_qwen.py`` instead uses one cheap weighted-error rule for every
codec, so its polar number is not directly comparable to the earlier figure.

This script reproduces the earlier draft's selection protocol exactly, and then
grants the *same* protocol to the Lloyd-fitted scalar control and to the paired
2-D codec.  If polar's reported advantage was a selection-effort artifact it
will survive here; if it was a control-strength artifact it will not.

Protocol (as in the earlier draft): each layer is screened independently with
all other weights at BF16, the winning candidate per layer is frozen, and the
frozen choices are then applied cumulatively and evaluated on the complete
validation stream.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from polar_codec import PAIRINGS, fit_scalar_grid, quantize_polar_legacy, _bucketize
from run_scq_qwen import atomic_json, evaluate, find_layers, get_module, paired_bootstrap
from scq import calibration_nll, capture_inputs
from vq_codec import quantize_vq2d

CLIP_FACTORS = (0.85, 0.95, 1.0, 1.05)


@torch.no_grad()
def real4_candidates(w, fitted: bool):
    """Four clip multipliers, uniform or Lloyd-fitted codebook."""
    base = w.abs().amax(1, keepdim=True).clamp_min(1e-8)
    grid = fit_scalar_grid((w / base).clamp(-1, 1), 16) if fitted else None
    out = []
    for f in CLIP_FACTORS:
        s = base * f
        if grid is None:
            q = s * ((w / s).clamp(-1, 1).add(1).mul(7.5).round().div(7.5).sub(1))
        else:
            u = (w / s).clamp(grid[0].item(), grid[-1].item())
            q = s * grid[_bucketize(u, grid)]
        out.append((f"clip{f}", q))
    return out


@torch.no_grad()
def polar_candidates(w):
    return [(p, quantize_polar_legacy(w, p).weight) for p in PAIRINGS]


@torch.no_grad()
def vq_candidates(w, h=None):
    """Four pairings; ``h`` switches the codebook fit to the activation metric."""
    return [(p, quantize_vq2d(w, p, h, fit_iters=12).weight) for p in PAIRINGS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/selection_fairness_v1")
    ap.add_argument("--eval-blocks", type=int, default=0)
    ap.add_argument("--cal-windows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--families", nargs="*", default=None)
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
    rng = np.random.default_rng(a.seed)
    starts = np.sort(rng.choice(len(train_ids) - 129, a.cal_windows, replace=False)).tolist()
    cal = [train_ids[s : s + 129] for s in starts]
    n = (len(val_ids) - 1) // 128
    if a.eval_blocks:
        n = min(n, a.eval_blocks)
    blocks = [val_ids[i * 128 : i * 128 + 129] for i in range(n)]
    blocks = [b for b in blocks if len(b) == 129]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    layers = find_layers(model)
    params = {i: get_module(model, i).weight for i in layers}
    originals = {i: p.detach().clone().cpu() for i, p in params.items()}

    def restore_all():
        with torch.no_grad():
            for i, p in params.items():
                p.copy_(originals[i].to(p.dtype).cuda())

    restore_all()
    bf = evaluate(model, blocks)
    results = {
        "plan": {
            "model": a.model,
            "layers": layers,
            "protocol": "per-layer candidate selection by calibration NLL with all "
            "other weights at BF16, then cumulative application (earlier draft's rule)",
            "cal_windows": a.cal_windows,
            "seed": a.seed,
            "cal_starts": starts,
            "eval_blocks": len(blocks),
            "eval_targets": 128 * len(blocks),
        },
        "bf16": {k: v for k, v in bf.items() if k != "block_nll"},
        "methods": {},
    }
    print(json.dumps({"method": "bf16", "nll": bf["nll"]}), flush=True)

    # Activation statistics from the BF16 model, for the codecs that use them.
    # Captured once so that every family sees the identical statistic.
    modules = {i: get_module(model, i) for i in layers}
    h_all = capture_inputs(model, modules, cal)

    families = {
        "polar_legacy_nllsel": lambda w, i: polar_candidates(w),
        "real4_uniform_nllsel": lambda w, i: real4_candidates(w, False),
        "real4_lloyd_nllsel": lambda w, i: real4_candidates(w, True),
        "vq2d_nllsel": lambda w, i: vq_candidates(w, None),
        "vq2d_h_nllsel": lambda w, i: vq_candidates(w, h_all.get(i)),
    }
    if a.families:
        families = {k: v for k, v in families.items() if k in a.families}

    for fam, gen in families.items():
        restore_all()
        choices = {}
        chosen_w = {}
        t0 = time.monotonic()
        for i in layers:
            w = originals[i].cuda().float()
            best = None
            for name, q in gen(w, i):
                with torch.no_grad():
                    params[i].copy_(q.to(params[i].dtype))
                nll = calibration_nll(model, cal)
                if best is None or nll < best[0]:
                    best = (nll, name, q.clone())
            with torch.no_grad():  # restore this layer before screening the next
                params[i].copy_(originals[i].to(params[i].dtype).cuda())
            choices[i] = {"candidate": best[1], "cal_nll": best[0]}
            chosen_w[i] = best[2]
            del w
            torch.cuda.empty_cache()
        with torch.no_grad():
            for i in layers:
                params[i].copy_(chosen_w[i].to(params[i].dtype))
        sel_s = time.monotonic() - t0
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev["block_nll"], bf["block_nll"])
        rec = {k: v for k, v in ev.items() if k != "block_nll"}
        rec.update(delta_nll=ev["nll"] - bf["nll"], delta_ci=[lo, hi],
                   choices={str(k): v for k, v in choices.items()}, selection_seconds=sel_s)
        results["methods"][fam] = rec
        results["methods"][fam]["block_nll"] = ev["block_nll"]
        print(json.dumps({"method": fam, "nll": ev["nll"], "delta": rec["delta_nll"],
                          "ci": [lo, hi]}), flush=True)
        atomic_json(out / "results.json", {**results, "methods": {
            k: {kk: vv for kk, vv in v.items() if kk != "block_nll"}
            for k, v in results["methods"].items()}})
        del chosen_w
        torch.cuda.empty_cache()

    comps = {}
    names = list(results["methods"])
    for i, m1 in enumerate(names):
        for m2 in names[i + 1 :]:
            d, lo, hi = paired_bootstrap(
                results["methods"][m1]["block_nll"], results["methods"][m2]["block_nll"]
            )
            comps[f"{m1}_minus_{m2}"] = {"delta": d, "ci": [lo, hi]}
    for v in results["methods"].values():
        v.pop("block_nll", None)
    results["comparisons"] = comps
    results["seconds"] = time.monotonic() - started
    atomic_json(out / "results.json", results)

    lines = [
        "# Selection-fairness check: the earlier draft's protocol, applied to every codec",
        "",
        f"Per-layer candidate selection by calibration NLL ({a.cal_windows} WikiText-2 training "
        f"windows), then cumulative application; {128 * len(blocks):,} validation targets.",
        f"BF16 NLL {bf['nll']:.6f}.",
        "",
        "| Method | NLL | ΔNLL vs BF16 | 95% CI |",
        "|---|---:|---:|:--|",
    ]
    for m, r in results["methods"].items():
        lines.append(
            f"| {m} | {r['nll']:.6f} | {r['delta_nll']:+.6f} "
            f"| [{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |"
        )
    lines += ["", "## Paired head-to-head differences", "",
              "| Comparison | ΔNLL | 95% CI |", "|---|---:|:--|"]
    for k, c in comps.items():
        lines.append(f"| {k.replace('_minus_', ' − ')} | {c['delta']:+.6f} "
                     f"| [{c['ci'][0]:+.6f}, {c['ci'][1]:+.6f}] |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
