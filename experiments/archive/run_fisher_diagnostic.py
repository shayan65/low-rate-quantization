"""Phase 1.1: does Fisher-weighted error predict task damage better than weight MSE?

Every phase of this project found that layer-local weight MSE fails to predict
task NLL.  The reformulation in `paper/scq_proposal.md` says the missing factor
is the downstream loss sensitivity

    S_j = E[(dL/dy_j)^2]

and that the right local surrogate for the end-to-end damage of perturbing one
layer is the Gauss-Newton/Fisher form

    cost = E_t[ sum_j S_j * ((dW x_t)_j)^2 ] .

This script tests that claim without building any new quantizer.  It quantizes
ONE layer at a time (all others BF16), measures the actual validation NLL
damage, and asks which predictor ranks the 18 layers correctly:

    weight_mse   mean((dW)^2)                        -- what earlier phases used
    h_weighted   mean_jc( h_c * dW_jc^2 )            -- diagonal forward only
    out_mse      E_t[ mean_j ((dW x_t)_j)^2 ]        -- exact forward only
    fisher       E_t[ mean_j S_j ((dW x_t)_j)^2 ]    -- forward x backward

Because only one layer is quantized at a time, everything downstream is BF16, so
S measured on the BF16 model is the self-consistent choice here.  This isolates
"is the Fisher metric the right local objective?" from "does self-consistency
help?", which is the next experiment, not this one.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_scf_phase0 import Budget, capture_sensitivity
from run_scq_qwen import atomic_json, evaluate, find_layers, get_module, make_codec, paired_bootstrap
from scq import capture_inputs

PREDICTORS = ("weight_mse", "h_weighted", "out_mse", "fisher")


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


@torch.no_grad()
def capture_activation_sample(model, modules, windows, max_tokens=2048, batch=4):
    """Keep a bounded sample of each layer's input rows for exact output error."""
    store: dict[str, list] = {k: [] for k in modules}
    count: dict[str, int] = {k: 0 for k in modules}
    handles = []

    def mk(name):
        def hook(_m, args):
            if count[name] >= max_tokens:
                return
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1])
            take = min(flat.shape[0], max_tokens - count[name])
            store[name].append(flat[:take].float())
            count[name] += take

        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_pre_hook(mk(name)))
    try:
        for i in range(0, len(windows), batch):
            x = torch.tensor(windows[i : i + batch], device="cuda")
            model(x[:, :-1], use_cache=False)
            del x
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v, 0) for k, v in store.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/fisher_diagnostic_v1")
    ap.add_argument("--eval-blocks", type=int, default=512)
    ap.add_argument("--fit-windows", type=int, default=32)
    ap.add_argument("--act-tokens", type=int, default=2048)
    ap.add_argument("--codec", default="vq2d")
    ap.add_argument("--seconds", type=float, default=1200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    tok = AutoTokenizer.from_pretrained(a.model)
    train_ids = tok(
        "\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()), add_special_tokens=False
    ).input_ids
    val_ids = tok(
        "\n".join(pd.read_parquet(f"{a.data}/validation.parquet").text.tolist()),
        add_special_tokens=False,
    ).input_ids
    rng = np.random.default_rng(20260921)
    starts = np.sort(rng.choice(len(train_ids) - 129, a.fit_windows, replace=False)).tolist()
    fit = [train_ids[s : s + 129] for s in starts]
    blocks = [val_ids[i * 128 : i * 128 + 129] for i in range(a.eval_blocks)]
    blocks = [b for b in blocks if len(b) == 129]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    modules = {f"L{i}": get_module(model, i) for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}
    codec = make_codec(a.codec)

    h = capture_inputs(model, modules, fit)
    S = capture_sensitivity(model, modules, fit)
    X = capture_activation_sample(model, modules, fit, a.act_tokens)
    print(json.dumps({"stage": "statistics_captured", "t": B.stamp()}), flush=True)

    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "t": B.stamp()}), flush=True)

    rows = []
    for k in params:
        if not B.check(f"layer {k}", 40):
            break
        w = originals[k].cuda().float()
        q, _, _ = codec(k, w, h.get(k), {})
        dW = (q - w)

        x = X[k]  # (tokens, in_dim)
        dy = x @ dW.T  # (tokens, out_dim)
        s = S[k]
        rec = {
            "layer": k,
            "weight_mse": dW.square().mean().item(),
            "h_weighted": (dW.square() * h[k].view(1, -1)).mean().item(),
            "out_mse": dy.square().mean().item(),
            "fisher": (dy.square() * s.view(1, -1)).mean().item(),
        }
        del dy

        with torch.no_grad():
            params[k].copy_(q.to(params[k].dtype))
        ev = evaluate(model, blocks)
        with torch.no_grad():
            params[k].copy_(originals[k].to(params[k].dtype).cuda())
        d, lo, hi = paired_bootstrap(ev["block_nll"], bf["block_nll"])
        rec.update(delta_nll=ev["nll"] - bf["nll"], delta_ci=[lo, hi])
        rows.append(rec)
        print(json.dumps({**{kk: vv for kk, vv in rec.items() if kk != "delta_ci"},
                          "t": B.stamp()}), flush=True)
        del w, q, dW, x

    actual = [r["delta_nll"] for r in rows]
    corr = {}
    for p in PREDICTORS:
        v = [r[p] for r in rows]
        corr[p] = {
            "spearman": spearman(v, actual),
            "pearson": pearson(v, actual),
            # Proportionality fit through the origin; the constant absorbs the
            # loss-reduction normalization used when S was measured.
            "slope_through_origin": float(np.dot(v, actual) / max(np.dot(v, v), 1e-30)),
        }
        pred = np.asarray(v) * corr[p]["slope_through_origin"]
        ss_res = float(((np.asarray(actual) - pred) ** 2).sum())
        ss_tot = float(((np.asarray(actual) - np.mean(actual)) ** 2).sum())
        corr[p]["r2_through_origin"] = 1 - ss_res / max(ss_tot, 1e-30)

    res = {
        "plan": {
            "model": a.model, "codec": a.codec, "layers": layers,
            "eval_targets": 128 * len(blocks), "fit_windows": a.fit_windows,
            "act_tokens": a.act_tokens,
            "protocol": "one layer quantized at a time, all others BF16; "
                        "S measured on the BF16 model (self-consistent for this case)",
            "note": "S uses mean-reduced loss, so absolute scale is arbitrary; "
                    "rank and linear correlation are the meaningful quantities",
        },
        "bf16": {k: v for k, v in bf.items() if k != "block_nll"},
        "layers": rows,
        "correlations": corr,
        "seconds": B.stamp(),
    }
    atomic_json(out / "results.json", res)

    L = ["# Does Fisher-weighted error predict task damage?", "",
         f"Qwen3.5-0.8B, {a.codec} codec, one layer quantized at a time, "
         f"{128 * len(blocks):,} validation targets. BF16 NLL {bf['nll']:.6f}.",
         f"Wall clock {res['seconds']}s.", "",
         "## Predictor quality across the 18 layers", "",
         "| Predictor | Spearman | Pearson | R² (through origin) |",
         "|---|---:|---:|---:|"]
    name = {"weight_mse": "weight MSE *(what earlier phases used)*",
            "h_weighted": "diagonal activation-weighted",
            "out_mse": "exact output error (forward only)",
            "fisher": "**Fisher: forward x backward**"}
    for p in PREDICTORS:
        c = corr[p]
        L.append(f"| {name[p]} | {c['spearman']:.3f} | {c['pearson']:.3f} "
                 f"| {c['r2_through_origin']:.3f} |")
    L += ["", "## Per-layer detail", "",
          "| Layer | ΔNLL | weight MSE | out MSE | Fisher |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['layer']} | {r['delta_nll']:+.6f} | {r['weight_mse']:.3e} "
                 f"| {r['out_mse']:.3e} | {r['fisher']:.3e} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
