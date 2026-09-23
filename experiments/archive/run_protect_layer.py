"""Phase 1.2: if you may leave exactly one layer in BF16, which one?

The Phase 1.1 diagnostic found that Fisher-weighted error has no general rank
skill across layers, but that it correctly identifies the single layer that
dominates total damage -- a layer that weight MSE ranks as the *safest* of all
18.  This turns that observation into a decision and tests it.

Arms (all quantize the 18 DeltaNet input projections except where noted):
    all_18            every layer quantized
    protect_fisher    leave the highest-Fisher layer in BF16
    protect_weightmse leave the highest-weight-MSE layer in BF16 (the choice the
                      earlier phases' criterion would have made)
    protect_random    a fixed arbitrary layer, as a floor

If Fisher's pick recovers substantially more damage than weight MSE's pick at
identical cost (one tensor left unquantized), the narrow claim survives even
though the general ranking claim did not.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, evaluate, find_layers, get_module, make_codec, paired_bootstrap
from scq import capture_inputs
from vq_codec import quantize_vq2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--diag", default="../results/fisher_diagnostic_v1/results.json")
    ap.add_argument("--out", default="../results/protect_layer_v1")
    ap.add_argument("--eval-blocks", type=int, default=512)
    ap.add_argument("--fit-windows", type=int, default=32)
    ap.add_argument("--codec", default="vq2d")
    ap.add_argument("--seconds", type=float, default=900)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    diag = json.loads(Path(a.diag).read_text())["layers"]
    pick_fisher = max(diag, key=lambda r: r["fisher"])["layer"]
    pick_wmse = max(diag, key=lambda r: r["weight_mse"])["layer"]
    pick_random = "L10"

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
    n = (len(val_ids) - 1) // 128
    if a.eval_blocks:  # 0 means the complete validation stream
        n = min(n, a.eval_blocks)
    blocks = [val_ids[i * 128 : i * 128 + 129] for i in range(n)]
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

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    restore()
    h = capture_inputs(model, modules, fit)
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "t": B.stamp()}), flush=True)

    # Bytes: leaving one tensor in BF16 costs its full BF16 size instead of 4-bit.
    per_tensor = {k: originals[k].numel() for k in params}
    idx_bytes = {k: n // 2 for k, n in per_tensor.items()}
    bf16_bytes = {k: n * 2 for k, n in per_tensor.items()}
    total_q = sum(idx_bytes.values())

    # Protecting one tensor costs 1.167x the all-18 payload, so a bigger
    # codebook applied to every layer brackets that cost from both sides:
    # k=512 is 1.125x and k=1024 is 1.25x. Without these controls "protecting
    # L0 helps" is confounded with "spending more bytes helps".
    arms = {"all_18": None, f"protect_fisher ({pick_fisher})": pick_fisher,
            f"protect_weightmse ({pick_wmse})": pick_wmse,
            f"protect_random ({pick_random})": pick_random,
            "all_18_k512 (4.5 bit)": ("k", 512),
            "all_18_k1024 (5 bit)": ("k", 1024)}
    res = {"plan": {"model": a.model, "codec": a.codec,
                    "eval_targets": 128 * len(blocks),
                    "pick_fisher": pick_fisher, "pick_weightmse": pick_wmse,
                    "pick_random": pick_random},
           "bf16": {k: v for k, v in bf.items() if k != "block_nll"}, "arms": {}}

    for name, skip in arms.items():
        if not B.check(name, 60):
            break
        restore()
        kbig = skip[1] if isinstance(skip, tuple) else None
        drop = skip if isinstance(skip, str) else None
        with torch.no_grad():
            for k in params:
                if k == drop:
                    continue
                w0 = originals[k].cuda().float()
                if kbig:
                    pairing = quantize_vq2d(w0, "adjacent", h.get(k), k=kbig)
                    q = pairing.weight
                else:
                    q, _, _ = codec(k, w0, h.get(k), {})
                params[k].copy_(q.to(params[k].dtype))
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev["block_nll"], bf["block_nll"])
        if kbig:
            bits = 8 if kbig <= 256 else (9 if kbig <= 512 else 10)
            payload = sum(n // 2 * bits // 8 for n in per_tensor.values())
        else:
            payload = total_q + (bf16_bytes[drop] - idx_bytes[drop] if drop else 0)
        rec = {"nll": ev["nll"], "delta_nll": ev["nll"] - bf["nll"], "delta_ci": [lo, hi],
               "payload_bytes": payload, "payload_vs_all18": payload / total_q}
        res["arms"][name] = rec
        print(json.dumps({"arm": name, **{k: v for k, v in rec.items()}}), flush=True)
        atomic_json(out / "results.json", res)

    base = res["arms"].get("all_18", {}).get("delta_nll")
    L = ["# Which single layer is worth leaving in BF16?", "",
         f"Qwen3.5-0.8B, {a.codec} codec, {128 * len(blocks):,} validation targets. "
         f"BF16 NLL {bf['nll']:.6f}.", "",
         "| Arm | ΔNLL vs BF16 | 95% CI | damage recovered | payload vs all-18 |",
         "|---|---:|:--|---:|---:|"]
    for name, r in res["arms"].items():
        rec = "—" if base is None or name == "all_18" else f"{(base - r['delta_nll']) / base:.1%}"
        L.append(f"| {name} | {r['delta_nll']:+.6f} "
                 f"| [{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] | {rec} "
                 f"| {r['payload_vs_all18']:.3f}x |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
