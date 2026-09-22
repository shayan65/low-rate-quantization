"""Three mechanism tests the review named, none of which needs a large run.

Each replaces an observation this project currently reports without a cause.

**A. Calibration domain x evaluation domain.** §13 found higher damage on
TinyStories than on wikitext and did not attribute it. Two explanations are
confounded there: the quantizer was calibrated on wikitext (mismatch), and
TinyStories may simply be more sensitive to these perturbations (intrinsic).
Crossing two calibration domains with two evaluation domains separates them.
If mismatch dominates, damage should be lowest on the diagonal; if sensitivity
dominates, the TinyStories column is high whatever it was calibrated on.

**B. Position-resolved loss.** §13 reported no large increase in aggregate
damage through 2048 tokens and correctly refused to call that evidence against
recurrent-state drift, because aggregate loss averages over positions. Damage
resolved by position within the block does address it: drift in a quantized
DeltaNet's recurrent state should show up as damage growing with distance from
the block start, which an average cannot reveal.

**C. Does the g64 instability come from refitting the codebook?** The leading
hypothesis is that g64 is not a superset of g128: scales are derived from group
maxima rather than chosen, *and* the shared codebook is refit in the new
normalized space. Those two causes are separable. Applying the codebook fitted
for g128 to g64 quantization holds the alphabet fixed and leaves only the scale
change; the reverse control does the same in the other direction. If the fixed
codebook variants behave, refitting is the culprit; if they do not, the scale
derivation and its interaction with GPTQ is.
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

from packed_lowrate import decode, encode_scalar3, encode_vq, gnorm_fp16
from run_rate_sweep import BLOCK, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_generalization_v1 import text_of, make_blocks
from run_ternary_task_v4 import (capture_hessians, cholesky_inv_upper, evaluate,
                                 paired_bootstrap, quantize_tensor, rot_signs,
                                 rotate, rotate_hessian, unrotate)


@torch.no_grad()
def evaluate_positions(model, blocks, batch=1, device="cuda"):
    """Mean loss at each position inside the block, plus the usual aggregate."""
    ctx = len(blocks[0]) - 1
    tot = torch.zeros(ctx, dtype=torch.float64)
    n = 0
    sums, counts = [], []
    for i in range(0, len(blocks), batch):
        grp = [b for b in blocks[i : i + batch] if len(b) == ctx + 1]
        if not grp:
            continue
        x = torch.tensor(grp, device=device)
        logits = model(x[:, :-1], use_cache=False).logits.float()
        y = x[:, 1:]
        tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").reshape(x.shape[0], -1)
        tot += tok.sum(0).double().cpu()
        n += x.shape[0]
        sums.extend(tok.sum(1).cpu().tolist())
        counts.extend([y.shape[1]] * x.shape[0])
        del x, y, logits, tok
    s, c = np.asarray(sums), np.asarray(counts, dtype=float)
    return {"per_position": (tot / max(n, 1)).tolist(), "blocks": n,
            "nll": float(s.sum() / c.sum()), "block_sums": sums,
            "block_counts": counts, "target_tokens": int(c.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/mechanism_v1")
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=262144)
    ap.add_argument("--seconds", type=float, default=10800)
    ap.add_argument("--parts", default="ABC")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    tok = AutoTokenizer.from_pretrained(a.model)
    wt_tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"), add_special_tokens=False).input_ids
    wt_te = tok(text_of(f"{a.data}/wikitext2/test.parquet"),
                add_special_tokens=False).input_ids[: a.max_tokens]
    ts_all = tok(text_of(f"{a.data}/tinystories/validation.parquet"),
                 add_special_tokens=False).input_ids
    half = len(ts_all) // 2
    ts_cal_src, ts_eval_src = ts_all[:half], ts_all[half : half + a.max_tokens]

    def slice_cal(src):
        c = [src[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
        return [x for x in c if len(x) == a.cal_len]

    CALS = {"wikitext": slice_cal(wt_tr), "tinystories": slice_cal(ts_cal_src)}
    EVALS = {"wt2_test": make_blocks(wt_te, 128),
             "tinystories": make_blocks(ts_eval_src, 128)}

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    key_of = {f"L{i}": i for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}
    cols = originals[f"L{layers[0]}"].shape[1]
    signs = rot_signs(cols, "cuda")

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    prep = {k: rotate(originals[k].cuda().float(), signs, BLOCK) for k in params}

    def hinv_for(cal):
        restore()
        Hraw, _ = capture_hessians(model, layers, cal)
        hv = {f"L{i}": cholesky_inv_upper(rotate_hessian(Hraw[i], signs, BLOCK))
              for i in layers}
        del Hraw
        torch.cuda.empty_cache()
        return hv

    def fit_book(dim, k, group):
        pool = torch.cat([gnorm_fp16(prep[kk], group)[0].reshape(-1, dim)[::17]
                          for kk in params], 0)
        bk = lloyd_scalar(pool.flatten(), 3) if dim == 1 else kmeans(pool, k)
        del pool
        return bk.to(torch.float16).float()

    def install(book, dim, k, group, hv):
        """Quantize every target with this codebook; returns weight MSE."""
        restore()
        wse, wn = 0.0, 0
        for kk in sorted(params, key=lambda x: key_of[x]):
            w = prep[kk]
            rows, _ = w.shape
            idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hv[kk])
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"{kk}: decode mismatch"
            wse += (dq - w).square().double().sum().item()
            wn += rows * cols
            params[kk].copy_(unrotate(dq, signs, BLOCK).to(params[kk].dtype))
            del w, idx, sc, ref, dq
        torch.cuda.empty_cache()
        return wse / wn

    res = {"plan": {"model": a.model, "layers": layers,
                    "cal_tokens": a.cal_blocks * a.cal_len,
                    "eval_targets": {k: sum(len(x) - 1 for x in v) for k, v in EVALS.items()}}}

    # ---------------- A: calibration domain x evaluation domain ----------------
    if "A" in a.parts and B.check("partA", 600):
        restore()
        bf = {n: evaluate(model, bl) for n, bl in EVALS.items()}
        cell = {}
        for cal_name, cal in CALS.items():
            hv = hinv_for(cal)
            for arm, dim, k, group in (("scalar3", 1, 3, 128), ("vq8", 8, 6561, 128)):
                book = fit_book(dim, k, group)
                install(book, dim, k, group, hv)
                for ev_name, bl in EVALS.items():
                    e = evaluate(model, bl)
                    d, lo, hi = paired_bootstrap(e, bf[ev_name])
                    cell[f"{cal_name}|{arm}|{ev_name}"] = {
                        "delta_nll": d, "delta_ci": [lo, hi]}
                    print(json.dumps({"A": f"cal={cal_name} arm={arm} eval={ev_name}",
                                      "delta": round(d, 6)}), flush=True)
            del hv
            torch.cuda.empty_cache()
        res["A_domain_cross"] = {"bf16": {n: bf[n]["nll"] for n in bf}, "cells": cell}
        atomic_json(out / "results.json", res)

    # ---------------- B: position-resolved loss at 2048 ----------------
    if "B" in a.parts and B.check("partB", 600):
        long_blocks = make_blocks(wt_te, 2048)
        hv = hinv_for(CALS["wikitext"])
        restore()
        bfp = evaluate_positions(model, long_blocks, batch=1)
        rows = {"bf16": bfp["per_position"]}
        for arm, dim, k, group in (("scalar3", 1, 3, 128), ("vq8", 8, 6561, 128)):
            book = fit_book(dim, k, group)
            install(book, dim, k, group, hv)
            rows[arm] = evaluate_positions(model, long_blocks, batch=1)["per_position"]
            print(json.dumps({"B": arm, "blocks": bfp["blocks"]}), flush=True)
        res["B_position"] = {"context": 2048, "blocks": bfp["blocks"], "per_position": rows}
        atomic_json(out / "results.json", res)
        del hv
        torch.cuda.empty_cache()

    # ---------------- C: is the g64 instability the codebook refit? ----------------
    if "C" in a.parts and B.check("partC", 600):
        hv = hinv_for(CALS["wikitext"])
        restore()
        bf_t = evaluate(model, EVALS["wt2_test"])
        book128 = fit_book(1, 3, 128)
        book64 = fit_book(1, 3, 64)
        variants = {
            "g128_own_book": (book128, 128),
            "g64_own_book": (book64, 64),
            "g64_with_g128_book": (book128, 64),   # scale change only
            "g128_with_g64_book": (book64, 128),   # alphabet change only
        }
        cell = {}
        evs = {}
        for name, (bk, grp) in variants.items():
            mse = install(bk, 1, 3, grp, hv)
            e = evaluate(model, EVALS["wt2_test"])
            d, lo, hi = paired_bootstrap(e, bf_t)
            cell[name] = {"weight_mse": mse, "delta_nll": d, "delta_ci": [lo, hi],
                          "group": grp, "levels": bk.flatten().tolist()}
            evs[name] = e
            print(json.dumps({"C": name, "weight_mse": mse, "delta": round(d, 6)}), flush=True)
        pairs = {}
        for m, r in (("g64_own_book", "g128_own_book"),
                     ("g64_with_g128_book", "g128_own_book"),
                     ("g64_with_g128_book", "g64_own_book"),
                     ("g128_with_g64_book", "g128_own_book")):
            d, lo, hi = paired_bootstrap(evs[m], evs[r])
            pairs[f"{m}_vs_{r}"] = {"delta": d, "ci": [lo, hi]}
        res["C_g64_mechanism"] = {"bf16": bf_t["nll"], "variants": cell, "contrasts": pairs}
        atomic_json(out / "results.json", res)

    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)
    print(json.dumps({"done": True, "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
