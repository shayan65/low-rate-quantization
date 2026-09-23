"""The calibration mixture, with the uncertainty the first run did not have.

§18 measured one calibration draw per condition, reported worst-domain damage
without an interval, and discarded per-block losses so no direct mixed-versus-
pure contrast could be formed afterwards. A review asked for all three. It also
pointed out a confound worth removing: the 50/50 mixture averages two
**32k-token** Hessian estimates, while each pure condition uses a single
**65k-token** estimate. So "mixing helps" and "two half-size estimates of
different things beat one full-size estimate of one thing" were not separated.

Five conditions, all at the same total calibration budget except where the
point is to vary it:

  * `wikitext` and `tinystories` -- 65,536 tokens, one domain (the §18 pure arms);
  * `mixed50` -- 32,768 tokens from each domain;
  * `wikitext_half` and `tinystories_half` -- 32,768 tokens, one domain.

The halves are the control the review asked for. If `mixed50` beats
`wikitext_half` on TinyStories, that is domain coverage rather than sample
count, because both fit on 32,768 wikitext tokens plus or minus what the second
domain adds. If instead `wikitext_half` is already most of the way there, the
effect was never about mixing.

Three **disjoint** calibration draws per condition, so the worst-domain figures
carry a spread across fits as well as a bootstrap interval over text. Per-block
losses are retained for every cell, which is what lets mixed-versus-pure be
contrasted directly rather than by subtracting two independently reported
numbers.

The worst-domain interval resamples each evaluation corpus's blocks
independently within a replicate and takes the maximum of the two resulting
deltas, which is the sampling distribution of the quantity actually reported --
not the larger of two separately computed intervals.

Scope unchanged from §18: the 50/50 split stays prespecified rather than tuned,
and the whole line of work remains exploratory because the crossed cells were
inspected before any of it was designed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from packed_lowrate import decode, encode_scalar3, encode_vq, gnorm_fp16
from run_rate_sweep import BLOCK, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_generalization_v1 import text_of, make_blocks
from run_ternary_task_v4 import (capture_hessians, cholesky_inv_upper, evaluate,
                                 paired_bootstrap, quantize_tensor, rot_signs,
                                 rotate, rotate_hessian, unrotate)

ARMS = (("scalar3", 1, 3, 128), ("vq8", 8, 6561, 128))


def worst_domain_ci(cells, refs, reps=20000, seed=23):
    """Sampling distribution of max over domains of (arm - BF16).

    Each corpus is resampled independently inside a replicate, because the two
    evaluation corpora are different text; taking the larger of two separately
    computed intervals would not be the interval of the maximum.
    """
    rng = np.random.default_rng(seed)
    per = []
    for name in cells:
        s = np.asarray(cells[name]["block_sums"])
        c = np.asarray(cells[name]["block_counts"], dtype=float)
        sb = np.asarray(refs[name]["block_sums"])
        idx = rng.integers(0, len(s), size=(reps, len(s)))
        den = c[idx].sum(1)
        per.append((s[idx].sum(1) - sb[idx].sum(1)) / den)
    M = np.max(np.stack(per), axis=0)
    point = max(float(np.asarray(cells[n]["block_sums"]).sum()
                      - np.asarray(refs[n]["block_sums"]).sum())
                / float(np.asarray(cells[n]["block_counts"], dtype=float).sum())
                for n in cells)
    return point, float(np.percentile(M, 2.5)), float(np.percentile(M, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/calib_mix_v2")
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=262144)
    ap.add_argument("--seconds", type=float, default=21600)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    wt_tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"),
                add_special_tokens=False).input_ids
    wt_te = tok(text_of(f"{a.data}/wikitext2/test.parquet"),
                add_special_tokens=False).input_ids[: a.max_tokens]
    ts_all = tok(text_of(f"{a.data}/tinystories/validation.parquet"),
                 add_special_tokens=False).input_ids
    half = len(ts_all) // 2
    ts_cal_src, ts_eval_src = ts_all[:half], ts_all[half : half + a.max_tokens]

    nb = a.cal_blocks

    def sl(src, n, off_blocks):
        """`n` calibration blocks starting `off_blocks` blocks into `src`."""
        o = off_blocks * a.cal_len
        c = [src[o + i * a.cal_len : o + (i + 1) * a.cal_len] for i in range(n)]
        c = [x for x in c if len(x) == a.cal_len]
        assert len(c) == n, f"only {len(c)} of {n} blocks available at offset {o}"
        return c

    def conditions(draw):
        """Disjoint per draw: draw d starts d*nb blocks into each source."""
        o = draw * nb
        return {
            "wikitext": sl(wt_tr, nb, o),
            "tinystories": sl(ts_cal_src, nb, o),
            "mixed50": sl(wt_tr, nb // 2, o) + sl(ts_cal_src, nb // 2, o),
            "wikitext_half": sl(wt_tr, nb // 2, o),
            "tinystories_half": sl(ts_cal_src, nb // 2, o),
        }

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
        restore()
        for kk in sorted(params, key=lambda x: key_of[x]):
            w = prep[kk]
            rows, _ = w.shape
            idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hv[kk])
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"{kk}: decode mismatch"
            params[kk].copy_(unrotate(dq, signs, BLOCK).to(params[kk].dtype))
            del w, idx, sc, ref, dq
        torch.cuda.empty_cache()

    restore()
    bf = {n: evaluate(model, bl) for n, bl in EVALS.items()}
    books = {arm: fit_book(dim, k, g) for arm, dim, k, g in ARMS}

    cond_names = list(conditions(0))
    res = {"plan": {"model": a.model, "layers": layers, "draws": a.draws,
                    "cal_tokens": {c: len(v) * a.cal_len
                                   for c, v in conditions(0).items()},
                    "eval_targets": {k: sum(len(x) - 1 for x in v)
                                     for k, v in EVALS.items()},
                    "note": "three disjoint calibration draws per condition; the "
                            "half-budget arms are the control separating domain "
                            "coverage from sample count"},
           "bf16": {n: bf[n]["nll"] for n in bf}, "cells": {}}

    store = {}          # (cond, arm, draw) -> {eval: evaluation dict}
    for draw in range(a.draws):
        cond = conditions(draw)
        for cname in cond_names:
            if not B.check(f"{cname}:d{draw}", 400):
                break
            hv = hinv_for(cond[cname])
            for arm, dim, k, group in ARMS:
                install(books[arm], dim, k, group, hv)
                cells = {}
                for ev, bl in EVALS.items():
                    e = evaluate(model, bl)
                    d, lo, hi = paired_bootstrap(e, bf[ev])
                    res["cells"][f"{cname}|{arm}|d{draw}|{ev}"] = {
                        "delta_nll": d, "delta_ci": [lo, hi]}
                    cells[ev] = e
                store[(cname, arm, draw)] = cells
                wp, wlo, whi = worst_domain_ci(cells, bf)
                res["cells"][f"{cname}|{arm}|d{draw}|WORST"] = {
                    "worst_domain": wp, "ci": [wlo, whi]}
                print(json.dumps({"cond": cname, "arm": arm, "draw": draw,
                                  "worst": round(wp, 6),
                                  "ci": [round(wlo, 6), round(whi, 6)]}), flush=True)
            del hv
            torch.cuda.empty_cache()
            atomic_json(out / "results.json", res)

    # --- direct paired contrasts: mixed against each pure condition ---
    contrasts = {}
    for arm, *_ in ARMS:
        for ev in EVALS:
            for other in ("wikitext", "tinystories", "wikitext_half",
                          "tinystories_half"):
                vals = []
                for draw in range(a.draws):
                    m = store.get(("mixed50", arm, draw))
                    o = store.get((other, arm, draw))
                    if m is None or o is None:
                        continue
                    d, lo, hi = paired_bootstrap(m[ev], o[ev])
                    vals.append({"draw": draw, "delta": d, "ci": [lo, hi]})
                if vals:
                    pts = [v["delta"] for v in vals]
                    contrasts[f"mixed50_vs_{other}|{arm}|{ev}"] = {
                        "per_draw": vals, "mean": float(np.mean(pts)),
                        "range": float(max(pts) - min(pts)),
                        "all_same_sign": bool(all(p > 0 for p in pts)
                                              or all(p < 0 for p in pts))}
    res["contrasts"] = contrasts

    # --- spread across draws of the reported quantity ---
    spread = {}
    for cname in cond_names:
        for arm, *_ in ARMS:
            w = [res["cells"][f"{cname}|{arm}|d{d}|WORST"]["worst_domain"]
                 for d in range(a.draws)
                 if f"{cname}|{arm}|d{d}|WORST" in res["cells"]]
            if w:
                spread[f"{cname}|{arm}"] = {
                    "worst_by_draw": w, "mean": float(np.mean(w)),
                    "range": float(max(w) - min(w))}
    res["worst_domain_by_draw"] = spread
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    np.savez_compressed(
        out / "per_block.npz",
        **{f"{c}|{a_}|d{d}|{e}|{k}": np.asarray(v[e][k], dtype=np.float64)
           for (c, a_, d), v in store.items() for e in v
           for k in ("block_sums", "block_counts")})

    L = ["# Calibration mixture, with draws and a sample-count control", "",
         f"{a.draws} disjoint calibration draws per condition. `mixed50` spends "
         "half its budget on each domain; the `_half` conditions spend the same "
         "reduced budget on a single domain, which separates domain coverage "
         "from sample count.", "",
         "| condition | code | worst-domain by draw | mean | range |",
         "|---|---|---|---:|---:|"]
    for key, v in spread.items():
        c, arm = key.split("|")
        L.append(f"| {c} | {arm} | " +
                 ", ".join(f"{x:.6f}" for x in v["worst_by_draw"]) +
                 f" | {v['mean']:.6f} | {v['range']:.6f} |")
    L += ["", "## Mixed against each pure condition, paired per draw", "",
          "| contrast | eval | mean Δ | range over draws | same sign |",
          "|---|---|---:|---:|:--|"]
    for key, v in contrasts.items():
        nm, arm, ev = key.split("|")
        L.append(f"| {nm} ({arm}) | {ev} | {v['mean']:+.6f} | {v['range']:.6f} | "
                 f"{'yes' if v['all_same_sign'] else '**no**'} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
