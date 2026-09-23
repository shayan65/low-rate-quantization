"""Does the grouping/codebook cancellation explain the g64 sign flips?

§18 showed that the near-zero g64-against-g128 contrast is a cancellation: at a
fixed alphabet, finer grouping hurts by $+0.014019$ or $+0.006568$; at fixed
grouping, adopting the finer-fitted alphabet helps by $-0.005831$ or
$-0.013281$. A review accepted that for the single fit but noted correctly that
it does not establish what makes the end-to-end contrast *change sign* across
refits. Cancellation explains smallness; it does not by itself explain
instability.

This run repeats the whole 2x2 across six refits and asks a specific question:
**when the diagonal flips sign, do the single-factor contrasts flip with it, or
do they hold steady while their difference wanders?** The second pattern is what
"a small difference between two larger opposed effects" predicts, and it is
falsifiable -- if the grouping contrast itself changes sign across refits, the
cancellation story is wrong and something else is moving.

One detail makes this cheap and sharp. Both codebooks here are fitted from the
*weights* by a deterministic Lloyd-Max procedure, so neither the calibration
draw nor the seed touches them: across refits the only thing that changes is
the Hessian used for GPTQ assignment and compensation. Any instability observed
is therefore attributable to the compensation statistics, not to codebook
refitting -- which also explains why §12's seed axis moved this contrast by
exactly zero.

Six refits: three disjoint calibration draws at 65,536 tokens, plus three
calibration sizes at draw 0, mirroring the §12 protocol minus the seed axis
that provably cannot matter here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from packed_lowrate import decode, encode_scalar3, gnorm_fp16
from run_rate_sweep import BLOCK, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_generalization_v1 import text_of, make_blocks
from run_ternary_task_v4 import (capture_hessians, cholesky_inv_upper, evaluate,
                                 paired_bootstrap, quantize_tensor, rot_signs,
                                 rotate, rotate_hessian, unrotate)

VARIANTS = ("g128_own", "g64_own", "g64_with_g128_book", "g128_with_g64_book")

# the four single-factor contrasts plus the confounded diagonal, as (a, b)
# meaning a - b
CONTRASTS = {
    "grouping_at_fixed_g128_book": ("g64_with_g128_book", "g128_own"),
    "grouping_at_fixed_g64_book": ("g64_own", "g128_with_g64_book"),
    "codebook_at_fixed_g128_grouping": ("g128_with_g64_book", "g128_own"),
    "codebook_at_fixed_g64_grouping": ("g64_own", "g64_with_g128_book"),
    "confounded_diagonal": ("g64_own", "g128_own"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/g64_refits_v1")
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=14400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"),
             add_special_tokens=False).input_ids
    ids = tok(text_of(f"{a.data}/wikitext2/validation.parquet"),
              add_special_tokens=False).input_ids
    blocks = make_blocks(ids, 128)
    n_targets = sum(len(b) - 1 for b in blocks)

    def sl(nblocks, off_blocks):
        o = off_blocks * a.cal_len
        c = [tr[o + i * a.cal_len : o + (i + 1) * a.cal_len] for i in range(nblocks)]
        c = [x for x in c if len(x) == a.cal_len]
        assert len(c) == nblocks, f"{len(c)} of {nblocks} blocks at offset {o}"
        return c

    # three disjoint draws at 65,536 tokens, then three sizes at draw 0
    REFITS = [("draw0_65k", sl(128, 0)), ("draw1_65k", sl(128, 128)),
              ("draw2_65k", sl(128, 256)), ("draw0_16k", sl(32, 0)),
              ("draw0_32k", sl(64, 0)), ("draw0_131k", sl(256, 0))]

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

    def fit_book(group):
        pool = torch.cat([gnorm_fp16(prep[kk], group)[0].reshape(-1, 1)[::17]
                          for kk in params], 0)
        bk = lloyd_scalar(pool.flatten(), 3)
        del pool
        return bk.to(torch.float16).float()

    # Fitted once, from weights only: no calibration draw or seed can move them,
    # which is what isolates the Hessian as the sole source of refit variation.
    BOOKS = {128: fit_book(128), 64: fit_book(64)}
    SPEC = {"g128_own": (128, 128), "g64_own": (64, 64),
            "g64_with_g128_book": (64, 128), "g128_with_g64_book": (128, 64)}

    def hinv_for(cal):
        restore()
        Hraw, _ = capture_hessians(model, layers, cal)
        hv = {f"L{i}": cholesky_inv_upper(rotate_hessian(Hraw[i], signs, BLOCK))
              for i in layers}
        del Hraw
        torch.cuda.empty_cache()
        return hv

    def install(group, book_group, hv):
        restore()
        book = BOOKS[book_group]
        for kk in sorted(params, key=lambda x: key_of[x]):
            w = prep[kk]
            rows, _ = w.shape
            idx, sc, ref = quantize_tensor(w, book, 1, group, hinv=hv[kk])
            p = encode_scalar3(idx, sc, (rows, cols), group)
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"{kk}: decode mismatch"
            params[kk].copy_(unrotate(dq, signs, BLOCK).to(params[kk].dtype))
            del w, idx, sc, ref, dq
        torch.cuda.empty_cache()

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"],
                      "eval_targets": n_targets}), flush=True)

    res = {"plan": {"model": a.model, "layers": layers, "eval_targets": n_targets,
                    "refits": [n for n, _ in REFITS],
                    "note": "both codebooks are deterministic Lloyd fits over "
                            "weights, so only the Hessian varies across refits"},
           "bf16": {"nll": bf["nll"]}, "refits": {}}

    for rname, cal in REFITS:
        if not B.check(rname, 400):
            break
        hv = hinv_for(cal)
        evs, cells = {}, {}
        for v in VARIANTS:
            group, bg = SPEC[v]
            install(group, bg, hv)
            e = evaluate(model, blocks)
            d, lo, hi = paired_bootstrap(e, bf)
            evs[v] = e
            cells[v] = {"delta_nll": d, "delta_ci": [lo, hi]}
        con = {}
        for cname, (x, y) in CONTRASTS.items():
            d, lo, hi = paired_bootstrap(evs[x], evs[y])
            con[cname] = {"delta": d, "ci": [lo, hi],
                          "excludes_zero": bool(lo * hi > 0)}
        g = {v: cells[v]["delta_nll"] for v in VARIANTS}
        con["interaction"] = {
            "delta": (g["g64_with_g128_book"] - g["g128_own"])
                     - (g["g64_own"] - g["g128_with_g64_book"])}
        res["refits"][rname] = {"cal_tokens": len(cal) * a.cal_len,
                                "variants": cells, "contrasts": con}
        print(json.dumps({"refit": rname,
                          **{k: round(v["delta"], 6) for k, v in con.items()}}),
              flush=True)
        del hv
        torch.cuda.empty_cache()
        atomic_json(out / "results.json", res)

    # --- the question: which contrasts are stable in sign, which are not? ---
    summary = {}
    for cname in list(CONTRASTS) + ["interaction"]:
        vals = [r["contrasts"][cname]["delta"] for r in res["refits"].values()
                if cname in r["contrasts"]]
        if not vals:
            continue
        excl = [r["contrasts"][cname].get("excludes_zero")
                for r in res["refits"].values() if cname in r["contrasts"]]
        summary[cname] = {
            "values": vals, "mean": float(np.mean(vals)),
            "range": float(max(vals) - min(vals)),
            "min": float(min(vals)), "max": float(max(vals)),
            "same_sign": bool(all(v > 0 for v in vals) or all(v < 0 for v in vals)),
            "n_excluding_zero": int(sum(1 for x in excl if x)),
            "n_refits": len(vals)}
    res["across_refits"] = summary
    res["verdict"] = {
        "single_factor_contrasts_all_stable": bool(all(
            summary[c]["same_sign"] for c in CONTRASTS if c != "confounded_diagonal"
            and c in summary)),
        "diagonal_stable": summary.get("confounded_diagonal", {}).get("same_sign"),
        "reading": "cancellation explains the instability if the single-factor "
                   "contrasts hold their signs while the diagonal does not",
    }
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Does cancellation explain the g64 sign flips?", "",
         f"The grouping/codebook 2x2 repeated across {len(res['refits'])} refits. "
         "Both codebooks are deterministic Lloyd fits over weights, so no draw "
         "or seed can move them: across refits the **only** thing that changes "
         "is the Hessian used for GPTQ assignment and compensation.", "",
         f"BF16 NLL {bf['nll']:.6f} over {n_targets:,} validation targets.", "",
         "| contrast | mean | min | max | range | same sign | excl. 0 |",
         "|---|---:|---:|---:|---:|:--|:--|"]
    for cname, v in summary.items():
        L.append(f"| {cname.replace('_', ' ')} | {v['mean']:+.6f} | {v['min']:+.6f} | "
                 f"{v['max']:+.6f} | {v['range']:.6f} | "
                 f"{'yes' if v['same_sign'] else '**no**'} | "
                 f"{v['n_excluding_zero']}/{v['n_refits']} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
