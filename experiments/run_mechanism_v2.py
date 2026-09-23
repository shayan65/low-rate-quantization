"""Three corrections to §14, and one new intervention.

§14 answered three mechanism questions. A later review showed that two of its
answers were stated more strongly than the design supports, and both defects
are in what was *retained* rather than in what was run. This re-run fixes the
retention and adds the experiment the identity below actually motivates.

**G. The g64 2x2 was read along the wrong diagonal.** §14C called
`g64 own book` against `g128 own book` "group size alone". It is not: each arm
fits its codebook in its own normalized space, so that contrast moves grouping
*and* codebook together, and its near-zero value is a cancellation rather than
an absence. Group size at a *fixed* codebook is +0.014019 with the g128 book
and +0.006568 with the g64 book -- both harmful, neither negligible, and the
second had no interval because it was never computed. This run reports all six
pairwise contrasts of the 2x2 with paired intervals, so the factorial can be
read as a factorial.

**P. Position-resolved damage had no intervals.** §14B reported per-position
means over 127 blocks and called the rise "recurrent-state drift". Means over
blocks cannot separate a position effect from the fact that different tokens
sit at different positions, and no block-level spread was kept. Here every
block's per-position losses are retained, so bucket contrasts get paired
bootstrap intervals over blocks and the finding can be stated as
position-dependent damage with a measured interval rather than as a mechanism.

**M. Balanced calibration, the intervention the mismatch identity suggests.**
For a fixed input distribution the layer reconstruction objective is
quadratic in the error E = What - W:

    L_D(E) = E_D || E x ||^2 = tr( E H_D E^T ),    H_D = E_D[ x x^T ],

so the penalty for evaluating a quantizer fitted on one domain against another
is exactly

    L_test(E) - L_cal(E) = tr[ E (H_test - H_cal) E^T ],

bounded by || H_test - H_cal ||_2 || E ||_F^2. This is an identity plus an
elementary bound, not a theorem and not a statement about NLL: it says only
that an error pattern which is cheap under one domain's second moments can be
expensive under another's, and that plain weight MSE, which ignores that
orientation entirely, cannot see the difference. Hessian-weighted reconstruction
objectives rest on the same quadratic (GPTQ, GPTVQ).

The intervention it suggests is a mixture. For the quadratic objective,

    H_mix = (1/2) H_WT + (1/2) H_TS

represents exactly the equally weighted average of the two domains'
reconstruction losses. Whether that *reduces worst-domain NLL damage* does not
follow -- the objective is a local proxy, and the mixture halves the data seen
from each domain -- so it is the experimental question. The split is
prespecified at 50/50 rather than tuned, the total calibration budget is held
at the same 65,536 tokens as every other condition here, and the calibration
material of each domain is disjoint from the text either is evaluated on.

This is an **exploratory intervention**: the crossed-domain results were
already inspected before it was designed, and it is reported as such.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from packed_lowrate import decode, encode_scalar3, encode_vq, gnorm_fp16
from run_rate_sweep import BLOCK, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_generalization_v1 import text_of, make_blocks
from run_ternary_task_v4 import (capture_hessians, cholesky_inv_upper, evaluate,
                                 paired_bootstrap, quantize_tensor, rot_signs,
                                 rotate, rotate_hessian, unrotate)

ARMS = (("scalar3", 1, 3, 128), ("vq8", 8, 6561, 128))


@torch.no_grad()
def evaluate_positions(model, blocks, device="cuda"):
    """Per-block, per-position token losses -- retained, not averaged away.

    §14B kept only the mean over blocks, which left its position trend without
    any measure of block-to-block spread. The full matrix is 127 x 2048 floats
    per arm; there was never a reason to throw it away.
    """
    ctx = len(blocks[0]) - 1
    mat = []
    for b in blocks:
        if len(b) != ctx + 1:
            continue
        x = torch.tensor([b], device=device)
        logits = model(x[:, :-1], use_cache=False).logits.float()
        tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              x[:, 1:].reshape(-1), reduction="none")
        mat.append(tok.double().cpu().numpy())
        del x, logits, tok
    M = np.stack(mat)                                   # (blocks, ctx)
    return {"per_block_position": M.tolist(), "blocks": M.shape[0],
            "per_position": M.mean(0).tolist(),
            "nll": float(M.sum() / M.size), "target_tokens": int(M.size)}


def bucket_contrast(A, B_, nb=8, reps=5000, seed=11):
    """Paired bootstrap over blocks of (arm - reference) within each bucket."""
    A, B_ = np.asarray(A), np.asarray(B_)
    nblk, ctx = A.shape
    edges = np.linspace(0, ctx, nb + 1).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nblk, size=(reps, nblk))
    out = []
    for i in range(nb):
        d = (A[:, edges[i]:edges[i + 1]] - B_[:, edges[i]:edges[i + 1]]).mean(1)
        boot = d[idx].mean(1)
        out.append({"positions": [int(edges[i]), int(edges[i + 1])],
                    "delta": float(d.mean()),
                    "ci": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/mechanism_v2")
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=262144)
    ap.add_argument("--long-blocks", type=int, default=127)
    ap.add_argument("--seconds", type=float, default=21600)
    ap.add_argument("--parts", default="MGP")
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

    def slice_cal(src, n, off=0):
        c = [src[off + i * a.cal_len : off + (i + 1) * a.cal_len] for i in range(n)]
        return [x for x in c if len(x) == a.cal_len]

    nb = a.cal_blocks
    # Every condition spends the same token budget; the mixture splits it 50/50
    # across domains rather than adding to it.
    CALS = {"wikitext": slice_cal(wt_tr, nb),
            "tinystories": slice_cal(ts_cal_src, nb),
            "mixed50": slice_cal(wt_tr, nb // 2) + slice_cal(ts_cal_src, nb // 2)}
    for name, c in CALS.items():
        assert len(c) == nb, f"{name}: {len(c)} blocks, expected {nb}"
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
                    "cal_tokens": nb * a.cal_len,
                    "cal_conditions": {k: len(v) * a.cal_len for k, v in CALS.items()},
                    "eval_targets": {k: sum(len(x) - 1 for x in v)
                                     for k, v in EVALS.items()},
                    "note": "mixed50 is prespecified at 50/50 and spends the same "
                            "token budget as the pure conditions; it is exploratory "
                            "because the crossed-domain results were seen first"}}

    # ---------------- M: calibration mixture against the pure conditions ------
    if "M" in a.parts and B.check("partM", 900):
        restore()
        bf = {n: evaluate(model, bl) for n, bl in EVALS.items()}
        cell = {}
        for cal_name, cal in CALS.items():
            if not B.check(f"M:{cal_name}", 600):
                break
            hv = hinv_for(cal)
            for arm, dim, k, group in ARMS:
                book = fit_book(dim, k, group)
                mse = install(book, dim, k, group, hv)
                for ev_name, bl in EVALS.items():
                    e = evaluate(model, bl)
                    d, lo, hi = paired_bootstrap(e, bf[ev_name])
                    cell[f"{cal_name}|{arm}|{ev_name}"] = {
                        "delta_nll": d, "delta_ci": [lo, hi], "weight_mse": mse}
                    print(json.dumps({"M": f"cal={cal_name} arm={arm} eval={ev_name}",
                                      "delta": round(d, 6)}), flush=True)
            del hv
            torch.cuda.empty_cache()
            atomic_json(out / "results.json", {**res, "M_calibration": {
                "bf16": {n: bf[n]["nll"] for n in bf}, "cells": cell}})
        worst = {}
        for cal_name in CALS:
            for arm, *_ in ARMS:
                vals = [cell[f"{cal_name}|{arm}|{e}"]["delta_nll"] for e in EVALS
                        if f"{cal_name}|{arm}|{e}" in cell]
                if vals:
                    worst[f"{cal_name}|{arm}"] = {"worst_domain": max(vals),
                                                  "mean_domain": sum(vals) / len(vals)}
        res["M_calibration"] = {"bf16": {n: bf[n]["nll"] for n in bf},
                                "cells": cell, "worst_domain": worst}
        atomic_json(out / "results.json", res)

    # ---------------- G: the g64 2x2, all six contrasts with intervals -------
    if "G" in a.parts and B.check("partG", 900):
        hv128 = hinv_for(CALS["wikitext"])
        restore()
        bf1 = evaluate(model, EVALS["wt2_test"])
        b128 = fit_book(1, 3, 128)
        b64 = fit_book(1, 3, 64)
        # g64 needs its own compensation statistics only through the group
        # size, so the Hessian is shared; only grouping and codebook vary.
        cells, evs = {}, {}
        for name, group, book in (("g128_own", 128, b128), ("g64_own", 64, b64),
                                  ("g64_with_g128_book", 64, b128),
                                  ("g128_with_g64_book", 128, b64)):
            mse = install(book, 1, 3, group, hv128)
            e = evaluate(model, EVALS["wt2_test"])
            d, lo, hi = paired_bootstrap(e, bf1)
            cells[name] = {"weight_mse": mse, "delta_nll": d, "delta_ci": [lo, hi],
                           "group": group}
            evs[name] = e
            print(json.dumps({"G": name, "delta": round(d, 6),
                              "wmse": mse}), flush=True)
        names = list(cells)
        contrasts = {}
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                d, lo, hi = paired_bootstrap(evs[names[i]], evs[names[j]])
                contrasts[f"{names[i]}_vs_{names[j]}"] = {"delta": d, "ci": [lo, hi]}
        # the factorial decomposition the diagonal contrast cannot give
        g = {n: cells[n]["delta_nll"] for n in names}
        factorial = {
            "grouping_at_fixed_g128_book":
                contrasts["g64_with_g128_book_vs_g128_own"],
            "grouping_at_fixed_g64_book":
                contrasts["g64_own_vs_g128_with_g64_book"],
            "codebook_at_fixed_g128_grouping":
                contrasts["g128_with_g64_book_vs_g128_own"],
            "codebook_at_fixed_g64_grouping":
                contrasts["g64_own_vs_g64_with_g128_book"],
            "confounded_diagonal": contrasts["g64_own_vs_g128_own"],
            "main_effect_grouping":
                (g["g64_with_g128_book"] + g["g64_own"]) / 2
                - (g["g128_own"] + g["g128_with_g64_book"]) / 2,
            "main_effect_codebook_g64book_minus_g128book":
                (g["g64_own"] + g["g128_with_g64_book"]) / 2
                - (g["g64_with_g128_book"] + g["g128_own"]) / 2,
            "interaction":
                (g["g64_with_g128_book"] - g["g128_own"])
                - (g["g64_own"] - g["g128_with_g64_book"]),
        }
        res["G_g64_factorial"] = {"bf16": bf1["nll"], "variants": cells,
                                  "contrasts": contrasts, "factorial": factorial}
        atomic_json(out / "results.json", res)
        print(json.dumps({"G_factorial": {k: v for k, v in factorial.items()
                                          if isinstance(v, float)}}), flush=True)
        del hv128
        torch.cuda.empty_cache()

    # ---------------- P: position-resolved damage, with block-level spread ---
    if "P" in a.parts and B.check("partP", 900):
        long_blocks = make_blocks(wt_te, 2048)[: a.long_blocks]
        hv = hinv_for(CALS["wikitext"])
        restore()
        base = evaluate_positions(model, long_blocks)
        mats = {"bf16": base["per_block_position"]}
        for arm, dim, k, group in ARMS:
            book = fit_book(dim, k, group)
            install(book, dim, k, group, hv)
            mats[arm] = evaluate_positions(model, long_blocks)["per_block_position"]
            print(json.dumps({"P": arm, "blocks": base["blocks"]}), flush=True)
        buckets = {
            "scalar3_damage": bucket_contrast(mats["scalar3"], mats["bf16"]),
            "vq8_damage": bucket_contrast(mats["vq8"], mats["bf16"]),
            "vq8_advantage": bucket_contrast(mats["vq8"], mats["scalar3"]),
        }
        res["P_position"] = {"context": 2048, "blocks": base["blocks"],
                             "buckets": buckets,
                             "per_position_mean": {k: np.asarray(v).mean(0).tolist()
                                                   for k, v in mats.items()},
                             "note": "per-block matrices retained in "
                                     "per_block_position.npz"}
        np.savez_compressed(out / "per_block_position.npz",
                            **{k: np.asarray(v, dtype=np.float32) for k, v in mats.items()})
        atomic_json(out / "results.json", res)
        for nm, bl in buckets.items():
            print(json.dumps({"P_bucket": nm,
                              "first": round(bl[0]["delta"], 6),
                              "last": round(bl[-1]["delta"], 6)}), flush=True)
        del hv
        torch.cuda.empty_cache()

    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)
    print(json.dumps({"done": True, "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
