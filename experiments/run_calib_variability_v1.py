"""How much do the reported contrasts move when the quantizer is refitted?

Every interval in this project is a paired block bootstrap over evaluation
text, conditional on one fitted quantizer. The 27B confirmation showed that
conditioning is not harmless: doubling calibration tokens moved the
storage-dominance contrast from -0.001736 [-0.004444, +0.000923] to
-0.007681 [-0.010467, -0.004984] on *identical* evaluation blocks, with BF16
loss matching to all printed digits. The quantizer moved, not the data.

That was a single pair of nested calibration sets on one checkpoint, which
cannot separate a size effect from draw-to-draw variability. This run separates
them on the cheap model, over the same 18 DeltaNet QKV projections as §5c, by
varying three things independently:

  * **calibration draw** -- three disjoint slices of the training stream at
    fixed size, which isolates sampling of the calibration text;
  * **codebook seed** -- three k-means initializations at fixed calibration,
    which isolates the fit. (Scalar codebooks are fitted by a deterministic
    Lloyd pass, so seed cannot move the scalar arms; that is a property of the
    method, and it is reported rather than hidden.)
  * **calibration size** -- 16k to 131k tokens on one draw, nested by
    construction, which is the factor the 27B comparison actually varied.

The quantity of interest is not any single configuration but the **spread of a
contrast across refits compared with the bootstrap half-width within one
refit**. If the spread is comparable to or larger than the half-width, the
published intervals understate the uncertainty a reader should attach to these
contrasts, and the paper has to say so.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import (bits_per_weight, decode, encode_scalar3, encode_vq,
                            gnorm_fp16)
from run_rate_sweep import BLOCK, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_ternary_task_v4 import (capture_hessians, cholesky_inv_upper, evaluate,
                                 paired_bootstrap, quantize_tensor, rot_signs,
                                 rotate, rotate_hessian, unrotate)

ARMS = [
    ("scalar3", 1, 3, 128),
    ("scalar3_g64", 1, 3, 64),
    ("vq4", 4, 81, 128),
    ("vq8", 8, 6561, 128),
]

# name, calibration draw index, codebook seed, calibration blocks (x512 tokens)
CONFIGS = [
    ("draw0_seed0_65k", 0, 0, 128),   # reference cell, shared by all three factors
    ("draw1_seed0_65k", 1, 0, 128),
    ("draw2_seed0_65k", 2, 0, 128),
    ("draw0_seed1_65k", 0, 1, 128),
    ("draw0_seed2_65k", 0, 2, 128),
    ("draw0_seed0_16k", 0, 0, 32),
    ("draw0_seed0_32k", 0, 0, 64),
    ("draw0_seed0_131k", 0, 0, 256),
]
CONTRASTS = [("vq8", "scalar3"), ("vq8", "scalar3_g64"), ("vq8", "vq4"),
             ("scalar3_g64", "scalar3")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/calib_variability_v1")
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=14400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok("\n".join(pd.read_parquet(f"{a.data}/validation.parquet").text.tolist()),
              add_special_tokens=False).input_ids
    full = (len(ids) - 1) // 128
    blocks = [ids[i * 128 : i * 128 + 129] for i in range(full)]
    tail = ids[full * 128 :]
    if len(tail) >= 2:
        blocks.append(tail)
    n_targets = sum(len(b) - 1 for b in blocks)

    tr = tok("\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()[:12000]),
             add_special_tokens=False).input_ids

    def cal_set(draw, nblocks):
        """Disjoint slices per draw; sizes within a draw are nested by design."""
        stride = 256 * a.cal_len          # room for the largest configuration
        base = draw * stride
        c = [tr[base + i * a.cal_len : base + (i + 1) * a.cal_len] for i in range(nblocks)]
        return [x for x in c if len(x) == a.cal_len]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    key_of = {f"L{i}": i for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}
    cols = originals[f"L{layers[0]}"].shape[1]
    assert cols == BLOCK
    signs = rot_signs(cols, "cuda")

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"targets": n_targets, "tensors": len(layers),
                      "bf16": bf["nll"], "configs": len(CONFIGS)}), flush=True)

    res = {"plan": {"model": a.model, "layers": layers, "eval_targets": n_targets,
                    "configs": [list(c) for c in CONFIGS],
                    "protocol": "same recipe throughout; only calibration draw, "
                                "codebook seed and calibration size vary"},
           "bf16": {"nll": bf["nll"]}, "cells": {}}

    prep = {k: rotate(originals[k].cuda().float(), signs, BLOCK) for k in params}

    for cname, draw, seed, nblk in CONFIGS:
        if not B.check(cname, 300):
            break
        cal = cal_set(draw, nblk)
        t0 = time.monotonic()
        restore()
        Hraw, _ = capture_hessians(model, layers, cal)
        Hrot = {f"L{i}": rotate_hessian(Hraw[i], signs, BLOCK) for i in layers}
        del Hraw
        hinv = {k: cholesky_inv_upper(Hrot[k]) for k in params}
        torch.cuda.empty_cache()

        cell = {"draw": draw, "seed": seed, "cal_tokens": len(cal) * a.cal_len, "arms": {}}
        evs = {}
        for aname, dim, k, group in ARMS:
            restore()
            pool = torch.cat([gnorm_fp16(prep[kk], group)[0].reshape(-1, dim)[::17]
                              for kk in params], 0)
            book = (lloyd_scalar(pool.flatten(), 3) if dim == 1
                    else kmeans(pool, k, seed=seed))
            del pool
            book = book.to(torch.float16).float()
            packs = []
            for kk in sorted(params, key=lambda x: key_of[x]):
                w = prep[kk]
                rows, _ = w.shape
                idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hinv[kk])
                p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                     else encode_vq(idx, sc, (rows, cols), group, dim, k))
                dq = decode(p, book, "cuda")
                assert torch.equal(dq, ref), f"{cname}/{aname}/{kk}: decode mismatch"
                params[kk].copy_(unrotate(dq, signs, BLOCK).to(params[kk].dtype))
                packs.append(p)
                del w, idx, sc, ref, dq
            torch.cuda.empty_cache()
            ev = evaluate(model, blocks)
            d, lo, hi = paired_bootstrap(ev, bf)
            cell["arms"][aname] = {
                "delta_nll": d, "delta_ci": [lo, hi],
                "bits_per_weight": bits_per_weight(packs, book.numel() * 2),
                "stored_bytes": sum(p.payload_bytes() for p in packs) + book.numel() * 2}
            evs[aname] = ev
            del packs

        cell["contrasts"] = {}
        for m, ref_ in CONTRASTS:
            d, lo, hi = paired_bootstrap(evs[m], evs[ref_])
            cell["contrasts"][f"{m}_vs_{ref_}"] = {
                "delta": d, "ci": [lo, hi], "half_width": (hi - lo) / 2}
        cell["seconds"] = time.monotonic() - t0
        res["cells"][cname] = cell
        print(json.dumps({"cell": cname, "cal_tokens": cell["cal_tokens"],
                          **{k2: round(v2["delta"], 6)
                             for k2, v2 in cell["contrasts"].items()}}), flush=True)
        atomic_json(out / "results.json", res)
        del hinv, Hrot, evs
        torch.cuda.empty_cache()

    # ---- the comparison the run exists for ----
    def spread(names, contrast):
        vals = [res["cells"][n]["contrasts"][contrast]["delta"]
                for n in names if n in res["cells"]]
        return (max(vals) - min(vals)) if len(vals) > 1 else float("nan"), vals

    groups = {
        "calibration draw (3 disjoint, 65k)":
            ["draw0_seed0_65k", "draw1_seed0_65k", "draw2_seed0_65k"],
        "codebook seed (3, fixed calibration)":
            ["draw0_seed0_65k", "draw0_seed1_65k", "draw0_seed2_65k"],
        "calibration size (16k-131k, nested)":
            ["draw0_seed0_16k", "draw0_seed0_32k", "draw0_seed0_65k", "draw0_seed0_131k"],
    }
    summary = {}
    L = ["# Refit variability: do the reported contrasts survive refitting?", "",
         f"Qwen3.5-0.8B, {len(layers)} DeltaNet QKV projections, {n_targets:,} "
         f"validation targets, BF16 NLL {bf['nll']:.6f}.",
         "Recipe fixed throughout; only the calibration draw, codebook seed and",
         "calibration size vary. `half-width` is the median paired-bootstrap",
         "half-width within a single refit, which is what the paper's intervals",
         "report.", ""]
    for contrast in [f"{m}_vs_{r}" for m, r in CONTRASTS]:
        hw = float(np.median([c["contrasts"][contrast]["half_width"]
                              for c in res["cells"].values()]))
        L += [f"## {contrast.replace('_vs_', ' vs ')}", "",
              f"Median within-refit bootstrap half-width: **{hw:.6f}**", "",
              "| factor varied | range of the contrast | spread / half-width | values |",
              "|---|---:|---:|:--|"]
        summary[contrast] = {"median_half_width": hw, "factors": {}}
        for gname, names in groups.items():
            sp, vals = spread(names, contrast)
            summary[contrast]["factors"][gname] = {"spread": sp, "values": vals}
            L.append(f"| {gname} | {sp:.6f} | {sp / hw:.2f}x | "
                     + ", ".join(f"{v:+.6f}" for v in vals) + " |")
        L.append("")
    res["summary"] = summary
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
