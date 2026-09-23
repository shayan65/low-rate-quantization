"""Testing the explanation §20 put in place of the one it falsified.

§20 killed the cancellation account of the unstable g64 contrast and replaced it
with an interpretation: since both codebooks there are deterministic fits over
weights, the Hessian is the only thing that varies across refits, so **GPTQ's
compensation statistics must themselves be noisy enough to move every contrast
in that family**. That was an interpretation of six numbers, not a measurement,
and it is the kind of story this project has repeatedly had to withdraw. So it
gets measured.

The interpretation makes three predictions that a finite-sample argument would
make, and all three are falsifiable:

  1. the Hessian's deviation from a large-sample reference should fall roughly
     as $N^{-1/2}$ in calibration tokens;
  2. the spread of the g64 contrast across disjoint draws should fall with $N$
     as well --- if it does not, the contrast's instability is not driven by
     Hessian sampling error;
  3. within a size, draws whose Hessian lands further from the reference should
     produce contrasts further from the reference contrast. This is the sharpest
     of the three, because a per-draw correlation cannot be produced by any
     confound that only varies with $N$.

Four calibration sizes (16k to 131k tokens), four disjoint draws at each,
against a reference Hessian estimated from 524k tokens of separate text. The
deviation is reported both for the Hessian itself and for
`cholesky_inv_upper(H)`, which is the operator GPTQ actually applies --- those
can differ, since inversion amplifies error in the small eigenvalues.

If prediction 3 fails the interpretation is wrong and §20's replacement story
goes the way of the one it replaced.
"""

import argparse
import json
import time
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

SIZES = [("16k", 32), ("32k", 64), ("65k", 128), ("131k", 256)]
DRAWS = 4


def rel_dev(A, B):
    """Relative Frobenius deviation of A from B."""
    return float((A - B).norm() / B.norm())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/hessian_noise_v1")
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--ref-blocks", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seconds", type=float, default=21600)
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

    # draws occupy blocks [0, 4*256); the reference sits after all of them
    ref_off = DRAWS * SIZES[-1][1]
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

    def hessians(cal):
        restore()
        Hraw, _ = capture_hessians(model, layers, cal)
        H = {f"L{i}": rotate_hessian(Hraw[i], signs, BLOCK) for i in layers}
        del Hraw
        torch.cuda.empty_cache()
        return H

    def fit_book(group):
        pool = torch.cat([gnorm_fp16(prep[kk], group)[0].reshape(-1, 1)[::17]
                          for kk in params], 0)
        bk = lloyd_scalar(pool.flatten(), 3)
        del pool
        return bk.to(torch.float16).float()

    BOOKS = {128: fit_book(128), 64: fit_book(64)}

    def install(group, H):
        restore()
        book = BOOKS[group]
        for kk in sorted(params, key=lambda x: key_of[x]):
            w = prep[kk]
            rows, _ = w.shape
            hi = cholesky_inv_upper(H[kk])
            idx, sc, ref = quantize_tensor(w, book, 1, group, hinv=hi)
            p = encode_scalar3(idx, sc, (rows, cols), group)
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"{kk}: decode mismatch"
            params[kk].copy_(unrotate(dq, signs, BLOCK).to(params[kk].dtype))
            del w, idx, sc, ref, dq, hi
        torch.cuda.empty_cache()

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"],
                      "eval_targets": n_targets}), flush=True)

    # --- reference statistics from a large, separate sample ---
    t0 = time.monotonic()
    Href = hessians(sl(a.ref_blocks, ref_off))
    Uref = {k: cholesky_inv_upper(v) for k, v in Href.items()}
    print(json.dumps({"stage": "reference_hessian",
                      "tokens": a.ref_blocks * a.cal_len,
                      "seconds": time.monotonic() - t0}), flush=True)

    # the contrast evaluated with the reference statistics, as the target value
    evs = {}
    for g in (128, 64):
        install(g, Href)
        evs[g] = evaluate(model, blocks)
    ref_contrast, rlo, rhi = paired_bootstrap(evs[64], evs[128])
    print(json.dumps({"reference_contrast": ref_contrast,
                      "ci": [rlo, rhi]}), flush=True)

    res = {"plan": {"model": a.model, "layers": layers, "eval_targets": n_targets,
                    "reference_tokens": a.ref_blocks * a.cal_len,
                    "sizes": [s for s, _ in SIZES], "draws": a.draws,
                    "note": "codebooks are deterministic Lloyd fits over weights, "
                            "so the Hessian is the only thing that varies"},
           "bf16": {"nll": bf["nll"]},
           "reference_contrast": {"delta": ref_contrast, "ci": [rlo, rhi]},
           "cells": {}}

    for sname, nblk in SIZES:
        for d in range(a.draws):
            key = f"{sname}|d{d}"
            if not B.check(key, 300):
                break
            H = hessians(sl(nblk, d * nblk))
            hdev = float(np.mean([rel_dev(H[k], Href[k]) for k in H]))
            udev = float(np.mean([rel_dev(cholesky_inv_upper(H[k]), Uref[k])
                                  for k in H]))
            e = {}
            for g in (128, 64):
                install(g, H)
                e[g] = evaluate(model, blocks)
            c, lo, hi = paired_bootstrap(e[64], e[128])
            res["cells"][key] = {
                "tokens": nblk * a.cal_len, "hessian_rel_dev": hdev,
                "cholinv_rel_dev": udev, "contrast": c, "contrast_ci": [lo, hi],
                "contrast_dev_from_reference": c - ref_contrast}
            print(json.dumps({"cell": key, "tokens": nblk * a.cal_len,
                              "h_dev": round(hdev, 5), "u_dev": round(udev, 5),
                              "contrast": round(c, 6)}), flush=True)
            del H
            torch.cuda.empty_cache()
            atomic_json(out / "results.json", res)

    # --- the three predictions ---
    cells = res["cells"]
    by_size = {}
    for sname, nblk in SIZES:
        vals = [cells[k] for k in cells if k.startswith(sname + "|")]
        if not vals:
            continue
        cs = [v["contrast"] for v in vals]
        by_size[sname] = {
            "tokens": nblk * a.cal_len,
            "mean_hessian_rel_dev": float(np.mean([v["hessian_rel_dev"] for v in vals])),
            "mean_cholinv_rel_dev": float(np.mean([v["cholinv_rel_dev"] for v in vals])),
            "contrast_values": cs, "contrast_mean": float(np.mean(cs)),
            "contrast_range": float(max(cs) - min(cs)),
            "contrast_sd": float(np.std(cs, ddof=1)) if len(cs) > 1 else None}

    def loglog_slope(xs, ys):
        if len(xs) < 2:
            return None
        lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
        return float(np.polyfit(lx, ly, 1)[0])

    sizes_present = [s for s, _ in SIZES if s in by_size]
    toks = [by_size[s]["tokens"] for s in sizes_present]
    pred = {
        "p1_hessian_slope_vs_tokens": loglog_slope(
            toks, [by_size[s]["mean_hessian_rel_dev"] for s in sizes_present]),
        "p1_cholinv_slope_vs_tokens": loglog_slope(
            toks, [by_size[s]["mean_cholinv_rel_dev"] for s in sizes_present]),
        "p1_expected_slope": -0.5,
        "p2_contrast_range_by_size": {s: by_size[s]["contrast_range"]
                                      for s in sizes_present},
        "p2_contrast_sd_slope_vs_tokens": loglog_slope(
            toks, [by_size[s]["contrast_sd"] for s in sizes_present
                   if by_size[s]["contrast_sd"]]) if all(
            by_size[s]["contrast_sd"] for s in sizes_present) else None,
    }
    # prediction 3: within-size correlation between Hessian deviation and
    # contrast deviation, pooled across sizes after centring each size
    xs, ys = [], []
    for s in sizes_present:
        v = [cells[k] for k in cells if k.startswith(s + "|")]
        h = np.array([x["cholinv_rel_dev"] for x in v])
        c = np.array([abs(x["contrast_dev_from_reference"]) for x in v])
        if len(v) > 1:
            xs.append(h - h.mean())
            ys.append(c - c.mean())
    if xs:
        X, Y = np.concatenate(xs), np.concatenate(ys)
        denom = float(np.sqrt((X ** 2).sum() * (Y ** 2).sum()))
        pred["p3_within_size_correlation"] = (float((X * Y).sum() / denom)
                                              if denom > 0 else None)
        pred["p3_n_pairs"] = int(len(X))
    res["predictions"] = pred
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Is GPTQ's Hessian noisy enough to explain the g64 instability?", "",
         f"Reference statistics from {a.ref_blocks * a.cal_len:,} tokens of "
         f"separate text; {a.draws} disjoint draws at each of "
         f"{len(sizes_present)} calibration sizes. Codebooks are deterministic "
         f"fits over weights, so the Hessian is the only thing that varies.", "",
         f"Reference contrast (g64 against g128): {ref_contrast:+.6f}.", "",
         "| tokens | mean ‖ΔH‖/‖H‖ | mean ‖ΔU‖/‖U‖ | contrast range | contrast sd |",
         "|---:|---:|---:|---:|---:|"]
    for s in sizes_present:
        v = by_size[s]
        sd = f"{v['contrast_sd']:.6f}" if v["contrast_sd"] else "--"
        L.append(f"| {v['tokens']:,} | {v['mean_hessian_rel_dev']:.5f} | "
                 f"{v['mean_cholinv_rel_dev']:.5f} | {v['contrast_range']:.6f} | {sd} |")
    L += ["", "## The three predictions", "",
          f"1. Hessian deviation against tokens, log-log slope "
          f"**{pred['p1_hessian_slope_vs_tokens']:+.3f}** "
          f"(compensation operator {pred['p1_cholinv_slope_vs_tokens']:+.3f}); "
          f"finite-sample noise predicts $-0.5$.",
          f"2. Contrast spread against tokens, log-log slope of the standard "
          f"deviation: "
          + (f"**{pred['p2_contrast_sd_slope_vs_tokens']:+.3f}**."
             if pred.get("p2_contrast_sd_slope_vs_tokens") is not None else "n/a."),
          f"3. Within-size correlation between compensation-operator deviation "
          f"and contrast deviation: "
          + (f"**{pred['p3_within_size_correlation']:+.3f}** over "
             f"{pred['p3_n_pairs']} draws."
             if pred.get("p3_within_size_correlation") is not None else "n/a.")]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
