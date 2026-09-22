"""The two 27B axes the frozen confirmation left unmeasured.

§11 measured one fitted quantizer on one corpus at one context length. Two
limitations were then stated rather than tested, and this run tests them.

**D. Domain and context at 27B.** §13 established on the 0.8B model that the
ordering survives a domain shift and context extension to 2048 tokens. Whether
that transfers to the 27B checkpoint was never checked; the 27B evidence is
entirely wikitext at 128 tokens.

**E. Refit variability at 27B.** §12 found that refitting the quantizer moves
the 0.8B contrasts by as much as the evaluation bootstrap does. The 27B
intervals carry the same conditioning, and the paper says so, but the size of
the effect there is unknown. Three refits is not eight; it bounds the effect
rather than characterizing it, and the run is labelled accordingly.

Both parts evaluate on a fixed 512-block prefix rather than the complete
stream. That is a deliberate trade: §13's prefix/remainder split showed the
prefix is representative of this corpus, an offloaded 27B forward pass moves
tens of GB across PCIe, and the questions here are about direction under new
conditions rather than about a sixth decimal place. Intervals are correspondingly
wider and that is stated with them.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import decode, encode_scalar3, encode_vq, gnorm_fp16
from run_rate_sweep import kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_mlp_task_v1 import largest_pow2_block
from run_generalization_v1 import text_of, make_blocks
from run_qwen38_confirm_v1 import (capture_hessians, evaluate, load_model,
                                   target_store)
from run_ternary_task_v4 import (cholesky_inv_upper, paired_bootstrap, quantize_tensor,
                                 rot_signs, rotate, rotate_hessian, unrotate)

ARMS = [("scalar3", 1, 3, 128), ("vq8", 8, 6561, 128)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.8-27B-metadata")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/qwen38_extend_v1")
    ap.add_argument("--gpu-gib", type=int, default=15)
    ap.add_argument("--cpu-gib", type=int, default=46)
    ap.add_argument("--eval-blocks", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--cal-batch", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=28800)
    ap.add_argument("--parts", default="DE")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    tok = AutoTokenizer.from_pretrained(a.model)
    cap = a.eval_blocks * 2048 + 4096
    wt_tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"), add_special_tokens=False).input_ids
    wt_te = tok(text_of(f"{a.data}/wikitext2/test.parquet"), add_special_tokens=False).input_ids
    ts_all = tok(text_of(f"{a.data}/tinystories/validation.parquet"),
                 add_special_tokens=False).input_ids
    half = len(ts_all) // 2

    EVALS = {
        "wt2_test_128": make_blocks(wt_te, 128, a.eval_blocks),
        "tiny_128": make_blocks(ts_all[half:], 128, a.eval_blocks),
        "wt2_test_2048": make_blocks(wt_te, 2048, max(32, a.eval_blocks // 16)),
    }

    def slice_cal(src, n, off=0):
        c = [src[off + i * a.cal_len : off + (i + 1) * a.cal_len] for i in range(n)]
        return [x for x in c if len(x) == a.cal_len]

    model = load_model(a.model, a.gpu_gib, a.cpu_gib)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    stores = {i: target_store(model, i) for i in layers}
    originals = {i: stores[i].detach().to("cpu", torch.bfloat16).clone() for i in layers}
    cols = originals[layers[0]].shape[1]
    blk = largest_pow2_block(cols)
    signs = rot_signs(cols, "cuda")

    def restore():
        with torch.no_grad():
            for i in layers:
                stores[i].copy_(originals[i].to(stores[i].device, stores[i].dtype))

    def hinv_for(cal):
        restore()
        Hs = capture_hessians(model, layers, cal, a.cal_batch)
        hv = {}
        for i in layers:
            hv[i] = cholesky_inv_upper(rotate_hessian(Hs[i].cuda(), signs, blk)).cpu()
            Hs[i] = None
        del Hs
        torch.cuda.empty_cache()
        return hv

    def install(dim, k, group, hv, seed=0):
        restore()
        pool = []
        for i in layers:
            w = rotate(originals[i].to("cuda", torch.float32), signs, blk)
            pool.append(gnorm_fp16(w, group)[0].reshape(-1, dim)[::521])
            del w
        pool = torch.cat(pool, 0)
        book = (lloyd_scalar(pool.flatten(), 3) if dim == 1 else kmeans(pool, k, seed=seed))
        del pool
        book = book.to(torch.float16).float()
        torch.cuda.empty_cache()
        for i in layers:
            w = rotate(originals[i].to("cuda", torch.float32), signs, blk)
            rows, _ = w.shape
            hi = hv[i].cuda()
            idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hi)
            del hi
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"layer {i}: decode mismatch"
            tw = stores[i]
            tw.copy_(unrotate(dq, signs, blk).to(tw.device, tw.dtype))
            del w, idx, sc, ref, dq, p
            torch.cuda.empty_cache()

    res = {"plan": {"model": a.model, "layers": len(layers), "hadamard_block": blk,
                    "eval_sets": {n: {"blocks": len(b),
                                      "targets": sum(len(x) - 1 for x in b)}
                                  for n, b in EVALS.items()},
                    "note": "512-block prefix, not the complete stream; intervals "
                            "are wider than the frozen confirmation's"}}

    # ---------------- D: domain and context at 27B ----------------
    if "D" in a.parts and B.check("partD", 1800):
        restore()
        bf = {}
        for n, bl in EVALS.items():
            batch = max(1, a.eval_batch * 128 // max(128, len(bl[0]) - 1))
            bf[n] = evaluate(model, bl, batch)
            print(json.dumps({"D": "bf16", "eval": n, "nll": bf[n]["nll"],
                              "targets": bf[n]["target_tokens"]}), flush=True)
        hv = hinv_for(slice_cal(wt_tr, a.cal_blocks))
        evs, cells = {}, {}
        for arm, dim, k, group in ARMS:
            t0 = time.monotonic()
            install(dim, k, group, hv)
            evs[arm] = {}
            for n, bl in EVALS.items():
                batch = max(1, a.eval_batch * 128 // max(128, len(bl[0]) - 1))
                e = evaluate(model, bl, batch)
                d, lo, hi = paired_bootstrap(e, bf[n])
                cells[f"{arm}|{n}"] = {"delta_nll": d, "delta_ci": [lo, hi]}
                evs[arm][n] = e
                print(json.dumps({"D": arm, "eval": n, "delta": round(d, 6)}), flush=True)
            print(json.dumps({"D_arm_seconds": time.monotonic() - t0}), flush=True)
        contrasts = {}
        for n in EVALS:
            d, lo, hi = paired_bootstrap(evs["vq8"][n], evs["scalar3"][n])
            contrasts[n] = {"delta": d, "ci": [lo, hi]}
            print(json.dumps({"D_contrast": n, "vq8_vs_scalar3": round(d, 6),
                              "ci": [round(lo, 6), round(hi, 6)]}), flush=True)
        res["D_domain_context"] = {"bf16": {n: bf[n]["nll"] for n in bf},
                                   "arms": cells, "vq8_vs_scalar3": contrasts}
        atomic_json(out / "results.json", res)
        del hv, evs
        torch.cuda.empty_cache()

    # ---------------- E: refit variability at 27B (3 refits, a bound) ----------------
    if "E" in a.parts and B.check("partE", 1800):
        bl = EVALS["wt2_test_128"]
        restore()
        bf1 = evaluate(model, bl, a.eval_batch)
        REFITS = [("draw0_seed0", slice_cal(wt_tr, a.cal_blocks, 0), 0),
                  ("draw1_seed0", slice_cal(wt_tr, a.cal_blocks, 256 * a.cal_len), 0),
                  ("draw0_seed1", slice_cal(wt_tr, a.cal_blocks, 0), 1)]
        cells, evs = {}, {}
        for rname, cal, seed in REFITS:
            if not B.check(f"E:{rname}", 1200):
                break
            hv = hinv_for(cal)
            evs[rname] = {}
            for arm, dim, k, group in ARMS:
                install(dim, k, group, hv, seed=seed)
                e = evaluate(model, bl, a.eval_batch)
                d, lo, hi = paired_bootstrap(e, bf1)
                cells[f"{rname}|{arm}"] = {"delta_nll": d, "delta_ci": [lo, hi]}
                evs[rname][arm] = e
            d, lo, hi = paired_bootstrap(evs[rname]["vq8"], evs[rname]["scalar3"])
            cells[f"{rname}|contrast"] = {"delta": d, "ci": [lo, hi]}
            print(json.dumps({"E": rname, "vq8_vs_scalar3": round(d, 6),
                              "half_width": round((hi - lo) / 2, 6)}), flush=True)
            del hv
            torch.cuda.empty_cache()
            atomic_json(out / "results.json", {**res, "E_refit": {"bf16": bf1["nll"],
                                                                  "cells": cells}})
        vals = [v["delta"] for k, v in cells.items() if k.endswith("|contrast")]
        res["E_refit"] = {"bf16": bf1["nll"], "cells": cells,
                          "contrast_range": (max(vals) - min(vals)) if len(vals) > 1 else None,
                          "contrast_values": vals,
                          "note": "three refits bound the effect; they do not "
                                  "characterize it as the eight 0.8B refits do"}
        atomic_json(out / "results.json", res)

    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)
    print(json.dumps({"done": True, "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
