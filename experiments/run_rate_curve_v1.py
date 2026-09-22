"""The rate-distortion curve above ternary, under the best recipe available.

Section 9 found that a 0.4-bit increase in index rate removes 45% of the
damage -- more than every codec refinement in this project combined -- but the
ladder stopped at 2.000 bits because that was all the allocation study needed.
Nothing here has ever measured the region between 2 and 4 bits, which is
exactly where a *usable* model is likely to live: at ternary the damage is
+0.51 NLL even with the best configuration, and at 4 bits earlier work found
total damage of 0.24%, so the interesting knee is somewhere in between and has
never been looked at.

This run sweeps uniform rate over the same 90 tensors as section 9
(`in_proj_qkv` + every MLP projection, 50.2% of the model) with everything else
held fixed: dimension-4 vector codes, randomized Hadamard rotation, FP16 group
scales, GPTQ block compensation, storage measured from real packed bytes, and
weights decoded back from those bytes.

Rate rungs come from the packing itself. With c codes per uint64 at dimension
d the rate is exactly 64/(c*d), and the best code at that rate uses the largest
K with K^c < 2^64:

    c=10  K=84     1.600 bits/weight
    c= 8  K=255    2.000
    c= 7  K=565    2.286
    c= 6  K=1625   2.667
    c= 5  K=7131   3.200

Every rung's packed rate matches log2(K)/4 to three decimals, so the ladder
wastes essentially nothing and the curve is a property of the codec rather than
of the container. K=81 is included as a sixth point purely to tie this sweep to
section 9, which used it.

Dimension is held at 4 throughout so the curve measures rate alone. The 4.000
rung (K=65535) is omitted deliberately: assignment against 65,535 centroids
over 94M vectors costs roughly 20 minutes per arm, and earlier work already
established that damage at 4 bits is negligible, so the rung would cost the
most and tell us the least.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import bits_per_weight, codes_per_word, decode, encode_vq
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json
from run_mixed_alloc_v1 import collect_targets, fit_book, index_bpw
from run_mlp_task_v1 import capture_hessians, largest_pow2_block
from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

DIM = 4
# (label, K); K is the largest value whose packing hits the stated rate exactly
LADDER = [
    ("r1600_k81", 81),      # ties to section 9
    ("r1600_k84", 84),
    ("r2000_k255", 255),
    ("r2286_k565", 565),
    ("r2667_k1625", 1625),
    ("r3200_k7131", 7131),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/rate_curve_v1")
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=9000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)
    group = a.group

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok("\n".join(pd.read_parquet(f"{a.data}/validation.parquet").text.tolist()),
              add_special_tokens=False).input_ids
    full = (len(ids) - 1) // 128
    blocks = [ids[i * 128 : i * 128 + 129] for i in range(full)]
    tail = ids[full * 128 :]
    if len(tail) >= 2:
        blocks.append(tail)
    n_targets = sum(len(b) - 1 for b in blocks)

    tr = tok("\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()[:4000]),
             add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = collect_targets(model)
    params = {nm: m.weight for nm, m, _ in targets}
    blk_of = {nm: b for nm, _, b in targets}
    originals = {nm: w.detach().clone().cpu() for nm, w in params.items()}
    n_all = sum(w.numel() for w in originals.values())
    total_params = sum(p.numel() for p in model.parameters())

    print(json.dumps({"eval_targets": n_targets, "tensors": len(targets),
                      "target_params": n_all, "share": n_all / total_params,
                      "ladder": [(lab, k, index_bpw(DIM, k)) for lab, k in LADDER]}), flush=True)

    def restore():
        with torch.no_grad():
            for nm, w in params.items():
                w.copy_(originals[nm].to(w.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "t": B.stamp()}), flush=True)

    signs = {nm: rot_signs(originals[nm].shape[1], "cuda") for nm in params}
    t0 = time.monotonic()
    Hraw, _ = capture_hessians(model, targets, cal)
    Hrot = {nm: rotate_hessian(Hraw[nm], signs[nm], blk_of[nm]) for nm in params}
    del Hraw
    torch.cuda.empty_cache()
    prep = {nm: rotate(originals[nm].cuda().float(), signs[nm], blk_of[nm]) for nm in params}
    hinv = {nm: cholesky_inv_upper(Hrot[nm]) for nm in params}  # rate-independent, cached
    print(json.dumps({"stage": "setup", "seconds": time.monotonic() - t0}), flush=True)

    res = {"plan": {"model": a.model, "tensors": len(targets), "target_params": n_all,
                    "share": n_all / total_params, "eval_targets": n_targets, "dim": DIM,
                    "protocol": "uniform rate sweep, dim-4 VQ + rotation + GPTQ; "
                                "bits/weight measured from packed bytes"},
           "bf16": {"nll": bf["nll"]}, "arms": {}}

    for label, k in LADDER:
        if not B.check(label, 400):
            break
        restore()
        t0 = time.monotonic()
        book = fit_book(prep, list(params), DIM, k, group)
        packs, maxdiff, wse, wn = [], 0.0, 0.0, 0
        for nm, _, _ in targets:
            w = prep[nm]
            rows, cols = w.shape
            idx, sc, ref = quantize_tensor(w, book, DIM, group, hinv=hinv[nm])
            p = encode_vq(idx, sc, (rows, cols), group, DIM, k)
            dq = decode(p, book, "cuda")
            maxdiff = max(maxdiff, (dq - ref).abs().max().item())
            wse += (dq - w).square().double().sum().item()
            wn += rows * cols
            params[nm].copy_(unrotate(dq, signs[nm], blk_of[nm]).to(params[nm].dtype))
            packs.append(p)
            del w, idx, sc, ref, dq
        assert maxdiff == 0.0, f"{label}: decoded weights differ by {maxdiff}"
        torch.cuda.empty_cache()

        cb = book.numel() * 2
        bpw = bits_per_weight(packs, cb)
        total_bytes = sum(p.payload_bytes() for p in packs) + cb
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev, bf)
        rec = {"k": k, "index_bpw": index_bpw(DIM, k), "bits_per_weight": bpw,
               "stored_bytes": total_bytes, "codebook_bytes": cb,
               "nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi],
               "weight_mse": wse / wn, "decode_exact": maxdiff == 0.0,
               "quantize_seconds": time.monotonic() - t0}
        res["arms"][label] = {**rec, "_ev": ev}
        print(json.dumps({"arm": label, **rec}), flush=True)
        atomic_json(out / "results.json",
                    {**res, "arms": {x: {p2: v2 for p2, v2 in y.items() if p2 != "_ev"}
                                     for x, y in res["arms"].items()}})

    comps = {}
    names = list(res["arms"])
    for i in range(len(names) - 1):  # each rung against the next one up
        m, ref = names[i], names[i + 1]
        d, lo, hi = paired_bootstrap(res["arms"][m]["_ev"], res["arms"][ref]["_ev"])
        comps[f"{m}_vs_{ref}"] = {
            "delta": d, "ci": [lo, hi],
            "byte_ratio": res["arms"][m]["stored_bytes"] / res["arms"][ref]["stored_bytes"]}
    for v in res["arms"].values():
        v.pop("_ev", None)
    res["comparisons"] = comps
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Rate-distortion curve above ternary (dim-4 VQ + rotation + GPTQ)", "",
         f"Qwen3.5-0.8B, {len(targets)} tensors, {n_all:,} params = "
         f"{n_all / total_params * 100:.1f}% of the model.",
         f"**{n_targets:,} validation targets**, token-weighted. BF16 NLL {bf['nll']:.6f}.",
         "Dimension held at 4 throughout, so the curve measures rate alone.", "",
         "| Arm | K | index bpw | total bpw | stored MB | weight MSE | ΔNLL vs BF16 | 95% CI |",
         "|---|---:|---:|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['k']} | {r['index_bpw']:.3f} | {r['bits_per_weight']:.4f} "
                 f"| {r['stored_bytes'] / 1e6:.2f} | {r['weight_mse']:.4e} "
                 f"| {r['delta_nll']:+.6f} "
                 f"| [{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |")
    L += ["", "## Each rung against the next one up", "",
          "| Comparison | ΔNLL | 95% CI | byte ratio | |", "|---|---:|:--|---:|:--|"]
    for kk, c in comps.items():
        ex = "excludes 0" if (c["ci"][0] > 0) == (c["ci"][1] > 0) else "**includes 0**"
        L.append(f"| {kk.replace('_vs_', ' vs ')} | {c['delta']:+.6f} "
                 f"| [{c['ci'][0]:+.6f}, {c['ci'][1]:+.6f}] | {c['byte_ratio']:.4f} | {ex} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
