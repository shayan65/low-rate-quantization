"""The comparison §17 identified as the important one and did not run.

§17 converted every non-embedding matrix of Qwen3.5-0.8B and found the result
dominated by what it did *not* touch: the embedding is 254,279,680 parameters,
it stays at BF16, and it is then 83% of the compressed model. The section ended
with a claim that was explicitly marked unmeasured --- that halving the
embedding to FP8 would save more bytes than moving the covered tensors from
2.667 bits to ternary, and would probably cost far less quality. This run
measures it.

The embedding is **tied** to the output head, so quantizing it perturbs the
input lookup *and* every logit. That is the opposite of a free ride: one tensor,
two jobs, and the output job is the one low-rate quantization is known to hurt.
Whether the tie makes the embedding more or less tolerant than the projections
is exactly what is unmeasured, and no argument here substitutes for running it.

Design: the two axes are crossed rather than studied separately, because the
question is a budget allocation, not a sensitivity.

  * **weight rate** --- the §17 ladder over all 186 non-embedding matrices,
    at 1.725, 2.125 and 2.792 covered bits/weight, plus BF16;
  * **embedding precision** --- BF16, FP8 (e4m3, per-row scaled), INT8 and INT4
    (group-128, symmetric).

Every cell reports total model bytes counted rather than derived, so the
frontier can be read directly: for a given model size, is it better to spend the
bytes on the weight codec or on embedding precision? A cheap embedding
quantizer is used deliberately --- per-row or per-group scaling, no rotation, no
compensation --- so the comparison is conservative. If a *naive* embedding
quantizer beats the carefully engineered weight codec on the byte-for-byte
trade, that is a stronger result than if it needed equal engineering.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from packed_lowrate import bits_per_weight, decode, encode_vq, gnorm_fp16
from run_rate_sweep import kmeans
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json
from run_mlp_task_v1 import capture_hessians, largest_pow2_block
from run_generalization_v1 import text_of, make_blocks
from run_fullmodel_v1 import SKIP, find_all_targets
from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

RATES = [("bf16", None, None), ("r1600_k81", 4, 81), ("r2000_k255", 4, 255),
         ("r2667_k1625", 4, 1625)]
EMBED = ["bf16", "fp8_e4m3", "int8_g128", "int4_g128"]


def quantize_embedding(W: torch.Tensor, mode: str, group: int = 128):
    """Dequantized copy of the embedding plus the bytes it would occupy.

    Deliberately simple: per-row scaling for FP8, per-group symmetric integers
    otherwise. No rotation and no error compensation, so this is a floor on what
    embedding quantization can achieve rather than a serious attempt at it.
    """
    rows, cols = W.shape
    w = W.float()
    if mode == "bf16":
        return W.clone(), rows * cols * 2
    if mode == "fp8_e4m3":
        amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        s = (amax / 448.0).to(torch.float16).float()          # per-row, FP16 scale
        q = (w / s).clamp(-448, 448).to(torch.float8_e4m3fn)
        deq = q.float() * s
        return deq.to(W.dtype), rows * cols * 1 + rows * 2
    bits = {"int8_g128": 8, "int4_g128": 4}[mode]
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(rows, cols // group, group)
    amax = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    s = (amax / qmax).to(torch.float16).float()
    q = torch.round(g / s).clamp(-qmax - 1, qmax)
    deq = (q * s).reshape(rows, cols)
    payload = rows * cols * bits // 8
    return deq.to(W.dtype), payload + rows * (cols // group) * 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/embed_precision_v1")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=28800)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(text_of(f"{a.data}/wikitext2/validation.parquet"),
              add_special_tokens=False).input_ids
    blocks = make_blocks(ids, 128)
    n_targets = sum(len(b) - 1 for b in blocks)
    tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"), add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is not None and head.weight.data_ptr() == emb.weight.data_ptr()
    emb_orig = emb.weight.detach().clone().cpu()
    n_emb = emb_orig.numel()

    targets = find_all_targets(model, a.group)
    params = {n: m.weight for n, m, _ in targets}
    blk_of = {n: b for n, _, b in targets}
    originals = {n: w.detach().clone().cpu() for n, w in params.items()}
    n_cov = sum(w.numel() for w in originals.values())
    n_all = sum(p.numel() for p in model.parameters())
    n_other = n_all - n_cov - n_emb
    print(json.dumps({"tied": tied, "embed_params": n_emb, "covered": n_cov,
                      "total": n_all, "other": n_other,
                      "eval_targets": n_targets}), flush=True)
    assert tied, "this model is expected to tie its embedding to the head"

    def restore_weights():
        with torch.no_grad():
            for n, w in params.items():
                w.copy_(originals[n].to(w.dtype).cuda())

    def restore_embed():
        with torch.no_grad():
            emb.weight.copy_(emb_orig.to(emb.weight.dtype).cuda())

    restore_weights(); restore_embed()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16/bf16", "nll": bf["nll"]}), flush=True)

    # --- embedding variants, prepared once ---
    emb_variants = {}
    for mode in EMBED:
        deq, nbytes = quantize_embedding(emb_orig.cuda(), mode, a.group)
        emb_variants[mode] = (deq.cpu(), nbytes)
        err = (deq.float() - emb_orig.cuda().float()).square().mean().item()
        print(json.dumps({"embed": mode, "bytes": nbytes,
                          "bits_per_param": nbytes * 8 / n_emb,
                          "mse": err}), flush=True)
        del deq
        torch.cuda.empty_cache()

    # --- weight variants: quantize once per rate, freeze the dequantized form ---
    signs = {n: rot_signs(originals[n].shape[1], "cuda") for n in params}
    t0 = time.monotonic()
    Hraw, _ = capture_hessians(model, targets, cal)
    hinv = {}
    for n in params:
        hinv[n] = cholesky_inv_upper(rotate_hessian(Hraw[n], signs[n], blk_of[n]))
        Hraw[n] = None
    del Hraw
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "hessians", "seconds": time.monotonic() - t0}), flush=True)

    weight_variants = {"bf16": (None, n_cov * 2, 16.0)}
    for label, dim, k in RATES[1:]:
        t0 = time.monotonic()
        pool = torch.cat([
            gnorm_fp16(rotate(originals[n].cuda().float(), signs[n], blk_of[n]),
                       a.group)[0].reshape(-1, dim)[::37] for n in params], 0)
        book = kmeans(pool, k).to(torch.float16).float()
        del pool
        torch.cuda.empty_cache()
        frozen, packs = {}, []
        for n, _, _ in targets:
            w = rotate(originals[n].cuda().float(), signs[n], blk_of[n])
            rows, cols = w.shape
            idx, sc, ref = quantize_tensor(w, book, dim, a.group, hinv=hinv[n])
            pk = encode_vq(idx, sc, (rows, cols), a.group, dim, k)
            dq = decode(pk, book, "cuda")
            assert torch.equal(dq, ref), f"{n}: decode mismatch"
            frozen[n] = unrotate(dq, signs[n], blk_of[n]).to(torch.bfloat16).cpu()
            packs.append(pk)
            del w, idx, sc, ref, dq
        cb = book.numel() * 2
        nbytes = sum(p.payload_bytes() for p in packs) + cb
        weight_variants[label] = (frozen, nbytes, bits_per_weight(packs, cb))
        print(json.dumps({"weights": label, "bytes": nbytes,
                          "covered_bpw": bits_per_weight(packs, cb),
                          "seconds": time.monotonic() - t0}), flush=True)
        del packs
        torch.cuda.empty_cache()

    dense_bytes = n_all * 2
    res = {"plan": {"model": a.model, "tied_embedding": tied,
                    "embed_params": n_emb, "covered_params": n_cov,
                    "other_params": n_other, "total_params": n_all,
                    "bf16_model_bytes": dense_bytes, "eval_targets": n_targets,
                    "note": "embedding quantization is deliberately naive: per-row "
                            "or per-group scaling, no rotation, no compensation"},
           "bf16": {"nll": bf["nll"]},
           "embedding_variants": {m: {"bytes": b, "bits_per_param": b * 8 / n_emb}
                                  for m, (_, b) in emb_variants.items()},
           "cells": {}}

    for wlabel, (frozen, wbytes, wbpw) in weight_variants.items():
        for elabel in EMBED:
            key = f"{wlabel}|{elabel}"
            if not B.check(key, 120):
                break
            restore_weights()
            if frozen is not None:
                with torch.no_grad():
                    for n, w in params.items():
                        w.copy_(frozen[n].to(w.dtype).cuda())
            deq, ebytes = emb_variants[elabel]
            with torch.no_grad():
                emb.weight.copy_(deq.to(emb.weight.dtype).cuda())
            e = evaluate(model, blocks)
            d, lo, hi = paired_bootstrap(e, bf)
            model_bytes = wbytes + ebytes + n_other * 2
            res["cells"][key] = {
                "weight_bytes": wbytes, "embed_bytes": ebytes,
                "model_bytes": model_bytes,
                "model_shrink_vs_bf16": dense_bytes / model_bytes,
                "effective_bits_per_weight": model_bytes * 8 / n_all,
                "covered_bpw": wbpw, "embed_bits": ebytes * 8 / n_emb,
                "nll": e["nll"], "delta_nll": d, "delta_ci": [lo, hi]}
            print(json.dumps({"cell": key, "MB": round(model_bytes / 1e6, 1),
                              "delta": round(d, 6)}), flush=True)
            atomic_json(out / "results.json", res)
    restore_weights(); restore_embed()

    # --- the frontier: is a byte better spent on the codec or the embedding? ---
    cells = res["cells"]
    pts = sorted(((v["model_bytes"], v["delta_nll"], k) for k, v in cells.items()))
    frontier, best = [], float("inf")
    for b, d, k in pts:
        if d < best:
            frontier.append({"cell": k, "model_bytes": b, "delta_nll": d})
            best = d
    res["frontier"] = frontier
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Embedding precision against weight rate: where the bytes should go", "",
         f"Qwen3.5-0.8B. The embedding is {n_emb:,} parameters and is **tied** to "
         f"the output head, so quantizing it perturbs both the input lookup and "
         f"every logit. {len(targets)} non-embedding matrices ({n_cov:,} params) "
         f"carry the weight codec. BF16 model {dense_bytes / 1e6:.0f} MB, BF16 NLL "
         f"{bf['nll']:.6f} over {n_targets:,} targets.", "",
         "Embedding quantization here is deliberately naive — per-row or "
         "per-group scaling, no rotation, no compensation — so the comparison is "
         "conservative against the weight codec, which has both.", "",
         "| weights | embedding | model MB | eff. bpw | shrink | ΔNLL | 95% CI |",
         "|---|---|---:|---:|---:|---:|:--|"]
    for k, v in cells.items():
        w, e = k.split("|")
        L.append(f"| {w} | {e} | {v['model_bytes'] / 1e6:.1f} | "
                 f"{v['effective_bits_per_weight']:.3f} | "
                 f"{v['model_shrink_vs_bf16']:.2f}x | {v['delta_nll']:+.6f} | "
                 f"[{v['delta_ci'][0]:+.6f}, {v['delta_ci'][1]:+.6f}] |")
    L += ["", "## Frontier (cheapest model at each new best loss)", "",
          "| cell | model MB | ΔNLL |", "|---|---:|---:|"]
    for f in frontier:
        L.append(f"| {f['cell']} | {f['model_bytes'] / 1e6:.1f} | {f['delta_nll']:+.6f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
