"""A full-coverage conversion, and what a whole model actually costs.

Every result in this project covers a subset: 15.1% of parameters, or 35.1%,
or 50.2%, or 9.36% at 27B. So the paper can say what a codec does to some
tensors and cannot say what a converted model weighs or how good it is. This
run removes that gap on the small model by quantizing **every two-dimensional
weight except the embeddings and the LM head** -- 615.6M of 873.4M parameters,
70.5% -- at several rates, and reporting total model bytes beside loss.

Coverage and its boundaries:

  * quantized: MLP, all `linear_attn` projections, `self_attn`, and the
    remaining 2-D weights (43.0M, mostly the vision tower);
  * left in BF16: embeddings and LM head (256.0M, 29.3%), which low-rate
    quantization damages disproportionately and which every practical recipe
    keeps at higher precision, and the norms (1.8M);
  * the vision tower is counted in the byte total but is **not exercised** by a
    text-only evaluation, so its quality contribution here is unmeasured. Its
    share is small enough (4.9%) not to move the storage conclusion.

Column counts all divide 128, and each tensor gets the largest power-of-two
Hadamard block that divides its input dimension, so no tensor needs a special
case. The Hessians for every target are held on the GPU in FP32, about 2.9 GB
for this model, which fits alongside it.

Total bytes are counted, not derived: packed payload plus FP16 group scales
plus the shared codebook for converted tensors, plus BF16 for everything
untouched. That is the number a deployment decision needs and the one this
project has so far only estimated.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from packed_lowrate import bits_per_weight, decode, encode_vq, gnorm_fp16
from run_rate_sweep import kmeans
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json
from run_mlp_task_v1 import capture_hessians, largest_pow2_block
from run_generalization_v1 import text_of, make_blocks
from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

SKIP = ("embed", "lm_head")
# (label, dim, K) -- index rate is 64/(codes_per_word * dim)
LADDER = [("r1600_k81", 4, 81), ("r2000_k255", 4, 255), ("r2667_k1625", 4, 1625)]


def find_all_targets(model, group):
    """Every 2-D weight except embeddings/head, with its Hadamard block."""
    out = []
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            continue
        if any(s in name for s in SKIP):
            continue
        if w.shape[1] % group:
            continue
        out.append((name, mod, largest_pow2_block(w.shape[1])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/fullmodel_v1")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=14400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)
    group = a.group

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
    targets = find_all_targets(model, group)
    params = {n: m.weight for n, m, _ in targets}
    blk_of = {n: b for n, _, b in targets}
    originals = {n: w.detach().clone().cpu() for n, w in params.items()}
    n_cov = sum(w.numel() for w in originals.values())
    n_all = sum(p.numel() for p in model.parameters())
    untouched = n_all - n_cov

    print(json.dumps({"tensors": len(targets), "covered_params": n_cov,
                      "total_params": n_all, "coverage": n_cov / n_all,
                      "untouched_params": untouched,
                      "eval_targets": n_targets}), flush=True)

    def restore():
        with torch.no_grad():
            for n, w in params.items():
                w.copy_(originals[n].to(w.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"]}), flush=True)

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

    dense_bytes = n_all * 2
    res = {"plan": {"model": a.model, "tensors": len(targets),
                    "covered_params": n_cov, "total_params": n_all,
                    "coverage": n_cov / n_all, "untouched_params": untouched,
                    "eval_targets": n_targets, "bf16_model_bytes": dense_bytes,
                    "note": "embeddings and LM head stay BF16; the vision tower is "
                            "counted in bytes but not exercised by a text evaluation"},
           "bf16": {"nll": bf["nll"]}, "arms": {}}

    for label, dim, k in LADDER:
        if not B.check(label, 600):
            break
        restore()
        t0 = time.monotonic()
        pool = torch.cat([
            gnorm_fp16(rotate(originals[n].cuda().float(), signs[n], blk_of[n]),
                       group)[0].reshape(-1, dim)[::37]
            for n in params], 0)
        book = kmeans(pool, k).to(torch.float16).float()
        del pool
        torch.cuda.empty_cache()

        packs = []
        for n, _, _ in targets:
            w = rotate(originals[n].cuda().float(), signs[n], blk_of[n])
            rows, cols = w.shape
            idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hinv[n])
            p = encode_vq(idx, sc, (rows, cols), group, dim, k)
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"{n}: decode mismatch"
            params[n].copy_(unrotate(dq, signs[n], blk_of[n]).to(params[n].dtype))
            packs.append(p)
            del w, idx, sc, ref, dq
        torch.cuda.empty_cache()

        cb = book.numel() * 2
        conv_bytes = sum(p.payload_bytes() for p in packs) + cb
        model_bytes = conv_bytes + untouched * 2
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev, bf)
        rec = {"dim": dim, "k": k,
               "covered_bits_per_weight": bits_per_weight(packs, cb),
               "converted_bytes": conv_bytes, "untouched_bytes": untouched * 2,
               "model_bytes": model_bytes,
               "model_shrink_vs_bf16": dense_bytes / model_bytes,
               "effective_bits_per_weight": model_bytes * 8 / n_all,
               "nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi],
               "seconds": time.monotonic() - t0}
        res["arms"][label] = rec
        print(json.dumps({"arm": label, **{kk: rec[kk] for kk in
                                           ("covered_bits_per_weight", "model_bytes",
                                            "effective_bits_per_weight", "delta_nll")}}),
              flush=True)
        atomic_json(out / "results.json", res)
        del packs

    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Full-coverage conversion of Qwen3.5-0.8B", "",
         f"Every 2-D weight except embeddings and the LM head: {len(targets)} tensors, "
         f"{n_cov:,} of {n_all:,} parameters (**{n_cov / n_all * 100:.1f}%**).",
         f"Embeddings and head ({untouched:,} params) stay BF16. BF16 model is "
         f"{dense_bytes / 1e6:.0f} MB; BF16 NLL {bf['nll']:.6f} over {n_targets:,} "
         "validation targets.",
         "The vision tower is inside the byte total but is not exercised by a "
         "text-only evaluation.", "",
         "| Arm | covered bpw | model MB | effective bpw | shrink | ΔNLL | 95% CI |",
         "|---|---:|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['covered_bits_per_weight']:.4f} | "
                 f"{r['model_bytes'] / 1e6:.1f} | {r['effective_bits_per_weight']:.3f} | "
                 f"{r['model_shrink_vs_bf16']:.2f}x | {r['delta_nll']:+.6f} | "
                 f"[{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |")
    L += ["", "`effective bpw` is the whole model's bytes divided by its parameter "
          "count, so it includes the BF16 embeddings; `covered bpw` is the converted "
          "tensors alone. The gap between them is what keeping embeddings at BF16 "
          "costs."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
