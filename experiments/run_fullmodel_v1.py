"""A full-coverage conversion, and what a whole model actually costs.

Every result in this project covers a subset: 15.1% of parameters, or 35.1%,
or 50.2%, or 9.36% at 27B. So the paper can say what a codec does to some
tensors and cannot say what a converted model weighs or how good it is. This
run removes that gap on the small model by quantizing **every two-dimensional
weight except the embedding** at several rates, and reporting total model bytes
beside loss.

What "the model" means here, stated precisely because an earlier version of
this docstring got it wrong. The checkpoint holds 873,438,784 parameters in
three pieces: the language model (752,393,024), a vision tower (100,592,896)
and a multi-token-prediction head (20,452,864). `AutoModelForCausalLM`
instantiates **only the language model**, so the other two are never loaded and
appear in no byte total below. Every ratio here is against the 752.4M text
model, not the checkpoint.

Coverage inside that model:

  * quantized: 497,614,848 parameters, 66.1% -- `in_proj_qkv`, `in_proj_z`,
    `in_proj_a`, `in_proj_b` and `out_proj` in the linear-attention layers,
    `q_proj`/`k_proj`/`v_proj`/`o_proj` in the attention layers, and
    `gate_proj`/`up_proj`/`down_proj` in the MLPs. That is *every* 2-D weight
    outside the embedding, not a selection from them;
  * left in BF16: the embedding, 254,279,680 parameters, which is 33.8% of the
    model on its own and which `tie_word_embeddings` also makes the output
    head, so it is charged once and serves twice; the depthwise causal
    convolutions (442,368), which are 3-D and outside this codec's shape
    assumption; and the norms (56,128).

The embedding's share is the whole point of the run: at low rates it dominates
what is left, and no codec applied to the other 66% can reach it.

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
import collections
import json
import time
from pathlib import Path

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
    # Imported here rather than at module scope so `--render`, which only reads
    # a finished results.json, runs anywhere -- including a laptop with no GPU
    # stack installed.
    global torch, bits_per_weight, decode, encode_vq, gnorm_fp16, kmeans, Budget
    global atomic_json, capture_hessians, largest_pow2_block, text_of, make_blocks
    global cholesky_inv_upper, evaluate, paired_bootstrap, quantize_tensor
    global rot_signs, rotate, rotate_hessian, unrotate
    import torch
    from packed_lowrate import bits_per_weight, decode, encode_vq, gnorm_fp16
    from run_rate_sweep import kmeans
    from run_scf_phase0 import Budget
    from run_scq_qwen import atomic_json
    from run_mlp_task_v1 import capture_hessians, largest_pow2_block
    from run_generalization_v1 import text_of, make_blocks
    from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                     quantize_tensor, rot_signs, rotate,
                                     rotate_hessian, unrotate)

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

    # What stays BF16, counted rather than asserted: an earlier writeup of this
    # run described the leftovers wrongly, so the breakdown is a reported
    # number that has to add up to `untouched`.
    covered_names = {f"{n}.weight" for n, _, _ in targets}
    rest = collections.Counter()
    for pname, p in model.named_parameters():
        if pname in covered_names:
            continue
        rest["embedding" if any(s in pname for s in SKIP)
             else "norms_and_biases_1d" if p.dim() == 1
             else "non_matrix_weights"] += p.numel()
    assert sum(rest.values()) == untouched, (dict(rest), untouched)

    print(json.dumps({"tensors": len(targets), "covered_params": n_cov,
                      "total_params": n_all, "coverage": n_cov / n_all,
                      "untouched_params": untouched, "untouched": dict(rest),
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
                    "untouched_breakdown": dict(rest),
                    "eval_targets": n_targets, "bf16_model_bytes": dense_bytes,
                    "note": "every ratio is against the language model that "
                            "AutoModelForCausalLM instantiates; the checkpoint's "
                            "vision tower and MTP head are not loaded and are in "
                            "no byte total here. The embedding stays BF16 and is "
                            "tied to the output head."},
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

    (out / "summary.md").write_text(render(res))
    print(render(res))


def render(res: dict) -> str:
    """The writeup, built from results.json alone.

    Kept separate from `main` so a wording correction can be applied to an
    existing run without rerunning it, and so the file on disk cannot drift
    from the code that claims to produce it.
    """
    p = res["plan"]
    n_all, n_cov = p["total_params"], p["covered_params"]
    rest = p["untouched_breakdown"]
    L = ["# Full-coverage conversion of Qwen3.5-0.8B", "",
         f"Every 2-D weight except the embedding: {p['tensors']} tensors, "
         f"{n_cov:,} of {n_all:,} parameters (**{n_cov / n_all * 100:.1f}%**).",
         f"BF16 model is {p['bf16_model_bytes'] / 1e6:.0f} MB; BF16 NLL "
         f"{res['bf16']['nll']:.6f} over {p['eval_targets']:,} validation targets.", "",
         f"`{p['model']}` holds three pieces; `AutoModelForCausalLM` instantiates "
         "only the language model, so the checkpoint's vision tower and "
         "multi-token-prediction head are **not loaded** and appear in no byte "
         "total here. Every figure below is against the text model.", "",
         f"What stays BF16 ({p['untouched_params']:,} params):", ""]
    L += [f"  * {k}: {v:,}" for k, v in sorted(rest.items(), key=lambda x: -x[1])]
    L += ["",
          "The embedding is tied to the output head, so it is stored once and "
          "used twice -- and it alone is "
          f"{rest['embedding'] / n_all * 100:.1f}% of the model.", "",
          "| Arm | covered bpw | model MB | effective bpw | shrink | ΔNLL | 95% CI |",
          "|---|---:|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['covered_bits_per_weight']:.4f} | "
                 f"{r['model_bytes'] / 1e6:.1f} | {r['effective_bits_per_weight']:.3f} | "
                 f"{r['model_shrink_vs_bf16']:.2f}x | {r['delta_nll']:+.6f} | "
                 f"[{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |")
    L += ["", "`effective bpw` is the whole model's bytes divided by its parameter "
          "count, so it includes the BF16 embedding; `covered bpw` is the converted "
          "tensors alone. The gap between them is what keeping the embedding at "
          "BF16 costs."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    if "--render" in sys.argv:          # re-render a finished run, no GPU needed
        d = Path(sys.argv[sys.argv.index("--render") + 1])
        r = json.loads((d / "results.json").read_text())
        (d / "summary.md").write_text(render(r))
        print(render(r))
    else:
        main()
