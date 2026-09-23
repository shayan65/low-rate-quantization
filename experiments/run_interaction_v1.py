"""Is coverage damage additive? The test the four-run comparison could not be.

§17 observed that damage from the QKV projections ($+0.088512$) and from the
MLP weights ($+0.552558$) sums to $0.641069$ against $+0.637829$ measured for
converting both, and called damage "additive to within 0.5%". A review was
right that the observation is supported but the conclusion is not: those three
numbers come from three separate runs, each fitting its shared codebook on its
own tensor pool. Expanding coverage therefore changes the perturbation applied
to the *previously covered* tensors too, so the comparison confounds an
interaction between tensor families with a change in the codec each family
receives.

This run removes that confound the way the review proposed. One Hessian
capture, one codebook fitted once over all 90 tensors, one quantization pass —
then the resulting dequantized weights are **frozen** and installed in three
combinations:

  * A  -- the 18 `in_proj_qkv` projections only;
  * B  -- the 72 MLP tensors only;
  * AB -- both.

Every tensor carries bit-identical weights in every arm that includes it, which
is what makes the interaction

    I = (L_AB - L_BF16) - (L_A - L_BF16) - (L_B - L_BF16)
      =  L_AB - L_A - L_B + L_BF16

a clean paired quantity rather than a difference of differently-fitted runs. It
is bootstrapped over the same resampled evaluation blocks as every other
interval here, so it carries an interval rather than a hand-waved tolerance.

A near-zero I means the two families' damage composes additively at this rate.
A positive I means converting both hurts more than the parts predict. Either
result is worth having; neither was available from §17's design.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from packed_lowrate import decode, encode_vq, gnorm_fp16
from run_rate_sweep import BLOCK, kmeans
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_mlp_task_v1 import capture_hessians as mlp_capture, largest_pow2_block
from run_generalization_v1 import text_of, make_blocks
from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

DIM, K, GROUP = 4, 81, 128          # ternary rate, the rate §17 compared at


def combo_bootstrap(ab, a, b, bf, reps=20000, seed=17):
    """Interaction L_AB - L_A - L_B + L_BF16, resampling blocks jointly.

    The four arms are evaluated on the same blocks in the same order, so one
    resampling index applies to all of them and the block-level correlation
    that makes these contrasts tight is preserved.
    """
    S = {k: np.asarray(v["block_sums"]) for k, v in
         (("ab", ab), ("a", a), ("b", b), ("bf", bf))}
    c = np.asarray(ab["block_counts"], dtype=float)
    n = len(c)
    for k, v in S.items():
        assert len(v) == n, f"{k}: block count differs"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    den = c[idx].sum(1)
    d = ((S["ab"][idx].sum(1) - S["a"][idx].sum(1)
          - S["b"][idx].sum(1) + S["bf"][idx].sum(1)) / den)
    point = float((S["ab"].sum() - S["a"].sum() - S["b"].sum() + S["bf"].sum())
                  / c.sum())
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def find_targets(model):
    """[(name, module, hadamard block)] for the 18 QKV and the 72 MLP tensors."""
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    qkv, mlp = [], []
    for i in find_layers(model):
        m = get_module(model, i)
        qkv.append((f"qkv.L{i}", m, largest_pow2_block(m.weight.shape[1])))
    for i, layer in enumerate(lm.layers):
        if not hasattr(layer, "mlp"):
            continue
        for nm in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(layer.mlp, nm, None)
            if mod is not None and getattr(mod, "weight", None) is not None:
                mlp.append((f"mlp.L{i}.{nm}", mod, largest_pow2_block(mod.weight.shape[1])))
    return qkv, mlp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/interaction_v1")
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=7200)
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
    qkv, mlp = find_targets(model)
    targets = qkv + mlp
    names_a = [n for n, _, _ in qkv]
    names_b = [n for n, _, _ in mlp]
    mods = {n: m for n, m, _ in targets}
    blk_of = {n: b for n, _, b in targets}
    originals = {n: m.weight.detach().clone().cpu() for n, m in mods.items()}
    n_a = sum(originals[n].numel() for n in names_a)
    n_b = sum(originals[n].numel() for n in names_b)
    print(json.dumps({"qkv_tensors": len(qkv), "qkv_params": n_a,
                      "mlp_tensors": len(mlp), "mlp_params": n_b,
                      "eval_targets": n_targets}), flush=True)

    def restore(which=None):
        with torch.no_grad():
            for n in (which or originals):
                mods[n].weight.copy_(originals[n].to(mods[n].weight.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"]}), flush=True)

    signs = {n: rot_signs(originals[n].shape[1], "cuda") for n in originals}
    t0 = time.monotonic()
    Hraw, _ = mlp_capture(model, targets, cal)
    hinv = {}
    for n in originals:
        hinv[n] = cholesky_inv_upper(rotate_hessian(Hraw[n], signs[n], blk_of[n]))
        Hraw[n] = None
    del Hraw
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "hessians", "seconds": time.monotonic() - t0}), flush=True)

    # --- one codebook, fitted once over all 90 tensors, used by every arm ---
    pool = torch.cat([
        gnorm_fp16(rotate(originals[n].cuda().float(), signs[n], blk_of[n]),
                   GROUP)[0].reshape(-1, DIM)[::37] for n in originals], 0)
    book = kmeans(pool, K).to(torch.float16).float()
    del pool
    torch.cuda.empty_cache()

    # --- one quantization pass; the results are frozen and reused ---
    quantized, bits = {}, []
    t0 = time.monotonic()
    for n, _, _ in targets:
        w = rotate(originals[n].cuda().float(), signs[n], blk_of[n])
        rows, cols = w.shape
        idx, sc, ref = quantize_tensor(w, book, DIM, GROUP, hinv=hinv[n])
        pk = encode_vq(idx, sc, (rows, cols), GROUP, DIM, K)
        dq = decode(pk, book, "cuda")
        assert torch.equal(dq, ref), f"{n}: decode mismatch"
        quantized[n] = unrotate(dq, signs[n], blk_of[n]).to(torch.bfloat16).cpu()
        bits.append(pk.payload_bytes() * 8 / (rows * cols))
        del w, idx, sc, ref, dq, pk
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "quantize", "seconds": time.monotonic() - t0,
                      "mean_bits_per_weight": float(np.mean(bits))}), flush=True)

    def install(which):
        restore()
        with torch.no_grad():
            for n in which:
                mods[n].weight.copy_(quantized[n].to(mods[n].weight.dtype).cuda())

    evs, arms = {}, {}
    for label, which in (("A_qkv", names_a), ("B_mlp", names_b),
                         ("AB_both", names_a + names_b)):
        if not B.check(label, 300):
            break
        install(which)
        e = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(e, bf)
        evs[label] = e
        arms[label] = {"tensors": len(which), "nll": e["nll"],
                       "delta_nll": d, "delta_ci": [lo, hi]}
        print(json.dumps({"arm": label, "delta": round(d, 6),
                          "ci": [round(lo, 6), round(hi, 6)]}), flush=True)

    res = {"plan": {"model": a.model, "dim": DIM, "k": K, "group": GROUP,
                    "qkv_tensors": len(qkv), "qkv_params": n_a,
                    "mlp_tensors": len(mlp), "mlp_params": n_b,
                    "cal_tokens": len(cal) * a.cal_len, "eval_targets": n_targets,
                    "mean_bits_per_weight": float(np.mean(bits)),
                    "note": "one codebook and one quantization pass shared by all "
                            "three arms; every tensor carries bit-identical weights "
                            "in every arm that includes it"},
           "bf16": {"nll": bf["nll"]}, "arms": arms}

    if len(evs) == 3:
        p, lo, hi = combo_bootstrap(evs["AB_both"], evs["A_qkv"], evs["B_mlp"], bf)
        add_ref = arms["A_qkv"]["delta_nll"] + arms["B_mlp"]["delta_nll"]
        res["interaction"] = {
            "value": p, "ci": [lo, hi],
            "additive_reference": add_ref,
            "measured_both": arms["AB_both"]["delta_nll"],
            "excludes_zero": bool(lo * hi > 0),
            "definition": "L_AB - L_A - L_B + L_BF16, paired over evaluation blocks",
        }
        print(json.dumps({"interaction": round(p, 6), "ci": [round(lo, 6), round(hi, 6)],
                          "excludes_zero": bool(lo * hi > 0)}), flush=True)

    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Is coverage damage additive? A frozen-artifact interaction test", "",
         f"One codebook and one quantization pass over {len(targets)} tensors "
         f"(dim {DIM}, K={K}, {np.mean(bits):.4f} bits/weight), then the "
         "dequantized weights installed in three combinations. Every tensor "
         "carries bit-identical weights in every arm that includes it, so the "
         "interaction is a clean paired quantity rather than a comparison "
         "between separately fitted runs.", "",
         f"BF16 NLL {bf['nll']:.6f} over {n_targets:,} validation targets.", "",
         "| arm | tensors | params | ΔNLL | 95% CI |", "|---|---:|---:|---:|:--|"]
    for label, r in arms.items():
        params = n_a if label == "A_qkv" else n_b if label == "B_mlp" else n_a + n_b
        L.append(f"| {label} | {r['tensors']} | {params:,} | {r['delta_nll']:+.6f} | "
                 f"[{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |")
    if "interaction" in res:
        it = res["interaction"]
        L += ["", "## Interaction", "",
              f"`L_AB - L_A - L_B + L_BF16` = **{it['value']:+.6f}** "
              f"[{it['ci'][0]:+.6f}, {it['ci'][1]:+.6f}], "
              f"{'excludes' if it['excludes_zero'] else 'includes'} zero.", "",
              f"Additive reference {it['additive_reference']:+.6f} against a "
              f"measured {it['measured_both']:+.6f}."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
