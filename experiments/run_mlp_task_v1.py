"""Step 1 of the 27B plan: does the ternary VQ recipe work on MLP weights?

Every result in this project so far was measured on `linear_attn.in_proj_qkv`
alone -- 15.1% of the parameters at 0.8B and 9.1% at 27B. The MLPs are 63.5% of
Qwen3.8-27B and completely unmeasured, so a full-coverage ternary conversion
rests on an assumption nobody has tested: that a recipe fitted to DeltaNet
input projections transfers to gate/up/down.

There is a specific reason to doubt it. `down_proj` reads the *activated*
intermediate, whose distribution is not remotely Gaussian, and it reads 3584
channels rather than 1024. Both cut against a shared codebook fitted mostly on
projection weights.

This run is the cheap falsification, before any 27B time is spent. It applies
the v4 recipe (rotation, FP16 group scales, GPTQ block compensation, storage
measured from real packed bytes, weights decoded back from those bytes) to all
72 MLP tensors while leaving `in_proj_qkv` at BF16, so the MLP question is
isolated from the result already established.

Two structural differences from v4, both forced by the shapes:

  * `down_proj` is (1024, 3584) and 3584 is not a multiple of the 1024-element
    Hadamard block. It uses a 512-element block instead (3584 = 7 x 512), which
    is the largest power of two that divides it.
  * `gate_proj` and `up_proj` read the same tensor, so their Hessians are
    identical by construction; they are still captured independently, which
    doubles as a consistency check on the capture code.

The `pertype` arm asks whether one codebook can serve all three projections or
whether `down_proj` needs its own. Three codebooks cost 3 x 104,976 bytes over
264M weights -- 0.0095 bits/weight -- so if per-type helps at all it is almost
free.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import (bits_per_weight, decode, encode_scalar3, encode_vq,
                            gnorm_fp16)
from run_rate_sweep import kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json
from run_ternary_task_v4 import (TINY, cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

TYPES = ("gate_proj", "up_proj", "down_proj")


def largest_pow2_block(cols: int, cap: int = 1024) -> int:
    """Biggest power-of-two Hadamard block dividing `cols` (<= cap)."""
    b = min(cap, 1 << (cols.bit_length() - 1))
    while b > 1 and cols % b:
        b //= 2
    assert b >= 2, f"cols={cols} admits no usable Hadamard block"
    return b


def find_mlp_targets(model):
    """[(name, module, block)] for every MLP projection, in forward order."""
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    out = []
    for i, layer in enumerate(lm.layers):
        if not hasattr(layer, "mlp"):
            continue
        for t in TYPES:
            mod = getattr(layer.mlp, t, None)
            if mod is not None and hasattr(mod, "weight"):
                out.append((f"L{i}.{t}", mod, largest_pow2_block(mod.weight.shape[1])))
    return out


@torch.no_grad()
def capture_hessians(model, targets, blocks, batch=4, device="cuda"):
    """H = E[x x^T] per target module, accumulated in float64."""
    acc, n, hooks = {}, {}, []

    def mk(name):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = (x.T @ x).double()
            acc[name] = g if name not in acc else acc[name] + g
            n[name] = n.get(name, 0) + x.shape[0]
        return hook

    for name, mod, _ in targets:
        hooks.append(mod.register_forward_pre_hook(mk(name)))
    try:
        for j in range(0, len(blocks), batch):
            x = torch.tensor(blocks[j : j + batch], device=device)
            model(x, use_cache=False)
            del x
    finally:
        for h in hooks:
            h.remove()
    return {k: (v / max(n[k], 1)).float() for k, v in acc.items()}, n


ARMS = [
    # name,                    dim, group, gptq,  pertype
    ("mlp_scalar3_rot",          1,   128, False, False),
    ("mlp_scalar3_rot_gptq",     1,   128, True,  False),
    ("mlp_vq4_rot_gptq",         4,   128, True,  False),
    ("mlp_vq8_rot_gptq",         8,   128, True,  False),
    ("mlp_vq8_rot_gptq_pertype", 8,   128, True,  True),
    ("mlp_scalar3_g64_rot_gptq", 1,    64, True,  False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/mlp_task_v1")
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=7200)
    ap.add_argument("--only", default="")
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
    if a.max_blocks:
        blocks = blocks[: a.max_blocks]
    n_targets = sum(len(b) - 1 for b in blocks)

    tr = tok("\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()[:4000]),
             add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = find_mlp_targets(model)
    params = {nm: mod.weight for nm, mod, _ in targets}
    blk_of = {nm: b for nm, _, b in targets}
    type_of = {nm: nm.split(".")[1] for nm, _, _ in targets}
    originals = {nm: w.detach().clone().cpu() for nm, w in params.items()}
    n_mlp = sum(w.numel() for w in originals.values())
    total_params = sum(p.numel() for p in model.parameters())

    print(json.dumps({"eval_blocks": len(blocks), "eval_targets": n_targets,
                      "cal_tokens": len(cal) * a.cal_len, "mlp_tensors": len(targets),
                      "mlp_params": n_mlp, "mlp_share": n_mlp / total_params,
                      "blocks_used": sorted({b for _, _, b in targets})}), flush=True)

    def restore():
        with torch.no_grad():
            for nm, w in params.items():
                w.copy_(originals[nm].to(w.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "targets": bf["target_tokens"],
                      "t": B.stamp()}), flush=True)

    # one independent sign per column; the Hadamard block only sets how the
    # columns are grouped for the transform
    signs = {nm: rot_signs(originals[nm].shape[1], "cuda") for nm in params}

    t0 = time.monotonic()
    Hraw, ntok = capture_hessians(model, targets, cal)
    gate_up_identical = None
    Hrot = {}
    for nm in params:
        Hrot[nm] = rotate_hessian(Hraw[nm], signs[nm], blk_of[nm])
    # gate and up read the same tensor, so their Hessians must agree
    g0, u0 = "L0.gate_proj", "L0.up_proj"
    if g0 in Hraw and u0 in Hraw:
        rel = (Hraw[g0] - Hraw[u0]).abs().max() / Hraw[g0].abs().max()
        gate_up_identical = float(rel)
        assert rel < 1e-5, f"gate/up Hessians differ ({rel:.2e}) -- capture is wrong"
    del Hraw
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "hessians", "seconds": time.monotonic() - t0,
                      "gate_up_rel_diff": gate_up_identical}), flush=True)

    res = {"plan": {"model": a.model, "mlp_tensors": len(targets), "mlp_params": n_mlp,
                    "mlp_share": n_mlp / total_params, "eval_targets": n_targets,
                    "protocol": "MLP only, in_proj_qkv left at BF16; v4 recipe"},
           "bf16": {"nll": bf["nll"], "target_tokens": bf["target_tokens"]}, "arms": {}}
    want = set(a.only.split(",")) if a.only else None

    for name, dim, group, gptq, pertype in ARMS:
        if want and name not in want:
            continue
        if not B.check(name, 300):
            break
        restore()
        t0 = time.monotonic()
        k = 3 ** dim

        prep = {nm: rotate(originals[nm].cuda().float(), signs[nm], blk_of[nm]) for nm in params}

        def fit(names):
            pool = torch.cat([gnorm_fp16(prep[nm], group)[0].reshape(-1, dim)[::37]
                              for nm in names], 0)
            lv = lloyd_scalar(pool.flatten(), 3) if dim == 1 else kmeans(pool, k)
            del pool
            return lv.to(torch.float16).float()

        if pertype:
            books = {t: fit([nm for nm in params if type_of[nm] == t]) for t in TYPES}
            cb_bytes = sum(b.numel() * 2 for b in books.values())
        else:
            shared = fit(list(params))
            books = {t: shared for t in TYPES}
            cb_bytes = shared.numel() * 2
        torch.cuda.empty_cache()

        packs, maxdiff, wse, wn, hse = [], 0.0, 0.0, 0, 0.0
        for nm, _, _ in targets:  # forward order
            w = prep[nm]
            rows, cols = w.shape
            lv = books[type_of[nm]]
            hi = cholesky_inv_upper(Hrot[nm]) if gptq else None
            idx, sc, ref = quantize_tensor(w, lv, dim, group, hinv=hi)
            del hi
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, lv, "cuda")
            maxdiff = max(maxdiff, (dq - ref).abs().max().item())
            d = dq - w
            wse += d.square().double().sum().item()
            hse += ((d @ Hrot[nm]) * d).double().sum().item()
            wn += rows * cols
            params[nm].copy_(unrotate(dq, signs[nm], blk_of[nm]).to(params[nm].dtype))
            packs.append(p)
            del w, d, dq, ref, idx, sc
            prep[nm] = None
        assert maxdiff == 0.0, f"decoded weights differ from quantized: {maxdiff}"
        prep.clear()
        torch.cuda.empty_cache()

        bpw = bits_per_weight(packs, cb_bytes)
        total_bytes = sum(p.payload_bytes() for p in packs) + cb_bytes
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev, bf)
        rec = {"nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi],
               "bits_per_weight": bpw, "stored_bytes": total_bytes,
               "codebook_bytes": cb_bytes, "decode_exact": maxdiff == 0.0,
               "weight_mse": wse / wn, "act_weighted_err": hse / wn,
               "dim": dim, "group": group, "gptq": gptq, "pertype": pertype,
               "quantize_seconds": time.monotonic() - t0}
        res["arms"][name] = {**rec, "_ev": ev}
        print(json.dumps({"arm": name, **rec}), flush=True)
        atomic_json(out / "results.json",
                    {**res, "arms": {x: {p2: v2 for p2, v2 in y.items() if p2 != "_ev"}
                                     for x, y in res["arms"].items()}})

    refs = ("mlp_scalar3_rot", "mlp_scalar3_rot_gptq", "mlp_scalar3_g64_rot_gptq",
            "mlp_vq8_rot_gptq")
    comps = {}
    for m in res["arms"]:
        for ref in refs:
            if m == ref or ref not in res["arms"]:
                continue
            d, lo, hi = paired_bootstrap(res["arms"][m]["_ev"], res["arms"][ref]["_ev"])
            comps[f"{m}_vs_{ref}"] = {
                "delta": d, "ci": [lo, hi],
                "byte_ratio": res["arms"][m]["stored_bytes"] / res["arms"][ref]["stored_bytes"]}
    for v in res["arms"].values():
        v.pop("_ev", None)
    res["comparisons"] = comps
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# MLP feasibility: does the ternary VQ recipe transfer off the projections?", "",
         f"Qwen3.5-0.8B, all {len(targets)} MLP tensors ({n_mlp:,} params, "
         f"{n_mlp / total_params * 100:.1f}% of the model); `in_proj_qkv` left at BF16.",
         f"**{n_targets:,} validation targets**, token-weighted. "
         f"Calibration: {len(cal) * a.cal_len:,} tokens from the train split.",
         f"BF16 NLL {bf['nll']:.6f}. `down_proj` uses a 512-element Hadamard block "
         "(3584 is not a multiple of 1024).", "",
         "| Arm | bits/wt | weight MSE | act-weighted err | NLL | ΔNLL vs BF16 | 95% CI |",
         "|---|---:|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['bits_per_weight']:.4f} | {r['weight_mse']:.4e} "
                 f"| {r['act_weighted_err']:.4e} | {r['nll']:.6f} | {r['delta_nll']:+.6f} "
                 f"| [{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |")
    L += ["", "## Head-to-head (byte ratio < 1 means the first arm stores fewer bytes)", "",
          "| Comparison | ΔNLL | 95% CI | byte ratio | |", "|---|---:|:--|---:|:--|"]
    for kk, c in comps.items():
        ex = "excludes 0" if (c["ci"][0] > 0) == (c["ci"][1] > 0) else "**includes 0**"
        L.append(f"| {kk.replace('_vs_', ' vs ')} | {c['delta']:+.6f} "
                 f"| [{c['ci'][0]:+.6f}, {c['ci'][1]:+.6f}] | {c['byte_ratio']:.4f} | {ex} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
