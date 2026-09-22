"""Step 2: does the ternary VQ margin survive a real quantization recipe?

v3 compared codecs under a deliberately stripped recipe: nearest-neighbour
assignment against a codebook fitted to the raw weights, no use of calibration
activations at all. That is a fair codec comparison but not a competitive
pipeline, and the honest worry is that a stronger recipe closes the gap -- that
vector quantization is only buying back error that error compensation would
have removed anyway.

This run adds the two things every serious low-rate pipeline has and v3 had
neither of:

  * **Activation weighting.** The quantity that matters is not |W - W'| but
    |(W - W')x| over the calibration distribution. Per target tensor we
    accumulate H = E[x x^T] and use it both to fit the codebook (per-coordinate
    weights h = diag H) and to weight the assignment.
  * **GPTQ-style error compensation.** Quantize in column order and push each
    block's residual onto the not-yet-quantized columns through the inverse
    Cholesky factor of H.

Generalizing GPTQ to a *vector* codec is the one piece that is not standard, so
it is worth stating exactly. For a block of columns J quantized jointly, the
cost charged by the GPTQ derivation is

    ||(W_J - Q_J) inv(U_JJ)||_F^2 ,   U = upper Cholesky factor of H^-1,

and the compensation applied to the remaining columns is

    W_rest -= (W_J - Q_J) inv(U_JJ) U_J,rest .

So the within-block decision is a *metric* nearest-neighbour search with
M = inv(U_JJ): transform both the points and the codebook by M and search as
usual. At d=1 this collapses to plain nearest-level with the error divided by
U_jj, i.e. exactly scalar GPTQ, which is the check that the generalization is
the right one.

The group scale is rounded to FP16 at the moment its group is reached, so
decode(encode(w)) stays bit-exact under compensation; every arm still asserts
it. Storage is untouched by both additions -- they change *which* codes are
chosen, never the format -- so bits/weight is identical to v3 by construction
and the comparison stays byte-for-byte.

Calibration uses the wikitext2 *train* split; evaluation uses the full
validation stream, so no activation statistic is fitted on the evaluation data.

Static vs sequential H: statistics are captured once from the BF16 model rather
than re-captured after each layer is quantized. The sequential variant is the
textbook choice, and rather than assume the difference is negligible this run
measures it directly (arm `vq8_rot_gptq_seq`), which is the same question the
retired SCF arc answered for calibration statistics generally.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import (bits_per_weight, decode, encode_scalar3, encode_vq,
                            gnorm_fp16)
from run_rate_sweep import BLOCK, assign, fwht, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module

TINY = float(torch.finfo(torch.float16).tiny)


# --------------------------------------------------------------------------
# rotation (identical to v3, repeated here so the arms reproduce exactly)
# --------------------------------------------------------------------------

def rot_signs(cols, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randint(0, 2, (cols,), generator=g, device=device) * 2 - 1).float()


def rotate(w, s, blk=BLOCK):
    r, c = w.shape
    return fwht((w * s).view(r, c // blk, blk)).view(r, c)


def unrotate(w, s, blk=BLOCK):
    r, c = w.shape
    return fwht(w.view(r, c // blk, blk)).view(r, c) * s


def rotate_hessian(H, s, blk=BLOCK):
    """H in the rotated basis: A H A^T with A = Hadamard @ diag(s).

    rotate(M, s) right-multiplies by A^T, so applying it, transposing, and
    applying it again gives A H A^T. Trace is invariant, which is asserted.
    """
    tr0 = torch.diagonal(H).sum().item()
    Hr = rotate(rotate(H, s, blk).T.contiguous(), s, blk)
    Hr = (Hr + Hr.T) * 0.5
    tr1 = torch.diagonal(Hr).sum().item()
    assert abs(tr1 - tr0) <= 1e-4 * max(1.0, abs(tr0)), f"rotation changed trace: {tr0} -> {tr1}"
    return Hr


# --------------------------------------------------------------------------
# calibration statistics
# --------------------------------------------------------------------------

@torch.no_grad()
def capture_hessians(model, layers, blocks, batch=4, device="cuda", only=None):
    """H = E[x x^T] for the input of each target projection.

    Accumulated in float64: the sum runs over ~10^5 tokens and the Cholesky
    downstream is sensitive to the tail of the spectrum.
    """
    want = layers if only is None else [only]
    acc = {i: None for i in want}
    n = {i: 0 for i in want}
    hooks = []

    def mk(i):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = (x.T @ x).double()
            acc[i] = g if acc[i] is None else acc[i] + g
            n[i] += x.shape[0]
        return hook

    for i in want:
        hooks.append(get_module(model, i).register_forward_pre_hook(mk(i)))
    try:
        for j in range(0, len(blocks), batch):
            x = torch.tensor(blocks[j : j + batch], device=device)
            model(x, use_cache=False)
            del x
    finally:
        for h in hooks:
            h.remove()
    return {i: (acc[i] / max(n[i], 1)).float() for i in want}, n


def cholesky_inv_upper(H, percdamp=0.01):
    """Upper Cholesky factor U of H^-1 (the GPTQ `Hinv`), with damping."""
    d = H.shape[0]
    H = H.double().clone()
    diag = torch.diagonal(H)
    dead = diag <= 0
    if dead.any():
        H[dead, dead] = 1.0
    H += torch.eye(d, device=H.device, dtype=H.dtype) * (percdamp * diag.mean())
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = (Hinv + Hinv.T) * 0.5
    return torch.linalg.cholesky(Hinv, upper=True).float()


# --------------------------------------------------------------------------
# weighted codebook fitting (per-coordinate weights)
# --------------------------------------------------------------------------

@torch.no_grad()
def wassign(pts, wts, cent, chunk=100_000):
    """Nearest centroid under a per-point diagonal metric.

    sum_t w_t (p_t - c_t)^2 = const - 2 (w*p).c + w.(c^2), so it is two matmuls
    rather than one and costs no more asymptotically than plain assignment.
    """
    c2 = cent.square()
    idx = torch.empty(pts.shape[0], dtype=torch.long, device=pts.device)
    for i in range(0, pts.shape[0], chunk):
        p, w = pts[i : i + chunk], wts[i : i + chunk]
        d = (w * p) @ cent.T * (-2.0) + w @ c2.T
        idx[i : i + chunk] = d.argmin(1)
        del d
    return idx


@torch.no_grad()
def kmeans_wcoord(pts, wts, k, iters=50, seed=0, sub=500_000):
    """Lloyd with per-coordinate weights; empty clusters respawn on worst error."""
    g = torch.Generator(device=pts.device).manual_seed(seed)
    n, d = pts.shape
    sel = torch.randperm(n, generator=g, device=pts.device)[: min(n, sub)]
    s, sw = pts[sel], wts[sel]
    cent = s[torch.randperm(s.shape[0], generator=g, device=pts.device)[:k]].clone()
    for it in range(iters):
        idx = wassign(s, sw, cent)
        num = torch.zeros(k, d, device=pts.device, dtype=torch.float64)
        den = torch.zeros(k, d, device=pts.device, dtype=torch.float64)
        num.index_add_(0, idx, (sw * s).double())
        den.index_add_(0, idx, sw.double())
        new = torch.where(den > 0, num / den.clamp_min(1e-30), cent.double()).float()
        cent = new
        alive = (den > 0).any(1)
        dead = (~alive).nonzero(as_tuple=True)[0]
        if dead.numel() and it < iters - 1:
            err = (sw * (s - cent[idx]).square()).sum(1)
            worst = err.topk(min(dead.numel(), s.shape[0])).indices
            cent[dead[: worst.numel()]] = s[worst]
    return cent


# --------------------------------------------------------------------------
# the quantizer: one tensor, optional activation weighting and/or compensation
# --------------------------------------------------------------------------

@torch.no_grad()
def quantize_tensor(w_rot, levels, dim, group, hdiag=None, hinv=None):
    """Quantize one rotated tensor; returns (indices, fp16 scales, reconstruction).

    hdiag  -- per-column activation weight, enables metric assignment.
    hinv   -- upper Cholesky factor of H^-1, enables GPTQ error compensation
              (and supersedes hdiag as the assignment metric, since the block
              metric inv(U_JJ) already carries the diagonal).
    """
    rows, cols = w_rot.shape
    W = w_rot.clone()
    ng = cols // group
    scales = torch.empty(rows, ng, 1, device=W.device)
    idx = torch.empty(rows, cols // dim, dtype=torch.long, device=W.device)
    eye = torch.eye(dim, device=W.device) if dim > 1 else None

    for gi in range(ng):
        g0 = gi * group
        s = W[:, g0 : g0 + group].abs().amax(1, keepdim=True).to(torch.float16).float()
        s = s.clamp_min(TINY)
        scales[:, gi, 0] = s.squeeze(1)
        for j0 in range(g0, g0 + group, dim):
            j1 = j0 + dim
            u = W[:, j0:j1] / s
            # M is both the assignment metric and the compensation transform;
            # inv(U_JJ) from the GPTQ derivation, or diag(sqrt(h)) for the
            # diagonal-only variant.
            if hinv is not None:
                M = (torch.linalg.solve_triangular(hinv[j0:j1, j0:j1], eye, upper=True)
                     if dim > 1 else hinv[j0, j0].reciprocal().view(1, 1))
            elif hdiag is not None:
                M = torch.diag(hdiag[j0:j1].clamp_min(1e-12).sqrt())
            else:
                M = None
            if dim == 1:
                # a 1-D argmin is invariant to any positive scalar metric, so
                # the sorted-level fast path stays exact in every configuration
                mid = (levels[1:] + levels[:-1]) * 0.5
                c = torch.bucketize(u.squeeze(1).contiguous(), mid)
            elif M is None:
                c, _ = assign(u, levels)
            else:
                c, _ = assign(u @ M, levels @ M)
            q = (levels[c].view(rows, dim) if dim > 1 else levels[c].view(rows, 1)) * s
            idx[:, j0 // dim : j1 // dim] = c.view(rows, 1)
            if hinv is not None and j1 < cols:
                E = (W[:, j0:j1] - q) @ M
                W[:, j1:] -= E @ hinv[j0:j1, j1:]
            W[:, j0:j1] = q
    return idx, scales, W


# --------------------------------------------------------------------------
# evaluation (identical protocol to v3: token-weighted, complete stream)
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, blocks, batch=8, device="cuda"):
    sums, counts = [], []
    by_len: dict[int, list] = {}
    for b in blocks:
        by_len.setdefault(len(b), []).append(b)
    for ln, group in by_len.items():
        for i in range(0, len(group), batch):
            x = torch.tensor(group[i : i + batch], device=device)
            logits = model(x[:, :-1], use_cache=False).logits.float()
            y = x[:, 1:]
            tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                  reduction="none").reshape(x.shape[0], -1)
            sums.extend(tok.sum(1).cpu().tolist())
            counts.extend([y.shape[1]] * x.shape[0])
            del x, y, logits, tok
    s, c = np.asarray(sums), np.asarray(counts, dtype=float)
    return {"nll": float(s.sum() / c.sum()), "target_tokens": int(c.sum()),
            "block_sums": sums, "block_counts": counts}


def paired_bootstrap(a, b, reps=5000, seed=7):
    rng = np.random.default_rng(seed)
    sa, ca = np.asarray(a["block_sums"]), np.asarray(a["block_counts"], dtype=float)
    sb = np.asarray(b["block_sums"])
    n = len(sa)
    idx = rng.integers(0, n, size=(reps, n))
    d = (sa[idx].sum(1) / ca[idx].sum(1)) - (sb[idx].sum(1) / ca[idx].sum(1))
    point = sa.sum() / ca.sum() - sb.sum() / ca.sum()
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------------------

ARMS = [
    # name,                      dim, group, actw,  gptq,  seq
    ("scalar3_rot",                1,   128, False, False, False),
    ("scalar3_rot_actw",           1,   128, True,  False, False),
    ("scalar3_rot_gptq",           1,   128, False, True,  False),
    ("vq8_rot",                    8,   128, False, False, False),
    ("vq8_rot_actw",               8,   128, True,  False, False),
    ("vq8_rot_gptq",               8,   128, False, True,  False),
    ("vq8_rot_gptq_actwfit",       8,   128, True,  True,  False),
    ("vq4_rot_gptq",               4,   128, False, True,  False),
    ("scalar3_g64_rot_gptq",       1,    64, False, True,  False),
    ("vq8_rot_gptq_seq",           8,   128, False, True,  True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/ternary_task_v4")
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

    # evaluation stream: full validation split, ragged tail included
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

    # calibration stream: train split, disjoint from evaluation
    tr = tok("\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()[:4000]),
             add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]
    print(json.dumps({"eval_blocks": len(blocks), "eval_targets": n_targets,
                      "cal_blocks": len(cal), "cal_tokens": len(cal) * a.cal_len}), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}
    key_of = {f"L{i}": i for i in layers}

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "targets": bf["target_tokens"],
                      "t": B.stamp()}), flush=True)

    cols = originals[f"L{layers[0]}"].shape[1]
    assert cols == BLOCK, f"rotation assumes one Hadamard block, cols={cols}"
    signs = rot_signs(cols, "cuda")

    # static statistics, captured once from the BF16 model
    t0 = time.monotonic()
    Hstat, ntok = capture_hessians(model, layers, cal)
    Hrot = {f"L{i}": rotate_hessian(Hstat[i], signs) for i in layers}
    del Hstat
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "hessians", "tokens_per_layer": ntok[layers[0]],
                      "seconds": time.monotonic() - t0}), flush=True)

    res = {"plan": {"model": a.model, "layers": layers, "eval_targets": n_targets,
                    "cal_tokens": len(cal) * a.cal_len,
                    "protocol": "activation-weighted fit and GPTQ block compensation; "
                                "weights decoded from packed bytes; bits/weight measured; "
                                "calibration on train split, evaluation on full validation"},
           "bf16": {"nll": bf["nll"], "target_tokens": bf["target_tokens"]}, "arms": {}}

    want = set(a.only.split(",")) if a.only else None

    for name, dim, group, actw, gptq, seq in ARMS:
        if want and name not in want:
            continue
        if not B.check(name, 300):
            break
        restore()
        t0 = time.monotonic()
        k = 3 ** dim

        # rotated weights and, for the sequential arm, statistics recaptured
        # against already-quantized earlier layers
        H = dict(Hrot)
        prep = {}
        for kk in params:
            prep[kk] = rotate(originals[kk].cuda().float(), signs)

        # --- codebook, shared across all 18 tensors, charged once ---
        # Fitted in g128-normalized space, exactly as in v3, so the baseline
        # arms reproduce rather than merely resemble the previous run.
        pool = torch.cat([gnorm_fp16(prep[kk], group)[0].reshape(-1, dim)[::17]
                          for kk in params], 0)
        if actw:
            # per-coordinate activation weight, normalized per tensor so no
            # tensor dominates the shared fit through its activation scale
            wpool = torch.cat([
                H[kk].diagonal().clamp_min(1e-12)
                .div(H[kk].diagonal().clamp_min(1e-12).mean())
                .view(1, cols // dim, dim).expand(prep[kk].shape[0], -1, -1)
                .reshape(-1, dim)[::17] for kk in params], 0)
        if dim == 1:
            if actw:
                levels, _ = torch.sort(kmeans_wcoord(pool, wpool, 3).flatten())
            else:
                levels = lloyd_scalar(pool.flatten(), 3)
            cb_bytes = levels.numel() * 2
        else:
            levels = kmeans_wcoord(pool, wpool, k) if actw else kmeans(pool, k)
            cb_bytes = levels.numel() * 2
        del pool
        if actw:
            del wpool
        levels = levels.to(torch.float16).float()
        torch.cuda.empty_cache()

        # --- quantize, in forward layer order (matters only for `seq`) ---
        packs, maxdiff = [], 0.0
        wse, wn, hse = 0.0, 0, 0.0
        for kk in sorted(params, key=lambda x: key_of[x]):
            if seq:
                Hs, _ = capture_hessians(model, layers, cal, only=key_of[kk])
                H[kk] = rotate_hessian(Hs[key_of[kk]], signs)
                del Hs
            w = prep[kk]
            rows, _ = w.shape
            hd = H[kk].diagonal().contiguous() if actw else None
            hi = cholesky_inv_upper(H[kk]) if gptq else None
            idx, sc, ref = quantize_tensor(w, levels, dim, group, hdiag=hd, hinv=hi)
            del hi
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, levels, "cuda")
            # the decisive check: the model gets weights read back from bytes,
            # compared against the reconstruction the quantizer actually chose
            maxdiff = max(maxdiff, (dq - ref).abs().max().item())
            d = dq - w
            wse += d.square().double().sum().item()
            hse += ((d @ H[kk]) * d).double().sum().item()
            wn += rows * cols
            params[kk].copy_(unrotate(dq, signs).to(params[kk].dtype))
            packs.append(p)
            del w, d, dq, ref, idx, sc
            prep[kk] = None
        assert maxdiff == 0.0, f"decoded weights differ from quantized: {maxdiff}"
        prep.clear()
        del H
        torch.cuda.empty_cache()

        bpw = bits_per_weight(packs, cb_bytes)
        total_bytes = sum(p.payload_bytes() for p in packs) + cb_bytes
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev, bf)
        rec = {"nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi],
               "bits_per_weight": bpw, "stored_bytes": total_bytes,
               "codebook_bytes": cb_bytes, "decode_exact": maxdiff == 0.0,
               "weight_mse": wse / wn, "act_weighted_err": hse / wn,
               "dim": dim, "group": group, "actw": actw, "gptq": gptq, "seq": seq,
               "quantize_seconds": time.monotonic() - t0}
        res["arms"][name] = {**rec, "_ev": ev}
        print(json.dumps({"arm": name, **rec}), flush=True)
        atomic_json(out / "results.json",
                    {**res, "arms": {x: {p2: v2 for p2, v2 in y.items() if p2 != "_ev"}
                                     for x, y in res["arms"].items()}})

    refs = ("scalar3_rot", "scalar3_rot_gptq", "scalar3_g64_rot_gptq", "vq8_rot_gptq")
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

    L = ["# Step 2: ternary rate with activation weighting and error compensation", "",
         f"Qwen3.5-0.8B, all {len(layers)} DeltaNet input projections, "
         f"**{n_targets:,} validation targets** (complete stream, token-weighted).",
         f"Calibration: {len(cal) * a.cal_len:,} tokens from the *train* split.",
         f"BF16 NLL {bf['nll']:.6f}. Every arm is rotated, stores FP16 group scales,",
         "asserts decode(encode(w)) bit-exact, and measures bits/weight from real bytes.", "",
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
