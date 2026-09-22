"""Step 1, corrected: ternary-rate codec comparison from real stored bytes.

Three corrections to `run_ternary_task.py`:

 1. **Storage is measured, not derived.** Every arm serializes to packed bytes
    (5 trits/byte; K^c codes per uint64) and bits/weight is computed from
    `len(bytes)`. The previous run used log2(3)=1.585, which no implementation
    can hit per-symbol.
 2. **The evaluated weights are decoded from those bytes.** Previously the model
    was loaded with the in-memory float quantization, so the byte count
    described a representation that was never actually tested. Scales are
    rounded to FP16 *before* quantizing, which makes decode(encode(w)) exact.
 3. **All 261,284 validation targets**, including the ragged final block of 36
    that 2041 whole blocks omit. NLL is token-weighted, so the short block
    cannot count as much as a full one.

Byte protocol: the vector codec must not spend more than the scalar control it
is compared against. Packed at 1.600 bits/weight for indices, the two formats
are identical to within 4 bytes per tensor; the codebook then puts dim-8 VQ
0.43% above scalar g128, so `scalar_ternary_g64_rot` is included as a control
that spends *more* than the VQ arm.
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

from packed_lowrate import bits_per_weight, decode, encode_scalar3, encode_vq, gnorm_fp16
from run_rate_sweep import BLOCK, assign, fwht, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module


@torch.no_grad()
def kmeans_weighted(pts, w, k, iters=50, seed=0, sub=500_000):
    """Lloyd with per-point weights.

    A normalized vector's contribution to *weight-space* MSE is scaled by its
    group scale squared, so fitting the codebook unweighted optimizes the wrong
    objective: groups with large scales are under-served. Weighting by scale^2
    makes the fit match the error we actually report.
    """
    g = torch.Generator(device=pts.device).manual_seed(seed)
    n = pts.shape[0]
    sel = torch.randperm(n, generator=g, device=pts.device)[: min(n, sub)]
    s, sw = pts[sel], w[sel].double()
    cent = s[torch.randperm(s.shape[0], generator=g, device=pts.device)[:k]].clone()
    d = pts.shape[1]
    for it in range(iters):
        idx, _ = assign(s, cent)
        num = torch.zeros(k, d, device=pts.device, dtype=torch.float64)
        den = torch.zeros(k, device=pts.device, dtype=torch.float64)
        num.index_add_(0, idx, s.double() * sw.unsqueeze(1))
        den.index_add_(0, idx, sw)
        alive = den > 0
        new = cent.double().clone()
        new[alive] = num[alive] / den[alive].unsqueeze(1)
        cent = new.float()
        dead = (~alive).nonzero(as_tuple=True)[0]
        if dead.numel() and it < iters - 1:
            err = (s - cent[idx]).square().sum(1) * sw.float()
            worst = err.topk(min(dead.numel(), s.shape[0])).indices
            cent[dead[: worst.numel()]] = s[worst]
    return cent


def rot_signs(cols, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randint(0, 2, (cols,), generator=g, device=device) * 2 - 1).float()


def rotate(w, s):
    r, c = w.shape
    return fwht((w * s).view(r, c // BLOCK, BLOCK)).view(r, c)


def unrotate(w, s):
    r, c = w.shape
    return fwht(w.view(r, c // BLOCK, BLOCK)).view(r, c) * s


@torch.no_grad()
def evaluate(model, blocks, batch=8, device="cuda"):
    """Token-weighted NLL plus per-block (sum, count) for a weighted bootstrap."""
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
    """Weighted paired bootstrap on (sum, count) blocks; handles the ragged block."""
    rng = np.random.default_rng(seed)
    sa, ca = np.asarray(a["block_sums"]), np.asarray(a["block_counts"], dtype=float)
    sb = np.asarray(b["block_sums"])
    n = len(sa)
    idx = rng.integers(0, n, size=(reps, n))
    d = (sa[idx].sum(1) / ca[idx].sum(1)) - (sb[idx].sum(1) / ca[idx].sum(1))
    point = sa.sum() / ca.sum() - sb.sum() / ca.sum()
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/ternary_task_v2")
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=7200)
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
    if len(tail) >= 2:  # the ragged remainder the previous run dropped
        blocks.append(tail)
    if a.max_blocks:
        blocks = blocks[: a.max_blocks]
    n_targets = sum(len(b) - 1 for b in blocks)
    print(json.dumps({"blocks": len(blocks), "targets": n_targets}), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    restore()
    bf = evaluate(model, blocks)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "targets": bf["target_tokens"],
                      "t": B.stamp()}), flush=True)
    res = {"plan": {"model": a.model, "layers": layers, "targets": n_targets,
                    "protocol": "weights decoded from packed bytes; bits/weight measured; "
                                "token-weighted NLL over the complete stream"},
           "bf16": {"nll": bf["nll"], "target_tokens": bf["target_tokens"]}, "arms": {}}

    ARMS = [
        ("scalar_ternary_g128",     dict(rot=False, dim=1, group=128)),
        ("scalar_ternary_g128_rot", dict(rot=True,  dim=1, group=128)),
        ("scalar_ternary_g64_rot",  dict(rot=True,  dim=1, group=64)),
        ("vq_dim4_g128_rot",        dict(rot=True,  dim=4, group=128)),
        ("vq_dim8_g128_rot",        dict(rot=True,  dim=8, group=128)),
        ("vq_dim8_g128_norot",      dict(rot=False, dim=8, group=128)),
        ("vq_dim4_g128_rot_w",      dict(rot=True,  dim=4, group=128, wfit=True)),
        ("vq_dim8_g128_rot_w",      dict(rot=True,  dim=8, group=128, wfit=True)),
    ]

    for name, cfg in ARMS:
        if not B.check(name, 240):
            break
        restore()
        t0 = time.monotonic()
        dim, group, rot = cfg["dim"], cfg["group"], cfg["rot"]
        k = 3 ** dim
        signs = {kk: rot_signs(originals[kk].shape[1], "cuda") for kk in params} if rot else None

        prep = {}
        for kk in params:
            w = originals[kk].cuda().float()
            if rot:
                w = rotate(w, signs[kk])
            prep[kk] = gnorm_fp16(w, group)  # scale rounded to FP16 before quantizing
            del w

        # one shared codebook for all 18 tensors, charged once
        if dim == 1:
            levels = lloyd_scalar(torch.cat([prep[kk][0].flatten()[::17] for kk in params]), 3)
            cb_bytes = levels.numel() * 2
        else:
            pool = torch.cat([prep[kk][0].reshape(-1, dim)[::17] for kk in params], 0)
            if cfg.get("wfit"):
                # each vector's weight-space error is scaled by its group scale^2
                pw = torch.cat([
                    prep[kk][1].view(-1, 1).expand(-1, group // dim).reshape(-1)[::17]
                    for kk in params], 0).square()
                levels = kmeans_weighted(pool, pw[: pool.shape[0]], k)
                del pw
            else:
                levels = kmeans(pool, k)
            cb_bytes = levels.numel() * 2
            del pool
        levels = levels.to(torch.float16).float()  # stored precision == evaluated precision

        packs, maxdiff = [], 0.0
        wse, wn = 0.0, 0  # weight-space MSE, to tell a fitting bug from a real effect
        with torch.no_grad():
            for kk in params:
                u, sc = prep[kk]
                rows, cols = u.shape
                if dim == 1:
                    mid = (levels[1:] + levels[:-1]) * 0.5
                    idx = torch.bucketize(u.contiguous(), mid)
                    q = (levels[idx].view(rows, cols // group, group) * sc).view(rows, cols)
                    p = encode_scalar3(idx, sc, (rows, cols), group)
                else:
                    pts = u.reshape(-1, dim)
                    ci = torch.empty(pts.shape[0], dtype=torch.long, device=pts.device)
                    for i in range(0, pts.shape[0], 2_000_000):
                        ci[i : i + 2_000_000], _ = assign(pts[i : i + 2_000_000], levels)
                    q = (levels[ci].view(rows, cols // group, group) * sc).view(rows, cols)
                    p = encode_vq(ci, sc, (rows, cols), group, dim, k)
                    del pts, ci
                # the decisive check: the model gets weights read back from bytes
                dq = decode(p, levels, "cuda")
                maxdiff = max(maxdiff, (dq - q).abs().max().item())
                wse += (dq - u.view(rows, cols // group, group).mul(sc).view(rows, cols)
                        ).square().double().sum().item()
                wn += rows * cols
                if rot:
                    dq = unrotate(dq, signs[kk])
                params[kk].copy_(dq.to(params[kk].dtype))
                packs.append(p)
                del q, dq
            prep.clear()
        assert maxdiff == 0.0, f"decoded weights differ from quantized: {maxdiff}"
        torch.cuda.empty_cache()

        bpw = bits_per_weight(packs, cb_bytes)
        total_bytes = sum(p.payload_bytes() for p in packs) + cb_bytes
        ev = evaluate(model, blocks)
        d, lo, hi = paired_bootstrap(ev, bf)
        rec = {"nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi],
               "bits_per_weight": bpw, "stored_bytes": total_bytes,
               "codebook_bytes": cb_bytes, "decode_exact": maxdiff == 0.0,
               "weight_mse": wse / wn,
               "quantize_seconds": time.monotonic() - t0}
        res["arms"][name] = {**rec, "_ev": ev}
        print(json.dumps({"arm": name, **rec}), flush=True)
        atomic_json(out / "results.json",
                    {**res, "arms": {x: {p2: v2 for p2, v2 in y.items() if p2 != "_ev"}
                                     for x, y in res["arms"].items()}})

    comps = {}
    for m in res["arms"]:
        for ref in ("scalar_ternary_g128_rot", "scalar_ternary_g64_rot"):
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

    L = ["# Step 1 (corrected): ternary rate, weights decoded from stored bytes", "",
         f"Qwen3.5-0.8B, all {len(layers)} DeltaNet input projections, "
         f"**{n_targets:,} validation targets** (complete stream, token-weighted).",
         f"BF16 NLL {bf['nll']:.6f}. `bits/wt` is measured from real packed bytes; every arm",
         "asserts decode(encode(w)) is bit-exact.", "",
         "| Arm | bits/wt | stored MB | NLL | ΔNLL vs BF16 | 95% CI |",
         "|---|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['bits_per_weight']:.4f} | {r['stored_bytes'] / 1e6:.2f} "
                 f"| {r['nll']:.6f} | {r['delta_nll']:+.6f} "
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
