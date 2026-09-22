"""Mixed-precision rate allocation, driven by a measured per-tensor proxy.

Section 8 measured that MLP weights are 2.79x more damaging per parameter than
the DeltaNet input projections at ternary rate. That is a direct instruction
about where to spend bits, and this run turns it into a procedure.

The question is not "which family should get more bits" -- that framing cannot
be budget-matched, because the families have very different sizes. It is:

    given a fixed number of extra bytes, which tensors should receive them?

**Budget.** The extra bytes available are exactly what it would cost to promote
every `in_proj_qkv` tensor from the low rate to the high rate. Every allocation
arm spends that same budget, so the comparison is byte-matched by construction
and `promote_proj` is simply one of the candidate answers rather than the
reference.

**Rates.** Both rates are dimension-4 vector codes, so promotion changes the
*rate* and nothing else:

    low   K=81   -> 10 codes/uint64 -> 40 weights / 8 B = 1.600 bits/weight
    high  K=243  ->  8 codes/uint64 -> 32 weights / 8 B = 2.000 bits/weight

Using dimension-8 for the low rate and dimension-4 for the high rate would
confound rate with dimension, so the main ladder avoids it. Two extra arms then
check that the dimension gain from section 5c and the allocation gain compose.

**The proxy.** Ranking tensors by measured NLL damage would need 90 separate
evaluations. Instead each tensor is quantized at both rates once and scored by
the activation-weighted error tr(dW H dW^T) -- summed, not averaged, since what
matters is a tensor's absolute contribution to output error. Section 5c found
this quantity tracks NLL ordering across recipe families where plain weight MSE
inverts. Tensors are then promoted greedily by

    (hErr_low - hErr_high) / (bytes_high - bytes_low),

i.e. error removed per extra byte. Whether the proxy actually earns its keep is
tested against a random allocation of the identical budget, which is the
control that makes a positive result mean something.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import bits_per_weight, codes_per_word, decode, encode_vq, gnorm_fp16
from run_rate_sweep import kmeans
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_mlp_task_v1 import capture_hessians, find_mlp_targets, largest_pow2_block
from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

LOW = (4, 81)     # 1.600 bits/weight of index
HIGH = (4, 243)   # 2.000 bits/weight of index
D8 = (8, 6561)    # 1.600 bits/weight of index, the section 5c codec


def index_bpw(dim, k):
    """Bits per weight of index under the multi-symbol packing."""
    return 64.0 / (codes_per_word(k) * dim)


def collect_targets(model):
    """[(name, module, block)] for in_proj_qkv and every MLP projection."""
    out = []
    for i in find_layers(model):
        m = get_module(model, i)
        out.append((f"L{i}.in_proj_qkv", m, largest_pow2_block(m.weight.shape[1])))
    out.extend(find_mlp_targets(model))
    return out


def fit_book(prep, names, dim, k, group, stride=37):
    pool = torch.cat([gnorm_fp16(prep[nm], group)[0].reshape(-1, dim)[::stride]
                      for nm in names], 0)
    lv = kmeans(pool, k)
    del pool
    torch.cuda.empty_cache()
    return lv.to(torch.float16).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/mixed_alloc_v1")
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=9000)
    ap.add_argument("--only", default="")
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
    targets = collect_targets(model)
    params = {nm: m.weight for nm, m, _ in targets}
    blk_of = {nm: b for nm, _, b in targets}
    fam_of = {nm: ("in_proj_qkv" if "in_proj_qkv" in nm else nm.split(".")[1])
              for nm, _, _ in targets}
    originals = {nm: w.detach().clone().cpu() for nm, w in params.items()}
    numel = {nm: w.numel() for nm, w in originals.items()}
    n_all = sum(numel.values())
    total_params = sum(p.numel() for p in model.parameters())
    proj_names = [nm for nm in params if fam_of[nm] == "in_proj_qkv"]

    print(json.dumps({"eval_targets": n_targets, "tensors": len(targets),
                      "target_params": n_all, "share": n_all / total_params,
                      "low_bpw": index_bpw(*LOW), "high_bpw": index_bpw(*HIGH),
                      "d8_bpw": index_bpw(*D8)}), flush=True)

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
    print(json.dumps({"stage": "hessians", "seconds": time.monotonic() - t0}), flush=True)

    prep = {nm: rotate(originals[nm].cuda().float(), signs[nm], blk_of[nm]) for nm in params}
    hinv = {}  # cached: the Cholesky is rate-independent and is reused by every arm

    def quantize(nm, lv, dim, k):
        """Quantize one tensor at (dim, k); returns (pack, decoded, hErr sum)."""
        w = prep[nm]
        rows, cols = w.shape
        if nm not in hinv:
            hinv[nm] = cholesky_inv_upper(Hrot[nm])
        idx, sc, ref = quantize_tensor(w, lv, dim, group, hinv=hinv[nm])
        p = encode_vq(idx, sc, (rows, cols), group, dim, k)
        dq = decode(p, lv, "cuda")
        assert (dq - ref).abs().max().item() == 0.0, f"{nm}: decode mismatch"
        d = dq - w
        herr = ((d @ Hrot[nm]) * d).double().sum().item()
        del idx, sc, ref, d
        return p, dq, herr

    # ---- phase 1: per-tensor cost/benefit at both rates, no model evaluation ----
    t0 = time.monotonic()
    books = {LOW: fit_book(prep, list(params), *LOW, group),
             HIGH: fit_book(prep, list(params), *HIGH, group),
             D8: fit_book(prep, list(params), *D8, group)}
    stats = {}
    for nm, _, _ in targets:
        rec = {}
        for cfg in (LOW, HIGH, D8):
            p, dq, herr = quantize(nm, books[cfg], *cfg)
            rec[str(cfg)] = {"herr": herr, "bytes": p.payload_bytes()}
            del p, dq
        rec["family"] = fam_of[nm]
        rec["numel"] = numel[nm]
        stats[nm] = rec
    print(json.dumps({"stage": "sensitivity", "seconds": time.monotonic() - t0}), flush=True)
    atomic_json(out / "sensitivity.json", stats)

    lo, hi = str(LOW), str(HIGH)
    gain = {nm: stats[nm][lo]["herr"] - stats[nm][hi]["herr"] for nm in params}
    cost = {nm: stats[nm][hi]["bytes"] - stats[nm][lo]["bytes"] for nm in params}
    budget = sum(cost[nm] for nm in proj_names)

    def greedy(pool_names, bud):
        order = sorted(pool_names, key=lambda n: gain[n] / max(cost[n], 1), reverse=True)
        chosen, spent = [], 0
        for n in order:
            if spent + cost[n] <= bud:
                chosen.append(n)
                spent += cost[n]
        return chosen, spent

    mlp_names = [nm for nm in params if fam_of[nm] != "in_proj_qkv"]
    g_mlp, s_mlp = greedy(mlp_names, budget)
    g_all, s_all = greedy(list(params), budget)
    rng = np.random.default_rng(11)
    perm = list(params)
    rng.shuffle(perm)
    r_sel, r_spent = [], 0
    for n in perm:
        if r_spent + cost[n] <= budget:
            r_sel.append(n)
            r_spent += cost[n]

    allocs = {
        "all_low":            ({nm: LOW for nm in params}, LOW),
        "promote_proj":       ({nm: (HIGH if nm in proj_names else LOW) for nm in params}, LOW),
        "promote_greedy_mlp": ({nm: (HIGH if nm in g_mlp else LOW) for nm in params}, LOW),
        "promote_greedy_all": ({nm: (HIGH if nm in g_all else LOW) for nm in params}, LOW),
        "promote_random":     ({nm: (HIGH if nm in r_sel else LOW) for nm in params}, LOW),
        "all_high":           ({nm: HIGH for nm in params}, HIGH),
        "all_low_d8":         ({nm: D8 for nm in params}, D8),
        "greedy_all_d8_low":  ({nm: (HIGH if nm in g_all else D8) for nm in params}, D8),
        "greedy_mlp_d8_low":  ({nm: (HIGH if nm in g_mlp else D8) for nm in params}, D8),
    }
    plan = {"budget_bytes": budget,
            "spent": {"promote_proj": budget, "promote_greedy_mlp": s_mlp,
                      "promote_greedy_all": s_all, "promote_random": r_spent},
            "promoted_params": {k: sum(numel[n] for n, c in v[0].items() if c == HIGH)
                                for k, v in allocs.items()},
            "greedy_all_families": {f: sum(1 for n in g_all if fam_of[n] == f)
                                    for f in sorted(set(fam_of.values()))},
            "family_counts": {f: sum(1 for n in params if fam_of[n] == f)
                              for f in sorted(set(fam_of.values()))}}
    print(json.dumps({"stage": "plan", **plan}), flush=True)

    res = {"plan": {"model": a.model, "tensors": len(targets), "target_params": n_all,
                    "share": n_all / total_params, "eval_targets": n_targets,
                    "rates": {"low": LOW, "high": HIGH, "d8": D8}, **plan},
           "bf16": {"nll": bf["nll"]}, "arms": {}}

    want = set(a.only.split(",")) if a.only else None
    for name, (cfgs, _) in allocs.items():
        if want and name not in want:
            continue
        if not B.check(name, 300):
            break
        restore()
        t0 = time.monotonic()
        packs, used = [], set()
        for nm, _, _ in targets:
            cfg = cfgs[nm]
            used.add(cfg)
            p, dq, _ = quantize(nm, books[cfg], *cfg)
            params[nm].copy_(unrotate(dq, signs[nm], blk_of[nm]).to(params[nm].dtype))
            packs.append(p)
            del dq
        torch.cuda.empty_cache()
        cb = sum(books[c].numel() * 2 for c in used)
        bpw = bits_per_weight(packs, cb)
        total_bytes = sum(p.payload_bytes() for p in packs) + cb
        ev = evaluate(model, blocks)
        d, l, h = paired_bootstrap(ev, bf)
        rec = {"nll": ev["nll"], "delta_nll": d, "delta_ci": [l, h],
               "bits_per_weight": bpw, "stored_bytes": total_bytes, "codebook_bytes": cb,
               "promoted_params": plan["promoted_params"][name],
               "quantize_seconds": time.monotonic() - t0}
        res["arms"][name] = {**rec, "_ev": ev}
        print(json.dumps({"arm": name, **rec}), flush=True)
        atomic_json(out / "results.json",
                    {**res, "arms": {x: {p2: v2 for p2, v2 in y.items() if p2 != "_ev"}
                                     for x, y in res["arms"].items()}})

    refs = ("all_low", "promote_proj", "promote_random", "all_low_d8", "promote_greedy_mlp")
    comps = {}
    for m in res["arms"]:
        for ref in refs:
            if m == ref or ref not in res["arms"]:
                continue
            d, l, h = paired_bootstrap(res["arms"][m]["_ev"], res["arms"][ref]["_ev"])
            comps[f"{m}_vs_{ref}"] = {
                "delta": d, "ci": [l, h],
                "byte_ratio": res["arms"][m]["stored_bytes"] / res["arms"][ref]["stored_bytes"]}
    for v in res["arms"].values():
        v.pop("_ev", None)
    res["comparisons"] = comps
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Mixed-precision allocation under a fixed extra-byte budget", "",
         f"Qwen3.5-0.8B, {len(targets)} tensors (`in_proj_qkv` + all MLP), "
         f"{n_all:,} params = {n_all / total_params * 100:.1f}% of the model.",
         f"**{n_targets:,} validation targets**, token-weighted. BF16 NLL {bf['nll']:.6f}.",
         f"Low rate dim-4 K=81 ({index_bpw(*LOW):.3f} bits of index), high rate dim-4 "
         f"K=243 ({index_bpw(*HIGH):.3f}).",
         f"Every promotion arm spends the same **{budget:,} extra bytes** — exactly what",
         "promoting all 18 projections costs.", "",
         "| Arm | bits/wt | stored MB | promoted params | NLL | ΔNLL vs BF16 | 95% CI |",
         "|---|---:|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['bits_per_weight']:.4f} | {r['stored_bytes'] / 1e6:.2f} "
                 f"| {r['promoted_params'] / 1e6:.1f}M | {r['nll']:.6f} "
                 f"| {r['delta_nll']:+.6f} "
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
