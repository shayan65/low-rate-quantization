"""Does the codec ordering hold off the corpus and context it was measured on?

Every contrast in this project was evaluated on wikitext-2 validation in
128-token blocks with state reset at each block. Two things are therefore
untested. The validation text has guided many decisions, so it is no longer
held out in any meaningful sense. And a 128-token reset protocol cannot say
anything about how a quantized recurrent model behaves over long contexts,
which is where DeltaNet state actually accumulates.

§12 showed that refitting the quantizer moves these contrasts by more than the
evaluation bootstrap does. This run asks the complementary question with the
quantizer **held fixed**: one calibration draw, one seed, one set of codes, and
only the evaluation varied. That separates evaluation sensitivity from refit
sensitivity instead of confounding them.

Five evaluation sets:

  * `wt2_val_128`   -- the corpus and protocol everything else used; it should
                       reproduce §5c exactly, which makes it a check on the
                       harness rather than a result.
  * `wt2_test_128`  -- the wikitext-2 test split, never used anywhere in this
                       project. Independent text, same domain.
  * `tiny_128`      -- TinyStories validation. Different domain entirely:
                       simple narrative prose rather than encyclopedic text.
  * `wt2_test_512`  -- 4x the context, same text as `wt2_test_128`.
  * `wt2_test_2048` -- 16x the context.

Calibration always comes from wikitext-2 train, which is the realistic setting
(calibrate once, deploy on whatever arrives) and is what makes the TinyStories
column informative: it asks whether a wikitext-calibrated codec ordering
survives a domain shift.

Longer contexts mean fewer, larger blocks at roughly fixed token count, so the
block bootstrap has fewer units and its intervals widen. That is a property of
the protocol, not a finding, and the block counts are reported alongside.
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
from run_rate_sweep import BLOCK, kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_ternary_task_v4 import (capture_hessians, cholesky_inv_upper, evaluate,
                                 paired_bootstrap, quantize_tensor, rot_signs,
                                 rotate, rotate_hessian, unrotate)

ARMS = [
    ("scalar3", 1, 3, 128),
    ("scalar3_g64", 1, 3, 64),
    ("vq4", 4, 81, 128),
    ("vq8", 8, 6561, 128),
]
CONTRASTS = [("vq8", "scalar3"), ("vq8", "scalar3_g64"), ("vq8", "vq4"),
             ("scalar3_g64", "scalar3")]


def text_of(path):
    df = pd.read_parquet(path)
    col = "text" if "text" in df.columns else df.columns[0]
    return "\n".join(str(x) for x in df[col].tolist())


def make_blocks(ids, ctx, max_blocks=0):
    n = (len(ids) - 1) // ctx
    b = [ids[i * ctx : i * ctx + ctx + 1] for i in range(n)]
    tail = ids[n * ctx :]
    if len(tail) >= 2:
        b.append(tail)
    return b[:max_blocks] if max_blocks else b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/generalization_v1")
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=262144,
                    help="cap per evaluation set so the sets are comparable in size")
    ap.add_argument("--seconds", type=float, default=10800)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    tok = AutoTokenizer.from_pretrained(a.model)
    src = {
        "wt2_val": f"{a.data}/wikitext2/validation.parquet",
        "wt2_test": f"{a.data}/wikitext2/test.parquet",
        "tiny": f"{a.data}/tinystories/validation.parquet",
    }
    ids = {}
    for k, p in src.items():
        if Path(p).exists():
            ids[k] = tok(text_of(p), add_special_tokens=False).input_ids[: a.max_tokens]

    EVALS = []
    for name, corpus, ctx in (("wt2_val_128", "wt2_val", 128),
                              ("wt2_test_128", "wt2_test", 128),
                              ("tiny_128", "tiny", 128),
                              ("wt2_test_512", "wt2_test", 512),
                              ("wt2_test_2048", "wt2_test", 2048)):
        if corpus in ids:
            EVALS.append((name, make_blocks(ids[corpus], ctx), ctx))

    tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"),
             add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    key_of = {f"L{i}": i for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}
    cols = originals[f"L{layers[0]}"].shape[1]
    signs = rot_signs(cols, "cuda")

    def restore():
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(originals[k].to(p.dtype).cuda())

    print(json.dumps({"evals": [(n, len(b), sum(len(x) - 1 for x in b), c)
                                for n, b, c in EVALS],
                      "cal_tokens": len(cal) * a.cal_len}), flush=True)

    restore()
    bf = {}
    for name, blocks, _ in EVALS:
        # long blocks are heavy per forward; shrink the batch as context grows
        batch = max(1, 8 * 128 // max(128, len(blocks[0]) - 1))
        bf[name] = evaluate(model, blocks, batch=batch)
        print(json.dumps({"arm": "bf16", "eval": name, "nll": bf[name]["nll"],
                          "targets": bf[name]["target_tokens"]}), flush=True)

    # one fitted quantizer for the whole run; only the evaluation varies
    Hraw, _ = capture_hessians(model, layers, cal)
    Hrot = {f"L{i}": rotate_hessian(Hraw[i], signs, BLOCK) for i in layers}
    del Hraw
    hinv = {k: cholesky_inv_upper(Hrot[k]) for k in params}
    prep = {k: rotate(originals[k].cuda().float(), signs, BLOCK) for k in params}
    torch.cuda.empty_cache()

    res = {"plan": {"model": a.model, "layers": layers,
                    "evals": {n: {"blocks": len(b), "targets": sum(len(x) - 1 for x in b),
                                  "context": c} for n, b, c in EVALS},
                    "cal_tokens": len(cal) * a.cal_len,
                    "protocol": "quantizer fitted once; only the evaluation set varies"},
           "bf16": {n: bf[n]["nll"] for n in bf}, "arms": {}}
    evs = {}

    for aname, dim, k, group in ARMS:
        if not B.check(aname, 300):
            break
        restore()
        t0 = time.monotonic()
        pool = torch.cat([gnorm_fp16(prep[kk], group)[0].reshape(-1, dim)[::17]
                          for kk in params], 0)
        book = (lloyd_scalar(pool.flatten(), 3) if dim == 1 else kmeans(pool, k))
        del pool
        book = book.to(torch.float16).float()
        packs = []
        for kk in sorted(params, key=lambda x: key_of[x]):
            w = prep[kk]
            rows, _ = w.shape
            idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hinv[kk])
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, book, "cuda")
            assert torch.equal(dq, ref), f"{aname}/{kk}: decode mismatch"
            params[kk].copy_(unrotate(dq, signs, BLOCK).to(params[kk].dtype))
            packs.append(p)
            del w, idx, sc, ref, dq
        torch.cuda.empty_cache()

        rec = {"bits_per_weight": bits_per_weight(packs, book.numel() * 2),
               "stored_bytes": sum(p.payload_bytes() for p in packs) + book.numel() * 2,
               "quantize_seconds": time.monotonic() - t0, "evals": {}}
        evs[aname] = {}
        for name, blocks, _ in EVALS:
            batch = max(1, 8 * 128 // max(128, len(blocks[0]) - 1))
            ev = evaluate(model, blocks, batch=batch)
            d, lo, hi = paired_bootstrap(ev, bf[name])
            rec["evals"][name] = {"nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi]}
            evs[aname][name] = ev
        res["arms"][aname] = rec
        print(json.dumps({"arm": aname,
                          **{n: round(v["delta_nll"], 6) for n, v in rec["evals"].items()}}),
              flush=True)
        atomic_json(out / "results.json", res)
        del packs

    res["contrasts"] = {}
    for m, ref_ in CONTRASTS:
        if m not in evs or ref_ not in evs:
            continue
        row = {}
        for name, _, _ in EVALS:
            d, lo, hi = paired_bootstrap(evs[m][name], evs[ref_][name])
            row[name] = {"delta": d, "ci": [lo, hi]}
        res["contrasts"][f"{m}_vs_{ref_}"] = row
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = ["# Generalization: fixed quantizer, varied evaluation", "",
         f"Qwen3.5-0.8B, {len(layers)} DeltaNet QKV projections. The quantizer is fitted",
         f"once ({len(cal) * a.cal_len:,} wikitext-2 train tokens, one seed); only the",
         "evaluation set changes. `wt2_val_128` is the corpus and protocol every other",
         "result used, so it is a harness check rather than new evidence.", "",
         "| Evaluation | corpus | context | blocks | targets | BF16 NLL |",
         "|---|---|---:|---:|---:|---:|"]
    for n, b, c in EVALS:
        corpus = "wikitext-2 val" if n.startswith("wt2_val") else (
            "wikitext-2 test" if n.startswith("wt2_test") else "TinyStories")
        L.append(f"| {n} | {corpus} | {c} | {len(b):,} | "
                 f"{sum(len(x) - 1 for x in b):,} | {bf[n]['nll']:.6f} |")
    for cname, row in res["contrasts"].items():
        L += ["", f"## {cname.replace('_vs_', ' vs ')}", "",
              "| Evaluation | ΔNLL | 95% CI | |", "|---|---:|:--|:--|"]
        for name, v in row.items():
            ex = "excludes 0" if (v["ci"][0] > 0) == (v["ci"][1] > 0) else "**includes 0**"
            L.append(f"| {name} | {v['delta']:+.6f} | "
                     f"[{v['ci'][0]:+.6f}, {v['ci'][1]:+.6f}] | {ex} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
