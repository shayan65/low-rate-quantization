"""The compressed matmul on a projection the quality results actually used.

`bench_packed_matmul.py` measures a decoder and a GEMV on Gaussian weights at
the 27B QKV shape. That is a legitimate microbenchmark and it is now labelled
as one, but it does not connect this project's loss numbers to compressed
inference: it skips the rotation, skips GPTQ, checks correctness against an
FP32 decoded product and compares latency against a BF16 product. A reviewer
was right that "a kernel that runs it" was not established by it.

This run closes that. It takes `in_proj_qkv` from Qwen3.5-0.8B, quantizes it
with the *same* recipe as the `vq8_rot_gptq` arm -- blockwise randomized
Hadamard rotation, a shared 6561-point eight-dimensional codebook pooled over
all 18 projections, GPTQ compensation against the real Hessian -- and then asks
whether the compressed path computes the same thing as the BF16 weights that
were installed to produce the reported loss.

**The activation transform is the part that was missing.** The stored codes
represent `rotate(W) = W A^T`, while the quality runs install
`unrotate(dq) = dq A`. So

    y = x (dq A)^T = x A^T dq^T = rotate(x) dq^T,

and a compressed path must rotate its input by the same blockwise Hadamard
before multiplying. That rotation is a real per-token cost and it is inside the
timed region here, not excluded from it.

Three vectors are compared on activations captured from the model itself:

  * `exact`     -- FP32 product with the dequantized weights: the mathematical
                   target the codec defines;
  * `installed` -- BF16 product with `unrotate(dq)`, which is literally the
                   operation behind every ΔNLL in this project;
  * `kernel`    -- the compressed path.

If the kernel's error against `exact` is at or below `installed`'s, then the
compressed path is at least as faithful as the weights the loss numbers were
measured with, which is the claim that matters. Reporting only the kernel's
error, as the synthetic benchmark did, cannot establish that.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from packed_lowrate import decode, encode_vq, gnorm_fp16
from packed_matmul import (HAVE_TRITON, matmul_streamed, matmul_triton,
                           resident_bytes, runtime_bits_per_weight,
                           storage_to_runtime)
from run_generalization_v1 import text_of, make_blocks
from run_rate_sweep import kmeans
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_ternary_task_v4 import (BLOCK, capture_hessians, cholesky_inv_upper,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)

DIM, K, GROUP = 8, 6561, 128


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def relerr(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/real_projection_v1")
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--act-blocks", type=int, default=8)
    ap.add_argument("--layer", type=int, default=-1, help="-1 = middle layer")
    ap.add_argument("--seconds", type=float, default=3600)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"),
             add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]
    va = tok(text_of(f"{a.data}/wikitext2/validation.parquet"),
             add_special_tokens=False).input_ids
    act_blocks = make_blocks(va, 128)[: a.act_blocks]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    pick = layers[len(layers) // 2] if a.layer < 0 else a.layer
    assert pick in layers, f"layer {pick} is not a target ({layers})"
    mod = get_module(model, pick)
    W0 = mod.weight.detach().clone()
    rows, cols = W0.shape
    assert cols == BLOCK, f"rotation assumes one Hadamard block, cols={cols}"
    signs = rot_signs(cols, "cuda")

    # ---- the real recipe: rotation, shared codebook over all 18, GPTQ ----
    t0 = time.monotonic()
    Hstat, ntok = capture_hessians(model, layers, cal)
    Hrot = rotate_hessian(Hstat[pick], signs)
    prep = {i: rotate(get_module(model, i).weight.detach().float(), signs)
            for i in layers}
    pool = torch.cat([gnorm_fp16(prep[i], GROUP)[0].reshape(-1, DIM)[::17]
                      for i in layers], 0)
    book = kmeans(pool, K).to(torch.float16)          # FP16 is the stored form
    del pool, Hstat
    torch.cuda.empty_cache()

    w_rot = prep[pick]
    hinv = cholesky_inv_upper(Hrot)
    idx, sc, ref = quantize_tensor(w_rot, book.float(), DIM, GROUP, hinv=hinv)
    pk = encode_vq(idx, sc, (rows, cols), GROUP, DIM, K)
    dq = decode(pk, book.float(), "cuda")
    assert torch.equal(dq, ref), "decode does not reproduce the quantizer's choice"
    del prep, hinv, Hrot
    torch.cuda.empty_cache()
    fit_s = time.monotonic() - t0

    # what the quality runs install, and what a compressed runtime holds
    W_inst = unrotate(dq, signs, BLOCK).to(torch.bfloat16)
    codes = storage_to_runtime(pk.index_bytes, pk.n_codes, K, "cuda")
    sc16 = sc.reshape(-1).to(torch.float16)

    # ---- real activations into this projection ----
    caught = []
    h = mod.register_forward_pre_hook(
        lambda _m, inp: caught.append(inp[0].detach().reshape(-1, cols).float()))
    with torch.no_grad():
        for b in act_blocks:
            model(torch.tensor([b[:-1]], device="cuda"))
    h.remove()
    X = torch.cat(caught, 0)
    del caught
    print(json.dumps({"layer": pick, "shape": [rows, cols], "act_rows": X.shape[0],
                      "cal_tokens": ntok[pick], "fit_seconds": fit_s}), flush=True)

    res = {"plan": {"model": a.model, "layer": pick, "shape": [rows, cols],
                    "dim": DIM, "k": K, "group": GROUP,
                    "recipe": "rotate + shared pooled codebook + GPTQ, as vq8_rot_gptq",
                    "cal_tokens": ntok[pick], "act_rows": X.shape[0],
                    "note": "a real projection with the real recipe; the compressed "
                            "path rotates its input, and that rotation is timed"}}

    # ---- fidelity against the weights the loss numbers were measured with ----
    xs = X[:1024]
    xr = rotate(xs, signs, BLOCK)
    y_exact = xr @ dq.float().T
    y_inst = (xs.to(torch.bfloat16) @ W_inst.T).float()
    y_stream = matmul_streamed(codes, book, sc16, xr, rows, cols, DIM, GROUP)
    fid = {"installed_bf16_vs_exact": relerr(y_inst, y_exact),
           "streamed_vs_exact": relerr(y_stream, y_exact)}
    if HAVE_TRITON:
        y_tri = torch.stack([matmul_triton(codes, book, sc16,
                                           rotate(xs[i : i + 1], signs, BLOCK)[0],
                                           rows, cols, DIM, GROUP)
                             for i in range(16)])
        fid["triton_vs_exact"] = relerr(y_tri, y_exact[:16])
        fid["triton_vs_installed"] = relerr(y_tri, y_inst[:16])
    fid["verdict"] = ("kernel at or below the BF16 install error"
                      if fid.get("triton_vs_exact", 1) <= fid["installed_bf16_vs_exact"]
                      else "kernel error exceeds the BF16 install error")
    res["fidelity"] = fid
    print(json.dumps({"fidelity": fid}), flush=True)

    # ---- resident bytes, from each tensor's own element size ----
    rb = resident_bytes(codes, book, sc16)
    dense_bytes = rows * cols * 2
    res["resident"] = {"bf16_bytes": dense_bytes, **rb,
                       "ratio": dense_bytes / rb["total"],
                       "predicted_bits_per_weight":
                           runtime_bits_per_weight(rows, cols, DIM, GROUP, K)}
    print(json.dumps({"resident_MB": {"bf16": dense_bytes / 1e6,
                                      "compressed": rb["total"] / 1e6},
                      "shrink": round(dense_bytes / rb["total"], 3),
                      "dtypes": rb["dtypes"]}), flush=True)

    # ---- latency at batch one, rotation included in the compressed path ----
    x1 = X[:1].contiguous()
    xb = x1.to(torch.bfloat16)
    lat = {"installed_bf16": timed(lambda: xb @ W_inst.T)}
    lat["rotate_only"] = timed(lambda: rotate(x1, signs, BLOCK))
    lat["streamed_with_rotation"] = timed(
        lambda: matmul_streamed(codes, book, sc16, rotate(x1, signs, BLOCK),
                                rows, cols, DIM, GROUP))
    if HAVE_TRITON:
        lat["triton_with_rotation"] = timed(
            lambda: matmul_triton(codes, book, sc16, rotate(x1, signs, BLOCK)[0],
                                  rows, cols, DIM, GROUP))
    res["latency_ms"] = lat
    print(json.dumps({"latency_ms": {k: round(v, 4) for k, v in lat.items()}}),
          flush=True)

    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    L = [f"# The compressed path on a real projection (layer {pick})", "",
         f"`in_proj_qkv` of Qwen3.5-0.8B, {rows}x{cols}, quantized with the same "
         f"recipe as the `vq8_rot_gptq` arm: blockwise randomized Hadamard "
         f"rotation, a shared {K}-point dimension-{DIM} codebook pooled over all "
         f"{len(layers)} projections, GPTQ against the Hessian from "
         f"{ntok[pick]:,} calibration tokens. Activations are captured from the "
         f"model on validation text ({X.shape[0]:,} rows).", "",
         "## Fidelity to the weights the loss numbers were measured with", "",
         "The compressed path rotates its input, since the stored codes are in "
         "the rotated basis: `y = rotate(x) @ dq^T`. `exact` is the FP32 product "
         "with the dequantized weights; `installed` is the BF16 product with "
         "`unrotate(dq)`, which is the operation behind every reported ΔNLL.", "",
         "| comparison | relative error |", "|---|---:|"]
    for k, v in fid.items():
        if k != "verdict":
            L.append(f"| {k.replace('_', ' ')} | {v:.2e} |")
    L += ["", f"**{fid['verdict']}.**", "",
          "## Resident bytes", "",
          f"| tensor | dtype | MB |", "|---|---|---:|"]
    for k in ("codes", "codebook", "scales"):
        L.append(f"| {k} | {rb['dtypes'][k]} | {rb[k] / 1e6:.3f} |")
    L += [f"| **total** | | **{rb['total'] / 1e6:.3f}** |",
          f"| BF16 dense | torch.bfloat16 | {dense_bytes / 1e6:.3f} |", "",
          f"Reduction: **{dense_bytes / rb['total']:.2f}x**.", "",
          "## Latency, batch one", "", "| path | ms |", "|---|---:|"]
    for k, v in lat.items():
        L.append(f"| {k.replace('_', ' ')} | {v:.4f} |")
    L += ["", "The rotation is a required part of the compressed path and is "
          "inside the timed region; it is also reported alone so its share is "
          "visible."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
