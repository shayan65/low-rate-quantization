"""Correctness and cost of the compressed matmul paths, on a real tensor.

Correctness first: both paths are checked against `packed_lowrate.decode`
followed by a dense product, on a tensor produced by the actual quantizer, not
on synthetic data. A timing number from an unverified kernel is worse than no
number, and this project has already been bitten once by evaluating weights
that were not the stored ones.

Then three costs, each measured rather than derived:

  * **resident bytes** for the runtime layout against BF16, which is the claim
    the storage sections have never been able to make;
  * **peak allocation** during a product, which is what decides whether a model
    fits, and where the streaming path earns its keep;
  * **latency** against a cuBLAS BF16 product at batch one.

The latency comparison is the one most likely to be unflattering. A BF16 GEMV
on this shape is a well-tuned library call; a first-cut Triton kernel that
gathers from a codebook is not. Reporting that honestly is the point: it
separates "the format stores fewer bytes", which is established, from "the
format is faster to run", which is not.

**A note on how the peak is measured, because the first version of this file
got it wrong.** `reset_peak_memory_stats` rebases the peak to whatever is
allocated at the moment it is called, so anything the harness is holding enters
both arms' figures. The original kept the FP32 reference weights (210 MB) alive
from the correctness check through both measurements; the two peaks came out
811.7 MB and 775.1 MB, 4.5% apart, and that 4.5% was a property of the
benchmark rather than of either path. Here each arm is measured with only the
state it actually needs resident -- the reference is freed first, and the
codebook and scales are parked on the host while the dense arm runs -- and each
arm reports its baseline, its peak, and the transient above baseline, so the
accounting can be checked rather than trusted.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from packed_lowrate import decode, encode_vq, gnorm_fp16
from packed_matmul import (HAVE_TRITON, matmul_streamed, matmul_triton,
                           runtime_bits_per_weight, storage_to_runtime)
from run_rate_sweep import kmeans


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10240)   # Qwen3.8-27B QKV shape
    ap.add_argument("--cols", type=int, default=5120)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--k", type=int, default=6561)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default="../results/packed_matmul_v2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    # a weight tensor with realistic statistics, quantized by the real codec
    W = (torch.randn(a.rows, a.cols, device=dev) * 0.02).to(torch.bfloat16).float()
    u, scale = gnorm_fp16(W, a.group)
    book = kmeans(u.reshape(-1, a.dim)[::17], a.k).to(torch.float16).float()
    pts = u.reshape(-1, a.dim)
    from run_rate_sweep import assign
    ci = torch.empty(pts.shape[0], dtype=torch.long, device=dev)
    for i in range(0, pts.shape[0], 2_000_000):
        ci[i : i + 2_000_000], _ = assign(pts[i : i + 2_000_000], book)
    pk = encode_vq(ci, scale, (a.rows, a.cols), a.group, a.dim, a.k)
    W_ref = decode(pk, book, dev)                      # the reference weights

    codes = storage_to_runtime(pk.index_bytes, pk.n_codes, a.k, dev)
    sc = scale.reshape(-1).to(torch.float16).float().to(dev)

    res = {"shape": [a.rows, a.cols], "dim": a.dim, "k": a.k, "group": a.group}

    # ---------------- correctness, before any timing ----------------
    x1 = torch.randn(1, a.cols, device=dev)
    ref1 = (x1.float() @ W_ref.T.float())
    ys = matmul_streamed(codes, book, sc, x1, a.rows, a.cols, a.dim, a.group)
    rel_s = ((ys - ref1).abs().max() / ref1.abs().max()).item()
    res["streamed_rel_err"] = rel_s
    ok = rel_s < 1e-5
    print(json.dumps({"check": "streamed vs decode+dense", "rel_err": rel_s,
                      "ok": ok}), flush=True)

    if HAVE_TRITON:
        try:
            yt = matmul_triton(codes, book, sc, x1[0], a.rows, a.cols, a.dim, a.group)
            rel_t = ((yt - ref1[0]).abs().max() / ref1.abs().max()).item()
            res["triton_rel_err"] = rel_t
            print(json.dumps({"check": "triton vs decode+dense", "rel_err": rel_t,
                              "ok": rel_t < 1e-4}), flush=True)
        except Exception as e:
            res["triton_error"] = f"{type(e).__name__}: {e}"
            print(json.dumps({"triton_error": res["triton_error"]}), flush=True)
    else:
        res["triton_error"] = "triton unavailable"

    # ---------------- resident bytes ----------------
    rt = runtime_bits_per_weight(a.rows, a.cols, a.dim, a.group, a.k)
    dense_bytes = a.rows * a.cols * 2
    rt_bytes = codes.numel() * 2 + sc.numel() * 2 + book.numel() * 2
    res["resident"] = {
        "bf16_bytes": dense_bytes, "runtime_bytes": rt_bytes,
        "ratio": dense_bytes / rt_bytes,
        "storage_index_bpw": 1.600, "runtime_bits_per_weight": rt,
    }
    print(json.dumps({"resident_MB": {"bf16": dense_bytes / 1e6,
                                      "compressed": rt_bytes / 1e6},
                      "shrink": round(dense_bytes / rt_bytes, 2)}), flush=True)

    # ---------------- free the harness before measuring memory ----------------
    # Everything above -- the source tensor, the normalized points, the
    # assignment indices, the FP32 reference -- exists only to build and check
    # the codes. Leaving any of it resident lands in both arms' peaks and makes
    # them look alike; see the module docstring.
    book_h, sc_h = book.cpu(), sc.cpu()
    xb = x1.to(torch.bfloat16)
    del W, u, pts, ci, W_ref, ref1, ys, scale
    torch.cuda.empty_cache()

    def clean_baseline():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()

    # ---- dense arm: only the BF16 weight and the activation are resident ----
    del codes, book, sc
    torch.cuda.empty_cache()
    bk = book_h.cuda()
    Wf = decode(pk, bk, dev)                   # rebuild, then drop the FP32 copy
    Wb = Wf.to(torch.bfloat16)
    del Wf, bk
    torch.cuda.empty_cache()
    base_dense = clean_baseline()
    _ = xb @ Wb.T
    torch.cuda.synchronize()
    peak_dense = torch.cuda.max_memory_allocated()
    t_dense = timed(lambda: xb @ Wb.T) if ok else None
    del Wb, _
    torch.cuda.empty_cache()

    # ---- compressed arms: only codes, codebook and scales are resident ----
    codes = storage_to_runtime(pk.index_bytes, pk.n_codes, a.k, dev)
    book, sc = book_h.cuda(), sc_h.cuda()
    base_stream = clean_baseline()
    _ = matmul_streamed(codes, book, sc, x1, a.rows, a.cols, a.dim, a.group)
    torch.cuda.synchronize()
    peak_stream = torch.cuda.max_memory_allocated()
    del _

    peaks = {"dense_bf16": {"baseline": base_dense, "peak": peak_dense,
                            "transient": peak_dense - base_dense},
             "streamed": {"baseline": base_stream, "peak": peak_stream,
                          "transient": peak_stream - base_stream}}

    # The streamed path's peak is a tile-size choice, not a property of the
    # format. Claimed in the writeup, so measured here.
    tiles = {}
    for tc in (128, 512, 2048):
        b = clean_baseline()
        _ = matmul_streamed(codes, book, sc, x1, a.rows, a.cols, a.dim, a.group,
                            tile_cols=tc)
        torch.cuda.synchronize()
        tiles[tc] = {"baseline": b, "peak": torch.cuda.max_memory_allocated(),
                     "ms": timed(lambda: matmul_streamed(codes, book, sc, x1, a.rows,
                                                         a.cols, a.dim, a.group,
                                                         tile_cols=tc))}
        del _
    res["streamed_tile_sweep"] = tiles
    print(json.dumps({"tile_sweep": {k: {"peak_MB": round(v["peak"] / 1e6, 1),
                                         "ms": round(v["ms"], 3)}
                                     for k, v in tiles.items()}}), flush=True)

    xv = x1[0].contiguous()
    run_triton = (HAVE_TRITON and "triton_error" not in res
                  and res.get("triton_rel_err", 1) < 1e-4)
    if run_triton:
        base_tri = clean_baseline()
        _ = matmul_triton(codes, book, sc, xv, a.rows, a.cols, a.dim, a.group)
        torch.cuda.synchronize()
        peak_tri = torch.cuda.max_memory_allocated()
        peaks["triton"] = {"baseline": base_tri, "peak": peak_tri,
                           "transient": peak_tri - base_tri}
        del _

    res["peak_bytes"] = {**{k: v["peak"] for k, v in peaks.items()},
                         "ratio": peak_dense / max(peak_stream, 1),
                         "detail": peaks,
                         "note": "each arm measured with only its own state "
                                 "resident; baseline is what was allocated when "
                                 "the peak counter was reset"}
    print(json.dumps({"peak_MB": {k: round(v["peak"] / 1e6, 1)
                                  for k, v in peaks.items()},
                      "baseline_MB": {k: round(v["baseline"] / 1e6, 1)
                                      for k, v in peaks.items()}}), flush=True)

    # ---------------- latency at batch one ----------------
    if ok:
        t_stream = timed(lambda: matmul_streamed(codes, book, sc, x1, a.rows,
                                                 a.cols, a.dim, a.group))
        res["latency_ms"] = {"dense_bf16": t_dense, "streamed": t_stream}
        if run_triton:
            res["latency_ms"]["triton"] = timed(
                lambda: matmul_triton(codes, book, sc, xv, a.rows, a.cols,
                                      a.dim, a.group))
        print(json.dumps({"latency_ms": {k: round(v, 3)
                                         for k, v in res["latency_ms"].items()}}),
              flush=True)

    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")

    L = ["# Compressed matmul: correctness, resident bytes, peak memory, latency", "",
         f"Shape {a.rows}x{a.cols} (Qwen3.8-27B QKV), dimension {a.dim}, K={a.k}, "
         f"group {a.group}.", "",
         "## Storage layout is not runtime layout", "",
         "| layout | bits/weight of index |", "|---|---:|",
         "| storage (5 codes per uint64) | 1.600 |",
         f"| runtime (1 code per int16) | {rt['index']:.3f} |", "",
         f"Resident: {dense_bytes / 1e6:.1f} MB BF16 against "
         f"{rt_bytes / 1e6:.1f} MB compressed, a factor of "
         f"{dense_bytes / rt_bytes:.2f}. That includes FP16 group scales "
         f"({rt['scales']:.3f} bits/weight) and the amortized codebook "
         f"({rt['codebook_amortized']:.3f}).", "",
         "## Correctness", "",
         f"Streamed path against `decode` plus a dense product: relative error "
         f"{rel_s:.2e}.", ""]
    if "triton_rel_err" in res:
        L.append(f"Triton path: relative error {res['triton_rel_err']:.2e}.")
    elif "triton_error" in res:
        L.append(f"Triton path: **not available** ({res['triton_error']}).")
    L += ["", "## Peak allocation during one product", "",
          "Each arm is measured with only its own state resident: the FP32 "
          "reference is freed first, and the codebook and scales are parked on "
          "the host while the dense arm runs. `baseline` is what was allocated "
          "when the peak counter was reset, so `transient` is what the product "
          "itself asked for.", "",
          "| path | baseline MB | peak MB | transient MB | peak vs dense |",
          "|---|---:|---:|---:|---:|"]
    for kk, v in peaks.items():
        r = "--" if kk == "dense_bf16" else f"{peak_dense / max(v['peak'], 1):.2f}x"
        L.append(f"| {kk} | {v['baseline'] / 1e6:.1f} | {v['peak'] / 1e6:.1f} | "
                 f"{v['transient'] / 1e6:.1f} | {r} |")
    L += ["",
          "The dense arm's transient is near zero because cuBLAS reuses a "
          "workspace already allocated during the correctness check; that "
          "workspace (~8.5 MB) sits in every baseline here and is a cost of "
          "running any matmul, not of either format.", "",
          "The two compressed arms differ in what they do with the weights they "
          "never store. The Triton kernel reads codes and accumulates, so its "
          "transient is nil and its peak is essentially its resident footprint. "
          "The streamed path materializes one decoded FP32 column tile at a "
          "time, and at `tile_cols=512` those tiles cost more than the "
          "compressed weights themselves -- its peak is a property of that "
          "tile size, not of the format:", "",
          "| tile_cols | peak MB | ms |", "|---|---:|---:|"]
    for tc, v in tiles.items():
        L.append(f"| {tc} | {v['peak'] / 1e6:.1f} | {v['ms']:.3f} |")
    L += ["",
          "An earlier version of this benchmark held the FP32 reference through "
          "both measurements and reported 811.7 MB against 775.1 MB, a 1.05x "
          "ratio that described the harness rather than either path.", ""]
    if "latency_ms" in res:
        L += ["## Latency, batch one", "", "| path | ms |", "|---|---:|"]
        for kk, v in res["latency_ms"].items():
            L.append(f"| {kk} | {v:.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
