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
    ap.add_argument("--out", default="../results/packed_matmul_v1")
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

    # ---------------- peak allocation during a product ----------------
    Wb = W_ref.to(torch.bfloat16)
    xb = x1.to(torch.bfloat16)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _ = xb @ Wb.T
    torch.cuda.synchronize()
    peak_dense = torch.cuda.max_memory_allocated()
    del Wb
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _ = matmul_streamed(codes, book, sc, x1, a.rows, a.cols, a.dim, a.group)
    torch.cuda.synchronize()
    peak_stream = torch.cuda.max_memory_allocated()
    res["peak_bytes"] = {"dense_bf16": peak_dense, "streamed": peak_stream,
                         "ratio": peak_dense / max(peak_stream, 1)}
    print(json.dumps({"peak_MB": {"dense": peak_dense / 1e6,
                                  "streamed": peak_stream / 1e6}}), flush=True)

    # ---------------- latency at batch one ----------------
    if ok:
        Wb = W_ref.to(torch.bfloat16)
        t_dense = timed(lambda: xb @ Wb.T)
        t_stream = timed(lambda: matmul_streamed(codes, book, sc, x1, a.rows,
                                                 a.cols, a.dim, a.group))
        res["latency_ms"] = {"dense_bf16": t_dense, "streamed": t_stream}
        if HAVE_TRITON and "triton_error" not in res and res.get("triton_rel_err", 1) < 1e-4:
            xv = x1[0].contiguous()
            res["latency_ms"]["triton"] = timed(
                lambda: matmul_triton(codes, book, sc, xv, a.rows, a.cols,
                                      a.dim, a.group))
        print(json.dumps({"latency_ms": {k: round(v, 3)
                                         for k, v in res["latency_ms"].items()}}),
              flush=True)
        del Wb

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
          f"Dense BF16 {peak_dense / 1e6:.1f} MB, streamed {peak_stream / 1e6:.1f} MB.", ""]
    if "latency_ms" in res:
        L += ["## Latency, batch one", "", "| path | ms |", "|---|---:|"]
        for kk, v in res["latency_ms"].items():
            L.append(f"| {kk} | {v:.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
