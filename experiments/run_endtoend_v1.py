"""A model that actually runs on compressed weights, and what it weighs.

Every quality number in this project so far was produced by *decoding* packed
bytes into BF16 weights, installing those, and measuring loss. That is the right
way to measure the codec, and it is why the paper has always said it establishes
no runtime claim: at inference the model still occupied its dense footprint, and
"packed weights would fit" was an arithmetic statement about bytes on disk.

This run removes that caveat for the weights it covers. Each target `nn.Linear`
is *replaced* by a module that holds only int16 codes, an FP16 codebook and FP16
group scales, and computes

    y = matmul(rotate_fused(x), W_hat^T)

with the Hadamard rotation fused into a single Triton kernel. The dense weights
are dropped, so the resident footprint reported here is measured from the
allocator rather than derived from a payload count.

Three questions, in order of how easily they could be faked:

  * **does it compute the same thing?** The compressed model's loss is compared
    against the decoded-BF16 model's loss on identical blocks. These should
    agree to the numerical error of the kernel, not merely be "close"; a
    mismatch means the served model is not the one the paper measured.
  * **what does it weigh?** `torch.cuda.memory_allocated()` before and after
    replacement, which counts the whole model and not just the covered tensors.
  * **what does a token cost?** Single-token decode latency against the BF16
    model, which is the regime where a low-rate format is supposed to help and
    where this one has so far been slower.

Scope, stated up front and not softened: the KV cache, the recurrent state, the
embedding and the norms stay BF16, so this is a model whose *covered weights*
are served compressed. Nothing here demonstrates a full low-precision serving
stack, and the latency figures come from two kernels tuned only far enough to
be correct.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fused_hadamard import FusedRotation
from packed_lowrate import decode, encode_vq, gnorm_fp16
from packed_matmul import (HAVE_TRITON, matmul_streamed, matmul_triton,
                           storage_to_runtime)
from run_rate_sweep import kmeans
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json
from run_mlp_task_v1 import capture_hessians, largest_pow2_block
from run_generalization_v1 import text_of, make_blocks
from run_fullmodel_v1 import find_all_targets
from run_ternary_task_v4 import (cholesky_inv_upper, evaluate, paired_bootstrap,
                                 quantize_tensor, rot_signs, rotate, rotate_hessian,
                                 unrotate)


class CompressedLinear(nn.Module):
    """A linear layer that never materializes its weight matrix.

    Holds the runtime layout -- one int16 code per `dim` weights, FP16 group
    scales, a shared FP16 codebook -- plus the sign vector for the rotation.
    The codebook is shared across every layer of a run and registered as a
    buffer on exactly one of them, so it is charged once in the byte total the
    allocator reports.
    """

    def __init__(self, codes, book, scale, signs, rot, rows, cols, dim, group,
                 bias=None):
        super().__init__()
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("signs", signs, persistent=False)
        self.book = book                       # shared; not a buffer per layer
        self.rot = rot                         # shared Hadamard factors
        self.rows, self.cols, self.dim, self.group = rows, cols, dim, group
        self.bias = bias

    @torch.no_grad()
    def forward(self, x):
        shp = x.shape
        flat = x.reshape(-1, self.cols)
        xr = self.rot(flat.float(), self.signs)
        if flat.shape[0] == 1 and HAVE_TRITON:
            # Single-token decode: the GEMV kernel reads codes and accumulates,
            # where the streamed path would build a decoded FP32 tile per
            # column block. Prefill and batched evaluation still take the
            # streamed path, which is the one checked against the reference.
            y = matmul_triton(self.codes, self.book, self.scale, xr[0],
                              self.rows, self.cols, self.dim,
                              self.group).unsqueeze(0)
        else:
            y = matmul_streamed(self.codes, self.book, self.scale, xr,
                                self.rows, self.cols, self.dim, self.group)
        y = y.to(x.dtype).reshape(*shp[:-1], self.rows)
        if self.bias is not None:
            y = y + self.bias
        return y


def replace(model, name, module):
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.rsplit(".", 1)[-1], module)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/endtoend_v1")
    ap.add_argument("--scope", default="all", choices=("qkv", "all"))
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--k", type=int, default=1625)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--eval-blocks", type=int, default=256)
    ap.add_argument("--cal-blocks", type=int, default=128)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=14400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(text_of(f"{a.data}/wikitext2/validation.parquet"),
              add_special_tokens=False).input_ids
    blocks = make_blocks(ids, 128)[: a.eval_blocks]
    n_targets = sum(len(b) - 1 for b in blocks)
    tr = tok(text_of(f"{a.data}/wikitext2/train.parquet"), add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    allt = find_all_targets(model, a.group)
    targets = ([t for t in allt if "in_proj_qkv" in t[0]] if a.scope == "qkv"
               else allt)
    params = {n: m.weight for n, m, _ in targets}
    blk_of = {n: b for n, _, b in targets}
    originals = {n: w.detach().clone().cpu() for n, w in params.items()}
    n_cov = sum(w.numel() for w in originals.values())
    n_all = sum(p.numel() for p in model.parameters())

    torch.cuda.synchronize(); torch.cuda.empty_cache()
    mem_bf16 = torch.cuda.memory_allocated()
    print(json.dumps({"scope": a.scope, "tensors": len(targets),
                      "covered_params": n_cov, "total_params": n_all,
                      "eval_blocks": len(blocks), "eval_targets": n_targets,
                      "resident_bf16_bytes": mem_bf16}), flush=True)

    # ---- quantize with the standard recipe ----
    signs = {n: rot_signs(originals[n].shape[1], "cuda") for n in params}
    t0 = time.monotonic()
    Hraw, _ = capture_hessians(model, targets, cal)
    hinv = {}
    for n in params:
        hinv[n] = cholesky_inv_upper(rotate_hessian(Hraw[n], signs[n], blk_of[n]))
        Hraw[n] = None
    del Hraw
    torch.cuda.empty_cache()

    pool = torch.cat([
        gnorm_fp16(rotate(originals[n].cuda().float(), signs[n], blk_of[n]),
                   a.group)[0].reshape(-1, a.dim)[::37] for n in params], 0)
    book32 = kmeans(pool, a.k).to(torch.float16).float()
    del pool
    torch.cuda.empty_cache()

    packed, deq_bf16 = {}, {}
    for n, _, _ in targets:
        w = rotate(originals[n].cuda().float(), signs[n], blk_of[n])
        rows, cols = w.shape
        idx, sc, ref = quantize_tensor(w, book32, a.dim, a.group, hinv=hinv[n])
        pk = encode_vq(idx, sc, (rows, cols), a.group, a.dim, a.k)
        dq = decode(pk, book32, "cuda")
        assert torch.equal(dq, ref), f"{n}: decode mismatch"
        packed[n] = (pk, sc.reshape(-1).to(torch.float16).clone())
        deq_bf16[n] = unrotate(dq, signs[n], blk_of[n]).to(torch.bfloat16).cpu()
        del w, idx, sc, ref, dq
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "quantize", "seconds": time.monotonic() - t0}),
          flush=True)

    # ---- reference: the decoded-BF16 model every earlier number came from ----
    with torch.no_grad():
        for n, w in params.items():
            w.copy_(deq_bf16[n].to(w.dtype).cuda())
    ref_eval = evaluate(model, blocks)
    print(json.dumps({"arm": "decoded_bf16", "nll": ref_eval["nll"]}), flush=True)

    # Baseline decode latency, measured on the same model object before the
    # modules are swapped. Without it the compressed figure means nothing.
    def decode_step():
        model(torch.zeros(1, 1, dtype=torch.long, device="cuda"), use_cache=False)

    def timed_decode(iters=20, warmup=5):
        for _ in range(warmup):
            decode_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            decode_step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    t_bf16 = timed_decode()
    print(json.dumps({"bf16_decode_ms": t_bf16}), flush=True)

    # ---- replace the modules and drop the dense weights ----
    book16 = book32.to(torch.float16)
    n_tensors = len(targets)
    rots = {}
    for n, mod, blk in targets:
        if blk not in rots:
            rots[blk] = FusedRotation(blk, "cuda")
        pk, sc16 = packed[n]
        rows, cols = originals[n].shape
        codes = storage_to_runtime(pk.index_bytes, pk.n_codes, a.k, "cuda")
        bias = getattr(mod, "bias", None)
        replace(model, n, CompressedLinear(
            codes, book16, sc16.cuda(), signs[n], rots[blk],
            rows, cols, a.dim, a.group,
            bias.detach().clone() if bias is not None else None))
    # Everything that still points at an original nn.Linear has to go, or its
    # BF16 weight stays resident and the "compressed" model measures larger
    # than the dense one. `targets` holds the module objects themselves, which
    # is exactly the harness contamination that made the first attempt at this
    # report a 0.85x shrink -- i.e. an expansion.
    del packed, deq_bf16, originals, params, signs, targets, allt, blk_of
    del book32, hinv
    import gc
    gc.collect()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    mem_comp = torch.cuda.memory_allocated()
    print(json.dumps({"resident_compressed_bytes": mem_comp,
                      "shrink": mem_bf16 / mem_comp}), flush=True)

    # ---- does the served model compute what the paper measured? ----
    comp_eval = evaluate(model, blocks)
    d, lo, hi = paired_bootstrap(comp_eval, ref_eval)
    print(json.dumps({"arm": "compressed_served", "nll": comp_eval["nll"],
                      "delta_vs_decoded_bf16": d,
                      "ci": [lo, hi]}), flush=True)

    # ---- single-token decode latency ----
    t_comp = timed_decode()

    res = {"plan": {"model": a.model, "scope": a.scope, "tensors": n_tensors,
                    "dim": a.dim, "k": a.k, "group": a.group,
                    "covered_params": n_cov, "total_params": n_all,
                    "eval_blocks": len(blocks), "eval_targets": n_targets,
                    "note": "covered weights served compressed; embedding, norms, "
                            "KV cache and recurrent state stay BF16"},
           "resident": {"bf16_bytes": mem_bf16, "compressed_bytes": mem_comp,
                        "shrink": mem_bf16 / mem_comp,
                        "measured_by": "torch.cuda.memory_allocated"},
           "quality": {"decoded_bf16_nll": ref_eval["nll"],
                       "compressed_served_nll": comp_eval["nll"],
                       "delta": d, "ci": [lo, hi],
                       "agrees": bool(abs(d) < 1e-4)},
           "latency_ms": {"bf16_decode_step": t_bf16,
                          "compressed_decode_step": t_comp,
                          "ratio": t_comp / t_bf16}}
    atomic_json(out / "results.json", res)

    L = [f"# A served compressed model ({a.scope} coverage)", "",
         f"{n_tensors} tensors, {n_cov:,} of {n_all:,} parameters, dimension "
         f"{a.dim}, K={a.k}. Each target `nn.Linear` is replaced by a module "
         f"holding int16 codes, an FP16 codebook and FP16 group scales; the "
         f"dense weights are dropped and the Hadamard rotation is fused into one "
         f"Triton kernel.", "",
         "## Does the served model compute what the paper measured?", "",
         f"| model | NLL over {n_targets:,} targets |", "|---|---:|",
         f"| decoded BF16 (what every earlier number used) | {ref_eval['nll']:.6f} |",
         f"| compressed, served | {comp_eval['nll']:.6f} |", "",
         f"Difference {d:+.2e}, 95% CI [{lo:+.2e}, {hi:+.2e}].", "",
         "## What it weighs", "",
         "| model | resident bytes | |", "|---|---:|---:|",
         f"| BF16 | {mem_bf16 / 1e6:.1f} MB | |",
         f"| compressed | {mem_comp / 1e6:.1f} MB | **{mem_bf16 / mem_comp:.2f}x** |", "",
         "Measured with `torch.cuda.memory_allocated`, so this is the whole "
         "model including the BF16 embedding and norms, not a payload count.", "",
         "## Single-token decode", "",
         f"| model | ms/step |", "|---|---:|",
         f"| BF16 | {t_bf16:.3f} |",
         f"| compressed | {t_comp:.3f} |", "",
         f"A factor of {t_comp / t_bf16:.1f}. Decode takes the GEMV kernel; "
         f"prefill and the evaluation above take the streamed path, which is "
         f"the one checked against the reference.", "",
         "Scope: the embedding, norms, KV cache and recurrent state stay BF16, "
         "and both kernels are tuned only far enough to be correct."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
