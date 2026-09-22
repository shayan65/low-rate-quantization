"""Scale transfer: does the ternary VQ margin hold at 27B?

Every codec result in this project was measured on Qwen3.5-0.8B. The claim that
matters for deployment is that the ordering survives a 32x scale jump, and that
has never been tested. This run applies the section 5c recipe to all 48
`linear_attn.in_proj_qkv` tensors of Qwen3.8-27B (10240 x 5120 each,
2,516,582,400 params, 9.1% of the model) and asks whether dimension-8 vector
quantization still beats learned scalar ternary at or below parity bytes.

It deliberately does *not* attempt full coverage. Section 8 established that
MLP weights are 2.79x more damaging per parameter and that post-training
quantization alone will not produce a good ternary 27B; converting 63.5% of a
27B model is a different project with a different answer. What is testable
cheaply, and what gates everything else, is whether the *codec ordering*
transfers.

Memory. The model is 55.6 GB in bf16 against 24 GB of VRAM and ~58 GB of host
RAM, so nothing fits the naive way:

  * Weights are split across GPU and CPU by `accelerate`, with disk offload
    disabled so every weight is a real tensor that can be read and written.
  * Hessians are accumulated on the **host**, one calibration pass for all 48.
    Keeping them on the GPU would cost 5 GB in fp32 on top of ~17 GB of
    resident weights; moving each batch's gram matrix across PCIe instead
    costs ~30 s in total.
  * Target weights are reached through a **meta-aware handle**. With
    `device_map="auto"` most of them are `meta` tensors whose real data sits in
    accelerate's offload store, so `module.weight` can be neither read nor
    written; `target_store` returns the underlying CPU tensor instead.
  * Original weights are kept on the host in bf16 (5 GB) so each arm can be
    restored without re-reading 55.6 GB from disk.

Evaluation uses a fixed prefix of the same validation stream rather than all
2,042 blocks: with offloaded weights each forward pass moves tens of GB across
PCIe, and the margins under test are large relative to the sampling error of
100k+ targets. The block count is reported and the paired bootstrap is over the
blocks actually used, so the intervals are honest about it.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_lowrate import (Packed, bits_per_weight, decode, encode_scalar3,
                            encode_vq, gnorm_fp16)
from run_rate_sweep import kmeans, lloyd_scalar
from run_scf_phase0 import Budget
from run_scq_qwen import atomic_json, find_layers, get_module
from run_mlp_task_v1 import largest_pow2_block
from run_ternary_task_v4 import (cholesky_inv_upper, paired_bootstrap, quantize_tensor,
                                 rot_signs, rotate, rotate_hessian, unrotate)

ARMS = [
    # name,                dim,   k, group
    ("scalar3_rot_gptq",     1,    3, 128),
    ("vq4_rot_gptq",         4,   81, 128),
    ("vq8_rot_gptq",         8, 6561, 128),
    ("scalar3_g64_rot_gptq", 1,    3,  64),
]


def manifest(args, model, layers, devs, blk, seed, extra):
    """Everything needed to attribute a result to a configuration.

    v1 saved none of this, so a saved result did not itself attest to the GPU
    budget, calibration size or code revision it was produced under -- and the
    27B and 0.8B runs in fact used different calibration sizes without saying so.
    """
    import subprocess, hashlib
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    idx = Path(args.model) / "model.safetensors.index.json"
    cfg = Path(args.model) / "config.json"
    def h(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None
    return {
        "args": vars(args),
        "seed": seed,
        "git_revision": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        # the cluster copy is not a git checkout, so hash the sources that
        # actually determine the result -- attribution must not depend on git
        "source_sha256_16": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
                             for f in ("run_qwen38_confirm_v2.py", "packed_lowrate.py",
                                       "run_ternary_task_v4.py", "run_rate_sweep.py")
                             if Path(f).exists()},
        "data_sha256_16": extra.get("data_hash"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_index_sha256_16": h(idx),
        "checkpoint_config_sha256_16": h(cfg),
        "target_layers": layers,
        "target_weight_devices": devs,
        "hadamard_block": blk,
        "cal_tokens": extra["cal_tokens"],
        "eval_blocks_used": extra["eval_blocks_used"],
        "eval_blocks_available": extra["eval_blocks_available"],
        "eval_targets_used": extra["eval_targets_used"],
        "eval_targets_available": extra["eval_targets_available"],
        "prespecified": {
            "primary": "vq8_rot_gptq vs scalar3_rot_gptq (near-equal payload bytes)",
            "secondary": ["vq8_rot_gptq vs scalar3_g64_rot_gptq (storage dominance)",
                          "vq8_rot_gptq vs vq4_rot_gptq (is dimension 8 needed?)"],
        },
    }


def save_artifacts(path, packs, book, signs, blk, group, dim, k):
    """Persist the encoded model and reload it independently.

    v1 reported byte counts computed from in-memory objects. Writing the packed
    payload, the rounded codebook and the rotation to disk and decoding from the
    reloaded files is what makes the storage claim about a real artifact, and it
    also yields complete-file bytes as distinct from payload bytes.
    """
    import numpy as np
    blob = {"codebook": book.cpu().numpy(), "signs": signs.cpu().numpy().astype(np.int8),
            "meta": np.frombuffer(json.dumps(
                {"blk": blk, "group": group, "dim": dim, "k": k,
                 "shapes": [list(p.shape) for p in packs],
                 "n_codes": [p.n_codes for p in packs],
                 "checksums": [p.checksum for p in packs]}).encode(), dtype=np.uint8)}
    for j, pk in enumerate(packs):
        blob[f"idx{j}"] = np.frombuffer(pk.index_bytes, dtype=np.uint8)
        blob[f"scl{j}"] = np.frombuffer(pk.scale_bytes, dtype=np.uint8)
    np.savez(path, **blob)
    return path.stat().st_size


def verify_reload(path, packs, book, device="cuda"):
    """Decode from the reloaded file and require equality with the in-memory pack."""
    import numpy as np
    z = np.load(path)
    bk = torch.from_numpy(z["codebook"]).to(device)
    assert torch.equal(bk, book.to(device)), "reloaded codebook differs"
    meta = json.loads(bytes(z["meta"]).decode())
    worst = 0.0
    for j, pk in enumerate(packs):
        assert meta["checksums"][j] == pk.checksum, f"pack {j}: checksum differs after reload"
        rp = Packed(pk.kind, tuple(meta["shapes"][j]), meta["group"], meta["dim"],
                    meta["n_codes"][j], z[f"idx{j}"].tobytes(), z[f"scl{j}"].tobytes(),
                    pk.checksum)
        a_, b_ = decode(rp, bk, device), decode(pk, book.to(device), device)
        worst = max(worst, (a_ - b_).abs().max().item())
        del a_, b_
    return worst


def check_exact(layer, w, idx, sc, ref, dq, book, dump_dir):
    """Strict, immediate verification of one tensor; dump the case if it fails.

    v1 accumulated `max(maxdiff, ...)` and asserted only after all 48 tensors,
    so a failure was neither localized nor preserved, and a NaN would have
    propagated silently through `max`. Finiteness and equality are checked
    explicitly, before the tensor is installed into the model.
    """
    import numpy as np
    bad = []
    if not torch.isfinite(dq).all():
        bad.append("decoded tensor is not finite")
    if not torch.isfinite(ref).all():
        bad.append("quantizer reference is not finite")
    if not torch.equal(dq, ref):
        d = (dq - ref).abs()
        bad.append(f"decoded != reference, max|diff|={d.max().item():.6e}, "
                   f"n_differing={int(d.gt(0).sum())}")
    if not bad:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    f = dump_dir / f"failing_layer{layer}.npz"
    d = (dq - ref).abs()
    nz = d.gt(0).nonzero()
    np.savez(f, idx=idx.cpu().numpy(), scale=sc.cpu().numpy(), ref=ref.cpu().numpy(),
             decoded=dq.cpu().numpy(), book=book.cpu().numpy(),
             weight=w.cpu().numpy(), first_bad=nz[:64].cpu().numpy())
    raise AssertionError(json.dumps({"layer": layer, "problems": bad,
                                     "saved": str(f),
                                     "scale_fp16_exact": bool(torch.equal(
                                         sc, sc.to(torch.float16).float())),
                                     "idx_range": [int(idx.min()), int(idx.max())]}))


def target_store(model, i):
    """A readable, in-place-writable handle on one target weight.

    With `device_map="auto"` most target weights are `meta` tensors: accelerate
    has moved the real data into the offload store and swaps it in at forward
    time. `module.weight` is therefore neither readable nor writable for them.
    The real tensor lives in `_hf_hook.weights_map.dataset.state_dict[key]`,
    which is a plain CPU tensor that can be written in place -- verified by
    zeroing one and observing the forward change.
    """
    mod = get_module(model, i)
    w = mod.weight
    if str(w.device) != "meta":
        return w
    ds = mod._hf_hook.weights_map.dataset
    sd = getattr(ds, "state_dict", None)
    assert sd is not None, "offload store has no in-memory state_dict"
    keys = [k for k in sd if k.endswith(f"layers.{i}.linear_attn.in_proj_qkv.weight")]
    assert len(keys) == 1, f"layer {i}: expected 1 offload key, got {keys}"
    return sd[keys[0]]


def load_model(path, gpu_gib, cpu_gib):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"},
        offload_folder=None,
    ).eval()


@torch.no_grad()
def evaluate(model, blocks, batch, device="cuda"):
    """Token-weighted NLL with per-block (sum, count) for the paired bootstrap."""
    import torch.nn.functional as F
    sums, counts, ids_out = [], [], []
    by_len: dict[int, list] = {}
    for bi, b in enumerate(blocks):
        by_len.setdefault(len(b), []).append((b, bi))
    for _, group in by_len.items():
        for i in range(0, len(group), batch):
            x = torch.tensor([b for b, _ in group[i : i + batch]], device=device)
            logits = model(x[:, :-1], use_cache=False).logits.float()
            y = x[:, 1:]
            tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                  reduction="none").reshape(x.shape[0], -1)
            sums.extend(tok.sum(1).cpu().tolist())
            counts.extend([y.shape[1]] * x.shape[0])
            ids_out.extend(bi for _, bi in group[i : i + batch])
            del x, y, logits, tok
    import numpy as np
    s, c = np.asarray(sums), np.asarray(counts, dtype=float)
    return {"nll": float(s.sum() / c.sum()), "target_tokens": int(c.sum()),
            "block_sums": sums, "block_counts": counts, "block_ids": ids_out}


@torch.no_grad()
def capture_hessians(model, layer_ids, blocks, batch):
    """H = E[x x^T] for every target, in one calibration pass.

    Accumulators live on the host: 48 tensors of 5120^2 would be 5 GB in fp32
    on top of ~17 GB of resident weights. Each batch's gram matrix is formed on
    the GPU and added host-side, which costs ~300 GB of PCIe traffic over the
    whole pass -- about 30 s, far cheaper than one extra forward pass per group.
    """
    acc, n, hooks = {}, {}, []

    def mk(i):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = (x.T @ x).cpu()
            acc[i] = g if i not in acc else acc[i] + g
            n[i] = n.get(i, 0) + x.shape[0]
        return hook

    for i in layer_ids:
        hooks.append(get_module(model, i).register_forward_pre_hook(mk(i)))
    try:
        for j in range(0, len(blocks), batch):
            x = torch.tensor(blocks[j : j + batch], device="cuda")
            model(x, use_cache=False)
            del x
    finally:
        for h in hooks:
            h.remove()
    return {i: acc[i] / max(n[i], 1) for i in layer_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.8-27B-metadata")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/qwen38_confirm_v1")
    ap.add_argument("--gpu-gib", type=int, default=15)  # 17 reproduces the exactness defect
    ap.add_argument("--cpu-gib", type=int, default=46)
    ap.add_argument("--eval-blocks", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--cal-blocks", type=int, default=64)
    ap.add_argument("--cal-len", type=int, default=512)
    ap.add_argument("--cal-batch", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=21600)
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
    if len(tail) >= 2:          # v1 dropped this regardless of --eval-blocks
        blocks.append(tail)
    stream_blocks, stream_targets = len(blocks), sum(len(b) - 1 for b in blocks)
    if a.eval_blocks:           # 0 == the complete stream for this tokenizer
        blocks = blocks[: a.eval_blocks]
    block_ids = list(range(len(blocks)))
    n_targets = sum(len(b) - 1 for b in blocks)

    tr = tok("\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()[:4000]),
             add_special_tokens=False).input_ids
    cal = [tr[i * a.cal_len : (i + 1) * a.cal_len] for i in range(a.cal_blocks)]
    cal = [c for c in cal if len(c) == a.cal_len]

    t0 = time.monotonic()
    model = load_model(a.model, a.gpu_gib, a.cpu_gib)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    devs = {}
    for i in layers:
        d = str(get_module(model, i).weight.device)
        devs[d] = devs.get(d, 0) + 1
    shape0 = tuple(get_module(model, layers[0]).weight.shape)
    n_tgt_params = len(layers) * shape0[0] * shape0[1]
    total_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"stage": "load", "seconds": time.monotonic() - t0,
                      "target_layers": len(layers), "tensor_shape": shape0,
                      "target_params": n_tgt_params, "total_params": total_params,
                      "share": n_tgt_params / total_params,
                      "target_weight_devices": devs,
                      "eval_blocks": len(blocks), "eval_targets": n_targets,
                      "cal_tokens": len(cal) * a.cal_len}), flush=True)

    stores = {i: target_store(model, i) for i in layers}
    originals = {i: stores[i].detach().to("cpu", torch.bfloat16).clone() for i in layers}

    def restore():
        with torch.no_grad():
            for i in layers:
                stores[i].copy_(originals[i].to(stores[i].device, stores[i].dtype))

    t0 = time.monotonic()
    bf = evaluate(model, blocks, a.eval_batch)
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "targets": bf["target_tokens"],
                      "eval_seconds": time.monotonic() - t0, "t": B.stamp()}), flush=True)

    cols = shape0[1]
    blk = largest_pow2_block(cols)
    signs = rot_signs(cols, "cuda")

    # Hessians, one group of layers at a time so the accumulators fit alongside
    # the resident weights.
    t0 = time.monotonic()
    Hs = capture_hessians(model, layers, cal, a.cal_batch)
    Hrot = {}
    for i in layers:
        Hrot[i] = rotate_hessian(Hs[i].cuda(), signs, blk).cpu()
        Hs[i] = None
    del Hs
    torch.cuda.empty_cache()
    print(json.dumps({"stage": "hessians", "seconds": time.monotonic() - t0}), flush=True)

    import hashlib as _h
    _dh = _h.sha256(json.dumps([blocks[0][:8], blocks[-1][-8:], n_targets]).encode()
                    ).hexdigest()[:16]
    man = manifest(a, model, layers, devs, blk, 0,
                   {"data_hash": _dh,
                    "cal_tokens": len(cal) * a.cal_len,
                    "eval_blocks_used": len(blocks),
                    "eval_blocks_available": stream_blocks,
                    "eval_targets_used": n_targets,
                    "eval_targets_available": stream_targets})
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"stage": "manifest", "cal_tokens": man["cal_tokens"],
                      "eval_targets_used": n_targets,
                      "eval_targets_available": stream_targets,
                      "gpu_gib": a.gpu_gib, "git": man["git_revision"]}), flush=True)

    res = {"manifest": man,
           "plan": {"model": a.model, "target_layers": layers, "tensor_shape": list(shape0),
                    "target_params": n_tgt_params, "total_params": total_params,
                    "share": n_tgt_params / total_params, "eval_blocks": len(blocks),
                    "eval_targets": n_targets, "hadamard_block": blk,
                    "protocol": "section 5c recipe at 27B; bits/weight measured from packed "
                                "bytes; weights decoded back from those bytes; calibration "
                                "on train split"},
           "bf16": {"nll": bf["nll"], "target_tokens": bf["target_tokens"]}, "arms": {}}
    want = set(a.only.split(",")) if a.only else None

    for name, dim, k, group in ARMS:
        if want and name not in want:
            continue
        if not B.check(name, 1800):
            break
        restore()
        t0 = time.monotonic()

        # shared codebook over a subsample of every target tensor
        pool = []
        for i in layers:
            w = rotate(originals[i].to("cuda", torch.float32), signs, blk)
            pool.append(gnorm_fp16(w, group)[0].reshape(-1, dim)[::521])
            del w
        pool = torch.cat(pool, 0)
        book = (lloyd_scalar(pool.flatten(), 3) if dim == 1 else kmeans(pool, k))
        del pool
        book = book.to(torch.float16).float()
        cb_bytes = book.numel() * 2
        torch.cuda.empty_cache()
        t_fit = time.monotonic() - t0

        packs, wse, wn = [], 0.0, 0
        for i in layers:
            w = rotate(originals[i].to("cuda", torch.float32), signs, blk)
            rows, _ = w.shape
            hi = cholesky_inv_upper(Hrot[i].cuda())
            idx, sc, ref = quantize_tensor(w, book, dim, group, hinv=hi)
            del hi
            p = (encode_scalar3(idx, sc, (rows, cols), group) if dim == 1
                 else encode_vq(idx, sc, (rows, cols), group, dim, k))
            dq = decode(p, book, "cuda")
            # strict and immediate: localize, preserve and stop before installing
            check_exact(i, w, idx, sc, ref, dq, book, out / "failures" / name)
            wse += (dq - w).square().double().sum().item()
            wn += rows * cols
            tw = stores[i]
            tw.copy_(unrotate(dq, signs, blk).to(tw.device, tw.dtype))
            packs.append(p)
            del w, idx, sc, ref, dq
            torch.cuda.empty_cache()

        # persist the encoded model, then decode it back from the files
        art = out / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        file_bytes = save_artifacts(art / f"{name}.npz", packs, book, signs,
                                    blk, group, dim, k)
        reload_diff = verify_reload(art / f"{name}.npz", packs, book)
        assert reload_diff == 0.0, f"{name}: reloaded artifact decodes differently"

        bpw = bits_per_weight(packs, cb_bytes)
        payload_bytes = sum(p.payload_bytes() for p in packs) + cb_bytes
        total_bytes = payload_bytes
        t_q = time.monotonic() - t0
        ev = evaluate(model, blocks, a.eval_batch)
        d, lo, hi = paired_bootstrap(ev, bf)
        rec = {"nll": ev["nll"], "delta_nll": d, "delta_ci": [lo, hi],
               "bits_per_weight": bpw, "stored_bytes": payload_bytes,
               "payload_bytes": payload_bytes, "artifact_file_bytes": file_bytes,
               "codebook_bytes": cb_bytes, "decode_exact": True,
               "artifact_reload_exact": reload_diff == 0.0,
               "weight_mse": wse / wn, "dim": dim, "k": k, "group": group,
               "fit_seconds": t_fit, "quantize_seconds": t_q}
        res["arms"][name] = {**rec, "_ev": ev}
        print(json.dumps({"arm": name, **rec}), flush=True)
        atomic_json(out / "results.json",
                    {**res, "arms": {x: {p2: v2 for p2, v2 in y.items() if p2 != "_ev"}
                                     for x, y in res["arms"].items()}})

    # per-block losses for independent interval reconstruction and dependence checks
    atomic_json(out / "blocks.json",
                {"bf16": {k2: bf[k2] for k2 in ("block_sums", "block_counts", "block_ids")},
                 **{nm: {k2: v["_ev"][k2] for k2 in ("block_sums", "block_counts", "block_ids")}
                    for nm, v in res["arms"].items()}})

    refs = ("scalar3_rot_gptq", "scalar3_g64_rot_gptq", "vq4_rot_gptq")
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

    L = ["# Scale transfer: the section 5c recipe on Qwen3.8-27B", "",
         f"All {len(layers)} `linear_attn.in_proj_qkv` tensors ({shape0[0]}x{shape0[1]} each), "
         f"{n_tgt_params:,} params = {n_tgt_params / total_params * 100:.1f}% of the model.",
         f"**{n_targets:,} validation targets** over {len(blocks)} blocks, token-weighted. "
         f"BF16 NLL {bf['nll']:.6f}.",
         f"Hadamard block {blk}; calibration {len(cal) * a.cal_len:,} tokens from the train "
         "split. Every arm verifies decode(encode(w)) per tensor and re-decodes",
         "the reloaded artifact from disk.", "",
         "| Arm | bits/wt | stored MB | weight MSE | NLL | ΔNLL vs BF16 | 95% CI |",
         "|---|---:|---:|---:|---:|---:|:--|"]
    for nm, r in res["arms"].items():
        L.append(f"| {nm} | {r['bits_per_weight']:.4f} | {r['stored_bytes'] / 1e6:.2f} "
                 f"| {r['weight_mse']:.4e} | {r['nll']:.6f} | {r['delta_nll']:+.6f} "
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
