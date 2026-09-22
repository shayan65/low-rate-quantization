"""Localize the budget-sensitive exactness defect from a saved failing tensor.

At a 17 GiB GPU weight budget the per-tensor bit-exactness check fires on the
27B model with a decoded-vs-reference difference of 0.036; at 15 GiB every
tensor is exact. Passing synthetic-shape tests and a clean layer 0 do not
exclude a data-dependent or later-layer fault, and "memory budget changes
arithmetic" is not a conclusion anyone should accept without evidence.

`check_exact` in the v2 runner saves the whole case (indices, FP16 scales,
codebook, quantizer reference, GPU decode, the rotated weight, and the first
differing coordinates). This script takes that dump and asks, in order, which
stage is responsible:

  A. packing        -- do the indices survive pack -> unpack unchanged?
  B. scale storage  -- do the FP16 scales survive the byte round trip?
  C. reconstruction -- does an independent CPU `book[idx] * scale` agree with
                       the GPU decode, and with the quantizer's own reference?
  D. reference      -- is the saved reference itself self-inconsistent, i.e.
                       does it differ from book[idx]*scale even on CPU?

These four are mutually exclusive explanations. A says the serialized bytes are
wrong; B says the metadata is; C says the two reconstructions disagree while
each is internally fine; D says the tensor the quantizer believed it produced
was already mutated before encoding. The structure of the differing positions
(clustered in one group, one column block, or scattered) discriminates further.

This script is pure analysis: it allocates no large GPU buffers of its own
beyond the reconstruction under test, so it can run under either budget, and it
reports rather than asserts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from packed_lowrate import Packed, codes_per_word, pack_codes, pack_trits, unpack_codes, unpack_trits, decode


def summarize_positions(bad, cols, group, dim):
    """Where do the differing elements live? Clustering is diagnostic."""
    if bad.size == 0:
        return {}
    rows_, cols_ = bad[:, 0], bad[:, 1]
    return {
        "n_shown": int(bad.shape[0]),
        "distinct_rows": int(len(set(rows_.tolist()))),
        "distinct_cols": int(len(set(cols_.tolist()))),
        "col_min": int(cols_.min()), "col_max": int(cols_.max()),
        "groups_touched": sorted({int(c) // group for c in cols_})[:12],
        "dim_blocks_touched": sorted({int(c) // dim for c in cols_})[:12],
        "col_mod_group": Counter((int(c) % group) for c in cols_).most_common(5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="failing_layerN.npz written by check_exact")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    z = np.load(a.dump)
    idx = z["idx"]            # (rows, cols/dim) int64
    scale = z["scale"]        # (rows, ng, 1) float32, already FP16-valued
    ref = z["ref"]            # quantizer reconstruction
    dec = z["decoded"]        # GPU decode from packed bytes
    book = z["book"]          # (k, dim) or (3,)
    rows, cols = ref.shape
    dim = 1 if book.ndim == 1 else book.shape[1]
    k = book.shape[0]
    out = {"dump": a.dump, "shape": [rows, cols], "dim": dim, "k": k,
           "group": a.group, "codes_per_word": codes_per_word(k) if dim > 1 else 5}

    # ---------- A. packing ----------
    flat = idx.reshape(-1).astype(np.int64)
    if dim == 1:
        rt = unpack_trits(pack_trits(flat.astype(np.uint8)), flat.size).astype(np.int64)
    else:
        rt = unpack_codes(pack_codes(flat, k), flat.size, k)
    a_ok = bool(np.array_equal(rt, flat))
    out["A_packing_roundtrip_exact"] = a_ok
    if not a_ok:
        d = np.nonzero(rt != flat)[0]
        out["A_first_bad_code_positions"] = d[:16].tolist()
        out["A_n_bad_codes"] = int(d.size)

    # ---------- B. scale storage ----------
    sb = scale.astype(np.float16).tobytes()
    sr = np.frombuffer(sb, np.float16).astype(np.float32).reshape(scale.shape)
    b_ok = bool(np.array_equal(sr, scale))
    out["B_scale_roundtrip_exact"] = b_ok
    out["B_scale_is_fp16_valued"] = bool(
        np.array_equal(scale, scale.astype(np.float16).astype(np.float32)))
    out["B_scale_min"] = float(scale.min())
    out["B_scale_max"] = float(scale.max())
    out["B_scale_finite"] = bool(np.isfinite(scale).all())

    # ---------- C/D. independent CPU reconstruction ----------
    ng = cols // a.group
    bk = book if dim > 1 else book.reshape(-1, 1)
    cpu = (bk[idx.reshape(-1)].reshape(rows, ng, a.group)
           * scale.reshape(rows, ng, 1)).reshape(rows, cols).astype(np.float32)

    def cmp(x, y):
        d = np.abs(x - y)
        n = int((d > 0).sum())
        return {"exact": n == 0, "n_differing": n, "max_abs": float(d.max()) if d.size else 0.0}

    out["C_cpu_vs_gpu_decode"] = cmp(cpu, dec)
    out["D_cpu_vs_quantizer_reference"] = cmp(cpu, ref)
    out["gpu_decode_vs_reference"] = cmp(dec, ref)
    out["reference_finite"] = bool(np.isfinite(ref).all())
    out["decoded_finite"] = bool(np.isfinite(dec).all())

    # ---------- re-decode from freshly packed bytes, CPU and GPU ----------
    import hashlib
    ib = pack_trits(flat.astype(np.uint8)) if dim == 1 else pack_codes(flat, k)
    sha = hashlib.sha256(); sha.update(ib); sha.update(sb)
    pk = Packed("scalar3" if dim == 1 else "vq", (rows, cols), a.group, dim,
                int(flat.size), ib, sb, sha.hexdigest()[:16])
    try:
        re_gpu = decode(pk, torch.from_numpy(book).to(a.device), a.device).cpu().numpy()
        out["redecode_gpu_vs_cpu"] = cmp(re_gpu, cpu)
        out["redecode_gpu_vs_original_decode"] = cmp(re_gpu, dec)
    except Exception as e:  # a failing device should not hide the rest
        out["redecode_gpu_error"] = f"{type(e).__name__}: {e}"

    # ---------- where are the differences ----------
    bad = np.argwhere(np.abs(dec - ref) > 0)
    out["positions_gpu_decode_vs_reference"] = summarize_positions(
        bad[:4096], cols, a.group, dim)
    badc = np.argwhere(np.abs(cpu - ref) > 0)
    out["positions_cpu_vs_reference"] = summarize_positions(
        badc[:4096], cols, a.group, dim)

    # ---------- verdict ----------
    if not out["A_packing_roundtrip_exact"]:
        v = "A: index packing is lossy for this data"
    elif not out["B_scale_roundtrip_exact"]:
        v = "B: scale byte round trip is lossy"
    elif out["D_cpu_vs_quantizer_reference"]["exact"] and \
            not out["C_cpu_vs_gpu_decode"]["exact"]:
        v = "C: GPU decode disagrees with an independent CPU reconstruction " \
            "that matches the quantizer -- fault is in the GPU decode path"
    elif not out["D_cpu_vs_quantizer_reference"]["exact"] and \
            out["C_cpu_vs_gpu_decode"]["exact"]:
        v = "D: the saved quantizer reference disagrees with book[idx]*scale, " \
            "while both reconstructions agree -- the reference was mutated " \
            "after the codes were chosen"
    elif not out["D_cpu_vs_quantizer_reference"]["exact"]:
        v = "C+D: both reconstructions differ from the reference"
    else:
        v = "no discrepancy reproduced from the dump (fault did not persist " \
            "into the saved arrays)"
    out["verdict"] = v

    print(json.dumps(out, indent=2))
    Path(a.dump).with_suffix(".analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
