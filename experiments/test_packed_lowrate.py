"""Round-trip and byte-accounting tests for the packed low-rate codecs.

The claim under test is the one the previous run did not establish: that the
bytes we count are bytes that actually reconstruct the weights we evaluate.
"""

import sys

import numpy as np
import torch

from packed_lowrate import (
    Packed, bits_per_weight, codes_per_word, decode, encode_scalar3, encode_vq,
    pack_codes, pack_trits, unpack_codes, unpack_trits,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main() -> int:
    rng = np.random.default_rng(0)

    # --- symbol packing round-trips, including the ragged tail ---
    for n in (1, 4, 5, 6, 1_000_003):
        t = rng.integers(0, 3, n).astype(np.uint8)
        check(f"trit round-trip n={n}", np.array_equal(unpack_trits(pack_trits(t), n), t))
    for n in (1, 4, 5, 6, 800_003):
        c = rng.integers(0, 6561, n).astype(np.int64)
        check(f"code round-trip n={n}", np.array_equal(unpack_codes(pack_codes(c, 6561), n, 6561), c))

    # the 5-per-word packing must not silently overflow uint64
    c = np.full(5, 6560, np.int64)
    check("max-value code word survives", np.array_equal(unpack_codes(pack_codes(c, 6561), 5, 6561), c),
          f"6561^5={6561**5:.4e} < 2^64={2**64:.4e}")

    # --- packing density is exactly 1.600 bits/weight for both formats ---
    N = 6144 * 1024
    trit_b = len(pack_trits(rng.integers(0, 3, N).astype(np.uint8)))
    code_b = len(pack_codes(rng.integers(0, 6561, N // 8).astype(np.int64), 6561))
    check("ternary index density = 1.600 bpw", abs(trit_b * 8 / N - 1.6) < 1e-4,
          f"{trit_b * 8 / N:.5f}")
    check("dim-8 VQ index density = 1.600 bpw", abs(code_b * 8 / N - 1.6) < 1e-4,
          f"{code_b * 8 / N:.5f}")
    check("index bytes differ by < 0.01%", abs(trit_b - code_b) / trit_b < 1e-4,
          f"{trit_b} vs {code_b} bytes")
    # dim-4 (K=81) must also reach 1.600 bpw, which needs 10 codes per word
    c4 = rng.integers(0, 81, N // 4).astype(np.int64)
    b4 = len(pack_codes(c4, 81))
    check("dim-4 VQ index density = 1.600 bpw", abs(b4 * 8 / N - 1.6) < 1e-3, f"{b4 * 8 / N:.5f}")
    check("codes_per_word adapts", codes_per_word(6561) == 5 and codes_per_word(81) == 10,
          f"K=6561 -> {codes_per_word(6561)}, K=81 -> {codes_per_word(81)}")
    check("dim-4 round-trip", np.array_equal(unpack_codes(pack_codes(c4, 81), c4.size, 81), c4))

    # --- full encode -> decode against the tensor actually quantized ---
    # Realistic tensor size: the shared codebook amortizes over actual weight
    # counts, so a toy size would overstate its share of the budget.
    rows, cols, group = 6144, 1024, 128
    scale = torch.rand(rows, cols // group, 1, device=DEV).add(0.1).to(torch.float16).float()

    lev = torch.tensor([-0.7, 0.0, 0.7], device=DEV)
    idx = torch.randint(0, 3, (rows, cols), device=DEV)
    want = (lev[idx].view(rows, cols // group, group) * scale).view(rows, cols)
    p = encode_scalar3(idx, scale, (rows, cols), group)
    got = decode(p, lev, DEV)
    check("scalar3 decode == quantized weights", torch.equal(got, want),
          f"max|diff|={(got - want).abs().max().item():.3e}")

    dim, k = 8, 6561
    cb = torch.randn(k, dim, device=DEV) * 0.3
    ci = torch.randint(0, k, (rows * cols // dim,), device=DEV)
    want = (cb[ci].view(rows, cols // group, group) * scale).view(rows, cols)
    p2 = encode_vq(ci, scale, (rows, cols), group, dim, k)
    got = decode(p2, cb, DEV)
    check("vq decode == quantized weights", torch.equal(got, want),
          f"max|diff|={(got - want).abs().max().item():.3e}")

    # --- checksum actually catches corruption ---
    bad = Packed(p2.kind, p2.shape, p2.group, p2.dim, p2.n_codes,
                 b"\x00" + p2.index_bytes[1:], p2.scale_bytes, p2.checksum)
    try:
        decode(bad, cb, DEV)
        check("checksum detects corruption", False, "no error raised")
    except AssertionError:
        check("checksum detects corruption", True)

    # --- measured bits/weight, 18 tensors, shared codebook ---
    packs_s = [encode_scalar3(idx, scale, (rows, cols), group) for _ in range(18)]
    packs_v = [encode_vq(ci, scale, (rows, cols), group, dim, k) for _ in range(18)]
    bs = bits_per_weight(packs_s, 3 * 2)
    bv = bits_per_weight(packs_v, k * dim * 2)
    print(f"\n  measured scalar3 : {bs:.4f} bits/weight")
    print(f"  measured dim-8 VQ: {bv:.4f} bits/weight   (+{(bv / bs - 1) * 100:.2f}%)")
    check("dim-8 VQ within 1% of scalar bits/weight", abs(bv / bs - 1) < 0.01,
          f"codebook is {k * dim * 2} B over {18 * rows * cols / 1e6:.0f}M weights")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
