"""Real packed storage for low-rate codecs, with a decoder and checksums.

Earlier runs reported bits/weight from an information-theoretic formula
(log2(3) = 1.585) rather than from bytes anyone could actually write, and the
evaluated weights came from the in-memory float path rather than from the stored
representation.  Both are fixed here: every codec serializes to real bytes, the
byte count is measured rather than derived, and the evaluated weights are
decoded back from those bytes.

Packing.  Multi-symbol packing makes the two formats cost *exactly* the same per
index, which is what makes the comparison byte-exact rather than approximate:

    ternary   3^5 = 243 <= 2^8     -> 5 trits per byte      = 1.600 bits/weight
    dim-8 VQ  6561^5 < 2^64        -> 5 codes per 8 bytes   = 1.600 bits/weight

Naive per-symbol packing would give ternary 2.000 and dim-8 VQ 1.625, i.e. it
would *favour* the vector codec; the multi-symbol scheme is the conservative
choice as well as the realistic one.

uint64 note: 6561^5 = 1.2158e19 exceeds int64's 9.223e18, so the 5-code packing
is done in numpy uint64, not torch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

TRITS_PER_BYTE = 5


def codes_per_word(k: int) -> int:
    """How many base-k symbols fit in one uint64 without overflow.

    Fixing this at 5 is right for K=6561 but wastes half a word for K=81
    (81^5 needs only 32 bits), which would inflate dim-4 to 3.2 bits/weight.
    K=6561 -> 5 codes (40 weights at dim 8); K=81 -> 10 codes (40 weights at
    dim 4); both land on exactly 1.600 bits/weight.
    """
    c = 0
    v = 1
    while v * k < 2 ** 64:
        v *= k
        c += 1
    assert c >= 1, f"K={k} does not fit in a 64-bit word"
    return c


@dataclass
class Packed:
    kind: str  # "scalar3" | "vq"
    shape: tuple
    group: int
    dim: int
    n_codes: int  # logical symbol count before padding
    index_bytes: bytes
    scale_bytes: bytes  # fp16, one per group
    checksum: str

    def payload_bytes(self) -> int:
        """Bytes that must be stored per tensor (codebook counted separately)."""
        return len(self.index_bytes) + len(self.scale_bytes)


def gnorm_fp16(w: torch.Tensor, group: int):
    """g128 normalization against the scale *as it will be stored* (FP16).

    Quantizing against an FP32 scale and then storing FP16 means the decoded
    weights differ from the ones evaluated -- small (~3e-4 here) but it makes
    the byte count describe a representation that was never actually tested.
    Rounding the scale first makes decode(encode(w)) exact.
    """
    r, c = w.shape
    g = w.view(r, c // group, group)
    # Clamp AFTER the FP16 cast: 1e-12 underflows to exactly zero in FP16, so
    # clamping first leaves all-zero groups dividing by zero. FP16's smallest
    # normal is exactly representable, so the stored scale round-trips.
    s = g.abs().amax(-1, keepdim=True).to(torch.float16).float()
    s = s.clamp_min(torch.finfo(torch.float16).tiny)
    return (g / s).view(r, c), s


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# ternary: 5 trits per byte
# --------------------------------------------------------------------------

def pack_trits(idx: np.ndarray) -> bytes:
    assert idx.min() >= 0 and idx.max() <= 2, "ternary indices must be 0..2"
    n = idx.size
    pad = (-n) % TRITS_PER_BYTE
    a = np.concatenate([idx.astype(np.uint8), np.zeros(pad, np.uint8)]).reshape(-1, TRITS_PER_BYTE)
    out = np.zeros(a.shape[0], np.uint16)
    for i in range(TRITS_PER_BYTE - 1, -1, -1):
        out = out * np.uint16(3) + a[:, i].astype(np.uint16)
    return out.astype(np.uint8).tobytes()


def unpack_trits(buf: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(buf, np.uint8).astype(np.uint16)
    out = np.empty((a.size, TRITS_PER_BYTE), np.uint8)
    for i in range(TRITS_PER_BYTE):
        out[:, i] = (a % 3).astype(np.uint8)
        a //= 3
    return out.reshape(-1)[:n]


# --------------------------------------------------------------------------
# vector codes: 5 codes of K per uint64 (valid while K^5 < 2^64)
# --------------------------------------------------------------------------

def pack_codes(idx: np.ndarray, k: int) -> bytes:
    assert idx.min() >= 0 and idx.max() < k
    cpw = codes_per_word(k)
    n = idx.size
    pad = (-n) % cpw
    a = np.concatenate([idx.astype(np.uint64), np.zeros(pad, np.uint64)]).reshape(-1, cpw)
    kk = np.uint64(k)
    out = np.zeros(a.shape[0], np.uint64)
    for i in range(cpw - 1, -1, -1):
        out = out * kk + a[:, i]
    return out.tobytes()


def unpack_codes(buf: bytes, n: int, k: int) -> np.ndarray:
    a = np.frombuffer(buf, np.uint64).copy()
    cpw = codes_per_word(k)
    kk = np.uint64(k)
    out = np.empty((a.size, cpw), np.int64)
    for i in range(cpw):
        out[:, i] = (a % kk).astype(np.int64)
        a //= kk
    return out.reshape(-1)[:n]


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------

def encode_scalar3(idx: torch.Tensor, scale: torch.Tensor, shape, group: int) -> Packed:
    ib = pack_trits(idx.flatten().to(torch.uint8).cpu().numpy())
    sb = scale.flatten().to(torch.float16).cpu().numpy().tobytes()
    return Packed("scalar3", tuple(shape), group, 1, int(idx.numel()), ib, sb, _sha(ib, sb))


def encode_vq(idx: torch.Tensor, scale: torch.Tensor, shape, group: int, dim: int, k: int) -> Packed:
    ib = pack_codes(idx.flatten().cpu().numpy().astype(np.int64), k)
    sb = scale.flatten().to(torch.float16).cpu().numpy().tobytes()
    return Packed("vq", tuple(shape), group, dim, int(idx.numel()), ib, sb, _sha(ib, sb))


def decode(p: Packed, levels: torch.Tensor, device="cuda") -> torch.Tensor:
    """Reconstruct the weight tensor from stored bytes alone.

    `levels` is the shared codebook: (3,) for scalar3, (K, dim) for vq. It is
    the only thing not carried inside `Packed`, because it is shared across
    tensors and charged once.
    """
    assert _sha(p.index_bytes, p.scale_bytes) == p.checksum, "checksum mismatch"
    rows, cols = p.shape
    scale = torch.from_numpy(
        np.frombuffer(p.scale_bytes, np.float16).copy()).to(device).float().view(rows, cols // p.group, 1)
    if p.kind == "scalar3":
        idx = torch.from_numpy(unpack_trits(p.index_bytes, p.n_codes).astype(np.int64)).to(device)
        u = levels.to(device)[idx].view(rows, cols // p.group, p.group)
    else:
        k = levels.shape[0]
        idx = torch.from_numpy(unpack_codes(p.index_bytes, p.n_codes, k)).to(device)
        u = levels.to(device)[idx].view(rows, cols // p.group, p.group)
    return (u * scale).view(rows, cols)


def bits_per_weight(packs: list[Packed], codebook_bytes: int) -> float:
    """Measured bits/weight: real serialized bytes, codebook amortized over the set."""
    n = sum(p.shape[0] * p.shape[1] for p in packs)
    b = sum(p.payload_bytes() for p in packs) + codebook_bytes
    return b * 8.0 / n
