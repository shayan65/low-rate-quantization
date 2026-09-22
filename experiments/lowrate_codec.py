"""Measured low-rate weight codecs, not a packed inference kernel.

Artifacts contain JSON plus little-endian raw arrays (never pickle).  Literal
ternary and learned scalar3 store five base-3 indices per byte, restarting at
each scale group.  VQ stores ceil(log2(3**dim)) bits per vector, without claiming
entropy-rate storage.  Rounded FP16 scales/codebooks are used during fitting,
assignment and decoding.  The optional Hadamard signs are stored explicitly.

Torch is imported lazily so packing and artifact validation work with NumPy
alone.  Fits are empirical, sampled weighted Lloyd fits, not distortion bounds.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class CodecConfig:
    kind: str
    group_size: int = 128
    dim: int = 8
    rotate: bool = True
    block_size: int = 1024
    seed: int = 0
    max_points: int = 65536
    iterations: int = 12
    memory_mb: float = 64.0
    random_restart: bool = False

    def __post_init__(self):
        if self.kind not in ("ternary", "scalar", "vq"):
            raise ValueError("kind must be ternary, scalar or vq")
        if self.group_size < 1 or self.dim < 1 or self.group_size % self.vector_dim:
            raise ValueError("vector dimension must divide group size")
        if self.block_size < 1 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a positive power of two")
        if self.max_points < 1 or self.iterations < 0 or self.memory_mb <= 0:
            raise ValueError("invalid fitting budget")
        if self.kind == "vq" and self.k > 65536:
            raise ValueError("explicit codebooks are capped at 65536 entries")

    @property
    def vector_dim(self):
        return self.dim if self.kind == "vq" else 1

    @property
    def k(self):
        return 3 ** self.vector_dim


@dataclass
class PreparedTensor:
    u: object
    scales: object
    signs: Optional[np.ndarray]
    shape: tuple


@dataclass
class EncodedTensor:
    shape: tuple
    scales: np.ndarray
    indices: np.ndarray
    signs: Optional[np.ndarray] = None
    codebook: Optional[np.ndarray] = None


def _check(check_budget):
    if check_budget is not None:
        check_budget()


def pack_trits(indices, group_size=128):
    """Five little-endian trits per byte, with per-group zero padding."""
    x = np.asarray(indices)
    if x.size % group_size or np.any(x < 0) or np.any(x > 2):
        raise ValueError("trit indices must lie in [0,2] and fill scale groups")
    x = x.astype(np.uint8).reshape(-1, group_size)
    padded = np.zeros((len(x), math.ceil(group_size / 5) * 5), dtype=np.uint8)
    padded[:, :group_size] = x
    return (padded.reshape(len(x), -1, 5).astype(np.uint16)
            @ np.array([1, 3, 9, 27, 81], dtype=np.uint16)).astype(np.uint8).tobytes()


def unpack_trits(payload, count, group_size=128):
    if count < 0 or count % group_size:
        raise ValueError("invalid trit count")
    size = math.ceil(group_size / 5)
    a = np.frombuffer(payload, dtype=np.uint8)
    if len(a) != (count // group_size) * size or np.any(a > 242):
        raise ValueError("invalid trit payload length or values")
    a = a.reshape(-1, size)
    if group_size % 5 and np.any(a[:, -1] >= 3 ** (group_size % 5)):
        raise ValueError("nonzero trit padding")
    result = ((a[:, :, None].astype(np.uint16) //
               np.array([1, 3, 9, 27, 81], dtype=np.uint16)) % 3)
    return result.reshape(len(a), -1)[:, :group_size].astype(np.uint8).reshape(-1)


def pack_indices(indices, bits):
    """Continuous little-endian fixed-width bit packing, zero tail padding."""
    x = np.asarray(indices).reshape(-1)
    if not 1 <= bits <= 16 or np.any(x < 0) or np.any(x >= (1 << bits)):
        raise ValueError("invalid index bit width or values")
    parts = []
    for start in range(0, len(x), 32768):
        chunk = x[start:start + 32768].astype(np.uint32)
        bit = ((chunk[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
        parts.append(np.packbits(bit.reshape(-1), bitorder="little").tobytes())
    return b"".join(parts)


def unpack_indices(payload, count, bits, k=None):
    if count < 0 or not 1 <= bits <= 16 or len(payload) != math.ceil(count * bits / 8):
        raise ValueError("invalid index payload length")
    if count * bits % 8 and payload and payload[-1] >> (count * bits % 8):
        raise ValueError("nonzero index padding")
    chunks = []
    for start in range(0, count, 32768):
        n = min(32768, count - start)
        begin = start * bits // 8
        raw = np.frombuffer(payload[begin:begin + math.ceil(n * bits / 8)], dtype=np.uint8)
        b = np.unpackbits(raw, bitorder="little")[:n * bits].reshape(n, bits)
        chunks.append((b.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(1).astype(np.uint16))
    result = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
    if k is not None and np.any(result >= k):
        raise ValueError("index outside codebook")
    return result


def fwht(x):
    """Orthonormal Walsh-Hadamard transform on the final power-of-two axis."""
    import torch
    n = x.shape[-1]
    if not n or n & (n - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    original = x.shape
    y = x.float().reshape(-1, n).clone()
    h = 1
    while h < n:
        z = y.reshape(-1, n // (2 * h), 2, h)
        a, b = z[:, :, 0, :].clone(), z[:, :, 1, :].clone()
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape(original)


def _fp16_scales(x):
    import torch
    if not torch.isfinite(x).all() or torch.any(x > 65504):
        raise ValueError("scales must be finite and representable as FP16")
    # Zero groups use scale 1; nonzero tiny groups use the smallest FP16 subnormal.
    return torch.where(x == 0, torch.ones_like(x), x.clamp_min(2 ** -24)).half()


def prepare(weight, config, check_budget=None):
    import torch
    _check(check_budget)
    if weight.ndim != 2 or weight.shape[1] % config.group_size:
        raise ValueError("weight must be a matrix with complete scale groups per row")
    w = weight.detach().float()
    if not torch.isfinite(w).all():
        raise ValueError("nonfinite weights")
    signs = None
    if config.rotate:
        if w.shape[1] % config.block_size:
            raise ValueError("column count must be divisible by Hadamard block size")
        # CPU NumPy generation only; artifacts store the resulting sign bits.
        signs = np.random.default_rng(config.seed).integers(0, 2, w.shape[1], dtype=np.uint8)
        sign = torch.from_numpy(signs.astype(np.float32) * 2 - 1).to(w.device)
        w = fwht((w * sign).reshape(-1, config.block_size)).reshape(weight.shape)
    g = w.reshape(-1, config.group_size)
    scale = _fp16_scales(g.abs().amax(1))
    if config.kind == "ternary":
        # Alternating groupwise LS fit, retaining best rounded scale per group.
        best = scale.clone()
        best_loss = torch.full_like(scale.float(), float("inf"))
        for _ in range(max(1, config.iterations)):
            _check(check_budget)
            s = scale.float()[:, None]
            selected = g.abs() > s * 0.5
            q = torch.where(selected, g.sign(), torch.zeros_like(g)) * s
            loss = (g - q).square().sum(1)
            improve = loss < best_loss
            best[improve] = scale[improve]
            best_loss = torch.minimum(loss, best_loss)
            count = selected.sum(1)
            new = (g.abs() * selected).sum(1) / count.clamp_min(1)
            scale = _fp16_scales(torch.where(count > 0, new, torch.zeros_like(new)))
        scale = best
    return PreparedTensor((g / scale.float()[:, None]).reshape(w.shape), scale,
                          signs, tuple(weight.shape))


def _nearest(points, book, memory_mb, check_budget=None):
    import torch
    book = book.float()
    # Allow three distance-sized FP32 buffers plus a small floor of one row.
    chunk = max(1, int(memory_mb * 1024 ** 2 / (12 * len(book))))
    indices = torch.empty(len(points), dtype=torch.long, device=points.device)
    norms = book.square().sum(1)
    for start in range(0, len(points), chunk):
        _check(check_budget)
        p = points[start:start + chunk].float()
        dist = p @ book.T
        dist.mul_(-2).add_(norms)
        indices[start:start + chunk] = dist.argmin(1)
    return indices


def _sample(prepared, config, check_budget=None):
    import torch
    if not prepared:
        raise ValueError("need at least one prepared tensor")
    points, weights = [], []
    rng = np.random.default_rng(config.seed)
    remaining = config.max_points
    for number, p in enumerate(prepared):
        _check(check_budget)
        flat = p.u.reshape(-1, config.vector_dim)
        count = min(len(flat), max(1, remaining // (len(prepared) - number)))
        if remaining <= 0:
            break
        sel = rng.choice(len(flat), size=count, replace=False)
        index = torch.from_numpy(sel).to(flat.device)
        points.append(flat[index])
        per_group = config.group_size // config.vector_dim
        weights.append(p.scales.float()[index // per_group].square())
        remaining -= count
    return torch.cat(points), torch.cat(weights)


def _scalar_fit(points, weights, config, check_budget=None):
    import torch
    v = points.reshape(-1).float()
    weight = weights[:, None].expand(-1, points.shape[1]).reshape(-1).float()
    c = torch.quantile(v, torch.tensor([0.15, 0.5, 0.85], device=v.device)).half().float()
    best, best_loss = c.clone(), float("inf")
    for iteration in range(config.iterations + 1):
        _check(check_budget)
        idx = _nearest(v[:, None], c[:, None], config.memory_mb, check_budget)
        loss = ((v - c[idx]).square() * weight).double().sum().item()
        if loss < best_loss:
            best, best_loss = c.clone(), loss
        if iteration == config.iterations:
            break
        num = torch.zeros(3, device=v.device, dtype=torch.float64)
        den = torch.zeros_like(num)
        num.scatter_add_(0, idx, (v * weight).double())
        den.scatter_add_(0, idx, weight.double())
        c = torch.where(den > 0, num / den.clamp_min(1e-300), c.double()).float()
        c = c.half().float().sort().values
    return best.half()


def fit_codebook(prepared, config, check_budget=None):
    """Weighted sampled Lloyd fit; product initialization retained as fallback.

    Fallback is selected on the sampled *raw-weight-MSE* objective.  It does not
    guarantee improvement on held-out tensors or task loss.  One shared VQ book
    should be fitted once, passed to ArtifactWriter and reused for every tensor.
    """
    import torch
    if config.kind == "ternary":
        return None
    points, weights = _sample(list(prepared), config, check_budget)
    levels = _scalar_fit(points, weights, config, check_budget)
    if config.kind == "scalar":
        return levels.reshape(3, 1)
    product = torch.cartesian_prod(*([levels.float()] * config.dim)).reshape(config.k, config.dim)
    initializations = [product]
    if config.random_restart:
        rng = np.random.default_rng(config.seed + 1)
        sel = rng.choice(len(points), config.k, replace=len(points) < config.k)
        initializations.append(points[torch.from_numpy(sel).to(points.device)].half().float())
    best, best_loss = product.clone(), float("inf")
    for centroids in initializations:
        centroids = centroids.half().float()
        for iteration in range(config.iterations + 1):
            _check(check_budget)
            idx = _nearest(points, centroids, config.memory_mb, check_budget)
            errors = (points - centroids[idx]).square().sum(1) * weights
            loss = errors.double().sum().item()
            if loss < best_loss:
                best, best_loss = centroids.clone(), loss
            if iteration == config.iterations:
                break
            num = torch.zeros_like(centroids, dtype=torch.float64)
            den = torch.zeros(config.k, device=points.device, dtype=torch.float64)
            num.index_add_(0, idx, (points * weights[:, None]).double())
            den.index_add_(0, idx, weights.double())
            alive = den > 0
            updated = centroids.clone()
            updated[alive] = (num[alive] / den[alive, None]).float()
            dead = (~alive).nonzero(as_tuple=True)[0]
            if len(dead):
                worst = errors.topk(min(len(dead), len(points))).indices
                updated[dead[:len(worst)]] = points[worst]
            centroids = updated.half().float()
    return best.half()


def encode_tensor(prepared, config, codebook=None, check_budget=None):
    import torch
    _check(check_budget)
    if config.kind == "ternary":
        idx = torch.where(prepared.u > 0.5, 2,
                          torch.where(prepared.u < -0.5, 0, 1)).reshape(-1)
        book_np = None
    else:
        if codebook is None:
            raise ValueError("learned codec requires a codebook")
        book = torch.as_tensor(codebook, device=prepared.u.device).half().float()
        if tuple(book.shape) != (config.k, config.vector_dim) or not torch.isfinite(book).all():
            raise ValueError("invalid codebook")
        idx = _nearest(prepared.u.reshape(-1, config.vector_dim), book,
                       config.memory_mb, check_budget)
        book_np = book.cpu().numpy().astype("<f2")
    return EncodedTensor(prepared.shape, prepared.scales.cpu().numpy().astype("<f2"),
                         idx.cpu().numpy().astype(np.uint16), prepared.signs, book_np)


class ArtifactWriter:
    """Collection format; shared codebook stored once and all bytes charged."""
    def __init__(self, directory, config, codebook=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError("artifact directory must be empty")
        self.config = config
        self.manifest = {"format": "lowrate-weight-collection-v1", "config": asdict(config),
                         "scope": "target_weight_tensors_only_not_a_deployed_model", "tensors": {}}
        self.shared = None
        if codebook is not None:
            self.shared = self._book_array(codebook)
            self.manifest["shared_codebook"] = self._write("shared-codebook.f16", self.shared.tobytes(), "codebooks")
        self.finished = False

    def _book_array(self, book):
        if hasattr(book, "detach"):
            book = book.detach().cpu().numpy()
        book = np.asarray(book, dtype="<f2")
        if book.shape != (self.config.k, self.config.vector_dim) or not np.isfinite(book).all():
            raise ValueError("invalid codebook shape or values")
        return book

    def _write(self, filename, payload, component):
        (self.directory / filename).write_bytes(payload)
        return {"file": filename, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "component": component}

    def add(self, name, encoded, codebook=None):
        if self.finished or name in self.manifest["tensors"]:
            raise ValueError("finished writer or duplicate tensor name")
        shape = tuple(encoded.shape)
        if len(shape) != 2 or any(n <= 0 for n in shape) or shape[1] % self.config.group_size:
            raise ValueError("invalid tensor shape")
        count = math.prod(shape)
        scales = np.asarray(encoded.scales, dtype="<f2").reshape(-1)
        if len(scales) != count // self.config.group_size or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("invalid rounded scales")
        indices = np.asarray(encoded.indices).reshape(-1)
        if len(indices) != count // self.config.vector_dim or np.any(indices >= self.config.k) or np.any(indices < 0):
            raise ValueError("invalid indices")
        prefix = f"tensor-{len(self.manifest['tensors']):04d}"
        entry = {"shape": list(shape), "scales": self._write(prefix + ".scales.f16", scales.tobytes(), "scales")}
        if self.config.kind == "vq":
            payload = pack_indices(indices, math.ceil(math.log2(self.config.k)))
        else:
            payload = pack_trits(indices, self.config.group_size)
        entry["indices"] = self._write(prefix + ".indices.bin", payload, "indices")
        if self.config.rotate:
            signs = np.asarray(encoded.signs)
            if signs.shape != (shape[1],) or np.any((signs != 0) & (signs != 1)) or shape[1] % self.config.block_size:
                raise ValueError("invalid stored rotation signs")
            entry["signs"] = self._write(prefix + ".signs.bin", np.packbits(signs.astype(np.uint8), bitorder="little").tobytes(), "signs")
        if self.config.kind != "ternary":
            book = codebook if codebook is not None else encoded.codebook
            if self.shared is not None:
                if book is not None and not np.array_equal(self._book_array(book), self.shared):
                    raise ValueError("encoded codebook differs from shared codebook")
                entry["codebook"] = "shared"
            else:
                if book is None:
                    raise ValueError("missing per-tensor codebook")
                entry["codebook"] = self._write(prefix + ".codebook.f16", self._book_array(book).tobytes(), "codebooks")
        self.manifest["tensors"][name] = entry

    def finalize(self):
        if not self.manifest["tensors"]:
            raise ValueError("cannot finalize an empty collection")
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(self.manifest, sort_keys=True, indent=2) + "\n")
        self.finished = True
        return artifact_storage(self.directory)


def _manifest(directory):
    root = Path(directory)
    m = json.loads((root / "manifest.json").read_text())
    if m.get("format") != "lowrate-weight-collection-v1" or not isinstance(m.get("tensors"), dict):
        raise ValueError("unsupported artifact format")
    return root, m, CodecConfig(**m["config"])


def _read(root, record):
    filename = record["file"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("artifact paths must be local basenames")
    path = root / filename
    if path.is_symlink():
        raise ValueError("artifact files must not be symlinks")
    raw = path.read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("artifact length or checksum mismatch")
    return raw


def artifact_storage(directory):
    root, m, config = _manifest(directory)
    records = ([m["shared_codebook"]] if "shared_codebook" in m else [])
    total_numel = 0
    for entry in m["tensors"].values():
        total_numel += math.prod(entry["shape"])
        records.extend(v for v in entry.values() if isinstance(v, dict) and "file" in v)
    components = {"manifest": (root / "manifest.json").stat().st_size}
    seen = set()
    for record in records:
        if record["file"] in seen:
            raise ValueError("duplicate payload file reference")
        seen.add(record["file"])
        _read(root, record)
        component = record["component"]
        components[component] = components.get(component, 0) + (root / record["file"]).stat().st_size
    expected = seen | {"manifest.json"}
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError("unaccounted files in artifact directory")
    total = sum(components.values())
    return {"total_bytes": total, "total_numel": total_numel, "bits_per_weight": total * 8 / total_numel,
            "component_bytes": components, "scope": m["scope"]}


def read_encoded(directory, name):
    """Validate and read raw arrays without requiring torch."""
    root, m, config = _manifest(directory)
    entry = m["tensors"][name]
    shape = tuple(entry["shape"])
    if len(shape) != 2 or any(not isinstance(n, int) or n <= 0 for n in shape) or shape[1] % config.group_size:
        raise ValueError("invalid tensor shape")
    count = math.prod(shape)
    raw = _read(root, entry["scales"])
    if len(raw) != count // config.group_size * 2:
        raise ValueError("scale length mismatch")
    scales = np.frombuffer(raw, dtype="<f2").copy()
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("invalid scales")
    raw = _read(root, entry["indices"])
    if config.kind == "vq":
        indices = unpack_indices(raw, count // config.dim, math.ceil(math.log2(config.k)), config.k)
    else:
        indices = unpack_trits(raw, count, config.group_size)
    signs = None
    if config.rotate:
        if shape[1] % config.block_size:
            raise ValueError("invalid rotation shape")
        raw = _read(root, entry["signs"])
        if len(raw) != math.ceil(shape[1] / 8):
            raise ValueError("sign length mismatch")
        signs = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:shape[1]].copy()
    book = None
    if config.kind != "ternary":
        record = m["shared_codebook"] if entry["codebook"] == "shared" else entry["codebook"]
        raw = _read(root, record)
        if len(raw) != config.k * config.vector_dim * 2:
            raise ValueError("codebook length mismatch")
        book = np.frombuffer(raw, dtype="<f2").reshape(config.k, config.vector_dim).copy()
        if not np.isfinite(book).all():
            raise ValueError("nonfinite codebook")
    return config, EncodedTensor(shape, scales, indices, signs, book)


def decode_numpy(directory, name):
    """Reference decoder; returns original-basis FP32 weights."""
    config, e = read_encoded(directory, name)
    if config.kind == "ternary":
        values = e.indices.astype(np.float32) - 1
    else:
        values = e.codebook[e.indices].astype(np.float32).reshape(-1)
    w = (values.reshape(-1, config.group_size) * e.scales.astype(np.float32)[:, None]).reshape(e.shape)
    if config.rotate:
        y = w.reshape(-1, config.block_size).copy()
        h = 1
        while h < config.block_size:
            z = y.reshape(-1, config.block_size // (2 * h), 2, h)
            a, b = z[:, :, 0, :].copy(), z[:, :, 1, :].copy()
            y = np.stack((a + b, a - b), axis=2).reshape(-1, config.block_size)
            h *= 2
        w = (y / math.sqrt(config.block_size)).reshape(e.shape) * (e.signs.astype(np.float32) * 2 - 1)
    return w.astype(np.float32)


def decode_tensor(directory, name, device="cpu", dtype=None):
    """Decode one persisted tensor, then cast to the requested evaluation dtype."""
    import torch
    config, e = read_encoded(directory, name)
    indices = torch.from_numpy(e.indices.astype(np.int64)).to(device)
    if config.kind == "ternary":
        values = indices.float() - 1
    else:
        book = torch.from_numpy(e.codebook.astype(np.float32)).to(device)
        values = book[indices].reshape(-1)
    scales = torch.from_numpy(e.scales.astype(np.float32)).to(device)
    w = (values.reshape(-1, config.group_size) * scales[:, None]).reshape(e.shape)
    if config.rotate:
        sign = torch.from_numpy(e.signs.astype(np.float32) * 2 - 1).to(device)
        w = fwht(w.reshape(-1, config.block_size)).reshape(e.shape) * sign
    return w.to(dtype=dtype or torch.bfloat16)
