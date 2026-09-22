"""Bounded, serialized low-rate VQ feasibility study (version 2).

The model remains BF16 at inference: this measures the quality of weights decoded
from honest packed artifacts, not a packed inference kernel or a speedup.  The
primary comparison is shared 8-D VQ versus per-tensor learned scalar3 g64, using
actual stored bytes as a Pareto constraint.  Scalar3 is not literal ternary.

Run --preflight to inspect the frozen protocol without importing torch, loading
a model, or creating output.  --smoke uses a SMALL TRAINING subset and one
projection only; its numbers are explicitly non-reportable.  A normal run uses
ALL validation targets (including the last partial block), all 18 DeltaNet input
projections, and refuses to overwrite any nonempty output directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

SCHEMA = "lowrate_feasibility_v2"
PRIMARY_CANDIDATE = "vq8_shared_g128_rot"
PRIMARY_REFERENCE = "scalar3_g64_rot"
ARMS = (
    ("ternary_g128_rot", "ternary", 128, 1),
    ("scalar3_g128_rot", "scalar", 128, 1),
    (PRIMARY_REFERENCE, "scalar", 64, 1),
    (PRIMARY_CANDIDATE, "vq", 128, 8),
)


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256_file(path: Path, check_budget=lambda: None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            check_budget()
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(root: Path, check_budget=lambda: None) -> dict:
    """Hash the actual local model/tokenizer/config inputs; no Hub auto-download."""
    if not root.is_dir():
        raise FileNotFoundError(f"Local model directory missing: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {
        ".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"})
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"No model weight files found in {root}")
    return {"path": str(root.resolve()), "files": {
        str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256_file(p, check_budget)}
        for p in files}}


def token_blocks(ids: list[int], context: int = 128) -> list[list[int]]:
    """Each target after token zero appears exactly once, including a short tail."""
    if context <= 0:
        raise ValueError("context must be positive")
    return [ids[start: start + context + 1]
            for start in range(0, max(0, len(ids) - 1), context)]


def aggregate_blocks(sums: list[float], counts: list[int]) -> dict:
    if len(sums) != len(counts) or not counts or any(n <= 0 for n in counts):
        raise ValueError("Paired nonempty block sums and positive target counts required")
    if not all(math.isfinite(x) for x in sums):
        raise ValueError("Nonfinite loss in evaluation")
    nll = math.fsum(sums) / sum(counts)
    return {"nll": nll, "ppl": math.exp(nll) if nll < 700 else None,
            "blocks": len(counts), "target_tokens": sum(counts),
            "block_loss_sums": sums, "block_target_counts": counts}


def paired_bootstrap(candidate: dict, reference: dict, reps: int = 5000,
                     seed: int = 7) -> dict:
    """Paired block bootstrap of token-weighted NLL; unequal tail lengths retained."""
    import numpy as np
    counts = np.asarray(candidate["block_target_counts"], dtype=np.int64)
    if candidate["block_target_counts"] != reference["block_target_counts"] or not len(counts):
        raise ValueError("Paired evaluations must have identical target counts in each block")
    if reps <= 0:
        raise ValueError("bootstrap replicates must be positive")
    diff = (np.asarray(candidate["block_loss_sums"], dtype=np.float64)
            - np.asarray(reference["block_loss_sums"], dtype=np.float64))
    if diff.shape != counts.shape or not np.isfinite(diff).all() or (counts <= 0).any():
        raise ValueError("Invalid paired loss records")
    rng = np.random.default_rng(seed)
    draws = []
    # Bound temporary bootstrap memory, independently of the evaluation length.
    for start in range(0, reps, 128):
        ix = rng.integers(0, len(counts), size=(min(128, reps - start), len(counts)))
        draws.extend((diff[ix].sum(1) / counts[ix].sum(1)).tolist())
    lo, hi = np.percentile(draws, [2.5, 97.5]).tolist()
    return {"delta_nll": float(diff.sum() / counts.sum()), "ci95": [lo, hi],
            "bootstrap_replicates": reps, "bootstrap_seed": seed,
            "resampling_unit": "paired context-reset blocks; token-weighted ratio per draw"}


def primary_gate(candidate: dict, reference: dict, bf16: dict, comparison: dict,
                 *, reportable: bool) -> dict:
    damage = reference["nll"] - bf16["nll"]
    gain = reference["nll"] - candidate["nll"]
    recovery = gain / damage if damage > 0 else None
    checks = {
        "practical_gain_at_least_0.005_nats_per_target": gain >= 0.005,
        "paired_ci_upper_below_zero": comparison["ci95"][1] < 0,
        "candidate_actual_bytes_no_greater_than_reference": (
            candidate["storage"]["total_bytes"] <= reference["storage"]["total_bytes"]),
        "reference_damage_positive": damage > 0,
        "recovers_at_least_10_percent_reference_damage": recovery is not None and recovery >= 0.10,
    }
    return {"status": ("PASS_EXPLORATORY" if all(checks.values()) else "STOP")
            if reportable else "NOT_APPLICABLE_SMOKE",
            "checks": checks, "reference_damage_nll": damage,
            "gain_nll": gain, "recovered_damage_fraction": recovery,
            "interpretation": "One quantization seed; a pass authorizes confirmation, not a paper claim."}


@contextmanager
def run_lock(path: Path):
    """Advisory lock shared by all runs of this runner, released on process exit."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another low-rate run owns {path}; refusing concurrent GPU work") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Deadline:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError(f"Hard experiment budget exhausted ({self.seconds:g} seconds)")

    def __enter__(self):
        if self.seconds <= 0:
            raise ValueError("seconds must be positive")
        self.previous = signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def _alarm(self, *_):
        raise TimeoutError(f"Hard experiment budget exhausted ({self.seconds:g} seconds)")

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


def refuse_nonempty_output(path: Path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output is not empty: {path}. Use a new directory; automatic resume is disabled.")
    path.mkdir(parents=True, exist_ok=True)


def check_gpu_idle(device: str):
    if not device.startswith("cuda"):
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10)
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    others = [pid for pid in pids if pid != os.getpid()]
    if others:
        raise RuntimeError(f"GPU compute processes already active (PIDs {others}); refusing competing work")


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def gpu_peak(device, reset=False):
    import torch
    if not str(device).startswith("cuda"):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def evaluate(model, blocks: list[list[int]], batch_size: int = 8, device="cuda",
             check_budget=lambda: None) -> dict:
    """Evaluate complete target coverage without padding the final short block."""
    import torch
    import torch.nn.functional as F
    if not blocks:
        raise ValueError("No evaluation targets")
    sums, counts = [], []
    synchronize(device)
    gpu_peak(device, reset=True)
    start = time.monotonic()
    with torch.no_grad():
        i = 0
        while i < len(blocks):
            check_budget()
            n = len(blocks[i])
            j = i + 1
            while j < min(len(blocks), i + batch_size) and len(blocks[j]) == n:
                j += 1
            x = torch.tensor(blocks[i:j], dtype=torch.long, device=device)
            logits = model(x[:, :-1], use_cache=False).logits.float()
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     x[:, 1:].reshape(-1), reduction="none").reshape(x.shape[0], -1)
            sums.extend(losses.double().sum(1).cpu().tolist())
            counts.extend([n - 1] * (j - i))
            del logits, losses, x
            i = j
    synchronize(device)
    return {**aggregate_blocks(sums, counts), "evaluation_seconds": time.monotonic() - start,
            "gpu_memory": gpu_peak(device), "context_reset_each_block": True}


def get_projections(model):
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return {f"layers.{i}.linear_attn.in_proj_qkv.weight": layer.linear_attn.in_proj_qkv.weight
            for i, layer in enumerate(lm.layers)
            if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "in_proj_qkv")}


def parser():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=root / "models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", type=Path, default=root / "data/wikitext2")
    ap.add_argument("--out", type=Path, default=root / "results/lowrate_task_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=None, help="Hard cap; default smoke 900, full 10800")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-points", type=int, default=None, help="Total fit sample cap; default smoke 8192, full 65536")
    ap.add_argument("--fit-iterations", type=int, default=None, help="Default smoke 2, full 20")
    ap.add_argument("--distance-memory-mb", type=int, default=128)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--expected-validation-targets", type=int, default=261284,
                    help="Reject unexpected full-stream tokenization; explicit dataset changes must change this")
    ap.add_argument("--lock-file", type=Path, default=root / "results/.lowrate_gpu.lock")
    ap.add_argument("--smoke", action="store_true", help="TRAIN only, 2048 targets, first projection; non-reportable")
    ap.add_argument("--preflight", action="store_true", help="Print protocol and input availability; no GPU/model load or writes")
    return ap


def build_protocol(a) -> dict:
    if min(a.context, a.batch_size, a.bootstrap_reps, a.distance_memory_mb) <= 0:
        raise ValueError("Context, batch size, bootstrap reps, and memory cap must be positive")
    seconds = a.seconds if a.seconds is not None else (900 if a.smoke else 10800)
    fit_points = a.fit_points if a.fit_points is not None else (8192 if a.smoke else 65536)
    fit_iterations = a.fit_iterations if a.fit_iterations is not None else (2 if a.smoke else 20)
    if seconds <= 0 or fit_iterations <= 0 or fit_points < 6561:
        raise ValueError("Positive budget/iterations and at least 6561 fit points required for dim8 codebook")
    split = "train" if a.smoke else "validation"
    codec_path = Path(__file__).with_name("lowrate_codec.py")
    protocol = {
        "schema": SCHEMA, "model": str(a.model.resolve()), "dataset": str(a.data.resolve()),
        "dataset_name": "WikiText-2", "split": split, "mode": "smoke" if a.smoke else "full_validation",
        "reportable": not a.smoke, "target_cap": 2048 if a.smoke else None,
        "expected_model_projections": 18, "selected_projections": "first only" if a.smoke else "all 18",
        "context": a.context, "batch_size": a.batch_size, "seed": a.seed,
        "seconds": seconds, "fit_points": fit_points, "fit_iterations": fit_iterations,
        "distance_memory_mb": a.distance_memory_mb, "device": a.device,
        "bootstrap_reps": a.bootstrap_reps, "bootstrap_seed": 7,
        "expected_validation_targets": a.expected_validation_targets,
        "arms": [{"name": n, "kind": k, "group_size": g, "dim": d,
                  "rotate": True, "block_size": 1024,
                  "codebook_scope": "shared_across_selected_projections" if k == "vq" else
                                    ("per_tensor" if k == "scalar" else "literal_ternary")}
                 for n, k, g, d in ARMS],
        "primary": {"candidate": PRIMARY_CANDIDATE, "reference": PRIMARY_REFERENCE,
                    "practical_gain_nll": 0.005, "max_ci_upper": 0,
                    "minimum_reference_damage_recovery": 0.10,
                    "requires_positive_reference_damage": True,
                    "byte_rule": "candidate total artifact bytes <= reference total artifact bytes"},
        "fit_data": "Weights only; no validation-dependent fitting or selection",
        "inference": "Decode serialized FP16 scales/codebooks, invert rotation, install BF16 weights",
        "runtime_claim": "No packed inference speed or resident VRAM advantage demonstrated",
        "statistical_scope": "Exploratory one-seed gate; secondary comparisons descriptive only",
        "resume": "Disabled; nonempty output rejected",
        "source_sha256": {Path(__file__).name: sha256_file(Path(__file__)),
                          codec_path.name: sha256_file(codec_path) if codec_path.exists() else None},
        "input_availability": {"model_directory": a.model.is_dir(),
                               "dataset_split": (a.data / f"{split}.parquet").is_file()},
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def write_summary(out: Path, results: dict):
    lines = ["# Serialized low-rate VQ feasibility study", "",
             f"Status: **{results['status']}**. "
             + ("TRAINING smoke only: NON-REPORTABLE." if not results['protocol']['reportable'] else
                "Exploratory, one quantization seed; all validation targets with a partial tail."), "",
             "Inference uses decoded BF16 weights; these timings do not measure packed inference.", "",
             "| Arm | Actual artifact bytes | Bits/quantized weight | NLL | Targets | Quantize s | Evaluate s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in results["arms"].items():
        st = row.get("storage", {})
        lines.append(f"| {name} | {st.get('total_bytes', '—')} | "
                     f"{st.get('bits_per_weight', '—')} | {row['nll']:.6f} | "
                     f"{row['target_tokens']} | {row.get('quantization_seconds', 0):.1f} | "
                     f"{row['evaluation_seconds']:.1f} |")
    if "primary_comparison" in results:
        p = results["primary_comparison"]
        lines += ["", f"Primary: `{PRIMARY_CANDIDATE}` minus `{PRIMARY_REFERENCE}`: "
                  f"ΔNLL {p['delta_nll']:+.6f}, 95% paired bootstrap CI "
                  f"[{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}].", "",
                  f"Gate: **{results['primary_gate']['status']}**."]
    lines += ["", "Artifact bytes include all serialized scales, indices, signs, codebooks, and manifests. "
              "Only selected projection weights are compressed; untouched model weights are excluded "
              "from this per-projection rate and retain BF16 residency.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def execute(a, protocol: dict, budget: Deadline, out: Path) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lowrate_codec import CodecConfig, ArtifactWriter, decode_tensor, encode_tensor, fit_codebook, prepare

    check_gpu_idle(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # Avoid TF32 changing distance/ranking across machines; reproducibility still
    # requires matching recorded torch/CUDA/GPU/software versions.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = a.data / f"{protocol['split']}.parquet"
    provenance = {"model": model_provenance(a.model, budget.check),
                  "data": {"path": str(source.resolve()), "bytes": source.stat().st_size,
                           "sha256": sha256_file(source, budget.check)},
                  "packages": {name: importlib.metadata.version(name)
                               for name in ("torch", "transformers", "numpy", "pandas")},
                  "cuda_version": torch.version.cuda,
                  "device": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else a.device}
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    texts = pd.read_parquet(source).text.tolist()
    # A smoke never reads the validation split.  The full run tokenizes every row.
    ids = tok("\n".join(texts), add_special_tokens=False).input_ids
    source_targets = len(ids) - 1
    if not a.smoke and source_targets != a.expected_validation_targets:
        raise ValueError(f"Full stream has {source_targets} targets, expected {a.expected_validation_targets}; "
                         "check dataset/tokenizer provenance before changing the protocol")
    if a.smoke:
        ids = ids[:protocol["target_cap"] + 1]
    blocks = token_blocks(ids, a.context)
    provenance["data"].update({"rows": len(texts), "source_target_tokens": source_targets,
                               "evaluated_target_tokens": len(ids) - 1,
                               "token_ids_sha256": hashlib.sha256(
                                   b"".join(struct.pack("<I", x) for x in ids)).hexdigest(),
                               "blocks": len(blocks), "tail_targets": len(blocks[-1]) - 1,
                               "join": "newline", "add_special_tokens": False})
    atomic_json(out / "provenance.json", provenance)
    budget.check()
    model = AutoModelForCausalLM.from_pretrained(
        str(a.model), dtype=torch.bfloat16, device_map=a.device, local_files_only=True).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    all_params = get_projections(model)
    if len(all_params) != 18:
        raise ValueError(f"Expected 18 Qwen3.5-0.8B DeltaNet input projections; got {len(all_params)}")
    params = dict(list(all_params.items())[:1]) if a.smoke else all_params
    originals = {name: param.detach().clone().cpu() for name, param in params.items()}
    provenance["quantized_tensors"] = {
        name: {"shape": list(w.shape), "source_dtype": str(w.dtype),
               "sha256_bf16_bytes": hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name, w in originals.items()}
    provenance["total_model_parameters"] = sum(p.numel() for p in model.parameters())
    provenance["quantized_parameters"] = sum(w.numel() for w in originals.values())
    provenance["base_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    provenance["untouched_parameter_bytes"] = (provenance["base_parameter_bytes"]
        - sum(w.numel() * w.element_size() for w in originals.values()))
    atomic_json(out / "provenance.json", provenance)
    result = {"schema": SCHEMA, "status": "RUNNING", "protocol": protocol,
              "provenance_file": "provenance.json", "arms": {}}

    def checkpoint():
        result["elapsed_seconds"] = time.monotonic() - budget.started
        atomic_json(out / "results.json", result)
        write_summary(out, result)

    def restore():
        with torch.no_grad():
            for name, param in params.items():
                budget.check()
                param.copy_(originals[name].to(device=param.device, dtype=param.dtype))

    try:
        baseline = evaluate(model, blocks, a.batch_size, a.device, budget.check)
        result["arms"]["bf16"] = baseline
        checkpoint()
        print(json.dumps({"arm": "bf16", "nll": baseline["nll"],
                          "targets": baseline["target_tokens"], "mode": protocol["mode"]}), flush=True)
        for name, kind, group, dim in ARMS:
            budget.check()
            restore()
            synchronize(a.device)
            gpu_peak(a.device, reset=True)
            start = time.monotonic()
            cfg = CodecConfig(kind=kind, group_size=group, dim=dim, rotate=True,
                              block_size=1024, seed=a.seed,
                              max_points=protocol["fit_points"], iterations=protocol["fit_iterations"],
                              memory_mb=a.distance_memory_mb)
            artifact_dir = out / "artifacts" / name
            # Only the shared codebook arm needs all prepared tensors concurrently.
            shared_prepared = None
            shared_book = None
            if kind == "vq":
                shared_prepared = {}
                for key, weight in originals.items():
                    budget.check()
                    shared_prepared[key] = prepare(weight.to(a.device).float(), cfg)
                shared_book = fit_codebook(list(shared_prepared.values()), cfg, check_budget=budget.check)
            writer = ArtifactWriter(artifact_dir, cfg, codebook=shared_book)
            for key, weight in originals.items():
                budget.check()
                prepared = (shared_prepared[key] if shared_prepared is not None else
                            prepare(weight.to(a.device).float(), cfg))
                book = shared_book
                if kind == "scalar":
                    book = fit_codebook([prepared], cfg, check_budget=budget.check)
                encoded = encode_tensor(prepared, cfg, codebook=book, check_budget=budget.check)
                writer.add(key, encoded, codebook=book if kind == "scalar" else None)
                del prepared, encoded
            storage = writer.finalize()
            storage["projected_model_parameter_bytes"] = (
                provenance["untouched_parameter_bytes"] + storage["total_bytes"])
            storage["projected_model_note"] = ("Accounting projection only: untouched BF16 parameters plus "
                                               "actual codec files; not a full-model artifact or VRAM claim")
            # Load exclusively from finalized files; no in-memory quantized tensor
            # is used by the evaluation model.
            del shared_prepared, shared_book, writer
            mse_sum, weight_count = 0.0, 0
            with torch.no_grad():
                for key, param in params.items():
                    budget.check()
                    decoded = decode_tensor(artifact_dir, key, device=a.device, dtype=param.dtype)
                    param.copy_(decoded)
                    mse_sum += (decoded.float() - originals[key].to(a.device).float()).square().sum().item()
                    weight_count += decoded.numel()
                    del decoded
            synchronize(a.device)
            quant_seconds = time.monotonic() - start
            quant_memory = gpu_peak(a.device)
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
            ev = evaluate(model, blocks, a.batch_size, a.device, budget.check)
            ev.update({"storage": storage, "codec_config": asdict(cfg),
                       "artifact_path": str(artifact_dir.relative_to(out)),
                       "quantization_seconds": quant_seconds, "quantization_gpu_memory": quant_memory,
                       "decoded_bf16_weight_mse": mse_sum / weight_count,
                       "versus_bf16": paired_bootstrap(ev, baseline, a.bootstrap_reps)})
            result["arms"][name] = ev
            checkpoint()
            print(json.dumps({"arm": name, "nll": ev["nll"], "storage": storage,
                              "quantization_seconds": quant_seconds,
                              "evaluation_seconds": ev["evaluation_seconds"]}), flush=True)
        candidate, reference = result["arms"][PRIMARY_CANDIDATE], result["arms"][PRIMARY_REFERENCE]
        comparison = paired_bootstrap(candidate, reference, a.bootstrap_reps)
        result["primary_comparison"] = comparison
        result["primary_gate"] = primary_gate(candidate, reference, baseline, comparison,
                                                reportable=protocol["reportable"])
        result["status"] = "COMPLETE_SMOKE_NONREPORTABLE" if a.smoke else "COMPLETE_EXPLORATORY"
        checkpoint()
        return result
    except BaseException as error:
        result["status"] = "TIMEOUT" if isinstance(error, TimeoutError) else "FAILED"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        checkpoint()
        raise


def main():
    a = parser().parse_args()
    protocol = build_protocol(a)
    if a.preflight:
        print(json.dumps(protocol, indent=2))
        return
    with run_lock(a.lock_file):
        refuse_nonempty_output(a.out)
        # Protocol exists before any model/data access or CUDA computation.
        atomic_json(a.out / "protocol.json", protocol)
        try:
            with Deadline(protocol["seconds"]) as budget:
                result = execute(a, protocol, budget, a.out)
        except BaseException as error:
            atomic_json(a.out / "failure.json", {"type": type(error).__name__, "message": str(error)})
            raise
        print(json.dumps({"status": result["status"], "primary_gate": result["primary_gate"]}), flush=True)


if __name__ == "__main__":
    main()
