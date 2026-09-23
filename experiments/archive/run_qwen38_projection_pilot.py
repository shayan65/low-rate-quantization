"""Bounded equal-payload, one-projection Qwen3.8-27B quality pilot."""
import argparse, json, math, time
from pathlib import Path
import pandas as pd
import torch
from torch.nn import functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import real4
from run_qwen_taskaware_pairing import polar35

TARGETS = [
    ("deltanet_qkv", 0, "linear_attn.in_proj_qkv", "split_half"),
    ("attention_q", 3, "self_attn.q_proj", "adjacent"),
    ("attention_mlp_up", 3, "mlp.up_proj", "split_half"),
]

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

@torch.no_grad()
def evaluate(model, windows, batch):
    total = 0.; count = 0; start = time.monotonic()
    for i in range(0, len(windows), batch):
        x = torch.tensor(windows[i:i+batch], device="cuda")
        y = x[:, 1:]
        logits = model(x[:, :-1], use_cache=False).logits.float()
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        count += y.numel()
        del x, y, logits
    torch.cuda.synchronize()
    return {"nll": total/count, "perplexity": math.exp(total/count), "target_tokens": count,
            "seconds": time.monotonic()-start, "gpu_peak_bytes": torch.cuda.max_memory_allocated()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.8-27B-metadata")
    ap.add_argument("--out", default="results/qwen38_27b_projection_pilot_v1")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--start-block", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--target", choices=["all", "deltanet_qkv", "attention_q", "attention_mlp_up"], default="all")
    ap.add_argument("--attention-layers", default="", help="Comma-separated additional attention Q layer indices; overrides --target")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok("\n".join(pd.read_parquet("data/wikitext2/validation.parquet").text.tolist()), add_special_tokens=False).input_ids
    windows = [ids[s:s+129] for s in range(a.start_block*128, (a.start_block+a.blocks)*128, 128)]
    assert all(len(w) == 129 for w in windows)
    targets = ([(f"attention_q_l{int(i)}", int(i), "self_attn.q_proj", "adjacent") for i in a.attention_layers.split(",") if i]
               if a.attention_layers else TARGETS if a.target == "all" else [t for t in TARGETS if t[0] == a.target])
    assert targets
    plan = {"model": "Qwen/Qwen3.8-27B", "dataset": "contiguous WikiText-2 validation windows", "start_block": a.start_block,
            "target_tokens": a.blocks*128, "batch_size": 8, "scope": "one tensor quantized at a time; all other weights BF16",
            "methods": ["bf16", "optimized_row_real4", "optimized_row_polar3_5"],
            "payload": "4 index bits per real weight plus one FP32 scale per output row; polar uses 3 magnitude + 5 phase bits per pair and one pairing ID",
            "pairing_selection": "fixed before task pilot from weight-only geometry", "runtime_cap_seconds": a.seconds,
            "targets": targets}
    save(out/"plan.json", plan)
    # Construct quantized candidates before the full model occupies GPU memory.
    candidates = {}
    weight_map = json.loads((Path(a.model)/"model.safetensors.index.json").read_text())["weight_map"]
    for label, layer, path, pairing in targets:
        key = f"model.language_model.layers.{layer}.{path}.weight"
        shard = Path(a.model)/weight_map[key]
        with safe_open(shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(key).float().cuda()
            qr, _, _ = real4(w, True)
            qp, _, _, _ = polar35(w, pairing)
            candidates[label] = {"real4": qr.to(torch.bfloat16).cpu(), "polar3_5": qp.to(torch.bfloat16).cpu(),
                                 "shape": list(w.shape), "real4_mse": (qr-w).square().mean().item(),
                                 "polar_mse": (qp-w).square().mean().item()}
            del w, qr, qp
            torch.cuda.empty_cache()
            print(json.dumps({"stage": "quantized", "tensor": label}), flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
             max_memory={0:"20GiB", "cpu":"40GiB"}, low_cpu_mem_usage=True,
             offload_folder=str(out/"offload")).eval()
    torch.cuda.reset_peak_memory_stats(); base = evaluate(model, windows, 8)
    rows = [{"tensor":"none", "method":"bf16", **base}]
    save(out/"results.json", rows); print(json.dumps(rows[-1]), flush=True)
    for label, layer, path, pairing in targets:
        obj = model.model.layers[layer]
        for part in path.split("."): obj = getattr(obj, part)
        p = obj.weight
        original = p.detach().cpu().clone()
        for method in ("real4", "polar3_5"):
            if time.monotonic()-started > a.seconds: raise TimeoutError("pilot wall cap")
            p.data.copy_(candidates[label][method])
            torch.cuda.reset_peak_memory_stats()
            metric = evaluate(model, windows, 8)
            row = {"tensor":label, "method":method, "pairing":pairing if method == "polar3_5" else None,
                   "delta_nll_vs_bf16":metric["nll"]-base["nll"],
                   "weight_mse":candidates[label]["polar_mse" if method == "polar3_5" else "real4_mse"],
                   "payload_bytes":p.numel()//2+p.shape[0]*4+1, **metric}
            rows.append(row); save(out/"results.json", rows); print(json.dumps(row), flush=True)
        p.data.copy_(original)
        del original, candidates[label]
    gate = {"complete":len(rows)==1+2*len(targets), "seconds":time.monotonic()-started,
            "polar_wins":sum(next(r for r in rows if r["tensor"]==t and r["method"]=="polar3_5")["nll"] < next(r for r in rows if r["tensor"]==t and r["method"]=="real4")["nll"] for t, *_ in targets),
            "note":"Single-projection screening only; no cumulative or full-dataset quality claim."}
    save(out/"gate.json", gate); print(json.dumps(gate), flush=True)

if __name__ == "__main__": main()
