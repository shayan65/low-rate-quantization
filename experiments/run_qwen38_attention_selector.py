"""Bounded train-calibrated polar/real4 selector with disjoint WikiText-2 validation."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from run_qwen38_projection_pilot import evaluate
from run_qwen_crosslayer_screen import real4
from run_qwen_taskaware_pairing import polar35

LAYERS = (7, 11, 15)
PAIRINGS = ("adjacent", "split_half", "reverse_half", "stride257")
REAL_FACTORS = (0.90, 1.00, 1.10)

def save(path, value):
    p = Path(path); q = p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(value, indent=2)+"\n"); q.replace(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.8-27B-metadata")
    ap.add_argument("--out", default="results/qwen38_27b_attention_selector_v1")
    ap.add_argument("--seconds", type=float, default=900)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.model)
    train_ids = tok("\n".join(pd.read_parquet("data/wikitext2/train.parquet").text.tolist()), add_special_tokens=False).input_ids
    val_ids = tok("\n".join(pd.read_parquet("data/wikitext2/validation.parquet").text.tolist()), add_special_tokens=False).input_ids
    rng = np.random.default_rng(20260920)
    starts = np.sort(rng.choice(len(train_ids)-129, 4, replace=False)).tolist()
    cal = [train_ids[s:s+129] for s in starts]
    held = [val_ids[s:s+129] for s in range(128*128, 160*128, 128)]
    assert len(held)==32 and all(len(w)==129 for w in held)
    plan = {"model":"Qwen/Qwen3.8-27B", "layers":LAYERS, "tensor":"self_attn.q_proj.weight",
            "calibration":"4 fixed 129-token WikiText-2 train windows", "train_starts":starts,
            "heldout":"WikiText-2 validation blocks 128–159, 4,096 targets, disjoint from previous Qwen3.8 screens",
            "polar_candidates":PAIRINGS, "real4_factors":REAL_FACTORS,
            "selection":"minimum train-calibration NLL within each codec family, then compare on held-out validation",
            "scope":"one tensor at a time; equal 4-bit index payload plus row scales and candidate ID; other weights BF16",
            "runtime_cap_seconds":a.seconds}
    save(out/"plan.json", plan)
    weight_map = json.loads((Path(a.model)/"model.safetensors.index.json").read_text())["weight_map"]
    candidates = {}
    for layer in LAYERS:
        key = f"model.language_model.layers.{layer}.self_attn.q_proj.weight"
        with safe_open(Path(a.model)/weight_map[key], framework="pt", device="cpu") as f:
            w = f.get_tensor(key).float().cuda()
        q0, _, s0 = real4(w, True)
        variants = []
        for factor in REAL_FACTORS:
            s = s0*factor
            q = s*((w/s).clamp(-1,1).add(1).mul(7.5).round().div(7.5).sub(1))
            variants.append({"codec":"real4", "candidate":factor, "weight":q.to(torch.bfloat16).cpu(),
                             "weight_mse":(q-w).square().mean().item()})
            del q
        for pairing in PAIRINGS:
            q, _, _, _ = polar35(w, pairing)
            variants.append({"codec":"polar3_5", "candidate":pairing, "weight":q.to(torch.bfloat16).cpu(),
                             "weight_mse":(q-w).square().mean().item()})
            del q
        candidates[layer] = variants
        del w, q0, s0
        torch.cuda.empty_cache()
        print(json.dumps({"stage":"quantized", "layer":layer}), flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
            max_memory={0:"20GiB", "cpu":"40GiB"}, low_cpu_mem_usage=True,
            offload_folder=str(out/"offload")).eval()
    results = []
    for layer in LAYERS:
        p = model.model.layers[layer].self_attn.q_proj.weight
        original = p.detach().cpu().clone()
        cal_rows = []
        for v in candidates[layer]:
            if time.monotonic()-start > a.seconds: raise TimeoutError("wall cap")
            p.data.copy_(v["weight"])
            metric = evaluate(model, cal, 4)
            cal_rows.append({"codec":v["codec"], "candidate":v["candidate"], "nll":metric["nll"],
                             "weight_mse":v["weight_mse"], "seconds":metric["seconds"]})
        p.data.copy_(original)
        best = {codec:min((r for r in cal_rows if r["codec"]==codec), key=lambda r:r["nll"])
                for codec in ("real4", "polar3_5")}
        held_rows = []
        for codec in ("real4", "polar3_5"):
            chosen = best[codec]["candidate"]
            v = next(v for v in candidates[layer] if v["codec"]==codec and v["candidate"]==chosen)
            p.data.copy_(v["weight"])
            metric = evaluate(model, held, 8)
            held_rows.append({"codec":codec, "candidate":chosen, **metric})
        p.data.copy_(original)
        row = {"layer":layer, "calibration":cal_rows, "selected":best, "heldout":held_rows,
               "polar_minus_real4_nll":held_rows[1]["nll"]-held_rows[0]["nll"]}
        results.append(row); save(out/"results.json", results); print(json.dumps({"layer":layer,"selected":best,
                 "heldout":held_rows,"polar_minus_real4_nll":row["polar_minus_real4_nll"]}), flush=True)
        del original, candidates[layer]
    gate = {"complete":len(results)==len(LAYERS), "seconds":time.monotonic()-start,
            "polar_heldout_wins":sum(r["polar_minus_real4_nll"]<0 for r in results),
            "note":"Train-calibrated single-projection screen; no cumulative model or full validation claim."}
    save(out/"gate.json", gate); print(json.dumps(gate),flush=True)

if __name__=="__main__":main()
