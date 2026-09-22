"""Bounded activation-aware attention quantization gate on Qwen3.8-27B."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import real4
from run_qwen_taskaware_pairing import polar35

LAYERS = (7, 11, 15)
PAIRINGS = ("adjacent", "split_half", "reverse_half", "stride257")
ALPHAS = (0., .25, .5, .75, 1.)

def save(path, obj):
    p = Path(path); t = p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj, indent=2)+"\n"); t.replace(p)

@torch.no_grad()
def evaluate(model, windows, batch=8):
    losses=[]; started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for i in range(0,len(windows),batch):
        x=torch.tensor(windows[i:i+batch],device="cuda"); y=x[:,1:]
        logits=model(x[:,:-1],use_cache=False).logits.float()
        per_token=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction="none")
        losses.extend(per_token.reshape(x.shape[0],-1).mean(1).cpu().tolist())
        del x,y,logits,per_token
    torch.cuda.synchronize()
    return {"nll":float(np.mean(losses)),"block_nll":losses,"target_tokens":128*len(windows),
            "seconds":time.monotonic()-started,"gpu_peak_bytes":torch.cuda.max_memory_allocated()}

def awq_style(w,h):
    act=h.sqrt().clamp_min(1e-8); best=None
    for alpha in ALPHAS:
        c=(act/act.mean()).pow(alpha).clamp(.1,10)
        wc=w*c; s=wc.abs().amax(1,keepdim=True).clamp_min(1e-8)
        ix=((wc/s).clamp(-1,1).add(1).mul(7.5)).round().clamp(0,15)
        q=s*(ix/7.5-1)/c
        score=((q-w).square()*h).mean().item()
        if best is None or score<best["score"]:
            best={"alpha":alpha,"score":score,"weight":q.to(torch.bfloat16).cpu(),
                  "weight_mse":(q-w).square().mean().item()}
    return best

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",default="models/Qwen3.8-27B-metadata")
    ap.add_argument("--out",default="results/qwen38_27b_attention_activation_gate_v1")
    ap.add_argument("--seconds",type=float,default=900)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(a.model)
    train_ids=tok("\n".join(pd.read_parquet("data/wikitext2/train.parquet").text.tolist()),add_special_tokens=False).input_ids
    val_ids=tok("\n".join(pd.read_parquet("data/wikitext2/validation.parquet").text.tolist()),add_special_tokens=False).input_ids
    rng=np.random.default_rng(20260921);starts=np.sort(rng.choice(len(train_ids)-129,16,replace=False)).tolist()
    cal=[train_ids[s:s+129] for s in starts]
    held=[val_ids[s:s+129] for s in range(160*128,224*128,128)]
    assert len(held)==64 and all(len(x)==129 for x in held)
    plan={"model":"Qwen/Qwen3.8-27B","layers":LAYERS,"tensor":"self_attn.q_proj.weight",
          "train_calibration_windows":16,"train_starts":starts,"heldout":"WikiText-2 validation blocks 160–223, 8,192 targets, disjoint from earlier screens",
          "selection":"diagonal activation-weighted projection error from train-only captured inputs",
          "methods":["bf16","optimized_row_real4","polar3_5_selected_pairing","awq_style_real4"],
          "polar_pairings":PAIRINGS,"awq_alphas":ALPHAS,
          "fairness":"same 4 index bits per real coefficient; AWQ-style adds 5120 FP32 channel scales per tensor, explicitly counted",
          "scope":"one projection at a time; all other weights BF16; decoded BF16 inference",
          "warning":"AWQ-style local control, not official AutoAWQ implementation; no packed-speed claim",
          "wall_cap_seconds":a.seconds}
    save(out/"plan.json",plan)
    weight_map=json.loads((Path(a.model)/"model.safetensors.index.json").read_text())["weight_map"]
    variants={}
    for layer in LAYERS:
        key=f"model.language_model.layers.{layer}.self_attn.q_proj.weight"
        with safe_open(Path(a.model)/weight_map[key],framework="pt",device="cpu") as f:w=f.get_tensor(key).float().cuda()
        qr,_,_=real4(w,True)
        row={"real4":{"weight":qr.to(torch.bfloat16).cpu(),"mse":(qr-w).square().mean().item()},"polar":{}}
        for name in PAIRINGS:
            qp,_,_,_=polar35(w,name)
            row["polar"][name]={"weight":qp.to(torch.bfloat16).cpu(),"mse":(qp-w).square().mean().item()}
            del qp
        variants[layer]=row;del w,qr;torch.cuda.empty_cache()
        print(json.dumps({"stage":"weights_prepared","layer":layer}),flush=True)
    model=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map="auto",
          max_memory={0:"20GiB","cpu":"40GiB"},low_cpu_mem_usage=True,offload_folder=str(out/"offload")).eval()
    captures={l:[] for l in LAYERS};hooks=[]
    for layer in LAYERS:
        def hook(module,args,layer=layer):captures[layer].append(args[0].detach().float().square().mean(dim=(0,1)).cpu())
        hooks.append(model.model.layers[layer].self_attn.q_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        for i in range(0,len(cal),8):
            x=torch.tensor(cal[i:i+8],device="cuda")
            logits=model(x[:,:-1],use_cache=False).logits
            del x,logits
    for hook in hooks:hook.remove()
    h={l:torch.stack(captures[l]).mean(0).cuda() for l in LAYERS}
    del captures;torch.cuda.empty_cache()
    print(json.dumps({"stage":"activation_capture_complete"}),flush=True)
    results=[]
    for layer in LAYERS:
        if time.monotonic()-start>a.seconds:raise TimeoutError("wall cap")
        p=model.model.layers[layer].self_attn.q_proj.weight;original=p.detach().cpu().clone()
        w=original.float().cuda();hh=h[layer]
        polar_scores={}
        for name,v in variants[layer]["polar"].items():
            e=v["weight"].float().cuda()-w
            polar_scores[name]=(e.square()*hh).mean().item()
            del e
        pairing=min(polar_scores,key=polar_scores.get)
        awq=awq_style(w,hh)
        del w;torch.cuda.empty_cache()
        payload=p.numel()//2+p.shape[0]*4+1
        methods=[("bf16",original,None,p.numel()*2),
                 ("real4",variants[layer]["real4"]["weight"],None,payload),
                 ("polar3_5",variants[layer]["polar"][pairing]["weight"],pairing,payload),
                 ("awq_style_real4",awq["weight"],awq["alpha"],payload+p.shape[1]*4)]
        evaluations=[]
        for name,q,selection,bytes_ in methods:
            if time.monotonic()-start>a.seconds:raise TimeoutError("wall cap")
            p.data.copy_(q)
            metric=evaluate(model,held)
            evaluations.append({"method":name,"selection":selection,"payload_bytes":bytes_,**metric})
            print(json.dumps({"stage":"evaluated","layer":layer,"method":name,"nll":metric["nll"],"seconds":metric["seconds"]}),flush=True)
        p.data.copy_(original)
        row={"layer":layer,"polar_calibration_scores":polar_scores,"selected_pairing":pairing,
             "awq_alpha":awq["alpha"],"awq_calibration_score":awq["score"],"evaluations":evaluations}
        results.append(row);save(out/"results.json",results)
        del original,awq,variants[layer],h[layer];torch.cuda.empty_cache()
    gate={"complete":len(results)==len(LAYERS),"seconds":time.monotonic()-start,
          "polar_beats_real4":sum(next(x for x in r["evaluations"] if x["method"]=="polar3_5")["nll"]<next(x for x in r["evaluations"] if x["method"]=="real4")["nll"] for r in results),
          "polar_beats_awq_style":sum(next(x for x in r["evaluations"] if x["method"]=="polar3_5")["nll"]<next(x for x in r["evaluations"] if x["method"]=="awq_style_real4")["nll"] for r in results),
          "note":"Single-projection held-out gate; no cumulative/full-dataset or end-to-end speed claim"}
    save(out/"gate.json",gate);print(json.dumps(gate),flush=True)

if __name__=="__main__":main()
