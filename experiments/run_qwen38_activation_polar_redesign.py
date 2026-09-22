"""One-layer bounded activation-weighted polar redesign gate for Qwen3.8-27B."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import real4
from run_qwen_taskaware_pairing import polar35, pairing_indices
from run_qwen38_attention_activation_gate import evaluate, awq_style

LAYER=11
PAIRINGS=("adjacent","reverse_half")
OFFSETS=(0., math.pi/32)  # zero or half of one 5-bit phase bin
RATIOS=(.55,.62,.69,.76,.83,.90,.97,1.)

def save(path,obj):
    p=Path(path);t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,indent=2)+"\n");t.replace(p)

@torch.no_grad()
def weighted_polar(w_cpu,h,pairing,offset,started,cap):
    rows,cols=w_cpu.shape
    ia,ib=pairing_indices(cols,pairing,"cuda")
    ha=h[ia][None,:]; hb=h[ib][None,:]
    out=torch.empty_like(w_cpu)
    scales=[]; score_sum=0.
    for start in range(0,rows,256):
        if time.monotonic()-started>cap:raise TimeoutError("quantization wall cap")
        w=w_cpu[start:start+256].float().cuda()
        x=w[:,ia];y=w[:,ib]
        phase=torch.atan2(y,x)
        code=torch.round(((phase-offset)%(2*math.pi))*(32/(2*math.pi))).remainder(32)
        theta=code*(2*math.pi/32)+offset
        c=theta.cos();sine=theta.sin()
        denom=(ha*c.square()+hb*sine.square()).clamp_min(1e-12)
        proj=(ha*x*c+hb*y*sine)/denom
        base=torch.hypot(x,y).amax(1,keepdim=True).clamp_min(1e-8)
        best_e=torch.full((w.shape[0],),float("inf"),device="cuda")
        best_x=torch.empty_like(x);best_y=torch.empty_like(y);best_scale=torch.empty_like(base)
        for ratio in RATIOS:
            scale=base*ratio
            index=(proj/scale*7).round().clamp(0,7)
            r=scale*(index/7)
            qx=r*c;qy=r*sine
            err=((qx-x).square()*ha+(qy-y).square()*hb).sum(1)
            take=err<best_e
            best_e=torch.where(take,err,best_e)
            best_x[take]=qx[take];best_y[take]=qy[take]
            best_scale[take]=scale[take]
        q=torch.empty_like(w);q[:,ia]=best_x;q[:,ib]=best_y
        out[start:start+len(w)]=q.to(torch.bfloat16).cpu()
        scales.append(best_scale.float().cpu());score_sum+=best_e.sum().item()
        del w,x,y,phase,code,theta,c,sine,denom,proj,base,best_e,best_x,best_y,best_scale,q
    return {"weight":out,"score":score_sum/(rows*cols),"row_scales":torch.cat(scales).flatten(),
            "pairing":pairing,"phase_offset":offset}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",default="models/Qwen3.8-27B-metadata")
    ap.add_argument("--out",default="results/qwen38_27b_activation_polar_redesign_v1")
    ap.add_argument("--seconds",type=float,default=900)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(a.model)
    train_ids=tok("\n".join(pd.read_parquet("data/wikitext2/train.parquet").text.tolist()),add_special_tokens=False).input_ids
    val_ids=tok("\n".join(pd.read_parquet("data/wikitext2/validation.parquet").text.tolist()),add_special_tokens=False).input_ids
    rng=np.random.default_rng(20260921)
    starts=np.sort(rng.choice(len(train_ids)-129,16,replace=False)).tolist()
    cal=[train_ids[s:s+129] for s in starts]
    held=[val_ids[s:s+129] for s in range(224*128,288*128,128)]
    assert len(held)==64 and all(len(x)==129 for x in held)
    plan={"model":"Qwen/Qwen3.8-27B","tensor":"layer 11 self_attn.q_proj.weight",
          "calibration":"16 fixed WikiText-2 training windows, activation diagonal proxy", "train_starts":starts,
          "heldout":"WikiText-2 validation blocks 224–287, 8,192 targets; untouched by earlier screens",
          "methods":["bf16","optimized_real4","legacy_polar3_5","activation_weighted_polar3_5","awq_style_real4"],
          "weighted_polar":"choose pairing, global phase-grid offset, and FP32 row scale by train-activation-weighted coefficient error; choose each 3-bit magnitude by weighted projection onto its 5-bit phase ray",
          "pairings":PAIRINGS,"offsets_radians":OFFSETS,"scale_ratios":RATIOS,
          "budget":"4 index bits per real weight; one FP32 row scale; one candidate ID byte for polar; AWQ-style has extra FP32 channel scales",
          "scope":"one tensor replaced at a time, other weights BF16; no packed kernel or full-model claim",
          "runtime_cap_seconds":a.seconds}
    save(out/"plan.json",plan)
    weight_map=json.loads((Path(a.model)/"model.safetensors.index.json").read_text())["weight_map"]
    key=f"model.language_model.layers.{LAYER}.self_attn.q_proj.weight"
    with safe_open(Path(a.model)/weight_map[key],framework="pt",device="cpu") as f:w_cpu=f.get_tensor(key).float()
    w=w_cpu.cuda()
    real,_,_=real4(w,True)
    legacy,_,_,_=polar35(w,"reverse_half")
    real_cpu=real.to(torch.bfloat16).cpu();legacy_cpu=legacy.to(torch.bfloat16).cpu()
    del w,real,legacy;torch.cuda.empty_cache()
    model=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map="auto",
          max_memory={0:"20GiB","cpu":"40GiB"},low_cpu_mem_usage=True,offload_folder=str(out/"offload")).eval()
    p=model.model.layers[LAYER].self_attn.q_proj.weight
    captures=[]
    def hook(module,args):captures.append(args[0].detach().float().square().mean(dim=(0,1)).cpu())
    handle=model.model.layers[LAYER].self_attn.q_proj.register_forward_pre_hook(hook)
    with torch.no_grad():
        for i in range(0,len(cal),8):
            x=torch.tensor(cal[i:i+8],device="cuda")
            z=model(x[:,:-1],use_cache=False).logits
            del x,z
    handle.remove();h=torch.stack(captures).mean(0).cuda();del captures
    print(json.dumps({"stage":"activation_capture_complete"}),flush=True)
    best=None;candidate_rows=[]
    for pairing in PAIRINGS:
        for offset in OFFSETS:
            cand=weighted_polar(w_cpu,h,pairing,offset,started,a.seconds)
            candidate_rows.append({"pairing":pairing,"phase_offset":offset,"weighted_score":cand["score"]})
            print(json.dumps({"stage":"polar_candidate","pairing":pairing,"offset":offset,"score":cand["score"]}),flush=True)
            if best is None or cand["score"]<best["score"]:best=cand
            else:del cand
    awq=awq_style(w_cpu.cuda(),h)
    del h;torch.cuda.empty_cache()
    payload=p.numel()//2+p.shape[0]*4+1
    original=p.detach().cpu().clone()
    methods=[("bf16",original,None,p.numel()*2),
             ("real4",real_cpu,None,payload),
             ("legacy_polar3_5",legacy_cpu,"reverse_half",payload),
             ("activation_weighted_polar3_5",best["weight"],{"pairing":best["pairing"],"phase_offset":best["phase_offset"]},payload),
             ("awq_style_real4",awq["weight"],awq["alpha"],payload+p.shape[1]*4)]
    rows=[]
    for name,weight,choice,bytes_ in methods:
        if time.monotonic()-started>a.seconds:raise TimeoutError("evaluation wall cap")
        p.data.copy_(weight)
        metric=evaluate(model,held)
        row={"method":name,"selection":choice,"payload_bytes":bytes_,**metric}
        rows.append(row);save(out/"results.json",rows)
        print(json.dumps({"stage":"evaluated","method":name,"nll":metric["nll"],"seconds":metric["seconds"]}),flush=True)
    p.data.copy_(original)
    gate={"complete":len(rows)==5,"seconds":time.monotonic()-started,"polar_candidates":candidate_rows,
          "selected_pairing":best["pairing"],"selected_phase_offset":best["phase_offset"],
          "awq_alpha":awq["alpha"],"redesigned_beats_legacy":rows[3]["nll"]<rows[2]["nll"],
          "redesigned_beats_awq":rows[3]["nll"]<rows[4]["nll"],
          "note":"One-layer held-out gate; no cumulative/full validation or end-to-end speed claim"}
    save(out/"gate.json",gate);print(json.dumps(gate),flush=True)

if __name__=="__main__":main()
