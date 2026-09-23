"""Task-calibrated pairing versus task-calibrated real4 at equal payload."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import real4,pack

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def pairing_indices(d,name,device):
 if name=='adjacent':return torch.arange(0,d,2,device=device),torch.arange(1,d,2,device=device)
 a=torch.arange(d//2,device=device)
 if name=='split_half':return a,a+d//2
 if name=='reverse_half':return a,d-1-a
 if name=='stride257':
  perm=(torch.arange(d,device=device)*257)%d;return perm[:d//2],perm[d//2:]
 raise ValueError(name)
def polar35(w,name):
 ia,ib=pairing_indices(w.shape[1],name,w.device);z=torch.complex(w[:,ia].float(),w[:,ib].float());mag=z.abs();phase=torch.angle(z)
 pi=((phase%(2*math.pi))/(2*math.pi)*32).round().remainder(32).to(torch.uint8);angle=pi.float()*(2*math.pi/32);base=mag.amax(1,keepdim=True).clamp_min(1e-8)
 best=None;best_s=None;best_mi=None;best_e=torch.full((w.shape[0],),float('inf'),device=w.device)
 for ratio in torch.linspace(.55,1,10,device=w.device):
  s=base*ratio;mi=(mag/s*7).round().clamp(0,7).to(torch.uint8);q=torch.polar(s*(mi.float()/7),angle);e=(q-z).abs().square().mean(1);take=e<best_e
  if best is None:best=q;best_s=s;best_mi=mi
  else:best[take]=q[take];best_s[take]=s[take];best_mi[take]=mi[take]
  best_e=torch.minimum(best_e,e)
 out=torch.empty_like(w,dtype=torch.float32);out[:,ia]=best.real;out[:,ib]=best.imag
 return out,best_mi,pi,best_s
@torch.no_grad()
def nll(model,windows):
 loss=0.;count=0
 for seq in windows:
  ids=torch.as_tensor(seq,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();y=ids[:,1:]
  loss+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').item();count+=y.numel();del logits
 return loss/count
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_taskaware_v1');ap.add_argument('--seconds',type=float,default=180);args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 raw=pd.read_parquet('data/wikitext2/validation.parquet').text.tolist();ids=tok('\n'.join(raw),add_special_tokens=False).input_ids;rng=np.random.default_rng(52016)
 starts=np.sort(rng.choice(len(ids)-129,20,replace=False));cal=[ids[s:s+129] for s in starts[:4]];ev=[ids[s:s+129] for s in starts[4:]]
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();base_cal=nll(model,cal);base_eval=nll(model,ev)
 layers=[0,8,16];pairings=['adjacent','split_half','reverse_half','stride257'];clip_factors=[.85,.95,1.0,1.05];rows=[]
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','layers':layers,'tensor':'linear_attn.in_proj_qkv.weight','calibration_targets':512,'evaluation_targets':2048,
  'sample_seed':52016,'starts':starts.tolist(),'polar_candidates':pairings,'real4_candidates':clip_factors,
  'selection':'minimum calibration NLL; final NLL on disjoint windows','budget':'3,170,305 bytes: equal 3,170,304-byte weights/scales plus one candidate-id byte',
  'scope':'one tensor at a time, frozen BF16 model','runtime_cap_seconds':args.seconds}
 atomic_json(out/'plan.json',plan);(out/'run_qwen_taskaware_pairing.py').write_bytes(Path(__file__).read_bytes())
 for layer in layers:
  p=model.model.layers[layer].linear_attn.in_proj_qkv.weight;original=p.detach().float().clone();real_base,_,real_scale=real4(original,True)
  real_candidates=[]
  for factor in clip_factors:
   s=real_scale*factor;q=s*((original/s).clamp(-1,1).add(1).mul(7.5).round().div(7.5).sub(1));ix=((q/s+1)*7.5).round().clamp(0,15).to(torch.uint8)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   real_candidates.append((nll(model,cal),q,ix,s))
  polar_candidates=[]
  for name in pairings:
   q,mi,pi,s=polar35(original,name)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   polar_candidates.append((nll(model,cal),q,mi,pi,s))
  chosen=[]
  ri=min(range(4),key=lambda i:real_candidates[i][0]);rc,q,ix,s=real_candidates[ri];chosen.append(('task_real4',ri,rc,q,[pack(ix,4)],s))
  pi_best=min(range(4),key=lambda i:polar_candidates[i][0]);pc,q,mi,phase,s=polar_candidates[pi_best];chosen.append(('task_polar3_5',pi_best,pc,q,[pack(mi,3),pack(phase,5)],s))
  for method,cid,cal_nll,q,chunks,s in chosen:
   payload=bytes([cid])+s.cpu().numpy().astype('<f4').tobytes()+b''.join(chunks);assert len(payload)==3170305;path=out/f'layer{layer}_{method}.bin';path.write_bytes(payload)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   eval_nll=nll(model,ev);rows.append({'layer':layer,'method':method,'candidate_id':cid,'candidate':clip_factors[cid] if method=='task_real4' else pairings[cid],
    'calibration_nll':cal_nll,'calibration_delta':cal_nll-base_cal,'evaluation_nll':eval_nll,'evaluation_delta':eval_nll-base_eval,
    'weight_mse':(q-original).square().mean().item(),'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
  with torch.no_grad():p.copy_(original.to(torch.bfloat16))
  atomic_json(out/'results.json',rows);del original,real_candidates,polar_candidates
  if time.monotonic()-started>args.seconds:break
 complete=len(rows)==6;text=f'# Qwen task-aware pairing screen\n\nBF16 evaluation NLL: {base_eval:.6f}. Calibration and evaluation windows are disjoint. Each method changes one tensor independently.\n\n| Layer | Method | Selected candidate | Calibration ΔNLL | Evaluation ΔNLL |\n|---:|---|---|---:|---:|\n'
 for r in rows:text+=f"| {r['layer']} | {r['method']} | {r['candidate']} | {r['calibration_delta']:+.6f} | {r['evaluation_delta']:+.6f} |\n"
 wins=sum(next(x for x in rows if x['layer']==l and x['method']=='task_polar3_5')['evaluation_nll']<next(x for x in rows if x['layer']==l and x['method']=='task_real4')['evaluation_nll'] for l in layers) if complete else 0
 gate={'complete':complete,'jobs':len(rows),'seconds':time.monotonic()-started,'polar_wins':wins,'layers':3,'promote':complete and wins==3,
  'action':'No cumulative or training run launched.'};atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
