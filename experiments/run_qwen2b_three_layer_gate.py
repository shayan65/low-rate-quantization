"""Bounded real-data gate for Qwen3.5-2B-Base on three DeltaNet projections."""
import argparse,json,time,math,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_full_delta_screen import real_candidate,payload_polar
from run_qwen_taskaware_pairing import polar35
from run_qwen_strong_controls import block_gptq
from run_qwen_crosslayer_screen import pack
def save(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def score(m,ws,start,cap,label):
 z=n=0
 for s in ws:
  if time.monotonic()-start>cap:raise TimeoutError(label)
  ids=torch.tensor(s,device='cuda')[None];y=ids[:,1:];v=m(ids[:,:-1],use_cache=False).logits.float();z+=F.cross_entropy(v.reshape(-1,v.shape[-1]),y.reshape(-1),reduction='sum').item();n+=y.numel();del v
 x=z/n;return {'nll':x,'perplexity':math.exp(x),'targets':n}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-2B-Base');ap.add_argument('--out',default='results/qwen2b_three_layer_gate_v1');ap.add_argument('--seconds',type=float,default=480);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=False);start=time.monotonic();tok=AutoTokenizer.from_pretrained(a.model)
 train=tok('\n'.join(pd.read_parquet('data/wikitext2/train.parquet').text.tolist()),add_special_tokens=False).input_ids;val=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids;rng=np.random.default_rng(220917);cs=np.sort(rng.choice(len(train)-129,4,False));vs=np.sort(rng.choice(len(val)-129,16,False));cal=[train[s:s+129] for s in cs];ev=[val[s:s+129] for s in vs]
 m=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[0,8,16];pairings=['adjacent','split_half','reverse_half','stride257'];clips=[.85,.95,1.,1.05];base=score(m,ev,start,a.seconds,'base');rows=[];selections=[]
 plan={'model':'Qwen/Qwen3.5-2B-Base','parameters':sum(p.numel() for p in m.parameters()),'layers':layers,'calibration':'four fixed WikiText-2 train windows','evaluation':'16 fixed disjoint WikiText-2 validation windows, 2048 targets','methods':['task_real4','task_polar3_5','block_gptq_real4'],'promotion':'polar has lowest evaluation NLL in all three layers','test_split_touched':False,'runtime_cap_seconds':a.seconds};save(o/'plan.json',plan);(o/'run_qwen2b_three_layer_gate.py').write_bytes(Path(__file__).read_bytes())
 for l in layers:
  p=m.model.layers[l].linear_attn.in_proj_qkv.weight;w=p.detach().float().clone();acts=[]
  h=p.register_hook(lambda g:g) if False else m.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(lambda mod,args:acts.append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu()))
  score(m,cal,start,a.seconds,f'capture{l}');h.remove();x=torch.cat(acts).cuda();pc=[];rc=[]
  for pairing in pairings:
   q,_,_,_=polar35(w,pairing);p.data.copy_(q.to(torch.bfloat16));pc.append(score(m,cal,start,a.seconds,f'pc{l}')['nll'])
  for clip in clips:
   q,_,_=real_candidate(w,clip);p.data.copy_(q.to(torch.bfloat16));rc.append(score(m,cal,start,a.seconds,f'rc{l}')['nll'])
  p.data.copy_(w.to(torch.bfloat16));pi=int(np.argmin(pc));ri=int(np.argmin(rc));qp,payload=payload_polar(w,pairings[pi]);qr,rix,rs=real_candidate(w,clips[ri]);qg,gix,gs=block_gptq(w,x);variants=[('task_real4',qr,rs.cpu().numpy().astype('<f4').tobytes()+pack(rix,4)),('task_polar3_5',qp,payload),('block_gptq_real4',qg,gs.cpu().numpy().astype('<f4').tobytes()+pack(gix,4))]
  selections.append({'layer':l,'polar_pairing':pairings[pi],'polar_calibration_nlls':pc,'real_clip':clips[ri],'real_calibration_nlls':rc});save(o/'selections.json',selections)
  for name,q,blob in variants:
   p.data.copy_(q.to(torch.bfloat16));metric=score(m,ev,start,a.seconds,name);path=o/f'layer{l}_{name}.bin';path.write_bytes(blob);rows.append({'layer':l,'method':name,'validation':metric,'delta_nll_vs_bf16':metric['nll']-base['nll'],'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()});save(o/'results.json',rows);p.data.copy_(w.to(torch.bfloat16));print(json.dumps(rows[-1]),flush=True)
 winners={l:min((r for r in rows if r['layer']==l),key=lambda r:r['validation']['nll'])['method'] for l in layers};gate={'complete':len(rows)==9,'seconds':time.monotonic()-start,'winners':winners,'promote':all(v=='task_polar3_5' for v in winners.values())};save(o/'gate.json',gate);(o/'summary.md').write_text('# Qwen3.5-2B three-layer gate\n\nBF16 NLL %.6f.\n\n| Layer | Method | Delta NLL | Bytes |\n|---:|---|---:|---:|\n'%base['nll']+''.join(f"| {r['layer']} | {r['method']} | {r['delta_nll_vs_bf16']:+.6f} | {r['payload_bytes']:,} |\n" for r in rows)+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
