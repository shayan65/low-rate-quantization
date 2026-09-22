"""Activation-aware and block-GPTQ controls for Qwen3.5 DeltaNet projections.

These are transparent local implementations, not AutoAWQ/GPTQModel backend runs.
"""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import pack

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def eval_windows(model,windows,started,cap,label):
 total=0.;count=0
 for i,seq in enumerate(windows):
  if time.monotonic()-started>cap:raise TimeoutError(label)
  ids=torch.as_tensor(seq,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();y=ids[:,1:]
  total+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').item();count+=y.numel();del logits
  if (i+1)%100==0:print(json.dumps({'progress':label,'blocks':i+1,'targets':count}),flush=True)
 return {'nll':total/count,'perplexity':math.exp(total/count),'target_tokens':count}
def quant_grid(w,s):
 ix=((w/s).clamp(-1,1).add(1).mul(7.5)).round().clamp(0,15).to(torch.uint8);return s*(ix.float()/7.5-1),ix
def activation_aware(w,h):
 # AWQ-style channel scaling, reconstructed back into the original basis.
 act=h.sqrt().clamp_min(1e-8);best=None
 for alpha in [0.,.25,.5,.75,1.]:
  c=(act/act.mean()).pow(alpha).clamp(.1,10);wc=w*c;scale=wc.abs().amax(1,keepdim=True).clamp_min(1e-8)
  q,ix=quant_grid(wc,scale);recon=q/c;score=((recon-w).square()*h).sum().item()
  if best is None or score<best[0]:best=(score,alpha,recon,ix,scale,c)
 return best
def block_gptq(w,x,group=128):
 # GPTQ error propagation within 128-column groups, with one FP32 row scale per group.
 qout=torch.empty_like(w);indices=torch.empty_like(w,dtype=torch.uint8);scales=[];d=w.shape[1]
 for start in range(0,d,group):
  end=min(start+group,d);xb=x[:,start:end].float();H=2*xb.T@xb;diag=torch.diag(H);damp=.01*diag.mean();H.diagonal().add_(damp)
  try:hinv=torch.linalg.cholesky(torch.linalg.inv(H),upper=True)
  except RuntimeError:hinv=torch.linalg.cholesky(torch.linalg.pinv(H)+torch.eye(end-start,device=w.device)*1e-6,upper=True)
  work=w[:,start:end].clone();scale=work.abs().amax(1,keepdim=True).clamp_min(1e-8);scales.append(scale)
  for j in range(end-start):
   q,ix=quant_grid(work[:,j:j+1],scale);qout[:,start+j:start+j+1]=q;indices[:,start+j:start+j+1]=ix
   err=(work[:,j]-q[:,0])/hinv[j,j];work[:,j:]-=err[:,None]*hinv[j,j:][None,:]
 return qout,indices,torch.cat(scales,1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_strong_controls_v1');ap.add_argument('--seconds',type=float,default=3600);args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 train_ids=tok('\n'.join(pd.read_parquet('data/wikitext2/train.parquet').text.tolist()),add_special_tokens=False).input_ids
 val_ids=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids
 rng=np.random.default_rng(72016);starts=np.sort(rng.choice(len(train_ids)-129,8,replace=False));cal=[train_ids[s:s+129] for s in starts]
 validation=[val_ids[s:min(s+129,len(val_ids))] for s in range(0,len(val_ids)-1,128)];assert sum(len(x)-1 for x in validation)==len(val_ids)-1
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[i for i,l in enumerate(model.model.layers) if hasattr(l,'linear_attn')]
 captures={l:[] for l in layers};hooks=[]
 for l in layers:
  def hook(module,args,l=l):captures[l].append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu())
  hooks.append(model.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(hook))
 with torch.no_grad():
  for seq in cal:model(torch.as_tensor(seq,device='cuda')[None],use_cache=False)
 for h in hooks:h.remove()
 captures={l:torch.cat(v).to('cuda') for l,v in captures.items()};originals={l:model.model.layers[l].linear_attn.in_proj_qkv.weight.detach().float().cpu() for l in layers}
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','scope':'all 18 linear_attn.in_proj_qkv weights cumulatively','calibration_targets':1024,'calibration_starts':starts.tolist(),
  'evaluation_targets':len(val_ids)-1,'test_split_touched':False,'controls':{
   'awq_style':'search alpha in {0,.25,.5,.75,1}; scale input channels by activation RMS^alpha; rowwise real4; choose by diagonal activation-weighted reconstruction; retain channel scales explicitly',
   'block_gptq':'128-column groups; activation Hessian with 1% damping; sequential error propagation inside each group; real4 row/group scales'},
  'warning':'Local transparent implementations for unsupported custom projections; not official AutoAWQ or GPTQModel backend outputs','runtime_cap_seconds':args.seconds}
 atomic_json(out/'plan.json',plan);(out/'run_qwen_strong_controls.py').write_bytes(Path(__file__).read_bytes());rows=[]
 for method in ['awq_style_real4','block_gptq_real4']:
  directory=out/method;directory.mkdir();layer_info=[]
  for l in layers:
   w=originals[l].to('cuda');x=captures[l];p=model.model.layers[l].linear_attn.in_proj_qkv.weight
   if method=='awq_style_real4':
    score,alpha,q,ix,scales,channel=activation_aware(w,x.square().mean(0));blob=scales.cpu().numpy().astype('<f4').tobytes()+channel.cpu().numpy().astype('<f4').tobytes()+pack(ix,4);meta={'alpha':alpha,'objective':score}
   else:
    q,ix,scales=block_gptq(w,x);blob=scales.cpu().numpy().astype('<f4').tobytes()+pack(ix,4);meta={'group_size':128}
   path=directory/f'layer{l}.bin';path.write_bytes(blob);meta.update({'layer':l,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'weight_mse':(q-w).square().mean().item()});layer_info.append(meta)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   print(json.dumps({'quantized':method,**meta}),flush=True)
  metric=eval_windows(model,validation,started,args.seconds,method);row={'method':method,'validation':metric,'payload_bytes':sum(x['bytes'] for x in layer_info),'layers':layer_info};rows.append(row);atomic_json(out/'results.json',rows)
  for l in layers:
   with torch.no_grad():model.model.layers[l].linear_attn.in_proj_qkv.weight.copy_(originals[l].to('cuda',dtype=torch.bfloat16))
 reference=json.loads(Path('results/qwen_full_delta_v1/results.json').read_text());base=reference[0]['validation']['nll']
 for r in rows:r['delta_nll_vs_bf16']=r['validation']['nll']-base
 atomic_json(out/'results.json',rows);text='# Strong classical controls for cumulative Qwen DeltaNet projection quantization\n\nThese are transparent local implementations on Qwen3.5 custom projections, not official AutoAWQ/GPTQModel backend runs. Evaluation uses the same complete 261,284-target WikiText-2 validation stream.\n\n| Method | NLL | Perplexity | Delta NLL | Export bytes |\n|---|---:|---:|---:|---:|\n'
 for r in reference[1:3]:text+=f"| {r['method']} | {r['validation']['nll']:.6f} | {r['validation']['perplexity']:.3f} | {r['delta_nll_vs_bf16']:+.6f} | {r['payload_bytes']:,} |\n"
 for r in rows:text+=f"| {r['method']} | {r['validation']['nll']:.6f} | {r['validation']['perplexity']:.3f} | {r['delta_nll_vs_bf16']:+.6f} | {r['payload_bytes']:,} |\n"
 gate={'complete':len(rows)==2,'seconds':time.monotonic()-started,'task_polar_beats_awq_style':reference[2]['validation']['nll']<rows[0]['validation']['nll'],
  'task_polar_beats_block_gptq':reference[2]['validation']['nll']<rows[1]['validation']['nll'],'action':'No training or test evaluation launched.'}
 atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
