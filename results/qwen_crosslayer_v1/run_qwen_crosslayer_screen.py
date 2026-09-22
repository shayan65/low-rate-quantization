"""Bounded cross-layer Qwen polar screen with equal-byte optimized real4."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM

def atomic_json(path,value):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def pack(ix,bits):
 a=ix.detach().cpu().numpy().astype(np.uint16).reshape(-1);b=((a[:,None]>>np.arange(bits,dtype=np.uint16))&1).astype(np.uint8)
 return np.packbits(b.reshape(-1),bitorder='little').tobytes()
def choose_row_scale(w,ratios,quantize):
 base=w.abs().amax(1,keepdim=True).clamp_min(1e-8);best_q=None;best_s=None;best_e=torch.full((w.shape[0],),float('inf'),device=w.device)
 for ratio in ratios:
  s=base*ratio;q=quantize(w,s);e=(q-w).square().mean(1);take=e<best_e
  if best_q is None:best_q=q;best_s=s
  else:best_q[take]=q[take];best_s[take]=s[take]
  best_e=torch.minimum(best_e,e)
 return best_q,best_s
def real4(w,optimized):
 def quant(x,s):return s*((x/s).clamp(-1,1).add(1).mul(7.5).round().div(7.5).sub(1))
 ratios=torch.linspace(.55,1,10,device=w.device) if optimized else [1.]
 q,s=choose_row_scale(w,ratios,quant);ix=((q/s+1)*7.5).round().clamp(0,15).to(torch.uint8);return q,ix,s
def polar35(w,pairing):
 n=w.shape[1]//2
 if pairing=='adjacent':a=w[:,0::2];b=w[:,1::2]
 else:a=w[:,:n];b=w[:,n:]
 z=torch.complex(a.float(),b.float());phase=torch.angle(z);mag=z.abs();base=mag.amax(1,keepdim=True).clamp_min(1e-8)
 pi=((phase%(2*math.pi))/(2*math.pi)*32).round().remainder(32).to(torch.uint8);angle=pi.float()*(2*math.pi/32)
 best=None;best_s=None;best_mi=None;best_e=torch.full((w.shape[0],),float('inf'),device=w.device)
 for ratio in torch.linspace(.55,1,10,device=w.device):
  s=base*ratio;mi=(mag/s*7).round().clamp(0,7).to(torch.uint8);q=torch.polar(s*(mi.float()/7),angle);e=(q-z).abs().square().mean(1);take=e<best_e
  if best is None:best=q;best_s=s;best_mi=mi
  else:best[take]=q[take];best_s[take]=s[take];best_mi[take]=mi[take]
  best_e=torch.minimum(best_e,e)
 out=torch.empty_like(w,dtype=torch.float32)
 if pairing=='adjacent':out[:,0::2]=best.real;out[:,1::2]=best.imag
 else:out[:,:n]=best.real;out[:,n:]=best.imag
 return out,best_mi,pi,best_s
@torch.no_grad()
def evaluate(model,windows):
 loss=0.;count=0
 for seq in windows:
  ids=torch.as_tensor(seq,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();target=ids[:,1:]
  loss+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='sum').item();count+=target.numel();del logits
 return {'nll':loss/count,'perplexity':math.exp(loss/count),'target_tokens':count}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_crosslayer_v1');ap.add_argument('--seconds',type=float,default=180);args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tokenizer=AutoTokenizer.from_pretrained(args.model)
 raw=pd.read_parquet('data/wikitext2/validation.parquet').text.tolist();ids=tokenizer('\n'.join(raw),add_special_tokens=False).input_ids
 rng=np.random.default_rng(42016);starts=np.sort(rng.choice(len(ids)-129,16,replace=False));windows=[ids[s:s+129] for s in starts]
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();base=evaluate(model,windows)
 layers=[0,8,16];methods=['real4_rtn','real4_optimized_clip','polar3_5_adjacent_optimized','polar3_5_split_half_optimized'];rows=[]
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','layers':layers,'tensor':'linear_attn.in_proj_qkv.weight','methods':methods,
  'sample_seed':42016,'sample_starts':starts.tolist(),'validation_target_tokens':2048,'quantization':'one tensor at a time; model otherwise BF16 and frozen',
  'budget':'8 index bits per pair plus one FP32 scale per output row; 3,170,304 bytes each','scale_search':'10 fixed ratios from 0.55 to 1.0 selected independently per row by weight MSE; no validation labels',
  'pairings':{'adjacent':'(w[2j],w[2j+1])','split_half':'(w[j],w[j+input_dim/2])'},'runtime_cap_seconds':args.seconds}
 atomic_json(out/'plan.json',plan);(out/'run_qwen_crosslayer_screen.py').write_bytes(Path(__file__).read_bytes())
 for layer in layers:
  p=model.model.layers[layer].linear_attn.in_proj_qkv.weight;original=p.detach().float().clone()
  variants={}
  for optimized in [False,True]:
   q,ix,s=real4(original,optimized);variants['real4_optimized_clip' if optimized else 'real4_rtn']=(q,[pack(ix,4)],s)
  for pairing in ['adjacent','split_half']:
   q,mi,pi,s=polar35(original,pairing);variants[f'polar3_5_{pairing}_optimized']=(q,[pack(mi,3),pack(pi,5)],s)
  for method in methods:
   if time.monotonic()-started>args.seconds:break
   q,chunks,s=variants[method];payload=s.cpu().numpy().astype('<f4').tobytes()+b''.join(chunks);assert len(payload)==3170304
   path=out/f'layer{layer}_{method}.bin';path.write_bytes(payload)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   metric=evaluate(model,windows);rows.append({'layer':layer,'method':method,'validation':metric,'delta_nll_vs_bf16':metric['nll']-base['nll'],
    'weight_mse':(q-original).square().mean().item(),'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
   with torch.no_grad():p.copy_(original.to(torch.bfloat16))
   atomic_json(out/'results.json',rows)
  del original,variants
  if time.monotonic()-started>args.seconds:break
 complete=len(rows)==len(layers)*len(methods);text='# Qwen3.5 cross-layer polar screen\n\nEach row changes one frozen DeltaNet projection independently and evaluates the same 2,048 real WikiText-2 validation targets. BF16 baseline NLL %.6f.\n\n| Layer | Method | Delta NLL | Perplexity | Weight MSE |\n|---:|---|---:|---:|---:|\n'%base['nll']
 for r in rows:text+=f"| {r['layer']} | {r['method']} | {r['delta_nll_vs_bf16']:+.6f} | {r['validation']['perplexity']:.3f} | {r['weight_mse']:.3g} |\n"
 wins={m:sum(next(x for x in rows if x['layer']==l and x['method']==m)['validation']['nll']<next(x for x in rows if x['layer']==l and x['method']=='real4_optimized_clip')['validation']['nll'] for l in layers) for m in methods[2:]} if complete else {}
 gate={'complete':complete,'jobs':len(rows),'planned_jobs':12,'seconds':time.monotonic()-started,'polar_wins_vs_optimized_real4_by_layer':wins,
  'promote':complete and max(wins.values())==len(layers),'action':'No cumulative, all-layer, or training run launched.'}
 atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
