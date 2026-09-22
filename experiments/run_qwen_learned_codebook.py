"""Learned Cartesian product-codebook baseline for all Qwen DeltaNet projections."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import pack
from run_qwen_taskaware_pairing import pairing_indices

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def kmeans1d(values,k=16,steps=12):
 sample=values.flatten();n=min(sample.numel(),262144);g=torch.Generator(device=values.device).manual_seed(82016);sample=sample[torch.randperm(sample.numel(),generator=g,device=values.device)[:n]]
 centers=torch.quantile(sample,torch.linspace(0,1,k+2,device=values.device)[1:-1])
 for _ in range(steps):
  ids=(sample[:,None]-centers[None]).abs().argmin(1);sums=torch.zeros(k,device=values.device).scatter_add_(0,ids,sample);counts=torch.bincount(ids,minlength=k).clamp_min(1);centers=sums/counts
 return centers.sort().values
def product_codebook(w,pairing):
 ia,ib=pairing_indices(w.shape[1],pairing,w.device);z=torch.complex(w[:,ia].float(),w[:,ib].float());scale=z.abs().amax(1,keepdim=True).clamp_min(1e-8);zn=z/scale
 cr=kmeans1d(zn.real);ci=kmeans1d(zn.imag);out=torch.empty_like(w);ir=torch.empty_like(zn.real,dtype=torch.uint8);ii=torch.empty_like(ir)
 for start in range(0,zn.numel(),262144):
  end=min(start+262144,zn.numel());r=zn.real.flatten()[start:end];im=zn.imag.flatten()[start:end];ir.flatten()[start:end]=(r[:,None]-cr).abs().argmin(1).to(torch.uint8);ii.flatten()[start:end]=(im[:,None]-ci).abs().argmin(1).to(torch.uint8)
 q=torch.complex(cr[ir.long()],ci[ii.long()])*scale;out[:,ia]=q.real;out[:,ib]=q.imag
 return out,ir,ii,scale,cr,ci
@torch.no_grad()
def evaluate(model,windows,started,cap):
 total=0.;count=0
 for i,seq in enumerate(windows):
  if time.monotonic()-started>cap:raise TimeoutError('runtime cap')
  ids=torch.as_tensor(seq,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();y=ids[:,1:];total+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').item();count+=y.numel();del logits
  if (i+1)%100==0:print(json.dumps({'progress':'learned_product_codebook','blocks':i+1,'targets':count}),flush=True)
 return {'nll':total/count,'perplexity':math.exp(total/count),'target_tokens':count}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_vector_codebook_v1');ap.add_argument('--seconds',type=float,default=1800);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 tok=AutoTokenizer.from_pretrained(args.model);val_ids=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids;windows=[val_ids[s:min(s+129,len(val_ids))] for s in range(0,len(val_ids)-1,128)]
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();selections=json.loads(Path('results/qwen_full_delta_v1/selections.json').read_text());rows=[];directory=out/'learned_product_codebook';directory.mkdir()
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','scope':'all 18 DeltaNet in_proj_qkv projections cumulatively','pairing':'reuse training-calibrated polar pairing per layer; no validation selection','codebook':'two learned 16-level scalar codebooks for real and imaginary coordinates after per-row complex-magnitude scaling; 4+4 index bits per pair','evaluation_targets':len(val_ids)-1,'test_split_touched':False,'runtime_cap_seconds':args.seconds};atomic_json(out/'plan.json',plan);(out/'run_qwen_learned_codebook.py').write_bytes(Path(__file__).read_bytes())
 for sel in selections:
  l=sel['layer'];p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=p.detach().float();q,ir,ii,scale,cr,ci=product_codebook(w,sel['polar_pairing']);blob=bytes([sel['polar_candidate_id']])+scale.cpu().numpy().astype('<f4').tobytes()+cr.cpu().numpy().astype('<f4').tobytes()+ci.cpu().numpy().astype('<f4').tobytes()+pack(ir,4)+pack(ii,4);path=directory/f'layer{l}.bin';path.write_bytes(blob);rows.append({'layer':l,'pairing':sel['polar_pairing'],'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'weight_mse':(q-w).square().mean().item()});p.data.copy_(q.to(torch.bfloat16));atomic_json(out/'layers.json',rows);print(json.dumps(rows[-1]),flush=True)
 metric=evaluate(model,windows,started,args.seconds);reference=json.loads(Path('results/qwen_full_delta_v1/results.json').read_text());base=reference[0]['validation']['nll'];result={'method':'learned_cartesian_product_codebook','validation':metric,'delta_nll_vs_bf16':metric['nll']-base,'payload_bytes':sum(x['bytes'] for x in rows)};atomic_json(out/'results.json',[result])
 polar=reference[2];gate={'complete':True,'seconds':time.monotonic()-started,'codebook_beats_polar':metric['nll']<polar['validation']['nll'],'action':'Proceed to second-dataset stage; retain whichever method wins.'};atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(f"# Learned product-codebook baseline\n\n| Method | NLL | Perplexity | Delta NLL | Export bytes |\n|---|---:|---:|---:|---:|\n| task-selected polar 3+5 | {polar['validation']['nll']:.6f} | {polar['validation']['perplexity']:.3f} | {polar['delta_nll_vs_bf16']:+.6f} | {polar['payload_bytes']:,} |\n| learned Cartesian product codebook | {metric['nll']:.6f} | {metric['perplexity']:.3f} | {result['delta_nll_vs_bf16']:+.6f} | {result['payload_bytes']:,} |\n\n{json.dumps(gate)}\n");print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
