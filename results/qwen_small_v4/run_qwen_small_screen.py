"""Bounded Qwen3.5-0.8B frozen one-projection quantization screen."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM

def sha(path):
 h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def pack(ix,bits):
 a=ix.detach().cpu().numpy().astype(np.uint16).reshape(-1);z=((a[:,None]>>np.arange(bits,dtype=np.uint16))&1).astype(np.uint8)
 return np.packbits(z.reshape(-1),bitorder='little').tobytes()
def real4(w):
 s=w.abs().amax(1,keepdim=True).clamp_min(1e-8);ix=((w/s+1)*7.5).round().clamp(0,15).to(torch.uint8);return s*(ix.float()/7.5-1),ix,s
def polar(w,mb,pb):
 z=torch.complex(w[:,0::2].float(),w[:,1::2].float());mag=z.abs();phase=torch.angle(z);s=mag.amax(1,keepdim=True).clamp_min(1e-8)
 mi=(mag/s*(2**mb-1)).round().clamp(0,2**mb-1).to(torch.uint8);pi=((phase%(2*math.pi))/(2*math.pi)*(2**pb)).round().remainder(2**pb).to(torch.uint8)
 q=torch.polar(s*(mi.float()/(2**mb-1)),pi.float()*(2*math.pi/(2**pb)));out=torch.empty_like(w,dtype=torch.float32);out[:,0::2]=q.real;out[:,1::2]=q.imag
 return out,mi,pi,s
@torch.no_grad()
def evaluate(model,windows):
 total=0.;count=0
 for x in windows:
  ids=torch.as_tensor(x,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();target=ids[:,1:]
  total+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='sum').item();count+=target.numel();del logits
 return {'nll':total/count,'perplexity':math.exp(total/count),'target_tokens':count}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_small_v1');ap.add_argument('--seconds',type=float,default=180);args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 raw=pd.read_parquet('data/wikitext2/validation.parquet').text.tolist();ids=tok('\n'.join(raw),add_special_tokens=False).input_ids
 rng=np.random.default_rng(20260916);starts=np.sort(rng.choice(len(ids)-129,8,replace=False));windows=[ids[s:s+129] for s in starts]
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();p=model.model.layers[0].linear_attn.in_proj_qkv.weight;original=p.detach().clone();rows=[]
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','parameters':sum(x.numel() for x in model.parameters()),'tensor':'model.layers.0.linear_attn.in_proj_qkv.weight','shape':list(p.shape),
  'dataset':'local real WikiText-2 validation parquet','sample_starts':starts.tolist(),'sequences':8,'sequence_tokens':129,'target_tokens':1024,
  'methods':['bf16','real4_row','polar_mag4_phase4_row','polar_mag3_phase5_row','polar_mag5_phase3_row'],'equal_index_bits_per_real_pair':8,
  'note':'Frozen post-training quantization of one projection only; screening metric, not full-dataset perplexity','runtime_cap_seconds':args.seconds}
 write_json(out/'plan.json',plan);(out/'run_qwen_small_screen.py').write_bytes(Path(__file__).read_bytes())
 base=evaluate(model,windows);rows.append({'method':'bf16','validation':base,'tensor_payload_bytes':p.numel()*2})
 variants={};q,ix,s=real4(original.float());variants['real4_row']=(q,[pack(ix,4)],s)
 for mb,pb in [(4,4),(3,5),(5,3)]:q,mi,pi,s=polar(original.float(),mb,pb);variants[f'polar_mag{mb}_phase{pb}_row']=(q,[pack(mi,mb),pack(pi,pb)],s)
 for name,(q,chunks,s) in variants.items():
  if time.monotonic()-started>args.seconds:break
  payload=s.cpu().numpy().astype('<f4').tobytes()+b''.join(chunks);path=out/f'{name}.bin';path.write_bytes(payload)
  with torch.no_grad():p.copy_(q.to(torch.bfloat16))
  metric=evaluate(model,windows);rows.append({'method':name,'validation':metric,'delta_nll_vs_bf16':metric['nll']-base['nll'],'tensor_payload_bytes':len(payload),'export_sha256':sha(path),'weight_mse':(q-original.float()).square().mean().item()})
  with torch.no_grad():p.copy_(original)
  write_json(out/'results.json',rows)
 complete=len(rows)==5;best=min(rows[1:],key=lambda x:x['validation']['nll']) if complete else None
 text='# Qwen3.5-0.8B one-projection screen\n\nFrozen model, one DeltaNet input projection, 1,024 real WikiText-2 validation targets. This is not full-dataset perplexity.\n\n| Method | NLL | Perplexity | Delta NLL | Payload bytes |\n|---|---:|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['method']} | {r['validation']['nll']:.6f} | {r['validation']['perplexity']:.3f} | {r.get('delta_nll_vs_bf16',0):+.6f} | {r['tensor_payload_bytes']:,} |\n"
 gate={'complete':complete,'seconds':time.monotonic()-started,'best_quantized':best['method'] if best else None,'polar_beats_equal_bit_real4':complete and any(r['validation']['nll']<rows[1]['validation']['nll'] for r in rows[2:]),'action':'No additional layer or training launched.'}
 write_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
