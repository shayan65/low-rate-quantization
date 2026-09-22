"""Full WikiText-2 validation of cumulative quantization in all Qwen DeltaNet input projections."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_crosslayer_screen import real4,pack
from run_qwen_taskaware_pairing import polar35

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def loss_on_windows(model,windows,started,cap,label,progress=False):
 total=0.;count=0
 for i,seq in enumerate(windows):
  if time.monotonic()-started>cap:raise TimeoutError(f'{label} exceeded wall cap')
  ids=torch.as_tensor(seq,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();y=ids[:,1:]
  total+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').item();count+=y.numel();del logits
  if progress and (i+1)%100==0:print(json.dumps({'progress':label,'blocks':i+1,'targets':count}),flush=True)
 return {'nll':total/count,'perplexity':math.exp(total/count),'target_tokens':count}
def real_candidate(w,factor):
 _,_,s0=real4(w,True);s=s0*factor;q=s*((w/s).clamp(-1,1).add(1).mul(7.5).round().div(7.5).sub(1));ix=((q/s+1)*7.5).round().clamp(0,15).to(torch.uint8)
 return q,ix,s
def payload_real(w,factor):
 q,ix,s=real_candidate(w,factor);return q,s.cpu().numpy().astype('<f4').tobytes()+pack(ix,4)
def payload_polar(w,pairing):
 q,mi,pi,s=polar35(w,pairing);return q,s.cpu().numpy().astype('<f4').tobytes()+pack(mi,3)+pack(pi,5)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_full_delta_v1');ap.add_argument('--seconds',type=float,default=3600);args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 train_text='\n'.join(pd.read_parquet('data/wikitext2/train.parquet').text.tolist());val_text='\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist())
 train_ids=tok(train_text,add_special_tokens=False).input_ids;val_ids=tok(val_text,add_special_tokens=False).input_ids
 rng=np.random.default_rng(62016);cal_starts=np.sort(rng.choice(len(train_ids)-129,4,replace=False));cal=[train_ids[s:s+129] for s in cal_starts]
 validation=[val_ids[s:min(s+129,len(val_ids))] for s in range(0,len(val_ids)-1,128)]
 assert sum(len(x)-1 for x in validation)==len(val_ids)-1
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval()
 linear_layers=[i for i,l in enumerate(model.model.layers) if hasattr(l,'linear_attn')];assert len(linear_layers)==18
 pairings=['adjacent','split_half','reverse_half','stride257'];clip_factors=[.85,.95,1.,1.05]
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','scope':'all 18 linear-attention layers; linear_attn.in_proj_qkv.weight; cumulative frozen PTQ',
  'linear_layers':linear_layers,'calibration':'4 fixed 129-token windows from WikiText-2 train','calibration_starts':cal_starts.tolist(),
  'evaluation':'complete Qwen-tokenized WikiText-2 validation stream in contiguous 128-target blocks with one-token input overlap; recurrent state resets each block','validation_source_tokens':len(val_ids),
  'validation_target_tokens':sum(len(x)-1 for x in validation),'test_split_touched':False,'pairing_candidates':pairings,'real_clip_candidates':clip_factors,
  'methods':['bf16','cumulative_task_real4','cumulative_task_polar3_5','cumulative_adjacent_polar3_5'],
  'per_tensor_payload_bytes':3170305,'metadata':'one candidate-id byte + per-row FP32 scales + 8 index bits per real pair',
  'runtime_cap_seconds':args.seconds}
 atomic_json(out/'plan.json',plan);(out/'run_qwen_full_delta_screen.py').write_bytes(Path(__file__).read_bytes())
 base_cal=loss_on_windows(model,cal,started,args.seconds,'bf16_cal')['nll'];selections=[]
 for layer in linear_layers:
  p=model.model.layers[layer].linear_attn.in_proj_qkv.weight;w=p.detach().float().clone();real_scores=[];polar_scores=[]
  for factor in clip_factors:
   q,_,_=real_candidate(w,factor)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   real_scores.append(loss_on_windows(model,cal,started,args.seconds,f'real_cal_l{layer}')['nll'])
  for pairing in pairings:
   q,_,_,_=polar35(w,pairing)
   with torch.no_grad():p.copy_(q.to(torch.bfloat16))
   polar_scores.append(loss_on_windows(model,cal,started,args.seconds,f'polar_cal_l{layer}')['nll'])
  with torch.no_grad():p.copy_(w.to(torch.bfloat16))
  row={'layer':layer,'real_candidate_id':int(np.argmin(real_scores)),'real_factor':clip_factors[int(np.argmin(real_scores))],
    'real_calibration_nlls':real_scores,'polar_candidate_id':int(np.argmin(polar_scores)),'polar_pairing':pairings[int(np.argmin(polar_scores))],
    'polar_calibration_nlls':polar_scores,'bf16_calibration_nll':base_cal};selections.append(row);atomic_json(out/'selections.json',selections);print(json.dumps(row),flush=True)
 originals={l:model.model.layers[l].linear_attn.in_proj_qkv.weight.detach().float().cpu() for l in linear_layers};rows=[]
 methods=plan['methods']
 for method in methods:
  export_dir=out/method;export_dir.mkdir()
  if method!='bf16':
   for sel in selections:
    l=sel['layer'];p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda')
    if method=='cumulative_task_real4':q,payload=payload_real(w,sel['real_factor']);cid=sel['real_candidate_id']
    else:
     pairing=sel['polar_pairing'] if method=='cumulative_task_polar3_5' else 'adjacent';q,payload=payload_polar(w,pairing);cid=pairings.index(pairing)
    blob=bytes([cid])+payload;assert len(blob)==3170305;path=export_dir/f'layer{l}.bin';path.write_bytes(blob)
    with torch.no_grad():p.copy_(q.to(torch.bfloat16))
  metric=loss_on_windows(model,validation,started,args.seconds,method,progress=True);payload_bytes=sum(x.stat().st_size for x in export_dir.glob('*.bin'))
  row={'method':method,'validation':metric,'delta_nll_vs_bf16':None if not rows else metric['nll']-rows[0]['validation']['nll'],
    'payload_bytes':payload_bytes,'layers_quantized':0 if method=='bf16' else 18};rows.append(row);atomic_json(out/'results.json',rows);print(json.dumps(row),flush=True)
  for l in linear_layers:
   with torch.no_grad():model.model.layers[l].linear_attn.in_proj_qkv.weight.copy_(originals[l].to('cuda',dtype=torch.bfloat16))
 complete=len(rows)==4;text='# Full cumulative Qwen3.5 DeltaNet projection screen\n\nAll 18 DeltaNet input projections are quantized simultaneously. Evaluation covers the complete tokenized WikiText-2 validation stream with fixed context resets; test data is untouched.\n\n| Method | NLL | Perplexity | Delta NLL | Export bytes |\n|---|---:|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['method']} | {r['validation']['nll']:.6f} | {r['validation']['perplexity']:.3f} | {(r['delta_nll_vs_bf16'] or 0):+.6f} | {r['payload_bytes']:,} |\n"
 gate={'complete':complete,'seconds':time.monotonic()-started,'task_polar_beats_task_real4':complete and rows[2]['validation']['nll']<rows[1]['validation']['nll'],
  'task_pairing_beats_adjacent':complete and rows[2]['validation']['nll']<rows[3]['validation']['nll'],'action':'No training or test-set evaluation launched.'}
 atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
