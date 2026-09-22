"""Frozen WikiText-2 mixed polar/GPTQ replication with direct paired intervals."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_full_delta_screen import loss_on_windows,payload_polar,payload_real
from run_qwen_taskaware_pairing import polar35
from run_qwen_strong_controls import block_gptq
from run_qwen_crosslayer_screen import pack

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def evaluate_blocks(model,windows,started,cap,label):
 blocks=[]
 for i,seq in enumerate(windows):
  if time.monotonic()-started>cap:raise TimeoutError(label)
  ids=torch.as_tensor(seq,device='cuda')[None];logits=model(ids[:,:-1],use_cache=False).logits.float();y=ids[:,1:];loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').item();blocks.append({'loss_sum':loss,'count':y.numel()});del logits
  if (i+1)%100==0:print(json.dumps({'progress':label,'blocks':i+1}),flush=True)
 total=sum(x['loss_sum'] for x in blocks);count=sum(x['count'] for x in blocks);nll=total/count;return {'nll':nll,'perplexity':math.exp(nll),'target_tokens':count},blocks
def paired_bootstrap(a_blocks,b_blocks,seed=91726,reps=5000):
 a=np.array([x['loss_sum'] for x in a_blocks]);b=np.array([x['loss_sum'] for x in b_blocks]);c=np.array([x['count'] for x in a_blocks]);rng=np.random.default_rng(seed);vals=[]
 for _ in range(reps//100):
  ix=rng.integers(0,len(a),size=(100,len(a)));vals.extend(((a[ix].sum(1)-b[ix].sum(1))/c[ix].sum(1)).tolist())
 return {'mean_delta':float((a.sum()-b.sum())/c.sum()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'bootstrap_reps':reps,'unit':'128-target contiguous block'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_wikitext_mixed_ci_v1');ap.add_argument('--seconds',type=float,default=1800);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 train_ids=tok('\n'.join(pd.read_parquet('data/wikitext2/train.parquet').text.tolist()),add_special_tokens=False).input_ids;val_ids=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids
 cal_starts=[59959,131396,1638120,2226031];cal=[train_ids[s:s+129] for s in cal_starts];windows=[val_ids[s:min(s+129,len(val_ids))] for s in range(0,len(val_ids)-1,128)];assert sum(len(x)-1 for x in windows)==261284
 selections=json.loads(Path('results/qwen_full_delta_v1/selections.json').read_text());model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[x['layer'] for x in selections]
 captures={l:[] for l in layers};hooks=[]
 for l in layers:
  def hook(module,args,l=l):captures[l].append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu())
  hooks.append(model.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(hook))
 loss_on_windows(model,cal,started,args.seconds,'capture')
 for h in hooks:h.remove()
 captures={l:torch.cat(v).to('cuda') for l,v in captures.items()};originals={l:model.model.layers[l].linear_attn.in_proj_qkv.weight.detach().float().cpu() for l in layers};mixed=[]
 plan={'model':'Qwen/Qwen3.5-0.8B-Base','dataset':'complete WikiText-2 validation, 261,284 targets','calibration_starts':cal_starts,'selection':'per layer polar versus block-GPTQ on frozen train-only calibration windows','test_split_touched':False,'methods':['bf16','task_polar3_5','block_gptq_real4','mixed_polar_gptq'],'uncertainty':'5,000 paired bootstrap replicates over contiguous 128-target blocks, including direct codec contrasts','runtime_cap_seconds':args.seconds};atomic_json(out/'plan.json',plan);(out/'run_qwen_wikitext_mixed_codec_ci.py').write_bytes(Path(__file__).read_bytes())
 for sel in selections:
  l=sel['layer'];p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda');qp,_,_,_=polar35(w,sel['polar_pairing']);p.data.copy_(qp.to(torch.bfloat16));pn=loss_on_windows(model,cal,started,args.seconds,f'polar_l{l}')['nll'];qg,_,_=block_gptq(w,captures[l]);p.data.copy_(qg.to(torch.bfloat16));gn=loss_on_windows(model,cal,started,args.seconds,f'gptq_l{l}')['nll'];p.data.copy_(w.to(torch.bfloat16));codec='polar' if pn<=gn else 'gptq';mixed.append({'layer':l,'codec':codec,'polar_calibration_nll':pn,'gptq_calibration_nll':gn,'polar_pairing':sel['polar_pairing'],'polar_id':sel['polar_candidate_id']});atomic_json(out/'mixed_selections.json',mixed);print(json.dumps(mixed[-1]),flush=True)
 rows=[];all_blocks={}
 for method in plan['methods']:
  directory=out/method;directory.mkdir();payload_bytes=0
  if method!='bf16':
   for sel,choice in zip(selections,mixed):
    l=sel['layer'];p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda');chosen=method
    if method=='mixed_polar_gptq':chosen='task_polar3_5' if choice['codec']=='polar' else 'block_gptq_real4'
    if chosen=='task_polar3_5':q,payload=payload_polar(w,sel['polar_pairing']);blob=bytes([sel['polar_candidate_id']])+payload
    else:q,ix,scales=block_gptq(w,captures[l]);blob=bytes([4])+scales.cpu().numpy().astype('<f4').tobytes()+pack(ix,4)
    (directory/f'layer{l}.bin').write_bytes(blob);payload_bytes+=len(blob);p.data.copy_(q.to(torch.bfloat16))
  metric,blocks=evaluate_blocks(model,windows,started,args.seconds,method);rows.append({'method':method,'validation':metric,'payload_bytes':payload_bytes});all_blocks[method]=blocks;atomic_json(out/'results.json',rows);print(json.dumps(rows[-1]),flush=True)
  for l in layers:model.model.layers[l].linear_attn.in_proj_qkv.weight.data.copy_(originals[l].to('cuda',dtype=torch.bfloat16))
 contrasts={}
 for method in plan['methods'][1:]:contrasts[f'{method}_minus_bf16']=paired_bootstrap(all_blocks[method],all_blocks['bf16'])
 contrasts['mixed_minus_polar']=paired_bootstrap(all_blocks['mixed_polar_gptq'],all_blocks['task_polar3_5'],seed=91727)
 contrasts['mixed_minus_gptq']=paired_bootstrap(all_blocks['mixed_polar_gptq'],all_blocks['block_gptq_real4'],seed=91728)
 atomic_json(out/'confidence_intervals.json',contrasts);base=rows[0]['validation']['nll']
 for r in rows:r['delta_nll_vs_bf16']=r['validation']['nll']-base
 atomic_json(out/'results.json',rows);gate={'complete':len(rows)==4,'seconds':time.monotonic()-started,'polar_layers':sum(x['codec']=='polar' for x in mixed),'gptq_layers':sum(x['codec']=='gptq' for x in mixed),'mixed_beats_polar':rows[3]['validation']['nll']<rows[1]['validation']['nll'],'mixed_beats_gptq':rows[3]['validation']['nll']<rows[2]['validation']['nll'],'direct_ci_mixed_vs_polar_excludes_zero':not(contrasts['mixed_minus_polar']['ci95'][0]<=0<=contrasts['mixed_minus_polar']['ci95'][1]),'direct_ci_mixed_vs_gptq_excludes_zero':not(contrasts['mixed_minus_gptq']['ci95'][0]<=0<=contrasts['mixed_minus_gptq']['ci95'][1])};atomic_json(out/'gate.json',gate)
 text='# WikiText-2 mixed polar/GPTQ replication\n\n| Method | NLL | Delta NLL | Export bytes |\n|---|---:|---:|---:|\n'+''.join(f"| {r['method']} | {r['validation']['nll']:.6f} | {r['delta_nll_vs_bf16']:+.6f} | {r['payload_bytes']:,} |\n" for r in rows);(out/'summary.md').write_text(text+'\nDirect contrasts:\n```json\n'+json.dumps(contrasts,indent=2)+'\n```\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
