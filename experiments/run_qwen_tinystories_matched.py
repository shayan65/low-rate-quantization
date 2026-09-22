"""Matched-protocol TinyStories pilot: same target count and resets as WikiText-2."""
import argparse,json,time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_full_delta_screen import loss_on_windows,payload_polar,payload_real,real_candidate
from run_qwen_taskaware_pairing import polar35
from run_qwen_strong_controls import block_gptq
from run_qwen_crosslayer_screen import pack

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_tinystories_matched_v1');ap.add_argument('--seconds',type=float,default=1800);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 train_rows=json.loads(Path('data/tinystories/train_first_rows.json').read_text())['rows'];train_ids=tok('\n'.join(x['row']['text'] for x in train_rows),add_special_tokens=False).input_ids;cal=[train_ids[s:s+129] for s in [0,256,512,768,1024,1280,1536,1792]]
 val_ids=tok('\n'.join(pd.read_parquet('data/tinystories/validation.parquet').text.tolist()),add_special_tokens=False).input_ids[:261285];validation=[val_ids[s:min(s+129,len(val_ids))] for s in range(0,len(val_ids)-1,128)];assert sum(len(x)-1 for x in validation)==261284
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[i for i,l in enumerate(model.model.layers) if hasattr(l,'linear_attn')];pairings=['adjacent','split_half','reverse_half','stride257'];factors=[.85,.95,1.,1.05]
 captures={l:[] for l in layers};hooks=[]
 for l in layers:
  def hook(module,args,l=l):captures[l].append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu())
  hooks.append(model.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(hook))
 base_cal=loss_on_windows(model,cal,started,args.seconds,'matched_cal_capture')['nll']
 for h in hooks:h.remove()
 captures={l:torch.cat(v).to('cuda') for l,v in captures.items()};originals={l:model.model.layers[l].linear_attn.in_proj_qkv.weight.detach().float().cpu() for l in layers};selections=[]
 plan={'dataset':'roneneldan/TinyStories','evaluation':'first 261,284 contiguous validation targets in 128-target blocks, identical target count/reset policy to WikiText-2','calibration':'eight fixed 128-target train windows','test_split_touched':False,'methods':['bf16','task_real4','task_polar3_5','block_gptq_real4'],'runtime_cap_seconds':args.seconds};atomic_json(out/'plan.json',plan);(out/'run_qwen_tinystories_matched.py').write_bytes(Path(__file__).read_bytes())
 for l in layers:
  p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda');rs=[];ps=[]
  for factor in factors:
   q,_,_=real_candidate(w,factor);p.data.copy_(q.to(torch.bfloat16));rs.append(loss_on_windows(model,cal,started,args.seconds,f'real_l{l}')['nll'])
  for pairing in pairings:
   q,_,_,_=polar35(w,pairing);p.data.copy_(q.to(torch.bfloat16));ps.append(loss_on_windows(model,cal,started,args.seconds,f'polar_l{l}')['nll'])
  p.data.copy_(w.to(torch.bfloat16));ri=min(range(4),key=lambda i:rs[i]);pi=min(range(4),key=lambda i:ps[i]);selections.append({'layer':l,'real_id':ri,'real_factor':factors[ri],'real_nlls':rs,'polar_id':pi,'polar_pairing':pairings[pi],'polar_nlls':ps,'bf16_calibration_nll':base_cal});atomic_json(out/'selections.json',selections);print(json.dumps(selections[-1]),flush=True)
 rows=[]
 for method in plan['methods']:
  directory=out/method;directory.mkdir();payload_bytes=0
  if method!='bf16':
   for sel in selections:
    l=sel['layer'];p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda')
    if method=='task_real4':q,payload=payload_real(w,sel['real_factor']);blob=bytes([sel['real_id']])+payload
    elif method=='task_polar3_5':q,payload=payload_polar(w,sel['polar_pairing']);blob=bytes([sel['polar_id']])+payload
    else:q,ix,scales=block_gptq(w,captures[l]);blob=scales.cpu().numpy().astype('<f4').tobytes()+pack(ix,4)
    (directory/f'layer{l}.bin').write_bytes(blob);payload_bytes+=len(blob);p.data.copy_(q.to(torch.bfloat16))
  metric=loss_on_windows(model,validation,started,args.seconds,f'matched_{method}',progress=True);row={'method':method,'validation':metric,'payload_bytes':payload_bytes,'delta_nll_vs_bf16':None if not rows else metric['nll']-rows[0]['validation']['nll']};rows.append(row);atomic_json(out/'results.json',rows);print(json.dumps(row),flush=True)
  for l in layers:model.model.layers[l].linear_attn.in_proj_qkv.weight.data.copy_(originals[l].to('cuda',dtype=torch.bfloat16))
 complete=len(rows)==4;gate={'complete':complete,'seconds':time.monotonic()-started,'polar_beats_block_gptq':complete and rows[2]['validation']['nll']<rows[3]['validation']['nll'],'polar_beats_task_real4':complete and rows[2]['validation']['nll']<rows[1]['validation']['nll'],'action':'If polar still loses GPTQ, revise to a layer-wise mixed codec.'};atomic_json(out/'gate.json',gate)
 text='# Matched-protocol TinyStories pilot\n\nExactly 261,284 contiguous validation targets and 128-target resets, matching the WikiText-2 protocol.\n\n| Method | NLL | Perplexity | Delta NLL | Export bytes |\n|---|---:|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['method']} | {r['validation']['nll']:.6f} | {r['validation']['perplexity']:.3f} | {(r['delta_nll_vs_bf16'] or 0):+.6f} | {r['payload_bytes']:,} |\n"
 (out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
