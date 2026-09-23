"""Second-dataset replication on complete TinyStories validation."""
import argparse,json,time,hashlib,math
from pathlib import Path
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_full_delta_screen import loss_on_windows,payload_polar
from run_qwen_taskaware_pairing import polar35
from run_qwen_strong_controls import block_gptq
from run_qwen_crosslayer_screen import pack

def atomic_json(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def evaluate_last_token(model,examples,pad_id,started,cap,label,batch=16):
 total=0.;count=0
 for start in range(0,len(examples),batch):
  if time.monotonic()-started>cap:raise TimeoutError(label)
  group=examples[start:start+batch];lengths=[len(x) for x in group];width=max(lengths)-1;ids=torch.full((len(group),width),pad_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
  for i,x in enumerate(group):ids[i,:len(x)-1]=torch.as_tensor(x[:-1],device='cuda');mask[i,:len(x)-1]=1
  logits=model(ids,attention_mask=mask,use_cache=False).logits.float();row=torch.arange(len(group),device='cuda');pos=torch.as_tensor(lengths,device='cuda')-2;target=torch.as_tensor([x[-1] for x in group],device='cuda');total+=F.cross_entropy(logits[row,pos],target,reduction='sum').item();count+=len(group);del logits
  if (start//batch+1)%100==0:print(json.dumps({'progress':label,'examples':count}),flush=True)
 return {'nll':total/count,'perplexity':math.exp(total/count),'target_examples':count}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_tinystories_v1');ap.add_argument('--seconds',type=float,default=3600);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model)
 train_rows=json.loads(Path('data/tinystories/train_first_rows.json').read_text())['rows'];train_text='\n'.join(x['row']['text'] for x in train_rows);train_ids=tok(train_text,add_special_tokens=False).input_ids;cal=[train_ids[s:s+129] for s in [0,256,512,768,1024,1280,1536,1792]]
 val_texts=pd.read_parquet('data/tinystories/validation.parquet').text.tolist();validation=[x for x in tok(val_texts,add_special_tokens=False,truncation=True,max_length=129).input_ids if len(x)>=2]
 model=Qwen3_5ForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[i for i,l in enumerate(model.model.layers) if hasattr(l,'linear_attn')];pairings=['adjacent','split_half','reverse_half','stride257']
 captures={l:[] for l in layers};hooks=[]
 for l in layers:
  def hook(module,args,l=l):captures[l].append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu())
  hooks.append(model.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(hook))
 base_cal=loss_on_windows(model,cal,started,args.seconds,'tinystories_cal_capture')['nll']
 for h in hooks:h.remove()
 captures={l:torch.cat(v).to('cuda') for l,v in captures.items()};originals={l:model.model.layers[l].linear_attn.in_proj_qkv.weight.detach().float().cpu() for l in layers}
 plan={'dataset':'roneneldan/TinyStories','calibration':'first 2,048 Qwen tokens from 100 train rows, eight fixed 128-target windows','validation':'last-token NLL for every nonempty example in complete official validation parquet; at most 128 preceding tokens','validation_examples':len(validation),'test_split_touched':False,'model':'Qwen/Qwen3.5-0.8B-Base','layers':layers,'methods':['bf16','task_polar3_5','block_gptq_real4'],'runtime_cap_seconds':args.seconds,'warning':'block-GPTQ is transparent local implementation, not official backend'};atomic_json(out/'plan.json',plan);(out/'run_qwen_tinystories_replication.py').write_bytes(Path(__file__).read_bytes())
 selections=[]
 for l in layers:
  p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda');scores=[]
  for pairing in pairings:
   q,_,_,_=polar35(w,pairing);p.data.copy_(q.to(torch.bfloat16));scores.append(loss_on_windows(model,cal,started,args.seconds,f'pair_l{l}')['nll'])
  p.data.copy_(w.to(torch.bfloat16));i=min(range(4),key=lambda j:scores[j]);selections.append({'layer':l,'candidate_id':i,'pairing':pairings[i],'calibration_nlls':scores,'bf16_calibration_nll':base_cal});atomic_json(out/'selections.json',selections);print(json.dumps(selections[-1]),flush=True)
 rows=[]
 for method in plan['methods']:
  directory=out/method;directory.mkdir();payload_bytes=0
  if method!='bf16':
   for sel in selections:
    l=sel['layer'];p=model.model.layers[l].linear_attn.in_proj_qkv.weight;w=originals[l].to('cuda')
    if method=='task_polar3_5':q,payload=payload_polar(w,sel['pairing']);blob=bytes([sel['candidate_id']])+payload
    else:
     q,ix,scales=block_gptq(w,captures[l]);blob=scales.cpu().numpy().astype('<f4').tobytes()+pack(ix,4)
    path=directory/f'layer{l}.bin';path.write_bytes(blob);payload_bytes+=len(blob);p.data.copy_(q.to(torch.bfloat16))
  metric=evaluate_last_token(model,validation,tok.pad_token_id or tok.eos_token_id,started,args.seconds,f'tinystories_{method}');row={'method':method,'validation':metric,'payload_bytes':payload_bytes,'delta_nll_vs_bf16':None if not rows else metric['nll']-rows[0]['validation']['nll']};rows.append(row);atomic_json(out/'results.json',rows);print(json.dumps(row),flush=True)
  for l in layers:model.model.layers[l].linear_attn.in_proj_qkv.weight.data.copy_(originals[l].to('cuda',dtype=torch.bfloat16))
 complete=len(rows)==3;gate={'complete':complete,'seconds':time.monotonic()-started,'polar_beats_block_gptq':complete and rows[1]['validation']['nll']<rows[2]['validation']['nll'],'action':'Proceed to second-checkpoint pilot only if complete.'};atomic_json(out/'gate.json',gate)
 text='# TinyStories second-dataset replication\n\nComplete official validation split; train-only pairing/Hessian calibration; test split untouched.\n\n| Method | NLL | Perplexity | Delta NLL | Export bytes |\n|---|---:|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['method']} | {r['validation']['nll']:.6f} | {r['validation']['perplexity']:.3f} | {(r['delta_nll_vs_bf16'] or 0):+.6f} | {r['payload_bytes']:,} |\n"
 (out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
