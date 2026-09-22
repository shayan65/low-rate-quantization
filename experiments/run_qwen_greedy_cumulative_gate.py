"""Bounded cumulative codec selector; full validation runs only after a train-only gate."""
import argparse,json,time,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_full_delta_screen import loss_on_windows,payload_polar
from run_qwen_taskaware_pairing import polar35
from run_qwen_strong_controls import block_gptq
from run_qwen_crosslayer_screen import pack

def save(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def blocks(model,windows,started,cap,label):
 out=[]
 for i,s in enumerate(windows):
  if time.monotonic()-started>cap:raise TimeoutError(label)
  ids=torch.tensor(s,device='cuda')[None];z=model(ids[:,:-1],use_cache=False).logits.float();y=ids[:,1:];out.append({'loss_sum':F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction='sum').item(),'count':y.numel()});del z
  if (i+1)%100==0:print(json.dumps({'progress':label,'blocks':i+1}),flush=True)
 n=sum(x['count'] for x in out);v=sum(x['loss_sum'] for x in out)/n;return {'nll':v,'perplexity':math.exp(v),'target_tokens':n},out
def ci(a,b,seed=91730,reps=5000):
 x=np.array([v['loss_sum'] for v in a]);y=np.array([v['loss_sum'] for v in b]);n=np.array([v['count'] for v in a]);rng=np.random.default_rng(seed);z=[]
 for _ in range(reps//100):
  ix=rng.integers(0,len(x),(100,len(x)));z.extend(((x[ix].sum(1)-y[ix].sum(1))/n[ix].sum(1)).tolist())
 return {'mean_delta':float((x.sum()-y.sum())/n.sum()),'ci95':[float(np.quantile(z,.025)),float(np.quantile(z,.975))],'reps':reps}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/qwen_greedy_cumulative_gate_v1');ap.add_argument('--seconds',type=float,default=1500);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=False);start=time.monotonic();tok=AutoTokenizer.from_pretrained(a.model)
 train=tok('\n'.join(pd.read_parquet('data/wikitext2/train.parquet').text.tolist()),add_special_tokens=False).input_ids;val=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids
 select_starts=[59959,131396,1638120,2226031];gate_starts=[310007,744211,1193033,2017009];select=[train[s:s+129] for s in select_starts];gate=[train[s:s+129] for s in gate_starts];validation=[val[s:min(s+129,len(val))] for s in range(0,len(val)-1,128)]
 sels=json.loads(Path('results/qwen_full_delta_v1/selections.json').read_text());m=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[x['layer'] for x in sels];orig={l:m.model.layers[l].linear_attn.in_proj_qkv.weight.detach().float().cpu() for l in layers}
 cap={l:[] for l in layers};hs=[]
 for l in layers:
  def hook(mod,args,l=l):cap[l].append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu())
  hs.append(m.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(hook))
 loss_on_windows(m,select,start,a.seconds,'capture')
 for h in hs:h.remove()
 cap={l:torch.cat(v).cuda() for l,v in cap.items()};choices=[]
 plan={'status':'exploratory after independent selector failed on WikiText-2','selection':'ascending-layer greedy cumulative NLL on four fixed training windows','gate':'four disjoint fixed training windows; full validation only if greedy NLL < uniform polar NLL','select_starts':select_starts,'gate_starts':gate_starts,'validation_targets':261284,'test_split_touched':False,'runtime_cap_seconds':a.seconds};save(o/'plan.json',plan);(o/'run_qwen_greedy_cumulative_gate.py').write_bytes(Path(__file__).read_bytes())
 for sel in sels:
  l=sel['layer'];p=m.model.layers[l].linear_attn.in_proj_qkv.weight;w=orig[l].cuda();qp,_,_,_=polar35(w,sel['polar_pairing']);qg,_,_=block_gptq(w,cap[l]);p.data.copy_(qp.to(torch.bfloat16));pn=loss_on_windows(m,select,start,a.seconds,f'polar_l{l}')['nll'];p.data.copy_(qg.to(torch.bfloat16));gn=loss_on_windows(m,select,start,a.seconds,f'gptq_l{l}')['nll'];codec='polar' if pn<=gn else 'gptq';p.data.copy_((qp if codec=='polar' else qg).to(torch.bfloat16));choices.append({'layer':l,'codec':codec,'polar_cumulative_select_nll':pn,'gptq_cumulative_select_nll':gn,'polar_pairing':sel['polar_pairing'],'polar_id':sel['polar_candidate_id']});save(o/'choices.json',choices);print(json.dumps(choices[-1]),flush=True)
 greedy_gate=loss_on_windows(m,gate,start,a.seconds,'greedy_gate')['nll']
 for sel in sels:
  l=sel['layer'];q,_,_,_=polar35(orig[l].cuda(),sel['polar_pairing']);m.model.layers[l].linear_attn.in_proj_qkv.weight.data.copy_(q.to(torch.bfloat16))
 polar_gate=loss_on_windows(m,gate,start,a.seconds,'polar_gate')['nll'];promote=greedy_gate<polar_gate;gate_result={'greedy_nll':greedy_gate,'polar_nll':polar_gate,'greedy_minus_polar':greedy_gate-polar_gate,'promote':promote,'polar_layers':sum(x['codec']=='polar' for x in choices),'gptq_layers':sum(x['codec']=='gptq' for x in choices)};save(o/'gate.json',gate_result);print(json.dumps(gate_result),flush=True)
 if not promote:(o/'summary.md').write_text('# Greedy cumulative selector\n\nStopped at the train-only gate.\n\n```json\n'+json.dumps(gate_result,indent=2)+'\n```\n');return
 results=[];raw={}
 for method in ['bf16','polar','greedy']:
  for l in layers:m.model.layers[l].linear_attn.in_proj_qkv.weight.data.copy_(orig[l].cuda().to(torch.bfloat16))
  payload=0
  if method!='bf16':
   d=o/method;d.mkdir()
   for sel,ch in zip(sels,choices):
    l=sel['layer'];w=orig[l].cuda();codec='polar' if method=='polar' else ch['codec']
    if codec=='polar':q,b=payload_polar(w,sel['polar_pairing']);blob=bytes([sel['polar_candidate_id']])+b
    else:q,ix,sc=block_gptq(w,cap[l]);blob=bytes([4])+sc.cpu().numpy().astype('<f4').tobytes()+pack(ix,4)
    (d/f'layer{l}.bin').write_bytes(blob);payload+=len(blob);m.model.layers[l].linear_attn.in_proj_qkv.weight.data.copy_(q.to(torch.bfloat16))
  metric,bb=blocks(m,validation,start,a.seconds,method);results.append({'method':method,'validation':metric,'payload_bytes':payload});raw[method]=bb;save(o/'results.json',results);print(json.dumps(results[-1]),flush=True)
 contrast=ci(raw['greedy'],raw['polar']);save(o/'confidence_interval.json',contrast);final={'complete':True,**gate_result,'seconds':time.monotonic()-start,'validation_greedy_minus_polar':contrast};save(o/'final.json',final);(o/'summary.md').write_text('# Greedy cumulative selector\n\n```json\n'+json.dumps(final,indent=2)+'\n```\n');print(json.dumps(final),flush=True)
if __name__=='__main__':main()
