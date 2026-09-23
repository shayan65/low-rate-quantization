"""Bounded 16-window BF16 forward gate for an offloaded Qwen3.8-27B text model."""
import argparse,json,math,time,os
from pathlib import Path
import pandas as pd,torch,psutil
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM

def swap_used():return psutil.swap_memory().used
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.8-27B-metadata');ap.add_argument('--out',default='results/qwen38_27b_forward_gate_v1');ap.add_argument('--seconds',type=float,default=1200);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();process=psutil.Process();swap0=swap_used()
 tok=AutoTokenizer.from_pretrained(a.model);ids=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids;windows=[ids[s:s+129] for s in range(0,16*128,128)];assert all(len(x)==129 for x in windows)
 plan={'model':'Qwen/Qwen3.8-27B','evaluation':'first 16 contiguous 128-target WikiText-2 validation blocks, context reset each block','batch_sizes':[1,4,8,16],'max_memory':{'gpu':'20GiB','cpu':'40GiB'},'hard_wall_seconds':a.seconds,'promotion':'no swap growth, no OOM, projected full pass under 2h'};(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'stage':'load_start','swap_bytes':swap0}),flush=True)
 t=time.monotonic();model=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='auto',max_memory={0:'20GiB','cpu':'40GiB'},low_cpu_mem_usage=True,offload_folder=str(out/'offload')).eval();load_seconds=time.monotonic()-t;print(json.dumps({'stage':'loaded','seconds':load_seconds,'device_map':model.hf_device_map,'rss_bytes':process.memory_info().rss,'gpu_bytes':torch.cuda.memory_allocated()}),flush=True)
 rows=[]
 for batch in [1,4,8,16]:
  if time.monotonic()-started>a.seconds:raise TimeoutError('wall cap')
  torch.cuda.reset_peak_memory_stats();t=time.monotonic();loss=0.;count=0
  for s in range(0,16,batch):
   xx=torch.tensor(windows[s:s+batch],device='cuda');yy=xx[:,1:]
   with torch.no_grad():z=model(xx[:,:-1],use_cache=False).logits.float();loss+=F.cross_entropy(z.reshape(-1,z.shape[-1]),yy.reshape(-1),reduction='sum').item();count+=yy.numel()
   del xx,yy,z
  torch.cuda.synchronize();sec=time.monotonic()-t;row={'batch_size':batch,'seconds_16_blocks':sec,'seconds_per_block':sec/16,'projected_seconds_2042_blocks':sec/16*2042,'nll':loss/count,'target_tokens':count,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'rss_bytes':process.memory_info().rss,'swap_growth_bytes':swap_used()-swap0};rows.append(row);(out/'results.json').write_text(json.dumps({'load_seconds':load_seconds,'rows':rows},indent=2)+'\n');print(json.dumps(row),flush=True)
  if row['swap_growth_bytes']>512*1024*1024:raise RuntimeError('swap growth gate')
 result={'load_seconds':load_seconds,'rows':rows,'complete':len(rows)==4,'seconds':time.monotonic()-started,'promote':len(rows)==4 and min(x['projected_seconds_2042_blocks'] for x in rows)<7200 and all(x['swap_growth_bytes']<512*1024*1024 for x in rows)};(out/'gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'complete','seconds':result['seconds'],'promote':result['promote']}),flush=True)
if __name__=='__main__':main()
