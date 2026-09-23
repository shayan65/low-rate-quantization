"""Short hardware feasibility probe on one official Qwen3.8-27B tensor."""
import argparse,json,time,os
from pathlib import Path
import torch
from safetensors import safe_open
from run_qwen_crosslayer_screen import real4,pack
from run_qwen_taskaware_pairing import polar35

KEY='model.language_model.layers.0.linear_attn.in_proj_qkv.weight'
def tick():torch.cuda.synchronize();return time.perf_counter()
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',required=True);p.add_argument('--out',default='results/qwen38_3090_probe_v1');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);result={'model':'Qwen/Qwen3.8-27B','tensor':KEY,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__};t0=time.perf_counter()
 with safe_open(a.shard,framework='pt',device='cpu') as sf:wcpu=sf.get_tensor(KEY).float()
 result['disk_to_cpu_seconds']=time.perf_counter()-t0;result['shape']=list(wcpu.shape);result['tensor_fp32_bytes']=wcpu.numel()*4
 torch.cuda.reset_peak_memory_stats();t=tick();w=wcpu.cuda();result['cpu_to_gpu_seconds']=tick()-t;result['gpu_baseline_bytes']=torch.cuda.memory_allocated()
 t=tick();qr,ri,rs=real4(w,True);result['real4_quantize_seconds']=tick()-t;result['real4_mse']=(qr-w).square().mean().item();result['real4_peak_gpu_bytes']=torch.cuda.max_memory_allocated();del qr,ri,rs;torch.cuda.empty_cache()
 torch.cuda.reset_peak_memory_stats();t=tick();qp,mi,pi,s=polar35(w,'split_half');result['polar_quantize_seconds']=tick()-t;result['polar_mse']=(qp-w).square().mean().item();result['polar_peak_gpu_bytes']=torch.cuda.max_memory_allocated()
 t=time.perf_counter();mb=pack(mi,3);pb=pack(pi,5);result['index_pack_seconds']=time.perf_counter()-t;result['packed_bytes']=len(mb)+len(pb)+s.numel()*4+1
 for tokens in (1,16,128):
  x=torch.randn(tokens,w.shape[1],device='cuda',dtype=torch.bfloat16);qb=qp.to(torch.bfloat16);torch.nn.functional.linear(x,qb);tick();starts=[]
  for _ in range(20):
   t=tick();torch.nn.functional.linear(x,qb);starts.append((tick()-t)*1000)
  result[f'projection_bf16_ms_tokens_{tokens}']=sorted(starts)[len(starts)//2]
 result['total_probe_seconds']=time.perf_counter()-t0;(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'probe_qwen38_3090.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result),flush=True)
if __name__=='__main__':main()
