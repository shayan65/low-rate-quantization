"""Fused Triton matmul that decodes packed polar weights inside the dot-product loop."""
import argparse,json
from pathlib import Path
import torch,triton,triton.language as tl
from transformers import Qwen3_5ForCausalLM
from run_qwen_taskaware_pairing import polar35
from run_qwen_crosslayer_screen import pack
@triton.jit
def fused(x,mag,phase,scale,y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,P:tl.constexpr,pairing:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BP:tl.constexpr):
 im=tl.program_id(0)*BM+tl.arange(0,BM);jn=tl.program_id(1)*BN+tl.arange(0,BN);acc=tl.zeros((BM,BN),tl.float32)
 for j0 in range(0,P,BP):
  j=j0+tl.arange(0,BP);q=jn[:,None]*P+j[None,:];valid=(jn[:,None]<N)&(j[None,:]<P)
  mb=q*3;mi=(tl.load(mag+mb//8,mask=valid,other=0).to(tl.uint16)|(tl.load(mag+mb//8+1,mask=valid,other=0).to(tl.uint16)<<8))>>(mb%8);mi=mi&7
  pb=q*5;pi=(tl.load(phase+pb//8,mask=valid,other=0).to(tl.uint16)|(tl.load(phase+pb//8+1,mask=valid,other=0).to(tl.uint16)<<8))>>(pb%8);pi=pi&31;a=pi.to(tl.float32)*0.19634954084936207;r=tl.load(scale+jn[:,None],mask=jn[:,None]<N,other=0.0)*mi.to(tl.float32)/7.0
  if pairing==0:ia=2*j;ib=2*j+1
  elif pairing==1:ia=j;ib=j+P
  elif pairing==2:ia=j;ib=K-1-j
  else:ia=(j*257)%K;ib=((j+P)*257)%K
  xa=tl.load(x+im[:,None]*K+ia[None,:],mask=(im[:,None]<M)&(j[None,:]<P),other=0.0).to(tl.float32);xb=tl.load(x+im[:,None]*K+ib[None,:],mask=(im[:,None]<M)&(j[None,:]<P),other=0.0).to(tl.float32)
  acc+=tl.sum(xa[:,None,:]*(r*tl.cos(a))[None,:,:]+xb[:,None,:]*(r*tl.sin(a))[None,:,:],axis=2)
 tl.store(y+im[:,None]*N+jn[None,:],acc,mask=(im[:,None]<M)&(jn[None,:]<N))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--out',default='results/fused_polar_matmul_v1');a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=False);m=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda').eval();w=m.model.layers[0].linear_attn.in_proj_qkv.weight.detach().float();pairing='reverse_half';q,mi,pi,s=polar35(w,pairing);qb=q.to(torch.bfloat16);N,K=w.shape;P=K//2;mb=torch.tensor(list(pack(mi,3))+[0,0],device='cuda',dtype=torch.uint8);pb=torch.tensor(list(pack(pi,5))+[0,0],device='cuda',dtype=torch.uint8);sc=s.flatten().contiguous();pid=2;rows=[]
 for M in [1,16,128]:
  x=torch.randn(M,K,device='cuda',dtype=torch.bfloat16);y=torch.empty(M,N,device='cuda',dtype=torch.float32);grid=(triton.cdiv(M,2),triton.cdiv(N,8));run=lambda:fused[grid](x,mb,pb,sc,y,M,N,K,P,pid,BM=2,BN=8,BP=32,num_warps=4);run();torch.cuda.synchronize();ref=torch.nn.functional.linear(x,qb).float();err=(y-ref).abs();fms=triton.testing.do_bench(run,warmup=25,rep=100);bms=triton.testing.do_bench(lambda:torch.nn.functional.linear(x,qb),warmup=25,rep=100);rows.append({'tokens':M,'fused_packed_ms':fms,'resident_bf16_ms':bms,'slowdown':fms/bms,'max_abs_error_vs_bf16_gemm':err.max().item(),'rmse_vs_bf16_gemm':err.square().mean().sqrt().item()})
 result={'model':a.model,'layer':0,'shape':[N,K],'pairing':pairing,'compressed_bytes':(mi.numel()*3+7)//8+(pi.numel()*5+7)//8+s.numel()*4,'bf16_bytes':N*K*2,'rows':rows,'note':'Packed magnitude/phase decoding is fused into matmul; no full weight matrix is materialized.'};(o/'results.json').write_text(json.dumps(result,indent=2)+'\n');(o/'benchmark_fused_polar_matmul.py').write_bytes(Path(__file__).read_bytes());(o/'summary.md').write_text('# Fused packed polar matmul\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
