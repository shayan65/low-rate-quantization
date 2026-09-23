"""Triton packed polar decoder plus GEMM latency benchmark on a real Qwen tensor."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch,triton,triton.language as tl
from transformers import Qwen3_5ForCausalLM
from run_qwen_taskaware_pairing import polar35
from run_qwen_crosslayer_screen import pack

@triton.jit
def decode_kernel(mag,phase,scale,out,n_pairs:tl.constexpr,k:tl.constexpr,total:tl.constexpr,pairing:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B);mask=q<total;row=q//n_pairs;j=q-row*n_pairs
 mb=q*3;mi=(tl.load(mag+mb//8,mask=mask,other=0).to(tl.uint16)|(tl.load(mag+mb//8+1,mask=mask,other=0).to(tl.uint16)<<8))>>(mb%8);mi=mi&7
 pb=q*5;pi=(tl.load(phase+pb//8,mask=mask,other=0).to(tl.uint16)|(tl.load(phase+pb//8+1,mask=mask,other=0).to(tl.uint16)<<8))>>(pb%8);pi=pi&31
 a=pi.to(tl.float32)*0.19634954084936207;r=tl.load(scale+row,mask=mask,other=0.0)*mi.to(tl.float32)/7.0
 if pairing==0: ia=2*j;ib=2*j+1
 elif pairing==1: ia=j;ib=j+n_pairs
 elif pairing==2: ia=j;ib=k-1-j
 else: ia=(j*257)%k;ib=((j+n_pairs)*257)%k
 tl.store(out+row*k+ia,r*tl.cos(a),mask=mask);tl.store(out+row*k+ib,r*tl.sin(a),mask=mask)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3.5-0.8B-Base');ap.add_argument('--layer',type=int,default=0);ap.add_argument('--pairing',default='reverse_half',choices=['adjacent','split_half','reverse_half','stride257']);ap.add_argument('--out',default='results/packed_polar_triton_v1');a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=False)
 m=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda').eval();w=m.model.layers[a.layer].linear_attn.in_proj_qkv.weight.detach().float();q,mi,pi,s=polar35(w,a.pairing);n,k=w.shape;p=k//2
 mb=torch.tensor(list(pack(mi,3))+[0,0],device='cuda',dtype=torch.uint8);pb=torch.tensor(list(pack(pi,5))+[0,0],device='cuda',dtype=torch.uint8);sc=s.flatten().contiguous();decoded=torch.empty((n,k),device='cuda',dtype=torch.float32);pid=['adjacent','split_half','reverse_half','stride257'].index(a.pairing);grid=(triton.cdiv(n*p,256),)
 def decode():decode_kernel[grid](mb,pb,sc,decoded,p,k,n*p,pid,B=256)
 decode();torch.cuda.synchronize();max_err=(decoded-q).abs().max().item();rmse=(decoded-q).square().mean().sqrt().item();assert max_err<2e-5,(max_err,rmse)
 decode_ms=triton.testing.do_bench(decode,warmup=25,rep=100)
 rows=[];qb=q.to(torch.bfloat16)
 for batch in [1,16,128]:
  x=torch.randn(batch,k,device='cuda',dtype=torch.bfloat16);torch.nn.functional.linear(x,qb);torch.cuda.synchronize()
  gemm=triton.testing.do_bench(lambda:torch.nn.functional.linear(x,qb),warmup=25,rep=100)
  def end_to_end():decode();torch.nn.functional.linear(x,decoded.to(torch.bfloat16))
  e2e=triton.testing.do_bench(end_to_end,warmup=25,rep=100);rows.append({'tokens':batch,'resident_bf16_gemm_ms':gemm,'packed_decode_plus_gemm_ms':e2e,'slowdown':e2e/gemm})
 compressed=mi.numel()*8//8+pi.numel()*8//8 # overwritten below with actual bit count
 compressed=(mi.numel()*3+7)//8+(pi.numel()*5+7)//8+s.numel()*4
 result={'model':a.model,'layer':a.layer,'shape':[n,k],'pairing':a.pairing,'correctness':{'max_abs_error':max_err,'rmse':rmse},'decode_ms':decode_ms,'compressed_bytes':compressed,'bf16_bytes':n*k*2,'compression_ratio':n*k*2/compressed,'latency':rows,'note':'Prototype decodes the packed tensor with Triton before cuBLAS GEMM; it is not a fused production matmul.'};(o/'results.json').write_text(json.dumps(result,indent=2)+'\n');(o/'benchmark_packed_polar_triton.py').write_bytes(Path(__file__).read_bytes());(o/'summary.md').write_text('# Packed polar Triton prototype\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
