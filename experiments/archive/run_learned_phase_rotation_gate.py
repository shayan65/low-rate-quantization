"""Activation-learned global phase-lattice rotation and pairing gate."""
import argparse,json,time,math,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from run_qwen_taskaware_pairing import polar35,pairing_indices
from run_qwen_full_delta_screen import payload_polar
from run_qwen_strong_controls import block_gptq
from run_qwen_crosslayer_screen import pack
def save(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def nll(m,ws,start,cap,label):
 z=n=0
 for s in ws:
  if time.monotonic()-start>cap:raise TimeoutError(label)
  ids=torch.tensor(s,device='cuda')[None];y=ids[:,1:];v=m(ids[:,:-1],use_cache=False).logits.float();z+=F.cross_entropy(v.reshape(-1,v.shape[-1]),y.reshape(-1),reduction='sum').item();n+=y.numel();del v
 return z/n
def rotated(w,pairing,offset_id):
 ia,ib=pairing_indices(w.shape[1],pairing,w.device);z=torch.complex(w[:,ia].float(),w[:,ib].float());mag=z.abs();step=2*math.pi/32;off=offset_id*step/16;pi=((torch.angle(z)-off)%(2*math.pi)/step).round().remainder(32).to(torch.uint8);angle=pi.float()*step+off;base=mag.amax(1,keepdim=True).clamp_min(1e-8);best_e=torch.full((w.shape[0],),float('inf'),device=w.device)
 for ratio in torch.linspace(.55,1,10,device=w.device):
  s=base*ratio;mi=(mag/s*7).round().clamp(0,7).to(torch.uint8);q=torch.polar(s*(mi.float()/7),angle);e=(q-z).abs().square().mean(1);take=e<best_e
  if 'best' not in locals():best=q;bs=s;bmi=mi
  else:best[take]=q[take];bs[take]=s[take];bmi[take]=mi[take]
  best_e=torch.minimum(best_e,e)
 out=torch.empty_like(w,dtype=torch.float32);out[:,ia]=best.real;out[:,ib]=best.imag;return out,bmi,pi,bs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seconds',type=float,default=600);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=False);start=time.monotonic();tok=AutoTokenizer.from_pretrained(a.model)
 train=tok('\n'.join(pd.read_parquet('data/wikitext2/train.parquet').text.tolist()),add_special_tokens=False).input_ids;val=tok('\n'.join(pd.read_parquet('data/wikitext2/validation.parquet').text.tolist()),add_special_tokens=False).input_ids;rng=np.random.default_rng(380917);cs=np.sort(rng.choice(len(train)-129,4,False));vs=np.sort(rng.choice(len(val)-129,16,False));cal=[train[s:s+129] for s in cs];ev=[val[s:s+129] for s in vs]
 m=Qwen3_5ForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda').eval();layers=[0,8,16];pairings=['adjacent','split_half','reverse_half','stride257'];base=nll(m,ev,start,a.seconds,'base');rows=[];sels=[]
 plan={'model':a.model,'method':'for each pairing, choose one of 16 global phase-lattice offsets by activation-weighted reconstruction MSE; choose pairing by train-only NLL','layers':layers,'calibration_targets':512,'evaluation_targets':2048,'metadata':'one pairing byte plus one offset byte per tensor','promotion':'rotated polar beats standard polar on all layers and wins at least two layers against recorded real4/GPTQ controls','test_split_touched':False};save(o/'plan.json',plan);(o/'run_learned_phase_rotation_gate.py').write_bytes(Path(__file__).read_bytes())
 for l in layers:
  p=m.model.layers[l].linear_attn.in_proj_qkv.weight;w=p.detach().float().clone();aa=[];h=m.model.layers[l].linear_attn.in_proj_qkv.register_forward_pre_hook(lambda mod,args:aa.append(args[0].detach().reshape(-1,args[0].shape[-1]).float().cpu()));nll(m,cal,start,a.seconds,'capture');h.remove();rms=torch.cat(aa).square().mean(0).sqrt().cuda();candidates=[]
  for pairing in pairings:
   scores=[]
   for off in range(16):
    q,mi,pi,s=rotated(w,pairing,off);scores.append((((q-w)*rms).square().mean().item(),off))
   off=min(scores)[1];q,mi,pi,s=rotated(w,pairing,off);p.data.copy_(q.to(torch.bfloat16));candidates.append((nll(m,cal,start,a.seconds,'pairing'),pairing,off,q,mi,pi,s,min(scores)[0]))
  choice=min(candidates,key=lambda x:x[0]);cn,pairing,off,q,mi,pi,s,wmse=choice;p.data.copy_(q.to(torch.bfloat16));rot_nll=nll(m,ev,start,a.seconds,'rot_eval');blob=bytes([pairings.index(pairing),off])+s.cpu().numpy().astype('<f4').tobytes()+pack(mi,3)+pack(pi,5);(o/f'layer{l}_rotated.bin').write_bytes(blob);p.data.copy_(w.to(torch.bfloat16))
  standard=[]
  for pn in pairings:
   sq,_,_,_=polar35(w,pn);p.data.copy_(sq.to(torch.bfloat16));standard.append((nll(m,cal,start,a.seconds,'std_cal'),pn,sq))
  _,sp,sq=min(standard,key=lambda x:x[0]);p.data.copy_(sq.to(torch.bfloat16));std_nll=nll(m,ev,start,a.seconds,'std_eval');p.data.copy_(w.to(torch.bfloat16));row={'layer':l,'rotated_pairing':pairing,'offset_id':off,'activation_weighted_mse':wmse,'rotated_delta_nll':rot_nll-base,'standard_pairing':sp,'standard_delta_nll':std_nll-base,'rotated_minus_standard':rot_nll-std_nll,'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()};rows.append(row);sels.append({'layer':l,'pairing':pairing,'offset_id':off});save(o/'results.json',rows);save(o/'selections.json',sels);print(json.dumps(row),flush=True)
 improved=sum(r['rotated_minus_standard']<0 for r in rows);gate={'complete':len(rows)==3,'seconds':time.monotonic()-start,'rotated_beats_standard_layers':improved,'promote_to_control_comparison':improved==3};save(o/'gate.json',gate);(o/'summary.md').write_text('# Learned phase-lattice rotation gate\n\nBF16 NLL %.6f.\n\n| Layer | Pairing | Offset/16 bin | Rotated ΔNLL | Standard polar ΔNLL | Difference |\n|---:|---|---:|---:|---:|---:|\n'%base+''.join(f"| {r['layer']} | {r['rotated_pairing']} | {r['offset_id']} | {r['rotated_delta_nll']:+.6f} | {r['standard_delta_nll']:+.6f} | {r['rotated_minus_standard']:+.6f} |\n" for r in rows)+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
