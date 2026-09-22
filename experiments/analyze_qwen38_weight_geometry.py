"""Bounded real-weight geometry screen for one official Qwen3.8-27B shard.

This is a weight-only diagnostic, not a language-model accuracy experiment.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from safetensors import safe_open
from run_qwen_crosslayer_screen import real4
from run_qwen_taskaware_pairing import polar35,pairing_indices

TARGETS={
 'deltanet_qkv':'model.language_model.layers.0.linear_attn.in_proj_qkv.weight',
 'attention_q':'model.language_model.layers.3.self_attn.q_proj.weight',
 'attention_mlp_up':'model.language_model.layers.3.mlp.up_proj.weight',
}
PAIRINGS=['adjacent','split_half','reverse_half','stride257']

def stats(w,q):
 e=(q-w).float();den=w.float().square().mean().item()
 return {'mse':e.square().mean().item(),'relative_mse':e.square().mean().item()/den,'max_abs_error':e.abs().amax().item(),'mean_abs_error':e.abs().mean().item()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',required=True);ap.add_argument('--out',default='results/qwen38_27b_weight_geometry_v1');ap.add_argument('--rows',type=int,default=256);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
 with safe_open(a.shard,framework='pt',device='cpu') as sf:
  available=set(sf.keys())
  for label,key in TARGETS.items():
   if key not in available:raise KeyError(key)
   shape=sf.get_slice(key).get_shape();idx=torch.linspace(0,shape[0]-1,min(a.rows,shape[0])).round().long().unique();w=sf.get_tensor(key).index_select(0,idx).float().cuda();n,k=w.shape
   qr,_,_=real4(w,True);real=stats(w,qr);cand=[]
   for pairing in PAIRINGS:
    qp,mi,pi,scale=polar35(w,pairing);s=stats(w,qp)
    ia,ib=pairing_indices(k,pairing,w.device);z=torch.complex(w[:,ia],w[:,ib]);mag=z.abs();phase=torch.angle(z);phase_q=pi.float()*(2*math.pi/32);angular=((phase-phase_q+math.pi)%(2*math.pi)-math.pi).abs();pair_error=torch.complex((qp-w)[:,ia],(qp-w)[:,ib]).abs().square();edges=torch.quantile(mag.flatten()[::max(1,mag.numel()//1000000)],torch.tensor([0.,.25,.5,.75,1.],device='cuda'));bins=[]
    for b in range(4):
     mask=(mag>=edges[b]) & ((mag<=edges[b+1]) if b==3 else (mag<edges[b+1]));bins.append({'quartile':b+1,'count':int(mask.sum()),'mean_pair_squared_error':pair_error[mask].mean().item(),'mean_angle_error_degrees':angular[mask].mean().item()*180/math.pi})
    cand.append({'pairing':pairing,**s,'polar_better_than_real4_mse':s['mse']<real['mse'],'phase_magnitude_bins':bins,'zero_magnitude_index_fraction':(mi==0).float().mean().item()})
    del qp,mi,pi,scale,z,mag,phase,phase_q,angular,pair_error
   compressed_bytes=(n*k+1)//2+n*4+1
   abs_sample=w.abs().flatten()[::max(1,w.numel()//1000000)];row={'tensor':key,'source_shape':shape,'sampled_rows':idx.tolist() if n<shape[0] else 'all','sample_shape':[n,k],'reference_bf16_bytes':n*k*2,'real4_payload_bytes':compressed_bytes,'polar_payload_bytes':compressed_bytes,'weight_rms':w.square().mean().sqrt().item(),'weight_abs_quantiles':torch.quantile(abs_sample,torch.tensor([.5,.9,.99,.999],device='cuda')).tolist(),'real4':real,'polar':cand,'best_polar_pairing_by_mse':min(cand,key=lambda v:v['mse'])['pairing']};records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'tensor':label,'shape':shape,'rows':n,'real4_mse':real['mse'],'best_polar_mse':min(x['mse'] for x in cand),'best_pairing':row['best_polar_pairing_by_mse']}),flush=True);del w,qr
 (out/'plan.json').write_text(json.dumps({'model':'Qwen/Qwen3.8-27B','source':'official BF16 safetensors shard 1','method':'256 evenly spaced actual rows per target tensor; full input width; 10 scale candidates per row; four fixed polar pairings','scope':'weight reconstruction only; no activations, task NLL, or full-model claim','payload':'actual equal-byte sampled-row payload including one FP32 row scale and one pairing-ID byte','runtime_seconds':time.monotonic()-start},indent=2)+'\n')
 text='# Qwen3.8-27B weight geometry screen\n\nReal official checkpoint weights. Weight-only reconstruction; no task accuracy measured.\n\n| Tensor | Rows | Real4 MSE | Best polar MSE | Best pairing | Polar/real MSE |\n|---|---:|---:|---:|---|---:|\n'
 for r in records:
  p=min(r['polar'],key=lambda x:x['mse']);text+=f"| {r['tensor']} | {r['sample_shape'][0]} | {r['real4']['mse']:.8g} | {p['mse']:.8g} | {p['pairing']} | {p['mse']/r['real4']['mse']:.3f} |\n"
 (out/'summary.md').write_text(text);print(json.dumps({'complete':True,'seconds':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()
