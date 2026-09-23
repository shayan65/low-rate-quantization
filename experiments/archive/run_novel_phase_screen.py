"""Equal-budget nonuniform and mode-adaptive phase screen on frozen hybrid LMs."""
import argparse,json,math,statistics,struct,time
from pathlib import Path
import numpy as np
import torch
from run_polar_lm import LanguageModel,digest,atomic_json
from run_coordinate_diagnostic import validation
from run_trained_transition_screen import response,relative,sample_windows

def nearest(phi,angles):
    return torch.cos(phi[...,None]-angles).argmax(-1)

def uniform(phi,offset,bits=4):
    angles=offset+torch.arange(2**bits,device=phi.device)*(2*math.pi/(2**bits))
    ix=nearest(phi,angles);return ix,angles[ix]

def warped(phi,alpha,bits=4):
    base=torch.arange(2**bits,device=phi.device)*(2*math.pi/(2**bits))
    angles=base+alpha*torch.sin(base)
    ix=nearest(phi,angles);return ix,angles[ix]

def adaptive(phi,rho,offset):
    # The high-bit group is recovered from stored decay ranks: no group mask.
    hi=rho>=torch.topk(rho,len(rho)//2).values.min();ix=torch.empty_like(phi,dtype=torch.long);q=torch.empty_like(phi)
    for mask,bits in [(hi,5),(~hi,3)]:
        ii,qq=uniform(phi[mask],offset,bits);ix[mask]=ii;q[mask]=qq
    return ix,q,hi

def variable_pack(ix,hi):
    bits=[]
    for value,is_hi in zip(ix.cpu().tolist(),hi.cpu().tolist()):
        bits.extend((value>>j)&1 for j in range(5 if is_hi else 3))
    assert len(bits)==256
    return np.packbits(np.asarray(bits,dtype=np.uint8),bitorder='little').tobytes()

def export(method,phi,rho,param,path):
    if method=='uniform_response':ix,q=uniform(phi,param);payload_ix=variable_pack(ix,torch.zeros_like(phi,dtype=torch.bool)) if False else np.packbits(((ix.cpu().numpy().astype(np.uint16)[:,None]>>np.arange(4))&1).astype(np.uint8).reshape(-1),bitorder='little').tobytes()
    elif method=='warped_response':ix,q=warped(phi,param);payload_ix=np.packbits(((ix.cpu().numpy().astype(np.uint16)[:,None]>>np.arange(4))&1).astype(np.uint8).reshape(-1),bitorder='little').tobytes()
    else:ix,q,hi=adaptive(phi,rho,param);payload_ix=variable_pack(ix,hi)
    payload=struct.pack('<f',float(param))+rho.cpu().numpy().astype('<f4').tobytes()+payload_ix
    assert len(payload)==292;path.write_bytes(payload);return q,digest(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/wikitext103_bpe8k');ap.add_argument('--checkpoints',default='results/coordinate_diagnostic_v1')
    ap.add_argument('--out',default='results/novel_phase_v1');ap.add_argument('--seconds',type=float,default=120);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();root=Path(args.data)
    manifest=json.loads((root/'manifest.json').read_text());prior=json.loads((Path(args.checkpoints)/'plan.json').read_text())
    train=np.memmap(root/'train.bin',dtype='<u2',mode='r');val=np.memmap(root/'validation.bin',dtype='<u2',mode='r');blocks=np.asarray(prior['validation_blocks'])
    device='cuda' if torch.cuda.is_available() else 'cpu';methods=['fp32','uniform_response','warped_response','adaptive_5_3_response'];rows=[]
    plan={'hypotheses':{'warped':'16 phase states theta_k=2pi*k/16+alpha*sin(2pi*k/16); one learned FP32 alpha, no offset',
      'adaptive':'top half of modes by already-stored decay receive 5-bit phase; bottom half receive 3-bit phase; 256 index bits total; one learned FP32 offset; group is deterministically reconstructed from decay ranks'},
      'controls':'response-calibrated uniform 4-bit circle with one FP32 offset','equal_export_bytes':292,'model':'frozen 697,088-parameter custom hybrid LM',
      'seeds':[0,1,2],'methods':methods,'validation_target_tokens':len(blocks)*128,'test_split_touched':False,'runtime_cap_seconds':args.seconds,'source_sha256':digest(__file__)}
    atomic_json(out/'plan.json',plan);(out/'run_novel_phase_screen.py').write_bytes(Path(__file__).read_bytes())
    for seed in plan['seeds']:
      if time.monotonic()-started>args.seconds:break
      ck=Path(args.checkpoints)/f'complex_fp32_seed{seed}/best.pt';model=LanguageModel(manifest['actual_vocab'],64,2,128,'complex_fp32','hybrid',4).to(device)
      model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval();block=model.blocks[0]
      phi=block.frequency.detach().float().clone();rho=block.decay.sigmoid().detach().float();windows=sample_windows(train,128,8,2700+seed);refs=[response(model,x) for x in windows]
      offsets=torch.linspace(-math.pi/16,math.pi/16,33,device=device);alphas=torch.linspace(-.45,.45,37,device=device)
      def score(q):return statistics.mean(relative(a,response(model,x,q)) for x,a in zip(windows,refs))
      us=[score(uniform(phi,x)[1]) for x in offsets];ws=[score(warped(phi,x)[1]) for x in alphas];ads=[score(adaptive(phi,rho,x)[1]) for x in offsets]
      params={'uniform_response':float(offsets[int(np.argmin(us))]),'warped_response':float(alphas[int(np.argmin(ws))]),'adaptive_5_3_response':float(offsets[int(np.argmin(ads))])}
      base=validation(model,val,blocks,128,manifest['pad_id'],device);rows.append({'seed':seed,'method':'fp32','validation':base,'transition_bytes':512})
      for method in methods[1:]:
        q,sha=export(method,phi,rho,params[method],out/f'{method}_seed{seed}.bin');block.frequency.data.copy_(q)
        metric=validation(model,val,blocks,128,manifest['pad_id'],device);rows.append({'seed':seed,'method':method,'parameter':params[method],
          'calibration_response_mse':{'uniform_response':min(us),'warped_response':min(ws),'adaptive_5_3_response':min(ads)}[method],
          'export_sha256':sha,'transition_bytes':292,'validation':metric,'delta_nll_vs_fp32':metric['nll']-base['nll']});block.frequency.data.copy_(phi)
      atomic_json(out/'results.json',rows);del model
    complete=len(rows)==12;groups={m:[r for r in rows if r['method']==m] for m in methods};text='# Equal-budget novel phase screen\n\nAll quantized methods use 292-byte decoded-and-evaluated transition exports.\n\n| Method | NLL | Perplexity | Delta NLL |\n|---|---:|---:|---:|\n'
    for m,g in groups.items():
      if g:text+=f"| {m} | {statistics.mean(x['validation']['nll'] for x in g):.6f} | {statistics.mean(x['validation']['perplexity'] for x in g):.3f} | {statistics.mean(x.get('delta_nll_vs_fp32',0) for x in g):+.6f} |\n"
    gate={'complete':complete,'jobs':len(rows),'seconds':time.monotonic()-started,'winner':min(methods[1:],key=lambda m:statistics.mean(x['validation']['nll'] for x in groups[m])) if complete else None,
      'novel_method_beats_uniform_all_seeds':complete and any(all(a['validation']['nll']<b['validation']['nll'] for a,b in zip(groups[m],groups['uniform_response'])) for m in methods[2:]),
      'action':'No Qwen or long training is automatically launched.'}
    atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
