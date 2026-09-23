"""Frozen language-model NLL after quantizing only the recurrent pole angle."""
import argparse,json,math,statistics,struct,time
from pathlib import Path
import numpy as np
import torch
from run_polar_lm import LanguageModel,pack_indices,unpack_indices,digest,atomic_json
from run_coordinate_diagnostic import validation
from run_trained_transition_screen import response,relative,sample_windows

def circular(phi,offset):
    step=2*math.pi/16
    ix=((phi-offset)/step).round().remainder(16).to(torch.uint8)
    return ix,offset+ix.float()*step

def square_directions(phi,offset):
    levels=torch.tensor([-1.,-1/3,1/3,1.],device=phi.device)
    directions=torch.angle((levels[:,None]+1j*levels[None,:]).reshape(-1))+offset
    ix=torch.cos(phi[...,None]-directions).argmax(-1).to(torch.uint8)
    return ix,directions[ix.long()]

def export_decode(phi,rho,method,offset,path):
    ix,q=(circular(phi,offset) if method.startswith(('phase','circular')) else square_directions(phi,offset))
    payload=struct.pack('<f',float(offset))+rho.cpu().numpy().astype('<f4').tobytes()+pack_indices(ix.cpu().numpy(),4)
    path.write_bytes(payload);restored=unpack_indices(payload[4+4*len(phi):],4,len(phi))
    assert np.array_equal(restored,ix.cpu().numpy()) and len(payload)==292
    if method.startswith(('phase','circular')): q=offset+torch.as_tensor(restored,device=phi.device)*(2*math.pi/16)
    else:
        levels=torch.tensor([-1.,-1/3,1/3,1.],device=phi.device)
        directions=torch.angle((levels[:,None]+1j*levels[None,:]).reshape(-1))+offset
        q=directions[torch.as_tensor(restored,device=phi.device)]
    return q,digest(path),len(payload)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/wikitext103_bpe8k')
    ap.add_argument('--checkpoints',default='results/coordinate_diagnostic_v1')
    ap.add_argument('--out',default='results/frozen_lm_transition_v1');ap.add_argument('--seconds',type=float,default=120)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    root=Path(args.data);manifest=json.loads((root/'manifest.json').read_text());prior=json.loads((Path(args.checkpoints)/'plan.json').read_text())
    train=np.memmap(root/'train.bin',dtype='<u2',mode='r');val=np.memmap(root/'validation.bin',dtype='<u2',mode='r')
    val_blocks=np.asarray(prior['validation_blocks'],dtype=np.int64);device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    methods=['fp32','phase_nearest','phase_weight_offset','phase_response_offset','circular_response_offset','square_direction_response_offset']
    plan={'model':'custom hybrid LM: 8K BPE vocab, width 64, depth 2, context 128; recurrent complex diagonal layer then causal-attention layer',
      'parameters':697088,'dataset':'Salesforce/wikitext wikitext-103-raw-v1','dataset_revision':manifest['revision'],
      'validation_blocks':val_blocks.tolist(),'validation_target_tokens':len(val_blocks)*128,'test_split_touched':False,
      'checkpoints':[f'complex_fp32_seed{s}/best.pt' for s in range(3)],'methods':methods,'seeds':[0,1,2],
      'calibration':'8 fixed original training-text windows per seed; 33 shared offsets; response MSE uses frozen first recurrent layer',
      'budget':'quantized methods: 64 four-bit angle indices + 64 unchanged FP32 decay values + one FP32 offset = 292 bytes',
      'circular_control':'algebraically identical 16-point circle and calibration to phase_response_offset; equality is required',
      'square_control':'16 Cartesian grid labels reduced to their pole directions; duplicate directions remain distinct stored labels, exposing the geometry limitation',
      'runtime_cap_seconds':args.seconds,'source_sha256':digest(__file__)}
    atomic_json(out/'plan.json',plan);(out/'run_frozen_lm_transition_screen.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for seed in plan['seeds']:
      if time.monotonic()-started>args.seconds:break
      ck=Path(args.checkpoints)/f'complex_fp32_seed{seed}/best.pt';torch.manual_seed(seed)
      model=LanguageModel(manifest['actual_vocab'],64,2,128,'complex_fp32','hybrid',4).to(device)
      model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval()
      block=model.blocks[0];phi=block.frequency.detach().float().clone();rho=block.decay.sigmoid().detach().float()
      windows=sample_windows(train,128,8,1700+seed);refs=[response(model,x) for x in windows]
      deltas=torch.linspace(-math.pi/16,math.pi/16,33,device=device)
      weight=[];circ=[];square=[]
      for d in deltas:
        if time.monotonic()-started>args.seconds:break
        _,pc=circular(phi,d);_,ps=square_directions(phi,d)
        weight.append((rho.square()*2*(1-(pc-phi).cos())).sum().item())
        circ.append(statistics.mean(relative(a,response(model,x,pc)) for x,a in zip(windows,refs)))
        square.append(statistics.mean(relative(a,response(model,x,ps)) for x,a in zip(windows,refs)))
      if len(weight)!=33:break
      offsets={'phase_nearest':0.,'phase_weight_offset':float(deltas[int(np.argmin(weight))]),
        'phase_response_offset':float(deltas[int(np.argmin(circ))]),'circular_response_offset':float(deltas[int(np.argmin(circ))]),
        'square_direction_response_offset':float(deltas[int(np.argmin(square))])}
      original=validation(model,val,val_blocks,128,manifest['pad_id'],device)
      rows.append({'seed':seed,'method':'fp32','checkpoint_sha256':digest(ck),'validation':original,'transition_bytes':64*8})
      for method in methods[1:]:
        q,sha,nbytes=export_decode(phi,rho,method,offsets[method],out/f'{method}_seed{seed}.bin')
        with torch.no_grad():block.frequency.copy_(q)
        metric=validation(model,val,val_blocks,128,manifest['pad_id'],device)
        rows.append({'seed':seed,'method':method,'checkpoint_sha256':digest(ck),'offset':offsets[method],
          'export_sha256':sha,'transition_bytes':nbytes,'validation':metric,'delta_nll_vs_fp32':metric['nll']-original['nll']})
        with torch.no_grad():block.frequency.copy_(phi)
      assert rows[-3]['validation']==rows[-2]['validation'], 'phase/circular identity failed'
      atomic_json(out/'results.json',rows);del model
    complete=len(rows)==18;groups={m:[r for r in rows if r['method']==m] for m in methods}
    text='# Frozen language-model transition quantization screen\n\nOnly the first recurrent transition angle changes. Metrics use 16,384 fixed held-out WikiText-103 validation targets; the test split is untouched.\n\n| Method | Seeds | Validation NLL | Perplexity | ΔNLL vs FP32 |\n|---|---:|---:|---:|---:|\n'
    for m,g in groups.items():
      if not g:continue
      n=[x['validation']['nll'] for x in g];p=[x['validation']['perplexity'] for x in g];d=[x.get('delta_nll_vs_fp32',0) for x in g]
      text+=f'| {m} | {len(g)} | {statistics.mean(n):.6f} | {statistics.mean(p):.3f} | {statistics.mean(d):+.6f} |\n'
    gate={'completed_jobs':len(rows),'planned_jobs':18,'complete':complete,'seconds':time.monotonic()-started,
      'phase_equals_circular':complete and all(a['validation']==b['validation'] for a,b in zip(groups['phase_response_offset'],groups['circular_response_offset'])),
      'promote':complete and statistics.mean(x['delta_nll_vs_fp32'] for x in groups['phase_response_offset'])<=0.01,
      'action':'No long run launched; this gate only decides whether to build a stronger small task-level QAT experiment.'}
    atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n');(out/('COMPLETE' if complete else 'STOPPED_BUDGET')).write_text(json.dumps(gate)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
