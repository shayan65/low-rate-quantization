"""Small, frozen trained recurrence: phase vs Cartesian at equal index bytes.

Real WikiText token embeddings drive the isolated layer. Calibration uses train
text; evaluation uses distinct validation text and lengths. No synthetic metrics.
"""
import argparse,json,math,time,statistics,struct
from pathlib import Path
import numpy as np
import torch
from run_polar_lm import LanguageModel,pack_indices,unpack_indices,digest,atomic_json


def phase_code(phi,offset):
    step=2*math.pi/16
    return (phi-offset).div(step).round().remainder(16)*step+offset

def cart_code(phi,offset):
    levels=torch.tensor([-1.,-1/3,1/3,1.],device=phi.device)
    poles=(levels[:,None]+1j*levels[None,:]).reshape(-1)/math.sqrt(2)
    rotated=poles*torch.polar(torch.ones((),device=phi.device),torch.as_tensor(offset,device=phi.device))
    target=torch.polar(torch.ones_like(phi),phi)
    ix=(target[...,None]-rotated).abs().argmin(-1)
    return rotated[ix]

@torch.no_grad()
def response(model,indices,phase=None,length=128):
    block=model.blocks[0];device=next(model.parameters()).device
    tokens=torch.as_tensor(indices[:length],device=device,dtype=torch.long)
    # A long isolated recurrence reuses the learned 128 positional table cyclically.
    z=model.emb(tokens)[None]+model.position(torch.arange(length,device=device)%model.position.num_embeddings)[None]
    r,i=block.input(block.norm1(z))
    if phase is None: phase=block.frequency
    decay=block.decay
    rho=decay.sigmoid();t=torch.arange(length,device=device,dtype=torch.float32)
    pole=phase if phase.is_complex() else torch.polar(torch.ones_like(rho),phase.float())
    kernel=(rho[:,None]*pole[:,None]).pow(t[None,:])*(1-rho[:,None])
    driven=torch.complex(r.float(),i.float()).transpose(1,2)
    state=torch.fft.ifft(torch.fft.fft(driven,n=2*length)*torch.fft.fft(kernel,n=2*length)[None],n=2*length)[...,:length].transpose(1,2)
    out,_=block.output(state.real,state.imag)
    return out[0].float()


def relative(reference,candidate):
    return (candidate-reference).square().sum().div(reference.square().sum().clamp_min(1e-12)).item()

def sample_windows(tokens,length,count,seed):
    rng=np.random.default_rng(seed);ends=len(tokens)-length-1
    return [np.asarray(tokens[s:s+length],dtype=np.int64).copy() for s in rng.choice(ends,count,replace=False)]

def exported_phase(method,phi,offset,path,rho):
    levels=torch.tensor([-1.,-1/3,1/3,1.],device=phi.device)
    poles=(levels[:,None]+1j*levels[None,:]).reshape(-1)/math.sqrt(2)
    if method.startswith('phase_'):
        indices=((phi-offset)/(2*math.pi/16)).round().remainder(16).to(torch.uint8)
        decode=lambda ix: offset+ix*(2*math.pi/16)
    else:
        rotated=poles*torch.polar(torch.ones((),device=phi.device),torch.as_tensor(offset,device=phi.device))
        indices=(torch.polar(torch.ones_like(phi),phi)[...,None]-rotated).abs().argmin(-1).to(torch.uint8)
        decode=lambda ix: rotated[ix.long()]
    packed=pack_indices(indices.cpu().numpy(),4)
    payload=struct.pack('<f',offset)+rho.cpu().numpy().astype('<f4').tobytes()+packed
    path.write_bytes(payload)
    assert len(payload)==4+4*len(phi)+math.ceil(len(phi)*4/8)
    restored=unpack_indices(payload[4+4*len(phi):],4,len(phi))
    assert np.array_equal(indices.cpu().numpy(),restored)
    return decode(torch.as_tensor(restored,device=phi.device)),len(payload)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/wikitext103_bpe8k')
    ap.add_argument('--checkpoints',default='results/coordinate_diagnostic_v1')
    ap.add_argument('--out',default='results/trained_transition_v1');ap.add_argument('--seconds',type=float,default=120.)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    root=Path(args.data);manifest=json.loads((root/'manifest.json').read_text())
    train=np.memmap(root/'train.bin',dtype='<u2',mode='r');validation=np.memmap(root/'validation.bin',dtype='<u2',mode='r')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    methods=['phase_nearest','phase_weight_offset','phase_memory_offset','cartesian_nearest','cartesian_memory_offset']
    plan={'dataset_revision':manifest['revision'],'dataset_manifest_sha256':digest(root/'manifest.json'),
          'checkpoint_root':args.checkpoints,'checkpoints':[f'complex_fp32_seed{s}/best.pt' for s in [0,1,2]],
          'seeds':[0,1,2],'methods':methods,'budget':'4 index bits per trained complex recurrent mode; FP32 learned decay unchanged, one FP32 shared offset for fitted controls; zero-offset nearest controls have same 4-byte offset field set to zero. Cartesian uses all 16 distinct complex grid points, scaled by 1/sqrt(2), with its pole radius varying by code.',
          'train_windows':'4 windows of 128 original training BPE tokens per seed, sampled with fixed seed 700+model seed',
          'validation_windows':'8 windows each at 128 and 512 from original held-out validation BPE tokens, distinct fixed seed 900+model seed',
          'metric':'relative isolated-layer output MSE against frozen trained FP32 transition; lower better; real text embeddings; no generated corpus',
          'training':'no new model training: frozen trained Cartesian complex_fp32 checkpoints','long_window_note':'512-position isolated recurrence cycles learned 128-position embedding; extrapolation diagnostic, not language-model long-context accuracy',
          'runtime_cap_seconds':args.seconds,'source_sha256':digest(__file__)}
    atomic_json(out/'plan.json',plan);(out/'run_trained_transition_screen.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for seed in plan['seeds']:
        if time.monotonic()-start>args.seconds:break
        checkpoint=Path(args.checkpoints)/f'complex_fp32_seed{seed}/best.pt'
        if not checkpoint.exists():break
        torch.manual_seed(seed);model=LanguageModel(manifest['actual_vocab'],64,2,128,'complex_fp32','hybrid',4).to(device)
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
        phi=model.blocks[0].frequency.detach().float();rho=model.blocks[0].decay.sigmoid().detach().float()
        train_windows=sample_windows(train,128,4,700+seed)
        val_windows={n:sample_windows(validation,n,8,900+seed+n) for n in [128,512]}
        refs_train=[response(model,x) for x in train_windows]
        refs_val={n:[response(model,x,length=n) for x in windows] for n,windows in val_windows.items()}
        deltas=torch.linspace(-math.pi/16,math.pi/16,33,device=device)
        # Equal candidate search for phase and radius-normalized Cartesian families.
        # Small train windows calibrate phase offset; no validation/test-driven fit.
        weight_scores=[];phase_memory_scores=[];cart_memory_scores=[]
        for delta in deltas:
            if time.monotonic()-start>args.seconds:break
            p=phase_code(phi,delta);c=cart_code(phi,delta)
            weight_scores.append((rho.square()*((p-phi).cos().neg()+1)*2).sum().item())
            phase_memory_scores.append(statistics.mean(relative(a,response(model,x,p)) for x,a in zip(train_windows,refs_train)))
            cart_memory_scores.append(statistics.mean(relative(a,response(model,x,c)) for x,a in zip(train_windows,refs_train)))
        if len(weight_scores)!=len(deltas):break
        choices={'phase_nearest':phase_code(phi,0.),
                 'phase_weight_offset':phase_code(phi,deltas[int(np.argmin(weight_scores))]),
                 'phase_memory_offset':phase_code(phi,deltas[int(np.argmin(phase_memory_scores))]),
                 'cartesian_nearest':cart_code(phi,0.),
                 'cartesian_memory_offset':cart_code(phi,deltas[int(np.argmin(cart_memory_scores))])}
        offsets={'phase_nearest':0.,'phase_weight_offset':float(deltas[int(np.argmin(weight_scores))]),
                 'phase_memory_offset':float(deltas[int(np.argmin(phase_memory_scores))]),
                 'cartesian_nearest':0.,'cartesian_memory_offset':float(deltas[int(np.argmin(cart_memory_scores))])}
        for method,p in choices.items():
            p,export_bytes=exported_phase(method,phi,offsets[method],out/f'{method}_seed{seed}.bin',rho)
            record={'seed':seed,'method':method,'model_checkpoint_sha256':digest(checkpoint),
              'offset':offsets[method],'index_bits_per_complex_mode':4,'modes':len(phi),
              'transition_index_and_radius_bytes':export_bytes,'export_sha256':digest(out/f'{method}_seed{seed}.bin'),
              'validation_relative_mse':{str(n):statistics.mean(relative(a,response(model,x,p,length=n)) for x,a in zip(val_windows[n],refs_val[n])) for n in [128,512]}}
            rows.append(record);atomic_json(out/'results.json',rows);print(json.dumps(record),flush=True)
        del model
    complete=len(rows)==len(plan['seeds'])*len(methods)
    groups={m:[r for r in rows if r['method']==m] for m in methods}
    text='# Frozen trained recurrent-transition screen\n\n'
    text+='Real WikiText-103 text drives a frozen learned recurrence from tiny sample-trained models. Calibration takes train windows; held-out original validation windows are distinct. No synthetic metrics, no test split, and no new model training. The Cartesian control has 16 distinct 2+2-bit complex grid points and quantizes pole radius as well as angle. Both families use the same 292-byte exported mode record; phase retains exact FP32 learned decay. The 512-token probe cycles an existing 128-position embedding and is isolated-layer output MSE, not LM perplexity.\n\n'
    text+='| Equal-byte 4-bit method | Seeds | Relative MSE 128 | Relative MSE 512 |\n|---|---:|---:|---:|\n'
    for m,g in groups.items():
        if not g:continue
        fields=[]
        for n in ['128','512']:
            values=[r['validation_relative_mse'][n] for r in g]
            fields.append(f'{statistics.mean(values):.4g} ± {statistics.stdev(values):.3g}' if len(values)>1 else f'{values[0]:.4g} (one seed)')
        text+=f'| {m} | {len(g)} | {fields[0]} | {fields[1]} |\n'
    gate={'completed_jobs':len(rows),'planned_jobs':15,'all_seeds_complete':complete,
          'seconds':time.monotonic()-start,'phase_memory_beats_cartesian_memory_at_512':complete and all(a['validation_relative_mse']['512']<b['validation_relative_mse']['512'] for a,b in zip(groups['phase_memory_offset'],groups['cartesian_memory_offset'])),
          'action':'No long run launched. A positive diagnostic warrants a trained task-model follow-up, not a paper claim.'}
    atomic_json(out/'gate.json',gate);(out/'summary.md').write_text(text+'\n'+json.dumps(gate)+'\n')
    (out/('COMPLETE' if complete else 'STOPPED_BUDGET')).write_text('Frozen transition screen finished.\n')
    print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
