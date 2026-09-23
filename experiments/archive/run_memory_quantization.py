"""Controlled operator-compression study. No pretrained LLM or QPU.
All compressed methods use the selected common index bits/mode, FP32 radii and one FP32 grid parameter.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import time
from pathlib import Path
import numpy as np
import torch

INDEX_BITS = 4

METHODS = ('fp32', 'cartesian_nearest', 'cartesian_weight', 'cartesian_memory',
           'cartesian_normalized_memory', 'phase_nearest', 'phase_weight', 'phase_memory')

def make_system(seed, modes, inputs, outputs, device):
    g=torch.Generator(device='cpu').manual_seed(seed)
    rho=torch.exp(-torch.exp(torch.empty(modes).uniform_(math.log(1e-4),math.log(.03),generator=g))).double().to(device)
    phi=torch.empty(modes).uniform_(-math.pi,math.pi,generator=g).double().to(device)
    def normal(shape):
        return torch.complex(torch.randn(shape,generator=g),torch.randn(shape,generator=g)).to(device=device,dtype=torch.complex128)
    B=normal((modes,inputs))/math.sqrt(inputs)
    C=normal((outputs,modes))/math.sqrt(2*modes)
    return rho,phi,B,C

def grid(method, parameter):
    if method.startswith('phase'):
        theta=torch.arange(2**INDEX_BITS,device=parameter.device,dtype=parameter.dtype)*(2*math.pi/(2**INDEX_BITS))+parameter
        return torch.polar(torch.ones_like(theta),theta)
    levels=torch.linspace(-1,1,2**(INDEX_BITS//2),device=parameter.device,dtype=parameter.dtype)
    a,b=torch.meshgrid(levels,levels,indexing='ij')
    z=torch.complex(a.flatten(),b.flatten())
    if 'normalized' in method:
        # Radius is exactly restored; same single parameter rotates the fixed directions.
        return z/z.abs()*torch.polar(parameter.new_ones(()),parameter)
    return z*parameter.exp()

def contributions(z,D,horizon):
    t=torch.arange(horizon,device=z.device,dtype=z.real.dtype)
    powers=z.unsqueeze(0)**t.unsqueeze(1)
    return (powers[:,:,None,None]*D.unsqueeze(0)).real

def kernel(z,D,horizon):
    return contributions(z,D,horizon).sum(1)

def relative_mse(a,b):
    return ((a-b).square().mean()/b.square().mean().clamp_min(1e-30)).item()

def rollout(z,B,C,x):
    h=torch.zeros((x.shape[0],len(z)),device=z.device,dtype=z.dtype)
    out=[]
    for t in range(x.shape[1]):
        h=h*z+x[:,t].to(B.dtype)@B.T
        out.append((h@C.T).real)
    return torch.stack(out,1)

def objective(z,reference,D,target,method):
    if method.endswith('weight'):
        return (z-reference).abs().square().mean()/reference.abs().square().mean()
    predicted=kernel(z,D,len(target))
    return (predicted-target).square().mean()/target.square().mean().clamp_min(1e-30)

def coordinate_sweep(indices,rho,parameter,method,reference,D,target):
    """Exact 16-way coordinate search at a fixed grid parameter."""
    choices=rho[:,None]*grid(method,parameter).detach()[None,:]
    if method.endswith('weight'):
        return (choices-reference[:,None]).abs().square().argmin(1)
    z=choices.gather(1,indices[:,None]).flatten()
    current=contributions(z,D,len(target)); residual=current.sum(1)-target
    for j in range(len(indices)):
        t=torch.arange(len(target),device=rho.device,dtype=rho.dtype)
        candidates=((choices[j][None,:]**t[:,None])[:,:,None,None]*D[j][None,None,:,:]).real
        base=residual-current[:,j]
        costs=(base[:,None]+candidates).square().mean((0,2,3))
        choice=costs.argmin(); indices[j]=choice
        residual=base+candidates[:,choice]
        current[:,j]=candidates[:,choice]
    return indices

def fit(method,rho,phi,B,C,horizon,steps,sweeps):
    reference=torch.polar(rho,phi)
    if method=='fp32': return reference, None, None, []
    D=C.T[:,:,None]*B[:,None,:]
    target=kernel(reference,D,horizon).detach()
    raw_cart=method.startswith('cartesian') and 'normalized' not in method
    start_parameter=-math.log(2)/2 if raw_cart and not method.endswith('nearest') else 0.
    parameter=torch.tensor(start_parameter,device=rho.device,dtype=rho.dtype,requires_grad=True)
    initial=grid(method,parameter).detach()
    indices=(rho[:,None]*initial[None,:]-reference[:,None]).abs().square().argmin(1)
    if method.endswith('nearest'):
        return rho*initial[indices],indices,parameter.detach(),[]
    opt=torch.optim.Adam([parameter],lr=.003)
    best=float('inf'); best_indices=indices.clone(); best_parameter=parameter.detach().clone(); history=[]
    per_sweep=max(1,steps//sweeps)
    for step in range(steps):
        if step%per_sweep==0:
            with torch.no_grad():
                indices=coordinate_sweep(indices,rho,parameter,method,reference,D,target)
        opt.zero_grad(set_to_none=True)
        z=rho*grid(method,parameter)[indices]
        loss=objective(z,reference,D,target,method)
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite training objective: {method}')
        value=loss.item()
        if value<best:
            best=value; best_indices=indices.clone(); best_parameter=parameter.detach().clone()
        loss.backward(); opt.step()
        if not method.startswith('phase') and 'normalized' not in method:
            with torch.no_grad(): parameter.clamp_(-4,-math.log(2)/2)
        if step%per_sweep==0 or step==steps-1:
            history.append({'step':step,'objective':value,'parameter':parameter.detach().item()})
    with torch.no_grad():
        candidate=coordinate_sweep(best_indices.clone(),rho,best_parameter,method,reference,D,target)
        z=rho*grid(method,best_parameter)[candidate]
        if objective(z,reference,D,target,method).item()<best: best_indices=candidate
    return rho*grid(method,best_parameter)[best_indices],best_indices,best_parameter,history

def export_operator(path,method,z,rho,indices,parameter):
    # Header 12 bytes: magic, format version, decoder, reserved, number of modes.
    decoder=0 if method=='fp32' else 1 if method.startswith('phase') else 3 if 'normalized' in method else 2
    data=struct.pack('<4sBBHI',b'MQOP',1,decoder,INDEX_BITS,len(rho))
    if decoder==0:
        pairs=torch.view_as_real(z).detach().cpu().numpy().astype('<f4')
        data+=pairs.tobytes()
    else:
        data+=rho.detach().cpu().numpy().astype('<f4').tobytes()
        data+=struct.pack('<f',parameter.item())
        idx=indices.cpu().numpy().astype(np.uint8)
        if INDEX_BITS==4:
            if len(idx)%2: idx=np.append(idx,np.uint8(0))
            data+=(idx[::2] | (idx[1::2]<<4)).tobytes()
        else: data+=idx.tobytes()
    path.write_bytes(data)
    return load_operator(path,z.device)

def load_operator(path,device):
    data=path.read_bytes(); magic,version,decoder,bits,n=struct.unpack('<4sBBHI',data[:12])
    assert magic==b'MQOP' and version==1
    if bits==0: bits=4
    assert bits==INDEX_BITS
    if decoder==0:
        a=np.frombuffer(data,dtype='<f4',offset=12,count=2*n).copy().reshape(n,2)
        a=torch.tensor(a,device=device,dtype=torch.float64)
        return torch.complex(a[:,0],a[:,1])
    rho=torch.tensor(np.frombuffer(data,dtype='<f4',offset=12,count=n).copy(),device=device,dtype=torch.float64)
    parameter=torch.tensor(struct.unpack('<f',data[12+4*n:16+4*n])[0],device=device,dtype=torch.float64)
    packed=np.frombuffer(data,dtype=np.uint8,offset=16+4*n)
    idx=(np.stack([packed&15,packed>>4],axis=1).flatten()[:n] if bits==4 else packed[:n]).copy()
    method={1:'phase_memory',2:'cartesian_memory',3:'cartesian_normalized_memory'}[decoder]
    return rho*grid(method,parameter)[torch.tensor(idx,device=device,dtype=torch.long)]

def verify():
    import tempfile
    device='cpu'; rho,phi,B,C=make_system(7,7,2,3,device)
    z=torch.polar(rho,phi); D=C.T[:,:,None]*B[:,None,:]
    x=torch.zeros(1,12,2,dtype=torch.float64); x[0,0,1]=1
    assert torch.allclose(rollout(z,B,C,x)[0],kernel(z,D,12)[:,:,1],atol=1e-10)
    T=13; delta=.017
    assert torch.allclose((z**T-torch.polar(rho,phi+delta)**T).abs(),2*rho**T*abs(math.sin(T*delta/2)),atol=1e-10)
    with tempfile.TemporaryDirectory() as tmp:
        for method in METHODS:
            zz,idx,p,_=fit(method,rho,phi,B,C,12,6,2)
            path=Path(tmp)/f'{method}.bin'; decoded=export_operator(path,method,zz,rho,idx,p)
            assert torch.allclose(zz,decoded,atol=2e-7,rtol=2e-7)
            if method!='fp32': assert path.stat().st_size==16+4*7+math.ceil(7*INDEX_BITS/8)
            if method.startswith('phase') or 'normalized' in method:
                assert torch.allclose(zz.abs(),rho,atol=1e-12)
                old=(rho*grid(method,torch.zeros((),dtype=rho.dtype))[((rho[:,None]*grid(method,torch.zeros((),dtype=rho.dtype))[None,:]-z[:,None]).abs().square().argmin(1))])
                if method.endswith('memory'):
                    assert objective(zz,z,D,kernel(z,D,12),method)<=objective(old,z,D,kernel(z,D,12),method)+1e-8
    print('PASS: impulse/recurrence equivalence, phase-drift identity, all controls, odd-mode nibble packing, decoder round trip, radius preservation, calibration improvement',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); p.add_argument('--index-bits',type=int,choices=[4,8],default=4); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--modes',type=int,default=64); p.add_argument('--inputs',type=int,default=4); p.add_argument('--outputs',type=int,default=4); p.add_argument('--calibration-horizon',type=int,default=128); p.add_argument('--horizons',type=int,nargs='+',default=[64,128,256,512]); p.add_argument('--steps',type=int,default=200); p.add_argument('--sweeps',type=int,default=5); p.add_argument('--out',default='results/memory_quantization'); a=p.parse_args()
    global INDEX_BITS
    INDEX_BITS=a.index_bits
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if a.verify: verify(); return
    if min(a.modes,a.inputs,a.outputs,a.calibration_horizon,a.steps,a.sweeps,*a.horizons)<1: p.error('all dimensions/budgets must be positive')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'args':vars(a),'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'methods':METHODS,'scope':'sampled linear recurrent systems; trained quantizers; no trained task model, LLM, attention or QPU','precision':'float64/complex128 optimization; decoded FP32 parameters for evaluation','selection':'minimum own calibration objective; no test selection','index_budget_bits_per_mode':INDEX_BITS,'raw_cartesian_fit_constraint':'start and remain at scale <= 1/sqrt(2); nearest control is unconstrained'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); rows=[]
    for seed in a.seeds:
        rho,phi,B,C=make_system(seed,a.modes,a.inputs,a.outputs,device)
        # Deployed projections are FP32, common to every decoder and reference.
        B=B.to(torch.complex64).to(torch.complex128); C=C.to(torch.complex64).to(torch.complex128)
        projection_bytes=torch.view_as_real(B).cpu().numpy().astype('<f4').tobytes()+torch.view_as_real(C).cpu().numpy().astype('<f4').tobytes()
        (out/f'shared_projections_seed{seed}.bin').write_bytes(projection_bytes)
        reference=torch.polar(rho,phi).to(torch.complex64).to(torch.complex128)
        # All modes' reference radii/phases derive from the same deployed reference.
        rho,phi=reference.abs(),reference.angle()
        D=C.T[:,:,None]*B[:,None,:]
        torch.save({'rho':rho.cpu(),'phi':phi.cpu(),'B':B.cpu(),'C':C.cpu()},out/f'system_seed{seed}.pt')
        for method in METHODS:
            start=time.time()
            if device=='cuda': torch.cuda.reset_peak_memory_stats()
            z,idx,parameter,history=fit(method,rho,phi,B,C,a.calibration_horizon,a.steps,a.sweeps)
            path=out/f'{method}_seed{seed}.bin'; z=export_operator(path,method,z,rho,idx,parameter)
            (out/f'{method}_seed{seed}_training.json').write_text(json.dumps(history,indent=2))
            elapsed=time.time()-start
            for horizon in a.horizons:
                g=torch.Generator(device='cpu').manual_seed(100000+seed*10000+horizon)
                x=torch.randn((16,horizon,a.inputs),generator=g,dtype=torch.float64).to(device)
                with torch.no_grad():
                    expected=rollout(reference,B,C,x); actual=rollout(z,B,C,x)
                    kref=kernel(reference,D,horizon); kactual=kernel(z,D,horizon)
                    row={'seed':seed,'method':method,'index_bits':INDEX_BITS,'horizon':horizon,'calibration_horizon':a.calibration_horizon,'weight_relative_mse':((z-reference).abs().square().mean()/reference.abs().square().mean()).item(),'kernel_relative_mse':relative_mse(kactual,kref),'rollout_relative_mse':relative_mse(actual,expected),'tail_rollout_relative_mse':relative_mse(actual[:,horizon//2:],expected[:,horizon//2:]),'max_radius':z.abs().max().item(),'unstable_modes':int((z.abs()>1).sum().item()),'operator_file_bytes':path.stat().st_size,'shared_projection_bytes':len(projection_bytes),'total_deployed_bytes':path.stat().st_size+len(projection_bytes),'fit_elapsed_s':elapsed,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
                if not all(math.isfinite(row[k]) for k in ['weight_relative_mse','kernel_relative_mse','rollout_relative_mse']): raise RuntimeError('nonfinite evaluation')
                rows.append(row)
            with (out/'results.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            print(json.dumps({'seed':seed,'method':method,'fit_s':elapsed,'last_horizon_result':rows[-1]}),flush=True)
    (out/'COMPLETE').write_text(f'{len(rows)} rows\n')
if __name__=='__main__': main()
