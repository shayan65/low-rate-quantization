"""Classical polar weights q*exp(i*phi), with per-layer learned q.
Complex tensors are represented by two real tensors; no physical qubits.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from run_pilot import evaluate

METHODS = ('complex_continuous', 'complex_phase4', 'complex_phase8', 'real_phase4')

def phase_quantize(phi, levels):
    if levels is None:
        return phi
    step = 2 * math.pi / levels
    rounded = torch.round(phi / step) * step
    return phi + (rounded - phi).detach()

class PolarLayer(nn.Module):
    def __init__(self, shape, convolution):
        super().__init__()
        self.phi = nn.Parameter(torch.empty(shape).uniform_(-math.pi, math.pi))
        fan_in = math.prod(shape[1:])
        q0 = 1 / math.sqrt(fan_in)
        self.log_q = nn.Parameter(torch.tensor(math.log(q0)))
        self.bias_r = nn.Parameter(torch.zeros(shape[0]))
        self.bias_i = nn.Parameter(torch.zeros(shape[0]))
        self.convolution = convolution
        self.levels = None
        self.real_only = False

    def weight_parts(self):
        theta = phase_quantize(self.phi, self.levels)
        q = self.log_q.exp()
        return q * theta.cos(), q * theta.sin()

    def forward(self, x):
        xr, xi = x
        wr, wi = self.weight_parts()
        def op(a, w):
            return F.conv2d(a, w, padding=1) if self.convolution else F.linear(a, w)
        bias_shape = (1, -1, 1, 1) if self.convolution else (1, -1)
        if self.real_only:
            return op(xr, wr) + self.bias_r.view(*bias_shape), None
        yr = op(xr, wr) - op(xi, wi) + self.bias_r.view(*bias_shape)
        yi = op(xr, wi) + op(xi, wr) + self.bias_i.view(*bias_shape)
        return yr, yi

def activate_pool(x, pool=False):
    xr, xi = x
    xr = F.relu(xr)
    if xi is None:
        return (F.max_pool2d(xr, 2) if pool else xr), None
    xi = F.relu(xi)
    if pool:
        # Select the same spatial element for both components, preserving phase.
        _, indices = F.max_pool2d(xr.square() + xi.square(), 2, return_indices=True)
        xr = xr.flatten(2).gather(2, indices.flatten(2)).reshape_as(indices)
        xi = xi.flatten(2).gather(2, indices.flatten(2)).reshape_as(indices)
    return xr, xi

class PolarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([PolarLayer((32,1,3,3),True), PolarLayer((64,32,3,3),True), PolarLayer((128,64*7*7),False), PolarLayer((10,128),False)])
        self.real_only = False
    def set_method(self, method):
        self.real_only = method == 'real_phase4'
        for layer in self.layers:
            layer.levels = None if method == 'complex_continuous' else 8 if method == 'complex_phase8' else 4
            layer.real_only = self.real_only
            layer.bias_i.requires_grad_(not self.real_only)
    def forward(self, x):
        z = x, None if self.real_only else torch.zeros_like(x)
        z = activate_pool(self.layers[0](z), True)
        z = activate_pool(self.layers[1](z), True)
        z = z[0].flatten(1), None if z[1] is None else z[1].flatten(1)
        z = activate_pool(self.layers[2](z))
        # Fixed real-part readout; imaginary contributions persist through hidden layers.
        return self.layers[3](z)[0]

def verify():
    torch.manual_seed(123)
    for levels in (4, 8):
        phi = torch.linspace(-10,10,1001,requires_grad=True)
        theta = phase_quantize(phi, levels)
        indices = torch.round(theta.detach()/(2*math.pi/levels)).long() % levels
        assert indices.unique().numel() == levels
        assert torch.equal(torch.autograd.grad(theta.sum(),phi)[0],torch.ones_like(phi))
    layer = PolarLayer((3,4),False).double()
    xr,xi = torch.randn(2,4,dtype=torch.double),torch.randn(2,4,dtype=torch.double)
    yr,yi = layer((xr,xi)); wr,wi = layer.weight_parts()
    expected = F.linear(torch.complex(xr,xi),torch.complex(wr,wi),torch.complex(layer.bias_r,layer.bias_i))
    assert torch.allclose(yr,expected.real) and torch.allclose(yi,expected.imag)
    conv = PolarLayer((2,1,3,3),True).double()
    xr,xi = torch.randn(2,1,5,5,dtype=torch.double),torch.randn(2,1,5,5,dtype=torch.double)
    yr,yi=conv((xr,xi)); wr,wi=conv.weight_parts()
    expected=F.conv2d(torch.complex(xr,xi),torch.complex(wr,wi),torch.complex(conv.bias_r,conv.bias_i),padding=1)
    assert torch.allclose(yr,expected.real) and torch.allclose(yi,expected.imag)
    for method in METHODS:
        net=PolarNet(); net.set_method(method)
        logits=net(torch.rand(2,1,28,28)); assert logits.shape==(2,10) and torch.isfinite(logits).all()
        F.cross_entropy(logits,torch.tensor([0,1])).backward()
        assert all(layer.phi.grad is not None and torch.isfinite(layer.phi.grad).all() for layer in net.layers)
        assert all(layer.log_q.grad is not None and torch.isfinite(layer.log_q.grad).all() for layer in net.layers)
    print('PASS: phase-grid cardinalities, STE, native complex linear/conv agreement, logits and trainable phase/magnitude gradients',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); p.add_argument('--epochs',type=int,default=3); p.add_argument('--seeds',type=int,nargs='+',default=[0]); p.add_argument('--dataset',choices=['mnist','fashion'],default='mnist'); p.add_argument('--data',default='data'); p.add_argument('--out',default='results/phase_pilot'); a=p.parse_args()
    torch.set_num_threads(4)
    if a.verify:
        verify(); return
    if a.epochs<1: p.error('epochs must be positive')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ds=datasets.MNIST if a.dataset=='mnist' else datasets.FashionMNIST
    train_all=ds(a.data,train=True,download=True,transform=transforms.ToTensor()); test=ds(a.data,train=False,download=True,transform=transforms.ToTensor())
    split=torch.randperm(60000,generator=torch.Generator().manual_seed(2026)); train_idx,val_idx=split[:55000],split[55000:]
    torch.save({'train':train_idx,'validation':val_idx},out/'split.pt')
    val_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(train_all,val_idx),batch_size=512)
    test_loader=torch.utils.data.DataLoader(test,batch_size=512)
    manifest={'args':vars(a),'torch':torch.__version__,'torchvision':torchvision.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'split_seed':2026,'train':55000,'validation':5000,'test':10000,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shared_magnitude_per_layer':True,'complex_activation':'split ReLU; magnitude-selected maxpool','readout':'real part of final complex linear output','optimizer':'Adam lr=0.001','batch_size':256,'quantum_hardware':False,'packed_storage':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); rows=[]
    for seed in a.seeds:
        torch.manual_seed(seed); initial=copy.deepcopy(PolarNet().state_dict())
        for method in METHODS:
            torch.manual_seed(seed); model=PolarNet().to(device); model.load_state_dict(initial); model.set_method(method)
            loader=torch.utils.data.DataLoader(torch.utils.data.Subset(train_all,train_idx),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
            opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=0.001); best=-1; start=time.time()
            if device=='cuda': torch.cuda.reset_peak_memory_stats()
            for epoch in range(a.epochs):
                model.train()
                for x,y in loader:
                    opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(x.to(device)),y.to(device))
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                    loss.backward(); opt.step()
                acc,vloss=evaluate(model,val_loader,device)
                print(json.dumps({'method':method,'seed':seed,'epoch':epoch+1,'val_accuracy':acc,'val_loss':vloss}),flush=True)
                if acc>best:
                    best=acc; best_epoch=epoch+1; torch.save(model.state_dict(),out/f'{method}_seed{seed}.pt')
            model.load_state_dict(torch.load(out/f'{method}_seed{seed}.pt',map_location=device,weights_only=True)); acc,loss=evaluate(model,test_loader,device)
            n=sum(layer.phi.numel() for layer in model.layers); biases=sum(layer.bias_r.numel()*(1 if model.real_only else 2) for layer in model.layers)
            bits=32 if method=='complex_continuous' else 3 if method=='complex_phase8' else 2
            payload_bits=n*bits+32*(biases+len(model.layers))
            row={'dataset':a.dataset,'method':method,'seed':seed,'epochs':a.epochs,'best_epoch':best_epoch,'val_accuracy':best,'test_accuracy':acc,'test_loss':loss,'weight_connections':n,'trainable_real_scalars':sum(p.numel() for p in model.parameters() if p.requires_grad),'phase_bits_per_weight':bits,'theoretical_payload_bits':payload_bits,'theoretical_bits_per_connection':payload_bits/n,'checkpoint_bytes':(out/f'{method}_seed{seed}.pt').stat().st_size,'elapsed_s':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'magnitudes':json.dumps([layer.log_q.exp().item() for layer in model.layers])}
            rows.append(row)
            with (out/'results.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            print('RESULT '+json.dumps(row),flush=True)
if __name__=='__main__': main()
