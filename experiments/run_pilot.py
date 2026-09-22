"""Matched weight-only QAT pilot; classical fake quantization, no QPU."""
import argparse, copy, csv, hashlib, json, math, platform, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

METHODS = ['fp32', 'uniform2', 'sawb2', 'bloch2', 'bloch_nearest2', 'binary1']

def quantize(w, method):
    if method == 'fp32': return w
    if method == 'binary1':
        a = w.detach().abs().mean().clamp_min(1e-8)
        q = torch.where(w >= 0, a, -a)
    else:
        # Same clipping scale isolates level placement for uniform2 vs bloch2.
        a = (2.587*w.detach().square().mean().sqrt()-1.693*w.detach().abs().mean()).clamp_min(1e-8)
        x = (w/a).clamp(-1, 1)
        if method == 'bloch2':
            # Uniform theta grid; Z expectation gives [-1,-.5,.5,1].
            theta = torch.acos(x)
            q = a * torch.cos(torch.round(theta/(math.pi/3))*(math.pi/3))
        elif method == 'bloch_nearest2':
            levels = w.new_tensor([-1., -.5, .5, 1.])
            q = a * levels[(x.unsqueeze(-1)-levels).abs().argmin(-1)]
        else:
            if method == 'uniform2': a = 2.5*w.detach().square().mean().sqrt().clamp_min(1e-8)
            x = (w/a).clamp(-1, 1)
            q = a * (torch.round((x+1)*1.5)/1.5-1)
    return w + (q-w).detach()  # identity STE; detached statistics

class QConv(nn.Conv2d):
    method = 'fp32'
    def forward(self,x): return F.conv2d(x,quantize(self.weight,self.method),self.bias,self.stride,self.padding)
class QLinear(nn.Linear):
    method = 'fp32'
    def forward(self,x): return F.linear(x,quantize(self.weight,self.method),self.bias)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(QConv(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),QConv(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),QLinear(64*7*7,128),nn.ReLU(),QLinear(128,10))
    def forward(self,x): return self.layers(x)
    def set_method(self,m):
        for layer in self.modules():
            if isinstance(layer,(QConv,QLinear)): layer.method=m

def evaluate(model, loader, device):
    model.eval(); correct=total=0; loss=0.
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x)
            correct+=(z.argmax(1)==y).sum().item(); total+=len(y)
            loss+=F.cross_entropy(z,y,reduction='sum').item()
    return correct/total,loss/total

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['mnist','fashion'],default='mnist'); p.add_argument('--epochs',type=int,default=3); p.add_argument('--seeds',type=int,nargs='+',default=[0]); p.add_argument('--data',default='data'); p.add_argument('--out',default='results/pilot'); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'results.csv').exists(): raise RuntimeError('Use a new output directory to preserve existing results')
    torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    ds=datasets.MNIST if a.dataset=='mnist' else datasets.FashionMNIST
    t=transforms.ToTensor(); train_all=ds(a.data,train=True,download=True,transform=t); test=ds(a.data,train=False,download=True,transform=t)
    split=torch.randperm(len(train_all),generator=torch.Generator().manual_seed(2026)); train_idx,val_idx=split[:55000],split[55000:]
    manifest={'args':vars(a),'torch':torch.__version__,'python':platform.python_version(),'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'split_seed':2026,'train':55000,'validation':5000,'test':10000,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fake_quantization':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); torch.save({'train':train_idx,'validation':val_idx},out/'split.pt')
    val_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(train_all,val_idx),batch_size=512)
    test_loader=torch.utils.data.DataLoader(test,batch_size=512)
    rows=[]
    for seed in a.seeds:
        torch.manual_seed(seed); random.seed(seed); initial=copy.deepcopy(Net().state_dict())
        for method in METHODS:
            torch.manual_seed(seed); random.seed(seed)
            model=Net().to(device); model.load_state_dict(initial); model.set_method(method)
            loader=torch.utils.data.DataLoader(torch.utils.data.Subset(train_all,train_idx),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
            opt=torch.optim.Adam(model.parameters(),lr=1e-3); best=-1.; start=time.time()
            if device=='cuda': torch.cuda.reset_peak_memory_stats()
            for epoch in range(a.epochs):
                model.train()
                for x,y in loader:
                    x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(x),y); loss.backward(); opt.step()
                acc,vloss=evaluate(model,val_loader,device)
                print(json.dumps({'method':method,'seed':seed,'epoch':epoch+1,'val_accuracy':acc,'val_loss':vloss}),flush=True)
                if acc>best:
                    best=acc; best_epoch=epoch+1; torch.save(model.state_dict(),out/f'{method}_seed{seed}.pt')
            model.load_state_dict(torch.load(out/f'{method}_seed{seed}.pt',map_location=device,weights_only=True))
            acc,loss=evaluate(model,test_loader,device)
            qweights=[m.weight for m in model.modules() if isinstance(m,(QConv,QLinear))]; n=sum(w.numel() for w in qweights); total=sum(v.numel() for v in model.parameters()); bits=32 if method=='fp32' else 1 if method=='binary1' else 2
            theoretical=(bits*n+32*(total-n)+(32*len(qweights) if bits<32 else 0))/total
            mse=sum((w-quantize(w,method)).square().sum().item() for w in qweights)/n
            row={'dataset':a.dataset,'method':method,'seed':seed,'epochs':a.epochs,'best_epoch':best_epoch,'val_accuracy':best,'test_accuracy':acc,'test_loss':loss,'weight_mse':mse,'parameters':total,'theoretical_bits_per_parameter':theoretical,'checkpoint_bytes':(out/f'{method}_seed{seed}.pt').stat().st_size,'elapsed_s':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
            rows.append(row)
            with (out/'results.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            print('RESULT '+json.dumps(row),flush=True)
if __name__=='__main__': main()
