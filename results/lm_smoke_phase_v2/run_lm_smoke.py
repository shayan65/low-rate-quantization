"""Byte-level hybrid LM smoke test; floating-point fake phase quantization."""
import argparse, json, math, time, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class PhaseLinear(nn.Module):
    def __init__(self, width, method):
        super().__init__()
        self.phase = nn.Parameter(torch.empty(width, width).uniform_(-math.pi, math.pi))
        self.log_scale = nn.Parameter(torch.tensor(-0.5 * math.log(width)))
        self.method = method
    def angles(self):
        if self.method == 'continuous': return self.phase
        if self.method == 'restricted180':
            base = torch.arange(1,91,device=self.phase.device) * math.pi / 180
            code = torch.cat((base, base + math.pi))
        else:
            n = 180 if self.method == 'full180' else 256
            code = torch.arange(n,device=self.phase.device) * (2*math.pi/n)
        # Circular nearest-neighbor assignment, processed by rows to bound memory.
        with torch.no_grad():
            assigned = torch.cat([code[torch.cos(x[...,None]-code).argmax(-1)] for x in self.phase.split(16)])
        return self.phase + (assigned-self.phase).detach()
    def forward(self, z):
        p=self.angles(); s=self.log_scale.exp()
        wr=s*p.cos(); wi=s*p.sin()
        r,i=z
        return F.linear(r,wr)-F.linear(i,wi), F.linear(r,wi)+F.linear(i,wr)

class Block(nn.Module):
    def __init__(self,d,method, recurrent):
        super().__init__(); self.recurrent=recurrent
        self.norm=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,4,batch_first=True)
        self.proj=PhaseLinear(d,method)
        self.decay=nn.Parameter(torch.full((d,),2.0)); self.frequency=nn.Parameter(torch.zeros(d))
    def forward(self,x):
        z=self.norm(x)
        if self.recurrent:
            r,i=self.proj((z,torch.zeros_like(z)))
            rho=self.decay.sigmoid(); ar=rho*self.frequency.cos(); ai=rho*self.frequency.sin()
            hr=torch.zeros_like(r[:,0]); hi=torch.zeros_like(hr); ys=[]
            for t in range(x.shape[1]):
                nr=ar*hr-ai*hi+(1-rho)*r[:,t]
                hi=ai*hr+ar*hi+(1-rho)*i[:,t]; hr=nr
                ys.append(hr)
            x=x+torch.stack(ys,1)
        else:
            mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
            x=x+self.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
            r,i=self.proj((self.norm(x),torch.zeros_like(x)))
            # Imaginary component contributes explicitly to the nonlinear output.
            x=x+(F.gelu(r)+F.gelu(i))/math.sqrt(2)
        return x

class Model(nn.Module):
    def __init__(self,d,depth,method,context):
        super().__init__(); self.emb=nn.Embedding(256,d); self.pos=nn.Embedding(context,d)
        self.blocks=nn.ModuleList([Block(d,method,j%2==0) for j in range(depth)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,256,bias=False); self.head.weight=self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
    def forward(self,x):
        z=self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device))
        for b in self.blocks: z=b(z)
        return self.head(self.norm(z))

def data(root):
    import pyarrow.parquet as pq
    root.mkdir(parents=True,exist_ok=True); arrays={}
    for split in ['train','validation','test']:
        path=root/f'{split}.parquet'
        if not path.exists():
            url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
            urllib.request.urlretrieve(url,path)
        text='\n'.join(pq.read_table(path,columns=['text'])['text'].to_pylist())
        arrays[split]=torch.from_numpy(np.frombuffer(text.encode('utf-8'),dtype=np.uint8).copy()).long()
    return arrays

def batch(tokens,batch_size,context,device,generator):
    starts=torch.randint(len(tokens)-context-1,(batch_size,),generator=generator)
    ix=starts[:,None]+torch.arange(context+1)
    seq=tokens[ix].to(device); return seq[:,:-1],seq[:,1:]

@torch.no_grad()
def evaluate(model,tokens,context,device):
    model.eval(); total=0.; count=0
    # Fixed first 16 disjoint windows: a smoke subset, not full test evaluation.
    for start in range(0,min(len(tokens)-context-1,16*context),context):
        seq=tokens[start:start+context+1].to(device)
        loss=F.cross_entropy(model(seq[:-1][None]).reshape(-1,256),seq[1:],reduction='sum')
        total+=loss.item(); count+=context
    return total/count

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--width',type=int,default=128); ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--context',type=int,default=128); ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--data',default='data/wikitext2'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'; tokens=data(Path(args.data))
    manifest=vars(args)|{'device':device,'tokenizer':'UTF-8 bytes (vocabulary 256)','dataset':'Salesforce/wikitext wikitext-2-raw-v1','dataset_revision':'main; downloaded parquet files retained','seed':0,'evaluation':'first 16 disjoint windows per held-out split','quantized_layers':'PhaseLinear only; remaining parameters FP32','note':'Smoke test, not standard tokenizer perplexity or packed low-bit training'}
    import hashlib
    manifest['data_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(args.data).glob('*.parquet')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); rows=[]
    for method in ['continuous','restricted180','full180','full256']:
        torch.manual_seed(0); model=Model(args.width,args.depth,method,args.context).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4); gen=torch.Generator().manual_seed(2026)
        torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
        started=time.time(); history=[]
        for step in range(args.steps):
            model.train(); x,y=batch(tokens['train'],args.batch,args.context,device,gen)
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            if (step+1)%20==0:
                record={'step':step+1,'train_nll':loss.item(),'validation_nll':evaluate(model,tokens['validation'],args.context,device)}
                history.append(record); print(method,json.dumps(record),flush=True)
        val=evaluate(model,tokens['validation'],args.context,device)
        row={'method':method,'parameters':sum(p.numel() for p in model.parameters()),'validation_nll':val,'validation_bits_per_byte':val/math.log(2),'seconds':time.time()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
        # Test is intentionally deferred until a substantive validation-selected study.
        rows.append(row); torch.save(model.state_dict(),out/f'{method}.pt')
        (out/f'{method}_history.json').write_text(json.dumps(history,indent=2))
        (out/'results.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(row),flush=True)
        del model,optimizer
    (out/'COMPLETE').write_text('Completed four-method smoke test; no test-set evaluation.\n')
if __name__=='__main__': main()
