"""Auditable full-split WikiText language-model quantization experiment.

Training is floating-point QAT. Exported quantized projections are packed and
reloaded for test evaluation. This is a custom model, not Qwen or IBM SAWB.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import struct
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ['real_fp32', 'real2', 'real4', 'real8', 'complex_fp32', 'continuous',
           'restricted180', 'full180', 'full256', 'phase4', 'phase8', 'phase16', 'polar_fp32', 'mag1phase1', 'mag2phase2', 'mag2phase6', 'mag4phase4']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)); temp.replace(path)


def atomic_torch(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp); temp.replace(path)


def prepare(root, subset='wikitext-103-raw-v1', vocab=8192):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['subset'] == subset and manifest['requested_vocab'] == vocab
        for split, meta in manifest['splits'].items():
            path = root/f'{split}.bin'
            assert path.stat().st_size == meta['tokens']*2
            assert digest(path) == meta['token_sha256'], f'Corrupted {split} tokens'
        assert digest(root/'tokenizer.json') == manifest['tokenizer_sha256']
        return manifest
    api = HfApi(); info = api.dataset_info('Salesforce/wikitext'); revision = info.sha
    files = api.list_repo_files('Salesforce/wikitext', repo_type='dataset', revision=revision)
    paths = {}
    for split in ['train','validation','test']:
        names = sorted(f for f in files if f.startswith(f'{subset}/{split}-') and f.endswith('.parquet'))
        assert names, f'Missing {split} parquet files'
        paths[split] = [(name, Path(hf_hub_download('Salesforce/wikitext', name,
            repo_type='dataset', revision=revision))) for name in names]
    def rows(split):
        for name, path in paths[split]:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=['text']):
                yield [text + '\n' for text in batch.column(0).to_pylist()]
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=['<pad>','<unk>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    def train_iterator():
        for batch in rows('train'): yield from batch
    print('Training BPE tokenizer on ALL train rows only', flush=True)
    tok.train_from_iterator(train_iterator(), trainer=trainer)
    tok.save(str(root/'tokenizer.json'))
    manifest = {'dataset':'Salesforce/wikitext', 'subset':subset, 'revision':revision,
        'requested_vocab':vocab, 'actual_vocab':tok.get_vocab_size(),
        'tokenizer':'ByteLevel BPE; trained on train only; each source row followed by newline',
        'tokenizer_sha256':digest(root/'tokenizer.json'), 'pad_id':tok.token_to_id('<pad>'),
        'dtype':'little-endian uint16', 'splits':{}}
    assert tok.get_vocab_size() <= 65536
    for split in paths:
        started=time.time(); n=0; byte_count=0; row_count=0
        temp=root/f'{split}.bin.tmp'; raw_hash=hashlib.sha256()
        with open(temp,'wb') as f:
            for batch in rows(split):
                for text in batch:
                    encoded=text.encode('utf-8'); raw_hash.update(encoded)
                    byte_count += len(encoded); row_count += 1
                encodings=tok.encode_batch(batch)
                values=np.asarray([i for e in encodings for i in e.ids], dtype='<u2')
                values.tofile(f); n+=len(values)
        temp.replace(root/f'{split}.bin')
        source=[{'name':name,'bytes':path.stat().st_size,'sha256':digest(path),
                 'rows':pq.ParquetFile(path).metadata.num_rows} for name,path in paths[split]]
        assert row_count == sum(x['rows'] for x in source)
        manifest['splits'][split]={'rows':row_count, 'utf8_bytes_with_newlines':byte_count,
            'tokens':n, 'target_tokens_per_pass':n-1, 'normalized_text_sha256':raw_hash.hexdigest(),
            'token_sha256':digest(root/f'{split}.bin'), 'source_files':source}
        print(f'Prepared {split}: {row_count:,} rows, {n:,} tokens, {time.time()-started:.1f}s',flush=True)
    atomic_json(manifest_path,manifest)
    return manifest


def circular_indices(phase, method):
    if method == 'restricted180':
        # Closest point on either discrete arc, including circular boundary cases.
        a=torch.atan2(phase.sin(),phase.cos())
        b=torch.atan2((phase-math.pi).sin(),(phase-math.pi).cos())
        ka=(a*(180/math.pi)).round().clamp(1,90)
        kb=(b*(180/math.pi)).round().clamp(1,90)
        pa=ka*(math.pi/180); pb=kb*(math.pi/180)+math.pi
        return torch.where((phase-pa).cos() >= (phase-pb).cos(),ka.long()-1,kb.long()+89)
    n={'full180':180,'full256':256,'phase4':4,'phase8':8,'phase16':16}[method]
    return (phase*(n/(2*math.pi))).round().long().remainder(n)


def code_angles(indices, method):
    if method == 'restricted180':
        return (indices.remainder(90)+1).float()*(math.pi/180)+(indices//90).float()*math.pi
    n={'full180':180,'full256':256,'phase4':4,'phase8':8,'phase16':16}[method]
    return indices.float()*(2*math.pi/n)


class Projection(nn.Module):
    def __init__(self, din, dout, method):
        super().__init__(); self.method=method; self.din=din; self.dout=dout
        self.register_buffer('deployed_real',None,persistent=False)
        self.register_buffer('deployed_imag',None,persistent=False)
        if method.startswith('mag') or method=='polar_fp32':
            if method.startswith('mag'):
                self.mag_bits=int(method[3]); self.phase_bits=int(method[-1])
            wr=torch.randn(dout,din)/math.sqrt(2*din)
            wi=torch.randn(dout,din)/math.sqrt(2*din)
            radius=torch.sqrt(wr.square()+wi.square())
            self.raw_magnitude=nn.Parameter(torch.log(torch.expm1(radius)))
            self.phase=nn.Parameter(torch.atan2(wi,wr))
            if method.startswith('mag'):
                self.log_clip=nn.Parameter(torch.tensor(math.log(3/math.sqrt(din))))
        elif method in ['continuous','restricted180','full180','full256','phase4','phase8','phase16']:
            self.phase=nn.Parameter(torch.empty(dout,din).uniform_(-math.pi,math.pi))
            self.log_scale=nn.Parameter(torch.tensor(-.5*math.log(2*din)))
        else:
            self.weight=nn.Parameter(torch.randn(dout,din)/math.sqrt(din))
            if method=='complex_fp32':
                self.weight.data /= math.sqrt(2)
                self.imag=nn.Parameter(torch.randn(dout,din)/math.sqrt(2*din))
            if method in ['real2','real4','real8']:
                # Shared trained clipping scale; this is a uniform STE baseline, not SAWB/LSQ.
                self.log_clip=nn.Parameter(torch.tensor(math.log(3/math.sqrt(din))))
    def coefficients(self):
        if self.deployed_real is not None: return self.deployed_real,self.deployed_imag
        if self.method=='polar_fp32':
            radius=F.softplus(self.raw_magnitude)
            return radius*self.phase.cos(),radius*self.phase.sin()
        if self.method.startswith('mag'):
            normalized=(F.softplus(self.raw_magnitude)/self.log_clip.exp()).clamp(0,1)
            levels=2**self.mag_bits-1
            with torch.no_grad(): q=(normalized*levels).round()/levels
            radius=self.log_clip.exp()*(q+(normalized-normalized.detach()))
            n=2**self.phase_bits
            with torch.no_grad(): angle=self.phase.mul(n/(2*math.pi)).round().remainder(n)*(2*math.pi/n)
            angle=angle+(self.phase-self.phase.detach())
            return radius*angle.cos(),radius*angle.sin()
        if hasattr(self,'phase'):
            p=self.phase
            if self.method!='continuous':
                with torch.no_grad(): q=code_angles(circular_indices(p,self.method),self.method)
                p=q+(p-p.detach())
            s=self.log_scale.exp()
            return s*p.cos(),s*p.sin()
        w=self.weight
        if self.method in ['real2','real4','real8']:
            bits=int(self.method[4:]); levels=2**bits; c=self.log_clip.exp()
            normalized=(w/c).clamp(-1,1)
            with torch.no_grad(): q=((normalized+1)*((levels-1)/2)).round()*(2/(levels-1))-1
            w=c*(q+(normalized-normalized.detach()))
        return w,self.imag if hasattr(self,'imag') else None
    def forward(self,r,i=None):
        wr,wi=self.coefficients(); real=F.linear(r,wr)
        if wi is None:
            return real, F.linear(i,wr) if i is not None else None
        imag=F.linear(r,wi)
        if i is not None:
            real=real-F.linear(i,wi); imag=imag+F.linear(i,wr)
        return real,imag


def recurrent_fft(r,i,decay,frequency):
    """Exact zero-state diagonal recurrence using causal FFT convolution."""
    with torch.autocast(device_type=r.device.type,enabled=False):
        r=r.float(); rho=decay.float().sigmoid()
        t=torch.arange(r.shape[1],device=r.device,dtype=torch.float32)
        if i is None:
            kernel=rho[:,None].pow(t[None,:])*(1-rho[:,None])
            inputs=r.transpose(1,2)
            out=torch.fft.irfft(torch.fft.rfft(inputs,n=2*r.shape[1])*
                 torch.fft.rfft(kernel,n=2*r.shape[1])[None],n=2*r.shape[1])[...,:r.shape[1]]
            return out.transpose(1,2),None
        kernel=torch.polar(rho[:,None].pow(t[None,:])*(1-rho[:,None]),frequency.float()[:,None]*t[None,:])
        inputs=torch.complex(r,i.float()).transpose(1,2)
        out=torch.fft.ifft(torch.fft.fft(inputs,n=2*r.shape[1])*
              torch.fft.fft(kernel,n=2*r.shape[1])[None],n=2*r.shape[1])[...,:r.shape[1]].transpose(1,2)
        return out.real,out.imag


class Attention(nn.Module):
    def __init__(self,d,heads):
        super().__init__(); self.heads=heads; self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
    def forward(self,x):
        b,l,d=x.shape
        q,k,v=self.qkv(x).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        z=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(z.transpose(1,2).reshape(b,l,d))


class Block(nn.Module):
    def __init__(self,d,method,recurrent,heads):
        super().__init__(); self.recurrent=recurrent
        self.norm1=nn.LayerNorm(d); self.norm2=nn.LayerNorm(d)
        if recurrent:
            self.input=Projection(d,d,method); self.output=Projection(d,d,method)
            self.decay=nn.Parameter(torch.empty(d).uniform_(1,5))
            if not method.startswith('real'): self.frequency=nn.Parameter(torch.empty(d).uniform_(-math.pi,math.pi))
            else: self.register_buffer('frequency',torch.zeros(d))
        else: self.attn=Attention(d,heads)
        self.up=Projection(d,4*d,method); self.down=Projection(4*d,d,method)
        self.residual_scale=.1
    def forward(self,x):
        z=self.norm1(x)
        if self.recurrent:
            r,i=self.input(z); r,i=recurrent_fft(r,i,self.decay,self.frequency)
            r,i=self.output(r,i); x=x+self.residual_scale*r
        else: x=x+self.residual_scale*self.attn(z)
        r,i=self.up(self.norm2(x)); r=F.gelu(r); i=F.gelu(i) if i is not None else None
        r,i=self.down(r,i); return x+self.residual_scale*r


class LanguageModel(nn.Module):
    def __init__(self,vocab=8192,width=384,depth=6,context=512,method='continuous',architecture='hybrid',heads=6):
        super().__init__(); self.emb=nn.Embedding(vocab,width); self.position=nn.Embedding(context,width)
        self.blocks=nn.ModuleList([Block(width,method,architecture=='hybrid' and j%2==0,heads) for j in range(depth)])
        self.norm=nn.LayerNorm(width); self.head=nn.Linear(width,vocab,bias=False); self.head.weight=self.emb.weight
        nn.init.normal_(self.emb.weight,std=.02); nn.init.normal_(self.position.weight,std=.02)
    def forward(self,ids):
        x=self.emb(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        for block in self.blocks: x=block(x)
        return self.head(self.norm(x))


def make_batch(tokens,blocks,context,pad,device):
    x=np.full((len(blocks),context),pad,dtype=np.int64)
    y=np.full_like(x,-100)
    for row,block in enumerate(blocks):
        start=int(block)*context; n=min(context,len(tokens)-1-start)
        assert n>0
        x[row,:n]=tokens[start:start+n]; y[row,:n]=tokens[start+1:start+n+1]
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)


def amp(device):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model,tokens,context,batch,pad,device):
    model.eval(); total=0.; count=0; blocks=math.ceil((len(tokens)-1)/context)
    for start in range(0,blocks,batch):
        x,y=make_batch(tokens,range(start,min(start+batch,blocks)),context,pad,device)
        with amp(device): logits=model(x)
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='sum')
        total+=loss.double().item(); count+=int((y!=-100).sum())
    assert count==len(tokens)-1, f'Incomplete evaluation: {count} != {len(tokens)-1}'
    nll=total/count
    return {'nll':nll,'perplexity':math.exp(nll),'target_tokens':count,'coverage':1.0,'context':context,'reset':'each contiguous context block'}


def pack_indices(indices,bits):
    a=np.asarray(indices,dtype=np.uint16).reshape(-1)
    assert np.all(a < 2**bits)
    if bits==8: return a.astype(np.uint8).tobytes()
    bitmatrix=((a[:,None]>>np.arange(bits,dtype=np.uint16))&1).astype(np.uint8)
    return np.packbits(bitmatrix.reshape(-1),bitorder='little').tobytes()


def unpack_indices(payload,bits,count):
    if bits==8: return np.frombuffer(payload,dtype=np.uint8,count=count).astype(np.int64)
    flat=np.unpackbits(np.frombuffer(payload,dtype=np.uint8),bitorder='little')[:count*bits]
    return (flat.reshape(count,bits)*(1<<np.arange(bits))).sum(1).astype(np.int64)


def export_model(model,directory):
    """Serialize actual deployment projections; avoid counting latent optimizer weights."""
    directory=Path(directory); directory.mkdir(exist_ok=True)
    quantized=set(); specs=[]; bytes_total=0
    with open(directory/'projections.bin','wb') as f:
        for name,module in model.named_modules():
            if not isinstance(module,Projection): continue
            method=module.method; count=module.din*module.dout; start=f.tell()
            if method.startswith('mag'):
                mb,pb=module.mag_bits,module.phase_bits
                clip=module.log_clip.exp().detach()
                radius=F.softplus(module.raw_magnitude.detach())
                mi=(radius/clip).clamp(0,1).mul(2**mb-1).round().long()
                pi=module.phase.detach().mul(2**pb/(2*math.pi)).round().long().remainder(2**pb)
                ix=(mi<<pb)|pi; bits=mb+pb
                payload=struct.pack('<f',float(clip))+pack_indices(ix.cpu().numpy(),bits)
                quantized.update([name+'.raw_magnitude',name+'.phase',name+'.log_clip'])
                spec={'kind':'polar','bits':bits,'mag_bits':mb,'phase_bits':pb,'scale_bytes':4}
            elif method in ['restricted180','full180','full256','phase4','phase8','phase16']:
                bits={'restricted180':8,'full180':8,'full256':8,'phase4':2,'phase8':3,'phase16':4}[method]
                ix=circular_indices(module.phase.detach(),method).cpu().numpy()
                scale=float(module.log_scale.exp().detach())
                payload=struct.pack('<f',scale)+pack_indices(ix,bits)
                quantized.update([name+'.phase',name+'.log_scale'])
                spec={'kind':'phase','bits':bits,'scale_bytes':4}
            elif method in ['real2','real4','real8']:
                bits=int(method[4:]); clip=float(module.log_clip.exp().detach())
                ix=((module.weight.detach()/clip).clamp(-1,1)+1)*((2**bits-1)/2)
                payload=struct.pack('<f',clip)+pack_indices(ix.round().long().cpu().numpy(),bits)
                quantized.update([name+'.weight',name+'.log_clip'])
                spec={'kind':'real','bits':bits,'scale_bytes':4}
            else: continue
            f.write(payload)
            spec.update({'name':name,'method':method,'shape':[module.dout,module.din],
                         'offset':start,'bytes':len(payload),'indices':count})
            specs.append(spec)
    # Tied weights serialized once; all nonquantized trainable parameters retained FP32.
    remaining={name:p.detach().cpu().float().contiguous() for name,p in model.named_parameters() if name not in quantized}
    atomic_torch(directory/'remaining_fp32.pt',remaining)
    payload_bytes=sum(p.numel()*4 for p in remaining.values())+(directory/'projections.bin').stat().st_size
    meta={'projections':specs,'remaining_fp32_payload_bytes':sum(p.numel()*4 for p in remaining.values()),
          'deployment_tensor_payload_bytes':payload_bytes,'note':'No optimizer or latent quantized weights; metadata/file overhead counted separately'}
    atomic_json(directory/'format.json',meta)
    meta['actual_directory_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    atomic_json(directory/'accounting.json',meta)
    meta['actual_directory_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    return meta


def load_export(model,directory):
    directory=Path(directory); meta=json.loads((directory/'format.json').read_text())
    remaining=torch.load(directory/'remaining_fp32.pt',map_location='cpu',weights_only=True)
    params=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in remaining.items(): params[name].copy_(value.to(params[name].device))
    modules=dict(model.named_modules()); data=(directory/'projections.bin').read_bytes()
    for spec in meta['projections']:
        module=modules[spec['name']]; raw=data[spec['offset']:spec['offset']+spec['bytes']]
        scale=struct.unpack('<f',raw[:4])[0]
        indices=torch.from_numpy(unpack_indices(raw[4:],spec['bits'],spec['indices'])).to(next(module.parameters()).device).reshape(spec['shape'])
        if spec['kind']=='polar':
            mb,pb=spec['mag_bits'],spec['phase_bits']
            mi=indices>>pb; pi=indices.remainder(2**pb)
            radius=scale*(mi.float()/(2**mb-1))
            angle=pi.float()*(2*math.pi/(2**pb))
            wr=radius*angle.cos();wi=radius*angle.sin()
        elif spec['kind']=='phase':
            angle=code_angles(indices,spec['method']); wr=scale*angle.cos(); wi=scale*angle.sin()
        else:
            wr=scale*(indices.float()*(2/(2**spec['bits']-1))-1); wi=None
        module.deployed_real=wr; module.deployed_imag=wi
    return meta


def verify():
    torch.set_num_threads(2); torch.manual_seed(10)
    for bits in [2,3,4,8]:
        a=np.arange(37)%2**bits; assert np.array_equal(a,unpack_indices(pack_indices(a,bits),bits,len(a)))
    phase=torch.linspace(-4*math.pi,4*math.pi,10001)
    base=torch.arange(1,91)*math.pi/180; codes=torch.cat((base,base+math.pi))
    chosen=code_angles(circular_indices(phase,'restricted180'),'restricted180')
    nearest=torch.cos(phase[:,None]-codes).max(1).values
    assert torch.allclose((phase-chosen).cos(),nearest,atol=2e-6)
    r=torch.randn(2,13,8,requires_grad=True); i=torch.randn_like(r)
    decay=torch.randn(8,requires_grad=True); freq=torch.randn(8,requires_grad=True)
    fr,fi=recurrent_fft(r,i,decay,freq)
    rho=decay.sigmoid(); ar=rho*freq.cos(); ai=rho*freq.sin()
    hr=torch.zeros_like(r[:,0]); hi=torch.zeros_like(hr); rr=[]; ii=[]
    for t in range(13):
        nr=ar*hr-ai*hi+(1-rho)*r[:,t]; hi=ai*hr+ar*hi+(1-rho)*i[:,t]; hr=nr
        rr.append(hr); ii.append(hi)
    assert torch.allclose(fr,torch.stack(rr,1),atol=2e-6)
    assert torch.allclose(fi,torch.stack(ii,1),atol=2e-6)
    (fr.square().sum()+fi.square().sum()).backward()
    assert torch.isfinite(freq.grad).all() and torch.isfinite(decay.grad).all()
    import tempfile
    for method in METHODS:
        torch.manual_seed(1); model=LanguageModel(258,24,2,16,method,'hybrid',4).eval()
        x=torch.randint(258,(2,16)); z=x.clone(); z[:,9:]=(z[:,9:]+7)%258
        y=model(x); assert torch.allclose(y[:,:9],model(z)[:,:9],atol=3e-6),method
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),method
        with tempfile.TemporaryDirectory() as tmp:
            export_model(model,tmp); restored=LanguageModel(258,24,2,16,method,'hybrid',4).eval()
            load_export(restored,tmp)
            assert torch.allclose(y,restored(x),atol=5e-6),method
    tokens=np.arange(1041,dtype=np.uint16)%258
    model=LanguageModel(258,24,2,16,'continuous','hybrid',4)
    assert evaluate(model,tokens,16,7,0,'cpu')['target_tokens']==1040
    seen=[]
    for block in range(math.ceil((len(tokens)-1)/16)):
        x,y=make_batch(tokens,[block],16,0,'cpu'); seen.extend(y[y!=-100].tolist())
    assert seen==tokens[1:].tolist()
    print('PASS: exact split coverage including tail; nearest restricted code; causal FFT/loop equivalence; all methods causal with finite gradients; packed exports reload equivalently',flush=True)


def train(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists(): print('Already complete',flush=True); return
    manifest=json.loads((Path(args.data)/'manifest.json').read_text())
    tokens={s:np.memmap(Path(args.data)/f'{s}.bin',dtype='<u2',mode='r') for s in manifest['splits']}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    if device=='cuda': torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    config=vars(args).copy(); source=digest(__file__)
    config.update({'dataset_manifest_sha256':digest(Path(args.data)/'manifest.json'),'source_sha256':source,
        'dataset':manifest,'precision':'BF16 autocast matmuls; FP32 latent weights/optimizer/FFT',
        'quantized_scope':'Projection layers only; attention/embeddings/norms/radii/frequencies remain FP32',
        'full_train_passes':args.epochs,'evaluation':'all N-1 next-token targets in each split; fixed contiguous blocks',
        'optimizer':'AdamW betas=(0.9,0.95), weight_decay=0.1, gradient clipping=1',
        'hardware':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),'torch':torch.__version__,
        'note':'Real-data custom LM; no synthetic metrics; no Qwen or IBM SAWB replication; quantizers trained with STE'})
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for field in ['method','architecture','seed','epochs','width','depth','context','batch','source_sha256','dataset_manifest_sha256']:
            assert old[field]==config[field], f'Resume config mismatch: {field}'
    else:
        atomic_json(out/'config.json',config); (out/'run_polar_lm.py').write_bytes(Path(__file__).read_bytes())
    model=LanguageModel(manifest['actual_vocab'],args.width,args.depth,args.context,args.method,args.architecture,args.heads).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.1)
    train_blocks=math.ceil((len(tokens['train'])-1)/args.context)
    steps_epoch=math.ceil(train_blocks/args.batch); total_steps=steps_epoch*args.epochs
    resume=out/'resume.pt'; epoch=0; offset=0; global_step=0; best=float('inf'); coverage=0
    completed_epochs=[]; elapsed=0.; train_loss_sum=0.; train_loss_tokens=0
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        epoch=state['epoch']; offset=state['offset']; global_step=state['global_step']; best=state['best']
        coverage=state['coverage']; completed_epochs=state['completed_epochs']; elapsed=state['elapsed']
        train_loss_sum=state['train_loss_sum']; train_loss_tokens=state['train_loss_tokens']
        print(f'Resuming epoch={epoch} offset={offset} step={global_step}',flush=True)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    started=time.time(); train_started=started
    resumed_targets=epoch*(len(tokens['train'])-1)+coverage
    def save(e,off):
        atomic_torch(resume,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':e,'offset':off,'global_step':global_step,'best':best,'coverage':coverage,
            'completed_epochs':completed_epochs,'elapsed':elapsed+time.time()-started,
            'train_loss_sum':train_loss_sum,'train_loss_tokens':train_loss_tokens})
    print(json.dumps({'event':'start','method':args.method,'architecture':args.architecture,'seed':args.seed,
        'parameters':sum(p.numel() for p in model.parameters()),'train_tokens':len(tokens['train']),
        'targets_per_epoch':len(tokens['train'])-1,'blocks':train_blocks,'steps_per_epoch':steps_epoch,'total_steps':total_steps}),flush=True)
    for e in range(epoch,args.epochs):
        order=np.random.default_rng(2026+args.seed*1000+e).permutation(train_blocks)
        begin=offset if e==epoch else 0
        for start in range(begin,train_blocks,args.batch):
            model.train(); ids=order[start:start+args.batch]
            x,y=make_batch(tokens['train'],ids,args.context,manifest['pad_id'],device)
            count=int((y!=-100).sum()); progress=global_step/max(1,total_steps-1)
            warmup=min(1.,(global_step+1)/min(500,max(1,total_steps//20)))
            lr=args.lr*warmup*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with amp(device): logits=model(x)
            loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {global_step}')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            global_step+=1; coverage+=count; train_loss_sum+=float(loss.detach())*count; train_loss_tokens+=count
            if global_step%args.log_every==0 or global_step==1:
                seconds=elapsed+time.time()-started
                status={'event':'train','epoch':e+1,'step':global_step,'total_steps':total_steps,
                    'current_batch_nll':float(loss.detach()),'train_targets_this_epoch':coverage,
                    'train_targets_required':len(tokens['train'])-1,'epoch_coverage':coverage/(len(tokens['train'])-1),
                    'elapsed_seconds':seconds,'lr':lr,'tokens_per_second_this_process':(e*(len(tokens['train'])-1)+coverage-resumed_targets)/max(.001,time.time()-train_started),
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
                atomic_json(out/'status.json',status); print(json.dumps(status),flush=True)
            if global_step%args.save_every==0: save(e,min(start+args.batch,train_blocks))
        assert coverage==len(tokens['train'])-1, f'Training did not cover full split: {coverage}'
        validation=evaluate(model,tokens['validation'],args.context,args.eval_batch,manifest['pad_id'],device)
        record={'epoch':e+1,'train_target_tokens':coverage,'train_coverage':1.,
                'train_nll':train_loss_sum/train_loss_tokens,'validation':validation,'step':global_step}
        completed_epochs.append(record)
        with open(out/'epochs.jsonl','a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps({'event':'epoch_complete',**record}),flush=True)
        if validation['nll']<best:
            best=validation['nll']; atomic_torch(out/'best.pt',{'model':model.state_dict(),'epoch':e+1,'validation':validation})
        coverage=0; train_loss_sum=0.; train_loss_tokens=0; offset=0
        save(e+1,0)
    best_state=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(best_state['model']); model.eval()
    probe=torch.from_numpy(np.asarray(tokens['validation'][:args.context],dtype=np.int64).copy())[None].to(device)
    with torch.no_grad(),amp(device): before=model(probe).float()
    coefficients={name:tuple(w.detach().clone() if w is not None else None for w in module.coefficients()) for name,module in model.named_modules() if isinstance(module,Projection)}
    export=export_model(model,out/'deployment'); load_export(model,out/'deployment')
    for name,module in model.named_modules():
        if isinstance(module,Projection):
            for original,decoded in zip(coefficients[name],module.coefficients()):
                assert (original is None and decoded is None) or torch.equal(original,decoded), f'Export coefficient mismatch: {name}'
    del coefficients
    with torch.no_grad(),amp(device): after=model(probe).float()
    assert torch.equal(before,after), 'Deployment roundtrip changed logits'
    test=evaluate(model,tokens['test'],args.context,args.eval_batch,manifest['pad_id'],device)
    result={'method':args.method,'architecture':args.architecture,'seed':args.seed,
        'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_state['epoch'],
        'validation':best_state['validation'],'test':test,'completed_epochs':completed_epochs,
        'deployment':export,'fp32_parameter_payload_bytes':sum(p.numel()*4 for p in model.parameters()),
        'elapsed_seconds':elapsed+time.time()-started,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        'test_model':'decoded packed deployment export','source_sha256':source,
        'dataset_manifest_sha256':config['dataset_manifest_sha256']}
    atomic_json(out/'result.json',result); atomic_json(out/'status.json',{'event':'complete',**result})
    (out/'COMPLETE').write_text('Full train epochs and complete validation/test evaluation finished.\n')
    print(json.dumps({'event':'complete','test':test,'best_epoch':best_state['epoch']}),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--verify',action='store_true')
    ap.add_argument('--data',default='data/wikitext103_bpe8k'); ap.add_argument('--subset',default='wikitext-103-raw-v1')
    ap.add_argument('--vocab',type=int,default=8192); ap.add_argument('--out')
    ap.add_argument('--method',choices=METHODS,default='continuous'); ap.add_argument('--architecture',choices=['hybrid','transformer'],default='hybrid')
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--width',type=int,default=384); ap.add_argument('--depth',type=int,default=6); ap.add_argument('--heads',type=int,default=6)
    ap.add_argument('--context',type=int,default=512); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--eval-batch',type=int,default=8)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--log-every',type=int,default=50); ap.add_argument('--save-every',type=int,default=500)
    args=ap.parse_args()
    if args.verify: verify(); return
    if args.prepare: prepare(args.data,args.subset,args.vocab); return
    assert args.out, '--out is required for training'
    train(args)
if __name__=='__main__': main()
