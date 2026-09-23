"""Time-limited validation-only screen on recorded real WikiText-103 samples."""
import argparse,hashlib,json,math,time,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run_polar_lm import LanguageModel,Projection,make_batch,amp,export_model,load_export,atomic_json,digest

@torch.no_grad()
def validation(model,tokens,blocks,context,pad,device):
    model.eval();loss_sum=0.;count=0
    for start in range(0,len(blocks),8):
        x,y=make_batch(tokens,blocks[start:start+8],context,pad,device)
        with amp(device): logits=model(x)
        loss_sum+=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='sum').item()
        count+=int((y!=-100).sum())
    return {'nll':loss_sum/count,'perplexity':math.exp(loss_sum/count),'sample_target_tokens':count}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/wikitext103_bpe8k')
    ap.add_argument('--out',default='results/coordinate_diagnostic_v1');ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume);started=time.monotonic()
    root=Path(args.data);manifest=json.loads((root/'manifest.json').read_text())
    tokens={s:np.memmap(root/f'{s}.bin',dtype='<u2',mode='r') for s in ['train','validation']}
    context=128;batch=16;width=64;depth=2
    rng=np.random.default_rng(20260914)
    train_blocks=rng.choice((len(tokens['train'])-1)//context,2048,replace=False)
    val_blocks=np.sort(rng.choice((len(tokens['validation'])-1)//context,128,replace=False))
    plan={'dataset':'Salesforce/wikitext wikitext-103-raw-v1','dataset_manifest_sha256':digest(root/'manifest.json'),
          'dataset_revision':manifest['revision'],'tokenizer':'existing train-only 8K BPE',
          'sample_seed':20260914,'train_blocks':train_blocks.tolist(),'validation_blocks':val_blocks.tolist(),
          'train_sample_target_tokens':len(train_blocks)*context,'validation_sample_target_tokens':len(val_blocks)*context,
          'context':context,'width':width,'depth':depth,'batch':batch,'steps':args.steps,'wall_time_limit_seconds':args.seconds,
          'methods':['real_fp32','complex_fp32','polar_fp32'],'seeds':[0,1,2],
          'gate':'Diagnose Cartesian complex versus polar training; no automatic promotion.',
          'note':'Real source blocks; no concatenation of unrelated samples; validation sample only; no test-set access; no full dataset result claim',
          'source_sha256':digest(__file__),'model_source_sha256':digest(Path(__file__).with_name('run_polar_lm.py'))}
    if args.resume:
        import ast
        previous=json.loads((out/'plan.json').read_text())
        for key in ['dataset_manifest_sha256','train_blocks','validation_blocks','steps','methods','seeds','width','depth','context','batch','gate']:
            assert previous[key]==plan[key], ('Resume experiment mismatch',key)
        old_tree=ast.parse((out/'run_polar_lm.py').read_text()); new_tree=ast.parse(Path(__file__).with_name('run_polar_lm.py').read_text())
        def forward_tree(tree):
            return [ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.ClassDef)]
        assert forward_tree(old_tree)==forward_tree(new_tree), 'Model training definition changed; cannot reuse checkpoints'
        atomic_json(out/'export_fix.json',{'old_model_source_sha256':previous['model_source_sha256'],'new_model_source_sha256':plan['model_source_sha256'],'model_classes_ast_identical':True,'change':'Real bin encoder uses the same tensor divisor as training; forward and gradient definitions unchanged'})
        (out/'run_polar_lm_exportfix.py').write_bytes(Path(__file__).with_name('run_polar_lm.py').read_bytes())
        (out/'run_polar_screen_resume.py').write_bytes(Path(__file__).read_bytes())
        plan=previous
    else:
        atomic_json(out/'plan.json',plan);(out/'run_polar_screen.py').write_bytes(Path(__file__).read_bytes())
        (out/'run_polar_lm.py').write_bytes(Path(__file__).with_name('run_polar_lm.py').read_bytes())
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    rows=json.loads((out/'results.json').read_text()) if args.resume and (out/'results.json').exists() else []; budget_hit=False
    for seed in plan['seeds']:
        for method in plan['methods']:
            if any(r['method']==method and r['seed']==seed for r in rows):continue
            if time.monotonic()-started>args.seconds:budget_hit=True;break
            torch.manual_seed(seed);model=LanguageModel(manifest['actual_vocab'],width,depth,context,method,'hybrid',4).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
            gen=np.random.default_rng(2026+seed);order=gen.permutation(train_blocks);position=0
            job=out/f'{method}_seed{seed}';job.mkdir(exist_ok=args.resume);history=[];best=float('inf');best_step=0;target_count=0
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            job_started=time.monotonic();initial=validation(model,tokens['validation'],val_blocks,context,manifest['pad_id'],device)
            recover=False
            if args.resume and (job/'history.json').exists() and (job/'best.pt').exists():
                history=json.loads((job/'history.json').read_text())
                recover=bool(history and history[-1]['step']==args.steps)
                if recover:
                    retained=min(history,key=lambda h:h['validation']['nll']);best=retained['validation']['nll'];best_step=retained['step']
                    target_count=args.steps*batch*context
                    print('RECOVER completed training checkpoint',method,seed,flush=True)
            for step in range(0 if recover else args.steps):
                if time.monotonic()-started>args.seconds:budget_hit=True;break
                if position==len(order):order=gen.permutation(train_blocks);position=0
                selected=order[position:position+batch];position+=len(selected)
                x,y=make_batch(tokens['train'],selected,context,manifest['pad_id'],device)
                optimizer.zero_grad(set_to_none=True);model.train()
                lr=3e-4*min(1,(step+1)/50)*(.1+.9*.5*(1+math.cos(math.pi*step/max(1,args.steps-1))))
                for group in optimizer.param_groups:group['lr']=lr
                with amp(device):logits=model(x)
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                target_count+=int((y!=-100).sum())
                if (step+1)%100==0 or step+1==args.steps:
                    val=validation(model,tokens['validation'],val_blocks,context,manifest['pad_id'],device)
                    entry={'step':step+1,'train_batch_nll':float(loss.detach()),'validation':val};history.append(entry)
                    if val['nll']<best:
                        best=val['nll'];best_step=step+1;torch.save(model.state_dict(),job/'best.pt')
                    print(method,seed,json.dumps(entry),flush=True)
            atomic_json(job/'history.json',history)
            if budget_hit:break
            model.load_state_dict(torch.load(job/'best.pt',map_location=device,weights_only=True));model.eval()
            probe,_=make_batch(tokens['validation'],val_blocks[:1],context,manifest['pad_id'],device)
            coefficients={n:tuple(v.detach().clone() if v is not None else None for v in m.coefficients()) for n,m in model.named_modules() if isinstance(m,Projection)}
            with torch.no_grad(),amp(device):before=model(probe).clone()
            export=export_model(model,job/'deployment');load_export(model,job/'deployment')
            for n,m in model.named_modules():
                if isinstance(m,Projection):
                    for a,b in zip(coefficients[n],m.coefficients()):assert (a is None and b is None) or torch.equal(a,b),(method,n)
            with torch.no_grad(),amp(device):after=model(probe)
            assert torch.equal(before,after),(method,'logit roundtrip')
            val=validation(model,tokens['validation'],val_blocks,context,manifest['pad_id'],device)
            assert val['nll']==best,(method,'decoded validation drift')
            row={'method':method,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                 'best_step':best_step,'initial_validation_nll':initial['nll'],'validation':val,
                 'train_targets_presented':target_count,'seconds':None if recover else time.monotonic()-job_started,'recovered_completed_training':recover,'recovery_export_seconds':time.monotonic()-job_started if recover else None,
                 'deployment_tensor_bytes':export['deployment_tensor_payload_bytes'],
                 'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                 'exact_coefficient_and_logit_roundtrip':True}
            rows.append(row);atomic_json(job/'result.json',row);atomic_json(out/'results.json',rows)
            print('COMPLETE',json.dumps(row),flush=True);del model,optimizer,coefficients
        if budget_hit:break
    complete=len(rows)==len(plan['methods'])*len(plan['seeds'])
    lookup={(r['method'],r['seed']):r['validation']['nll'] for r in rows}
    gaps={}
    if complete:
        for control in ['real_fp32','complex_fp32']:
            gaps[control]=statistics.mean(lookup['polar_fp32',s]-lookup[control,s] for s in plan['seeds'])
    gate={'completed_jobs':len(rows),'planned_jobs':len(plan['methods'])*len(plan['seeds']),'budget_exhausted':budget_hit,
          'this_invocation_wall_seconds':time.monotonic()-started,'mean_paired_nll_gaps':gaps,
          'passed':complete and gaps.get('complex_fp32',float('inf'))<=.05,'action':'No full-data job is launched by this screen.'}
    atomic_json(out/'gate.json',gate)
    text='# Small real-data coordinate-training diagnostic\n\n'
    text+='0.6M-scale custom hybrid, real WikiText-103 source blocks, train-only 8K tokenizer. Only 262,144 selected training targets and 16,384 selected validation targets. Test set untouched. These are screening results, not full-dataset perplexity.\n\n'
    text+='| Method | Seeds completed | Validation sample perplexity mean ± SD | Deployment tensor bytes |\n|---|---:|---:|---:|\n'
    for method in plan['methods']:
        values=[r for r in rows if r['method']==method]
        if not values:continue
        ppls=[r['validation']['perplexity'] for r in values];sd=statistics.stdev(ppls) if len(ppls)>1 else 0
        text+=f"| {method} | {len(values)} | {statistics.mean(ppls):.3f} ± {sd:.3f} | {values[0]['deployment_tensor_bytes']:,} |\n"
    text+=f"\nGate: {json.dumps(gate)}\n\nThese are FP32 trained variants with equal exported size for the two complex coordinates; real versus complex state capacity and exported size differ. This small-sample gate tests whether polar training approaches Cartesian complex training. It does not establish novelty, task superiority, or justify an automatic long run.\n"
    (out/'summary.md').write_text(text)
    if complete:(out/'COMPLETE').write_text('All validation-only screen jobs complete.\n')
    else:(out/'STOPPED_BUDGET').write_text('Stopped at explicit wall-time budget; no fabricated pending metrics.\n')
    print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
