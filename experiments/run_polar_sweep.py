"""Sequential single-GPU full-data sweep. Completed jobs are not repeated."""
import argparse, csv, hashlib, json, math, os, statistics, subprocess, sys, time
from pathlib import Path
from run_polar_lm import atomic_json, digest, METHODS


def collect(root):
    root=Path(root); plan=json.loads((root/'plan.json').read_text()); records=[]; pending=[]
    for job in plan['jobs']:
        directory=root/job['id']; result=directory/'result.json'
        if not (directory/'COMPLETE').exists(): pending.append(job['id']); continue
        r=json.loads(result.read_text()); config=json.loads((directory/'config.json').read_text())
        assert r['source_sha256']==config['source_sha256']==digest(directory/'run_polar_lm.py')
        assert r['dataset_manifest_sha256']==config['dataset_manifest_sha256']
        assert config['dataset']['subset']=='wikitext-103-raw-v1'
        assert len(r['completed_epochs'])==plan['epochs']
        train_targets=config['dataset']['splits']['train']['target_tokens_per_pass']
        for epoch in r['completed_epochs']:
            assert epoch['train_target_tokens']==train_targets and epoch['train_coverage']==1
            assert epoch['validation']['target_tokens']==config['dataset']['splits']['validation']['target_tokens_per_pass']
        for split in ['validation','test']:
            assert r[split]['target_tokens']==config['dataset']['splits'][split]['target_tokens_per_pass']
            assert r[split]['coverage']==1
        assert r['test_model']=='decoded packed deployment export'
        row={'job':job['id'],'architecture':r['architecture'],'method':r['method'],'seed':r['seed'],
             'parameters':r['parameters'],'best_epoch':r['best_epoch'],
             'validation_nll':r['validation']['nll'],'test_nll':r['test']['nll'],
             'test_perplexity':r['test']['perplexity'],'test_targets':r['test']['target_tokens'],
             'deployment_tensor_bytes':r['deployment']['deployment_tensor_payload_bytes'],
             'actual_deployment_bytes':sum(p.stat().st_size for p in (directory/'deployment').iterdir() if p.is_file()),
             'fp32_parameter_bytes':r['fp32_parameter_payload_bytes'],'peak_cuda_bytes':r['peak_cuda_bytes'],
             'elapsed_seconds':r['elapsed_seconds']}
        records.append(row)
    if records:
        with open(root/'results.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    text='# Full-data WikiText-103 sweep\n\n'
    text+=f"Completed {len(records)} / {len(plan['jobs'])} runs. Pending runs have no reported metrics.\n\n"
    text+='All reported results come from full real WikiText-103 splits. BPE token perplexity uses the train-only 8K tokenizer and is not directly comparable to a different tokenizer. All training is floating-point QAT.\n\n'
    text+='| Architecture | Method | Seeds completed | Test perplexity mean ± SD | Deployment tensor bytes |\n|---|---|---:|---:|---:|\n'
    groups={}
    for r in records: groups.setdefault((r['architecture'],r['method']),[]).append(r)
    for (arch,method),group in groups.items():
        values=[r['test_perplexity'] for r in group]; std=statistics.stdev(values) if len(values)>1 else None
        value=f'{statistics.mean(values):.3f}' + (f' ± {std:.3f}' if std is not None else ' (one seed)')
        text+=f"| {arch} | {method} | {len(group)} | {value} | {group[0]['deployment_tensor_bytes']:,} |\n"
    if not records: text+='\nNo full runs have completed yet.\n'
    text+='\nReal and polar models differ in effective state capacity; the unconstrained complex model also has more parameters. These comparisons cannot attribute all differences to quantization. The matched phase variants isolate codebook coverage. Temporal calibration and downstream Qwen validation are separate, uncompleted experiments.\n'
    (root/'summary.md').write_text(text)
    atomic_json(root/'progress.json',{'completed':len(records),'total':len(plan['jobs']),'pending':pending})
    return len(records),len(plan['jobs'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/wikitext103_polar_v2')
    ap.add_argument('--data',default='data/wikitext103_bpe8k');ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--collect',action='store_true')
    ap.add_argument('--screen-report',default='results/polar_screen_v2/gate.json')
    ap.add_argument('--allow-full-run',action='store_true',help='Deliberately promote a previously validated small experiment')
    args=ap.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if args.collect: collect(root);return
    if not args.allow_full_run:
        raise SystemExit('Full runs are disabled by default. First complete small/medium validation; no GPU job launched.')
    screen=json.loads(Path(args.screen_report).read_text())
    if not screen.get('passed'):
        raise SystemExit('Small-screen gate failed. No full-data GPU job launched.')
    # A PID flock prevents duplicate GPU sweeps. Release is automatic on process exit.
    import fcntl
    lock=open(root/'sweep.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    training=Path(__file__).with_name('run_polar_lm.py').resolve()
    # First seed prioritizes user's representation and matched controls.
    hybrid=['mag4phase4','polar_fp32','mag2phase6','mag2phase2','mag1phase1','real_fp32','real2','real4','real8','complex_fp32','continuous']
    transformer=['real_fp32','mag4phase4']
    jobs=[]
    for seed in args.seeds:
        for architecture,methods in [('hybrid',hybrid),('transformer',transformer)]:
            for method in methods:
                jobs.append({'id':f'{architecture}_{method}_seed{seed}','architecture':architecture,'method':method,'seed':seed})
    plan={'dataset':'Salesforce/wikitext','subset':'wikitext-103-raw-v1','epochs':args.epochs,'seeds':args.seeds,
          'width':384,'depth':6,'context':512,'batch':16,'heads':6,'training_source_sha256':digest(training),
          'scope':'Magnitude-phase version 2; full real-data comparison; separate from the earlier shared-radius sweep','jobs':jobs}
    plan_path=root/'plan.json'
    if plan_path.exists(): assert json.loads(plan_path.read_text())==plan,'Sweep plan/source changed; choose a new root'
    else: atomic_json(plan_path,plan)
    (root/'run_polar_sweep.py').write_bytes(Path(__file__).read_bytes())
    print('Verifying implementation before experiment',flush=True)
    subprocess.run([sys.executable,str(training),'--verify'],check=True)
    print('Preparing/verifying full real dataset',flush=True)
    # No duplicate preparation while a previously launched preparation process is active.
    prep_lock=open(Path(args.data).parent/'wikitext103_prepare.lock','w')
    fcntl.flock(prep_lock,fcntl.LOCK_EX)
    subprocess.run([sys.executable,str(training),'--prepare','--data',args.data],check=True)
    fcntl.flock(prep_lock,fcntl.LOCK_UN)
    manifest_path=Path(args.data)/'manifest.json'
    (root/'dataset_manifest.json').write_bytes(manifest_path.read_bytes())
    collect(root)
    for job in jobs:
        directory=root/job['id'];directory.mkdir(exist_ok=True)
        if (directory/'COMPLETE').exists():continue
        command=[sys.executable,str(training),'--data',args.data,'--out',str(directory),
            '--method',job['method'],'--architecture',job['architecture'],'--seed',str(job['seed']),
            '--epochs',str(args.epochs),'--width','384','--depth','6','--heads','6','--context','512','--batch','16','--allow-full-run']
        atomic_json(root/'current_job.json',{'job':job,'command':command,'started_unix':time.time()})
        print(f"START {job['id']}",flush=True)
        with open(directory/'train.log','a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(root/'FAILED.json',{'job':job,'returncode':result.returncode,'log':str(directory/'train.log')})
            raise RuntimeError(f"Run failed: {job['id']}; see its log. Resume retains checkpoint.")
        print(f"COMPLETE {job['id']}",flush=True); collect(root)
    done,total=collect(root);assert done==total
    (root/'COMPLETE').write_text(f'All {total} full-data experiments complete.\n')
    print(f'ALL {total} RUNS COMPLETE',flush=True)
if __name__=='__main__':main()
