"""Fetch audited full-run metadata from the authorized workstation; no checkpoints.

The workstation address comes from $QQ_WORKSTATION so no private host is
recorded in the repository.
"""
import os
import subprocess
from pathlib import Path

def main():
    workstation=os.environ.get('QQ_WORKSTATION', 'user@workstation')
    remote='/home/shayan/quantum_quantization'
    local=Path(__file__).resolve().parent.parent/'results'/'wikitext103_full_v1'
    local.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ssh','-o','BatchMode=yes',workstation,
        f'cd {remote} && PYTHONPATH={remote}/.lm_deps /home/shayan/miniconda3/bin/python experiments/run_full_sweep.py --collect'],check=True)
    command=['rsync','-r','--prune-empty-dirs','--include=*/','--include=*.json','--include=*.jsonl',
        '--include=*.csv','--include=*.md','--include=*.log','--include=*.py','--include=COMPLETE',
        '--exclude=*',f'{workstation}:{remote}/results/wikitext103_full_v1/',str(local)+'/']
    subprocess.run(command,check=True)
    print((local/'summary.md').read_text())
if __name__=='__main__':main()
