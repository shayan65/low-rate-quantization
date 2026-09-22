"""Generate descriptive tables and plot from completed, immutable experiment runs."""
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES={'fp32':'FP32 reference','cartesian_nearest':'Cartesian nearest','cartesian_weight':'Stable Cartesian weight','cartesian_memory':'Stable Cartesian memory','cartesian_normalized_memory':'Normalized Cartesian memory','phase_nearest':'Phase nearest','phase_weight':'Phase weight','phase_memory':'Phase memory'}

def main():
    records=[]
    for bits in (4,8):
        folder=ROOT/'results'/f'memory_quantization_v2_b{bits}'
        assert (folder/'COMPLETE').exists(),f'Incomplete run: {folder}'
        manifest=json.loads((folder/'manifest.json').read_text())
        source=folder/'run_memory_quantization.py'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
        rows=list(csv.DictReader((folder/'results.csv').open()))
        assert len(rows)==96 and len({(r['seed'],r['method'],r['horizon']) for r in rows})==96
        records.extend(rows)
    grouped=collections.defaultdict(list)
    for r in records: grouped[int(r['index_bits']),r['method'],int(r['horizon'])].append(float(r['rollout_relative_mse']))
    def stats(bits,method,h):
        values=grouped[bits,method,h]; return statistics.mean(values),statistics.stdev(values)
    def fmt(value):
        return f'{value:.3g}'
    def texnum(value):
        if abs(value)>=1e4:
            a,b=f'{value:.2e}'.split('e'); return rf'{a}\!\times\!10^{{{int(b)}}}'
        return f'{value:.3f}'
    lines=['# Controlled recurrent-memory quantization results','','Scope: trained quantizers on three sampled, frozen linear systems; no task-model, Qwen or quantum-hardware training. All evaluations use exported and decoded operators. Relative output MSE is lower-is-better; predicting zero gives exactly 1. Values are mean ± sample SD across three systems, not significance estimates.','','| Method | 4-bit, L=128 | 4-bit, L=512 | 8-bit, L=128 | 8-bit, L=512 |','|---|---:|---:|---:|---:|']
    tex=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Stable-control decoded-operator rollout relative MSE, mean $\pm$ sample SD over three sampled systems. Calibration length is 128; lower is better. A zero-output predictor has error 1. Index bits exclude the stored FP32 radii; all compressed controls use identical bytes at each budget.}',r'\label{tab:memory}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{4 index bits/mode} & \multicolumn{2}{c}{8 index bits/mode} \\',r'Method & $L=128$ & $L=512$ & $L=128$ & $L=512$ \\',r'\midrule']
    for method,name in NAMES.items():
        entries=[stats(bits,method,h) for bits,h in [(4,128),(4,512),(8,128),(8,512)]]
        lines.append('| '+name+' | '+' | '.join(f'{fmt(m)} ± {fmt(s)}' for m,s in entries)+' |')
        if method!='cartesian_nearest': tex.append(name+' & '+' & '.join('$'+texnum(m)+r'\pm'+texnum(s)+'$' for m,s in entries)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    lines+=['','## Interpretation','', 'At four index bits, phase-memory calibration improves over phase-weight calibration, but it does not outperform the stable Cartesian memory control at L=512. Phase-memory output error exceeds 1 at that horizon, so it is worse than predicting zero. This is evidence against claiming a useful long-context phase advantage from this pilot. See the eight-bit results above to assess the higher-rate setting separately.','', 'The unstable Cartesian nearest row is a diagnostic, not a competitive practical baseline. The fitted Cartesian weight and memory controls start from the same stable scale and enforce candidate radii no larger than the reference radii. The normalized Cartesian control also preserves radii.','', 'At four bits: operator files 304 bytes vs FP32 524; complete system including unchanged projections 4,400 bytes vs 4,620. At eight bits: operator files 336 bytes, whole system 4,432. Whole-system savings are about 4.8% and 4.1%, respectively. This study does not demonstrate whole-model low-bit quantization or inference acceleration.','', 'The pilot varies system seed; Gaussian probes are fresh and deterministic for each horizon. It has no learned free codebook, nonlinear hybrid model, or task accuracy result. Calibration and candidate search budgets are fixed, not tuned using test outcomes.','', 'Version 1 is retained as a numerical diagnostic with unconstrained fitted Cartesian controls; version 2 stabilizes these controls and is the version tabulated here. The phase-family results are unchanged between versions at four bits. Full raw metrics and training histories are retained in each run directory.']
    summary=ROOT/'results'/'memory_quantization_summary.md'; summary.write_text('\n'.join(lines)+'\n')
    (ROOT/'paper'/'memory_table.tex').write_text('\n'.join(tex)+'\n')
    # Plot the strong stable controls; catastrophic nearest values would flatten this comparison.
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for ax,bits in zip(axes,(4,8)):
        for method in ['cartesian_memory','cartesian_normalized_memory','phase_nearest','phase_weight','phase_memory']:
            hs=[64,128,256,512]; means=[stats(bits,method,h)[0] for h in hs]; sd=[stats(bits,method,h)[1] for h in hs]
            ax.errorbar(hs,means,yerr=sd,marker='o',capsize=2,label=NAMES[method],linewidth=1.2)
        ax.axhline(1,color='gray',linestyle=':',label='Zero-output predictor')
        ax.axvline(128,color='gray',linestyle='--',alpha=.5)
        ax.set_xscale('log',base=2); ax.set_xticks([64,128,256,512],labels=['64','128','256','512'])
        ax.set_title(f'{bits} index bits per mode'); ax.set_xlabel('Evaluation length'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Rollout relative MSE (lower is better)')
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.02))
    fig.tight_layout(rect=(0,.14,1,1)); path=ROOT/'paper'/'memory_comparison.png'; fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig)
    text=r'''\input{memory_table.tex}
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{memory_comparison.png}
\caption{Stable-control comparisons over evaluation length. Error bars show sample standard deviation over systems. The dashed vertical line marks calibration length 128; the horizontal dotted line is a zero-output predictor.}
\end{figure}
The four-bit phase-memory control improves relative to phase-weight calibration, but the stable Cartesian memory control has lower mean error at length 512. Phase-memory relative error exceeds one at that horizon, indicating worse output reconstruction than predicting zero. The pilot therefore does not establish a useful long-context phase advantage. At eight index bits, phase-memory error at length 512 is approximately 0.791, compared with 0.811 for phase-weight calibration and 0.922 for stable Cartesian memory. The extra memory-objective improvement over phase-weight calibration is modest; only three systems are tested. The eight-bit setting is reported separately rather than pooled with the four-bit setting. The observed benefit of the memory objective within a representation must not be conflated with superiority of that representation.

The nearest Cartesian decoder can have radii above one and exhibits catastrophic growth. This diagnostic should not dominate the assessment: fitted Cartesian controls are explicitly stabilized, and the normalized Cartesian control provides a fixed-radius comparison. Reported sample deviations are descriptive over only three systems; no significance claim is justified. Version 1, with unconstrained fitted Cartesian controls and numerically unreliable large-error coordinate comparisons, is retained as a diagnostic. Version 2 uses stable initialization and constraints for fitted raw Cartesian controls and supplies all tabulated results.
'''
    (ROOT/'paper'/'memory_results.tex').write_text(text)
    print(summary)
if __name__=='__main__':main()
