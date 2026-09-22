"""Render the paper's results tables from the raw experiment JSON."""

import json
import sys
from pathlib import Path

LABEL = {
    "real4_uniform": r"real-4, uniform codebook$^{a}$",
    "real4_lloyd": r"real-4, Lloyd codebook$^{b}$",
    "polar_legacy": r"polar 3+5, decoupled rounding$^{c}$",
    "polar_fitted": r"polar 3+5, exact NN + fitted radii",
    "vq2d": r"paired 2-D codebook (256 points)",
    "real4_lloyd+scf": r"\quad + self-consistent calibration",
    "polar_fitted+scf": r"\quad + self-consistent calibration",
    "vq2d+scf": r"\quad + self-consistent calibration",
    "real4_lloyd+seq": r"\quad + sequential calibration",
    "polar_legacy+seq": r"\quad + sequential calibration",
    "polar_fitted+seq": r"\quad + sequential calibration",
    "vq2d+seq": r"\quad + sequential calibration",
}
ORDER = [
    "real4_uniform",
    "real4_lloyd", "real4_lloyd+scf", "real4_lloyd+seq",
    "polar_legacy", "polar_legacy+seq",
    "polar_fitted", "polar_fitted+scf", "polar_fitted+seq",
    "vq2d", "vq2d+scf", "vq2d+seq",
]


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/scq_qwen08b_v1/results.json")
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "paper")
    d = json.loads(src.read_text())
    m = d["methods"]
    bf = m["bf16"]
    plan = d["plan"]

    rows = []
    for k in ORDER:
        if k not in m:
            continue
        r = m[k]
        rows.append(
            f"{LABEL[k]} & {r['nll']:.6f} & {r['ppl']:.3f} & {r['delta_nll']:+.6f} & "
            f"$[{r['delta_ci'][0]:+.6f},\\,{r['delta_ci'][1]:+.6f}]$\\\\"
        )

    tbl = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Frozen Qwen3.5-0.8B with all 18 DeltaNet input projections quantized",
        rf"simultaneously, evaluated on all {plan['eval'].split()[0]} contiguous 128-target",
        r"WikiText-2 validation blocks. Every codec spends four index bits per real weight",
        r"and one FP32 row scale. Intervals are paired block bootstraps against BF16",
        r"(5{,}000 replicates). Lower is better.",
        r"$^{a}$the earlier draft's control; $^{b}$the matched control;",
        r"$^{c}$the earlier draft's method.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrrrl}\toprule",
        r"Method & NLL & Perplexity & $\Delta$NLL & 95\% CI\\\midrule",
        f"BF16 reference & {bf['nll']:.6f} & {bf['ppl']:.3f} & --- & ---\\\\",
        r"\midrule",
    ]
    tbl += rows
    tbl += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Head-to-head comparisons that carry the paper's two claims.
    comps = d.get("comparisons", {})
    interesting = [
        ("polar_legacy_minus_real4_lloyd", r"polar (earlier) $-$ real-4 Lloyd"),
        ("polar_fitted_minus_real4_lloyd", r"polar (repaired) $-$ real-4 Lloyd"),
        ("vq2d_minus_real4_lloyd", r"paired 2-D $-$ real-4 Lloyd"),
        ("vq2d_minus_polar_legacy", r"paired 2-D $-$ polar (earlier)"),
        ("vq2d+scf_minus_real4_lloyd", r"paired 2-D + SCF $-$ real-4 Lloyd"),
        ("vq2d+scf_minus_polar_legacy", r"paired 2-D + SCF $-$ polar (earlier)"),
    ]
    crows = []
    for key, label in interesting:
        if key not in comps:
            continue
        c = comps[key]
        sign = "favours the first" if c["delta"] < 0 else "favours the second"
        excl = "excludes 0" if (c["ci"][0] > 0) == (c["ci"][1] > 0) else "includes 0"
        crows.append(
            f"{label} & {c['delta']:+.6f} & $[{c['ci'][0]:+.6f},\\,{c['ci'][1]:+.6f}]$ & {excl}\\\\"
        )
    if crows:
        tbl += [
            r"\begin{table}[ht]",
            r"\centering\small",
            r"\caption{Paired head-to-head differences on the same validation blocks.",
            r"A negative value favours the first-named method.}",
            r"\label{tab:head}",
            r"\begin{tabular}{lrll}\toprule",
            r"Comparison & $\Delta$NLL & 95\% CI & Interval\\\midrule",
        ]
        tbl += crows
        tbl += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # SCF convergence trace.
    scf_rows = []
    for k in ("real4_lloyd+scf", "polar_fitted+scf", "vq2d+scf"):
        if k not in m or "scf" not in m[k]:
            continue
        s = m[k]["scf"]
        short = {"real4_lloyd": "real-4 Lloyd", "polar_fitted": "polar (repaired)",
                 "vq2d": "paired 2-D"}[k.replace("+scf", "")]
        scf_rows.append(
            f"{short} & {s['iterations']} & {s['rejected_steps']} & "
            f"{s['one_shot_energy']:.6f} & {s['final_energy']:.6f} & "
            f"{(s['final_churn'] if s['final_churn'] is not None else float('nan')):.2e}\\\\"
        )
    if scf_rows:
        tbl += [
            r"\begin{table}[ht]",
            r"\centering\small",
            r"\caption{Self-consistent field behaviour. Energy is calibration NLL;",
            r"iteration zero is conventional one-shot calibration. Churn is the fraction",
            r"of codebook assignments that moved in the final iteration.}",
            r"\label{tab:scf}",
            r"\begin{tabular}{lrrrrr}\toprule",
            r"Codec & Iters & Rejected & One-shot energy & Final energy & Final churn\\\midrule",
        ]
        tbl += scf_rows
        tbl += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    (outdir / "scq_tables.tex").write_text("\n".join(tbl) + "\n")
    print("\n".join(tbl))


if __name__ == "__main__":
    main()
