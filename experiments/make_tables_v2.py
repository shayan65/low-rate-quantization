"""Generate every reported table directly from saved results.

The 2026-09-22 review caught a storage-dominance margin quoted from
`ternary_task_v3` (no GPTQ) beside a 27B number produced under the v4 GPTQ
recipe. Hand-transcribed tables make that class of error easy to commit and
hard to see, so the paper's tables are generated here instead, each one
carrying the run directory it came from.

Two rules are enforced mechanically rather than by care:

  * every row records its source run, so cross-recipe comparisons are visible;
  * byte ratios are printed to enough digits to show that "byte-matched" means
    near-equal payload rather than literal parity (VQ8 is +0.43% at 0.8B and
    +0.0193% at 27B, neither of which is 1.0000).

Intervals are reproduced verbatim from the saved paired bootstrap. Ratios of
point estimates are labelled as such, never given an interval they do not have.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "paper" / "generated"

RUNS = {
    "v3": ("ternary_task_v3", "0.8B, no compensation"),
    "v4": ("ternary_task_v4", "0.8B, GPTQ recipe"),
    "mlp": ("mlp_task_v1", "0.8B MLP, GPTQ recipe"),
    "alloc": ("mixed_alloc_v1", "0.8B, allocation"),
    "rate": ("rate_curve_v1", "0.8B, rate sweep"),
    "q27": ("qwen38_confirm_v1", "27B, GPTQ recipe"),
    "q27b": ("qwen38_confirm_v2", "27B, GPTQ recipe, frozen full stream"),
}


def load(key):
    d = RESULTS / RUNS[key][0] / "results.json"
    return json.loads(d.read_text()) if d.exists() else None


def esc(s):
    return s.replace("_", r"\_")


def fmt_ci(ci):
    return f"$[{ci[0]:+.6f}, {ci[1]:+.6f}]$"


def arm_table(key, caption, label, rows=None):
    r = load(key)
    if r is None:
        return f"% {RUNS[key][0]} not present\n"
    arms = r["arms"]
    names = rows or list(arms)
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrl}", r"\toprule",
         r"Arm & bits/wt & payload MB & $\Delta$NLL vs BF16 & 95\% CI \\", r"\midrule"]
    for n in names:
        if n not in arms:
            continue
        a = arms[n]
        L.append(f"{esc(n)} & {a['bits_per_weight']:.4f} & "
                 f"{a['stored_bytes'] / 1e6:.2f} & ${a['delta_nll']:+.6f}$ & {fmt_ci(a['delta_ci'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          f"\\caption{{{caption} Source: \\texttt{{results/{RUNS[key][0]}}} ({RUNS[key][1]}); "
          f"BF16 NLL {r['bf16']['nll']:.6f}.}}",
          f"\\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(L)


def comparison_table(entries, caption, label):
    """entries: [(run_key, comparison_key, human_label)] -- run shown per row."""
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{llrlr}", r"\toprule",
         r"Contrast & Run & $\Delta$NLL & 95\% CI & byte ratio \\", r"\midrule"]
    for key, ck, human in entries:
        r = load(key)
        if r is None or ck not in r.get("comparisons", {}):
            L.append(f"% missing: {key} {ck}")
            continue
        c = r["comparisons"][ck]
        L.append(f"{esc(human)} & {esc(RUNS[key][1])} & ${c['delta']:+.6f}$ & "
                 f"{fmt_ci(c['ci'])} & {c['byte_ratio']:.5f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          "\\caption{" + caption + " Each row names the run it comes from; rows from "
          "different runs are not matched comparisons. Byte ratios are printed to "
          "five decimals because ``byte-matched\'\' here means near-equal payload, "
          "not literal parity.}",
          f"\\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(L)


def rate_table():
    r = load("rate")
    if r is None:
        return ""
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{rrrrrl}", r"\toprule",
         r"$K$ & index bpw & total bpw & payload MB & $\Delta$NLL & 95\% CI \\", r"\midrule"]
    for n, a in r["arms"].items():
        L.append(f"{a['k']} & {a['index_bpw']:.3f} & {a['bits_per_weight']:.4f} & "
                 f"{a['stored_bytes'] / 1e6:.2f} & ${a['delta_nll']:+.6f}$ & {fmt_ci(a['delta_ci'])} \\\\")
    lo = min(a["stored_bytes"] for a in r["arms"].values())
    hi = max(a["stored_bytes"] for a in r["arms"].values())
    L += [r"\bottomrule", r"\end{tabular}",
          f"\\caption{{Uniform rate sweep at fixed coverage (50.2\\% of parameters), "
          f"dimension 4 throughout so rate varies alone. Stored-byte range "
          f"{hi / lo:.2f}$\\times$. Source: \\texttt{{results/{RUNS['rate'][0]}}}; "
          f"BF16 NLL {r['bf16']['nll']:.6f}.}}",
          r"\label{tab:rate}", r"\end{table}", ""]
    return "\n".join(L)


def coverage_table():
    """Protocol and coverage, the table the review asked to lead with."""
    rows = []
    for key in ("v4", "mlp", "alloc", "rate", "q27"):
        r = load(key)
        if r is None:
            continue
        pl = r["plan"]
        tp = pl.get("target_params") or pl.get("mlp_params") or 0
        share = pl.get("share") or pl.get("mlp_share") or 0
        tgt = pl.get("eval_targets") or pl.get("eval_targets", 0)
        cal = pl.get("cal_tokens", r.get("manifest", {}).get("cal_tokens", "--"))
        rows.append((RUNS[key][0], tp, share, tgt, cal))
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Run & target params & share & eval targets & calib.\ tokens \\", r"\midrule"]
    for nm, tp, sh, tg, cal in rows:
        L.append(f"\\texttt{{{esc(nm)}}} & {tp:,} & {sh * 100:.1f}\\% & {tg:,} & "
                 f"{cal if isinstance(cal, str) else format(cal, ',')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Protocol and coverage for every run reported. Coverage and "
          r"calibration differ between checkpoints, so cross-run differences are "
          r"not attributable to scale alone.}", r"\label{tab:coverage}",
          r"\end{table}", ""]
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "tab_coverage.tex": coverage_table(),
        "tab_rate.tex": rate_table(),
        "tab_v4_arms.tex": arm_table(
            "v4", "Codec and recipe ablation on the 0.8B DeltaNet QKV projections.",
            "tab:v4"),
        "tab_mlp_arms.tex": arm_table(
            "mlp", "The same recipe on all 72 MLP tensors of the 0.8B model.",
            "tab:mlp"),
        "tab_q27_arms.tex": arm_table(
            "q27", "Scale transfer: 48 DeltaNet QKV projections of Qwen3.8-27B.",
            "tab:q27"),
        "tab_primary.tex": comparison_table([
            ("v4", "vq8_rot_gptq_vs_scalar3_rot_gptq", "VQ8 vs scalar (0.8B)"),
            ("q27", "vq8_rot_gptq_vs_scalar3_rot_gptq", "VQ8 vs scalar (27B)"),
            ("v4", "vq4_rot_gptq_vs_scalar3_rot_gptq", "VQ4 vs scalar (0.8B)"),
            ("q27", "vq4_rot_gptq_vs_scalar3_rot_gptq", "VQ4 vs scalar (27B)"),
            ("v4", "vq8_rot_gptq_vs_scalar3_g64_rot_gptq", "VQ8 vs scalar g64 (0.8B)"),
            ("q27", "vq8_rot_gptq_vs_scalar3_g64_rot_gptq", "VQ8 vs scalar g64 (27B)"),
        ], "Primary and secondary contrasts, matched recipe across checkpoints. "
           "The 0.8B rows come from the GPTQ run, not the earlier uncompensated one.",
           "tab:primary"),
    }
    for name, body in files.items():
        (OUT / name).write_text(body)
        print(f"wrote {name} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
