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
    "rate": ("rate_curve_v2", "0.8B, rate sweep incl. four bits"),
    "q27": ("qwen38_confirm_v1", "27B, GPTQ recipe"),
    "q27b": ("qwen38_confirm_v2", "27B, GPTQ recipe, frozen full stream"),
    "refit": ("calib_variability_v1", "0.8B, 8 refits"),
    "gen": ("generalization_v1", "0.8B, fixed quantizer, varied evaluation"),
    "full": ("fullmodel_v1", "0.8B, every non-embedding matrix"),
    "mech2": ("mechanism_v2", "0.8B, calibration mixture and the g64 factorial"),
    "inter": ("interaction_v1", "0.8B, frozen-artifact coverage interaction"),
    "mix": ("calib_mix_v2", "0.8B, calibration mixture, 3 draws + budget control"),
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
          f"\\caption{{{caption} Source: \\texttt{{results/{esc(RUNS[key][0])}}} ({RUNS[key][1]}); "
          f"BF16 NLL {r['bf16']['nll']:.6f}.}}",
          f"\\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(L)


SHORT = {"v3": "0.8B/no-comp", "v4": "0.8B/GPTQ", "mlp": "0.8B-MLP/GPTQ",
         "alloc": "0.8B/alloc", "rate": "0.8B/rate", "q27": "27B/pilot",
         "q27b": "27B/frozen", "refit": "0.8B/refit", "gen": "0.8B/gen"}


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
        L.append(f"{esc(human)} & {esc(SHORT[key])} & ${c['delta']:+.6f}$ & "
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
          f"dimension 4 throughout so rate varies alone; six rates from 1.600 to "
          f"4.000 index bits. Stored-byte range "
          f"{hi / lo:.2f}$\\times$. Source: \\texttt{{results/{esc(RUNS['rate'][0])}}}; "
          f"BF16 NLL {r['bf16']['nll']:.6f}.}}",
          r"\label{tab:rate}", r"\end{table}", ""]
    return "\n".join(L)


def refit_table():
    """Text-bootstrap half-width beside refit ranges -- never divided.

    The two are different statistics; an earlier draft quoted their quotient as
    an uncertainty inflation factor, which has no coverage interpretation.
    Calibration size is a condition change, not a further sample, so it is
    labelled separately.
    """
    r = load("refit")
    if r is None:
        return "% calib_variability_v1 not present\n"
    sm = r["summary"]
    nice = {"vq8_vs_scalar3": "VQ8 vs scalar", "vq8_vs_scalar3_g64": "VQ8 vs scalar g64",
            "vq8_vs_vq4": "VQ8 vs VQ4", "scalar3_g64_vs_scalar3": "g64 vs g128"}
    keys = ["calibration draw (3 disjoint, 65k)", "codebook seed (3, fixed calibration)",
            "calibration size (16k-131k, nested)"]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Contrast & bootstrap half-width & range: draws & seeds & size$^{*}$ \\",
         r"\midrule"]
    for ck, label in nice.items():
        if ck not in sm:
            continue
        f = sm[ck]["factors"]
        L.append(f"{esc(label)} & {sm[ck]['median_half_width']:.6f} & "
                 + " & ".join(f"{f[k]['spread']:.6f}" for k in keys if k in f) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Conditional text-bootstrap half-width beside the range of each "
          r"contrast over refits. These are different statistics and are not divided: "
          r"a range over a few point estimates has no coverage interpretation. "
          r"$^{*}$Calibration size changes the experimental condition rather than "
          r"resampling it. The zero for seed on the scalar-only contrast is a harness "
          r"check: both arms use a deterministic Lloyd fit, so a seed cannot move it. "
          f"Source: \\texttt{{results/{esc(RUNS['refit'][0])}}}.}}",
          r"\label{tab:refit}", r"\end{table}", ""]
    return "\n".join(L)


def generalization_table():
    """One fitted quantizer, five evaluation configurations."""
    r = load("gen")
    if r is None:
        return "% generalization_v1 not present\n"
    ev = r["plan"]["evals"]
    order = [k for k in ("wt2_val_128", "wt2_test_128", "tiny_128",
                         "wt2_test_512", "wt2_test_2048") if k in ev]
    nice = {"vq8_vs_scalar3": "VQ8 vs scalar", "vq8_vs_scalar3_g64": "VQ8 vs scalar g64",
            "vq8_vs_vq4": "VQ8 vs VQ4", "scalar3_g64_vs_scalar3": "g64 vs g128"}
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "r" * len(order) + "}", r"\toprule",
         "Contrast & " + " & ".join(esc(o) for o in order) + r" \\", r"\midrule"]
    for ck, label in nice.items():
        if ck not in r.get("contrasts", {}):
            continue
        row = r["contrasts"][ck]
        cells = []
        for o in order:
            v = row[o]
            star = "" if (v["ci"][0] > 0) == (v["ci"][1] > 0) else r"$^{\dagger}$"
            cells.append(f"${v['delta']:+.6f}${star}")
        L.append(f"{esc(label)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{One fitted quantizer; only the evaluation varies. These are two "
          r"corpora and three context lengths, not five independent replications --- the "
          r"three \texttt{wt2\_test} columns share their text and differ only in "
          r"blocking. $\dagger$ marks an interval including zero. "
          f"Source: \\texttt{{results/{esc(RUNS['gen'][0])}}}.}}",
          r"\label{tab:gen}", r"\end{table}", ""]
    return "\n".join(L)


def coverage_table():
    """Protocol and coverage, the table the review asked to lead with."""
    rows = []
    for key in ("v4", "mlp", "alloc", "rate", "q27", "q27b"):
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


def fullmodel_table():
    """Full coverage, and the additive reference that kills super-linearity."""
    r = load("full")
    if r is None:
        return ""
    pl = r["plan"]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrrl}", r"\toprule",
         r"Arm & covered bpw & model MB & effective bpw & shrink & $\Delta$NLL \\",
         r"\midrule"]
    for nm, a in r["arms"].items():
        L.append(f"\\texttt{{{esc(nm)}}} & {a['covered_bits_per_weight']:.4f} & "
                 f"{a['model_bytes'] / 1e6:.1f} & "
                 f"{a['effective_bits_per_weight']:.3f} & "
                 f"{a['model_shrink_vs_bf16']:.2f}$\\times$ & "
                 f"${a['delta_nll']:+.6f}$ {fmt_ci(a['delta_ci'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Every two-dimensional weight outside the embedding of "
          f"Qwen3.5-0.8B: {pl['tensors']} tensors, {pl['covered_params']:,} of "
          f"{pl['total_params']:,} parameters "
          f"({pl['coverage'] * 100:.1f}\\%), from \\texttt{{{esc(RUNS['full'][0])}}}. "
          r"\texttt{effective bpw} includes the BF16 embedding, which "
          r"\texttt{tie\_word\_embeddings} also makes the output head.}",
          r"\label{tab:fullmodel}", r"\end{table}", ""]
    return "\n".join(L)


def additivity_table():
    """Damage against coverage at one rate, with the additive reference.

    Four runs share a BF16 baseline, an evaluation stream and a rate, which is
    what makes the comparison legitimate; the table asserts that rather than
    leaving it to the reader, and refuses to print if it stops being true.
    """
    src = [("v4", "vq4_rot_gptq", "in\\_proj\\_qkv (18)", 113246208, "bits_per_weight"),
           ("mlp", "mlp_vq4_rot_gptq", "MLP (72)", 264241152, "bits_per_weight"),
           ("alloc", "all_low", "both (90)", 377487360, "bits_per_weight"),
           ("full", "r1600_k81", "all non-embedding (186)", None,
            "covered_bits_per_weight")]
    rows, base, targets = [], None, None
    for key, arm, label, params, bitkey in src:
        r = load(key)
        if r is None:
            return ""
        # The BF16 NLL is a float computed over the entire evaluation stream,
        # so agreement to full double precision is a far stronger fingerprint
        # than a matching target count -- which an earlier version of this
        # function checked, and then skipped entirely for the full-model row.
        b = r["bf16"]["nll"]
        base = b if base is None else base
        assert b == base, f"{key}: BF16 baseline differs ({b!r} vs {base!r})"
        t = r["plan"]["eval_targets"]
        targets = t if targets is None else targets
        assert t == targets, f"{key}: evaluation target count differs"
        a = r["arms"][arm]
        rows.append((RUNS[key][0], label,
                     params if params is not None else r["plan"]["covered_params"],
                     a[bitkey], a["delta_nll"]))
    additive = rows[0][4] + rows[1][4]
    measured = rows[2][4]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Run & target set & params & bpw & $\Delta$NLL & avg.\ per 100M \\",
         r"\midrule"]
    for nm, label, params, bpw, d in rows:
        L.append(f"\\texttt{{{esc(nm)}}} & {label} & {params:,} & {bpw:.4f} & "
                 f"${d:+.6f}$ & ${d / params * 1e8:.3f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Damage against coverage at ternary rate. Every row is "
          r"asserted at generation time to share the same BF16 baseline to full "
          f"double precision (${base:.6f}$) and the same target count "
          f"({targets:,}); a matching count alone would not establish stream "
          f"identity, but a loss computed over the whole stream agreeing in all "
          f"digits does. The two disjoint families give an apparent additive "
          f"reference, ${rows[0][4]:.6f} + {rows[1][4]:.6f} = {additive:.6f}$ "
          f"against a measured ${measured:.6f}$, but these rows fit their "
          f"codebooks on different pools; the frozen-artifact test reports an "
          f"interaction of $+0.003818$ $[+0.000769,+0.006835]$, so damage is "
          f"slightly super-additive. The "
          r"jump at full coverage is composition, not breadth: going from the "
          f"90-tensor set to all 186 adds {rows[3][2] - rows[2][2]:,} parameters "
          f"for ${rows[3][4] - rows[2][4]:+.6f}$, i.e.\\ "
          f"${(rows[3][4] - rows[2][4]) / (rows[3][2] - rows[2][2]) * 1e8:.3f}$ per "
          f"100M, the highest marginal cost in the model. The last column is an "
          r"average over each row's own target set, not a marginal rate.}"
          if len(rows) > 3 else r"}",
          r"\label{tab:additivity}", r"\end{table}", ""]
    return "\n".join(L)


def calibration_table():
    """Calibration domain x evaluation domain, including the 50/50 mixture."""
    r = load("mech2")
    if r is None or "M_calibration" not in r:
        return ""
    M = r["M_calibration"]
    cals = ["wikitext", "tinystories", "mixed50"]
    evals = ["wt2_test", "tinystories"]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Calibration & code & wikitext-2 test & TinyStories & worst \\",
         r"\midrule"]
    for c in cals:
        for arm in ("scalar3", "vq8"):
            cells = [M["cells"].get(f"{c}|{arm}|{e}") for e in evals]
            if any(x is None for x in cells):
                continue
            vals = [x["delta_nll"] for x in cells]
            L.append(f"{esc(c)} & {arm} & " +
                     " & ".join(f"${v:+.6f}$" for v in vals) +
                     f" & $\\mathbf{{{max(vals):+.6f}}}$ \\\\")
        if c != cals[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Calibration domain crossed with evaluation domain, "
          r"identical recipe, storage and evaluated text throughout, from "
          f"\\texttt{{{esc(RUNS['mech2'][0])}}} "
          f"({r['plan']['cal_tokens']:,} calibration tokens in every condition). "
          r"\texttt{mixed50} spends half its budget on each domain and is "
          r"prespecified at 50/50; it is an exploratory intervention, designed "
          r"after the pure conditions were seen. The last column is the "
          r"worst-domain damage, which is what a mixture is meant to reduce.}",
          r"\label{tab:calibration}", r"\end{table}", ""]
    return "\n".join(L)


def g64_table():
    """The grouping/codebook 2x2, read as a factorial rather than a diagonal."""
    r = load("mech2")
    if r is None or "G_g64_factorial" not in r:
        return ""
    G = r["G_g64_factorial"]
    named = [("grouping_at_fixed_g128_book", "grouping (g128 book fixed)"),
             ("grouping_at_fixed_g64_book", "grouping (g64 book fixed)"),
             ("codebook_at_fixed_g128_grouping", "codebook (g128 grouping fixed)"),
             ("codebook_at_fixed_g64_grouping", "codebook (g64 grouping fixed)"),
             ("confounded_diagonal", "\\emph{both at once (the diagonal)}")]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrl}", r"\toprule",
         r"Contrast & $\Delta$NLL & 95\% CI \\", r"\midrule"]
    for key, label in named:
        c = G["factorial"].get(key)
        if not isinstance(c, dict):
            continue
        L.append(f"{label} & ${c['delta']:+.6f}$ & {fmt_ci(c['ci'])} \\\\")
    f = G["factorial"]
    L += [r"\midrule",
          f"main effect, grouping & ${f['main_effect_grouping']:+.6f}$ & --- \\\\",
          "main effect, codebook & $"
          f"{f['main_effect_codebook_g64book_minus_g128book']:+.6f}$ & --- \\\\",
          f"interaction & ${f['interaction']:+.6f}$ & --- \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{The scalar group-size control is a $2\times2$ over grouping "
          r"and codebook, not a single knob, from "
          f"\\texttt{{{esc(RUNS['mech2'][0])}}}. Moving to g64 at a fixed "
          r"alphabet hurts; refitting the codebook in the new normalized space "
          r"helps by a similar amount. An earlier draft read the diagonal as "
          r"``group size alone'' and concluded grouping was irrelevant; the two "
          r"effects are of the same order and opposed, which is why the "
          r"end-to-end contrast is small and unstable. Main effects and the "
          r"interaction are differences of point estimates and carry no "
          r"interval.}",
          r"\label{tab:g64}", r"\end{table}", ""]
    return "\n".join(L)


def interaction_table():
    """Coverage interaction with the codec held bit-identical across arms."""
    r = load("inter")
    if r is None or "interaction" not in r:
        return ""
    pl, it = r["plan"], r["interaction"]
    lab = {"A_qkv": "A --- \\texttt{in\\_proj\\_qkv}",
           "B_mlp": "B --- MLP", "AB_both": "AB --- both"}
    par = {"A_qkv": pl["qkv_params"], "B_mlp": pl["mlp_params"],
           "AB_both": pl["qkv_params"] + pl["mlp_params"]}
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrl}", r"\toprule",
         r"Arm & tensors & params & $\Delta$NLL & 95\% CI \\", r"\midrule"]
    for k, v in r["arms"].items():
        L.append(f"{lab.get(k, esc(k))} & {v['tensors']} & {par[k]:,} & "
                 f"${v['delta_nll']:+.6f}$ & {fmt_ci(v['delta_ci'])} \\\\")
    L += [r"\midrule",
          f"interaction & & & ${it['value']:+.6f}$ & {fmt_ci(it['ci'])} \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Does coverage damage compose additively? One Hessian "
          r"capture, one codebook fitted over all 90 tensors and one "
          f"quantization pass at {pl['mean_bits_per_weight']:.4f} bits/weight, "
          r"then the dequantized weights installed in three combinations, so "
          r"every tensor carries bit-identical weights in every arm that "
          r"includes it. The interaction is "
          r"$L_{AB}-L_A-L_B+L_{\mathrm{BF16}}$, bootstrapped over the same "
          f"resampled blocks. It excludes zero, so damage is slightly "
          f"\\emph{{super}}-additive --- though at ${it['value']:.6f}$ against a "
          f"total of ${it['measured_both']:.6f}$ it is "
          f"{it['value'] / it['measured_both'] * 100:.1f}\\% of the effect, so "
          r"``approximately additive'' remains a fair description. From "
          f"\\texttt{{{esc(RUNS['inter'][0])}}}.}}",
          r"\label{tab:interaction}", r"\end{table}", ""]
    return "\n".join(L)


def mixture_table():
    """Worst-domain damage by calibration condition, with the budget control."""
    r = load("mix")
    if r is None or "worst_domain_by_draw" not in r:
        return ""
    tok = r["plan"]["cal_tokens"]
    order = ["wikitext", "tinystories", "mixed50", "wikitext_half",
             "tinystories_half"]
    pretty = {"wikitext": "wikitext", "tinystories": "TinyStories",
              "mixed50": r"\textbf{mixed 50/50}",
              "wikitext_half": "wikitext, half budget",
              "tinystories_half": "TinyStories, half budget"}
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Calibration & code & tokens & mean worst-domain & range over draws \\",
         r"\midrule"]
    for arm in ("scalar3", "vq8"):
        for c in order:
            v = r["worst_domain_by_draw"].get(f"{c}|{arm}")
            if v is None:
                continue
            L.append(f"{pretty[c]} & {arm} & {tok[c]:,} & "
                     f"${v['mean']:.6f}$ & ${v['range']:.6f}$ \\\\")
        if arm == "scalar3":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Worst-domain damage by calibration condition, "
          f"{r['plan']['draws']} disjoint calibration draws each, from "
          f"\\texttt{{{esc(RUNS['mix'][0])}}}. The half-budget rows are the "
          r"control that separates domain coverage from sample count: for the "
          r"vector code, halving the wikitext budget costs $+0.008921$ while "
          r"spending that same halved budget on a second domain instead gains "
          r"$-0.060724$, so coverage is worth about seven times what sample "
          r"count is worth here. The $50/50$ split is prespecified, not tuned; "
          r"the line of work is exploratory because the crossed cells were "
          r"inspected first.}",
          r"\label{tab:mixture}", r"\end{table}", ""]
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "tab_coverage.tex": coverage_table(),
        "tab_fullmodel.tex": fullmodel_table(),
        "tab_additivity.tex": additivity_table(),
        "tab_interaction.tex": interaction_table(),
        "tab_calibration.tex": calibration_table(),
        "tab_mixture.tex": mixture_table(),
        "tab_g64.tex": g64_table(),
        "tab_rate.tex": rate_table(),
        "tab_v4_arms.tex": arm_table(
            "v4", "Codec and recipe ablation on the 0.8B DeltaNet QKV projections.",
            "tab:v4"),
        "tab_mlp_arms.tex": arm_table(
            "mlp", "The same recipe on all 72 MLP tensors of the 0.8B model.",
            "tab:mlp"),
        "tab_q27_arms.tex": arm_table(
            "q27b", "Scale transfer, frozen confirmation: 48 DeltaNet QKV "
            "projections of Qwen3.8-27B over the complete validation stream.",
            "tab:q27"),
        "tab_refit.tex": refit_table(),
        "tab_gen.tex": generalization_table(),
        "tab_primary.tex": comparison_table([
            ("v4", "vq8_rot_gptq_vs_scalar3_rot_gptq", "VQ8 vs scalar (0.8B)"),
            ("q27b", "vq8_rot_gptq_vs_scalar3_rot_gptq", "VQ8 vs scalar (27B)"),
            ("v4", "vq4_rot_gptq_vs_scalar3_rot_gptq", "VQ4 vs scalar (0.8B)"),
            ("q27b", "vq4_rot_gptq_vs_scalar3_rot_gptq", "VQ4 vs scalar (27B)"),
            ("v4", "vq8_rot_gptq_vs_scalar3_g64_rot_gptq", "VQ8 vs scalar g64 (0.8B)"),
            ("q27b", "vq8_rot_gptq_vs_scalar3_g64_rot_gptq", "VQ8 vs scalar g64 (27B)"),
            ("v4", "vq8_rot_gptq_vs_vq4_rot_gptq", "VQ8 vs VQ4 (0.8B)"),
            ("q27b", "vq8_rot_gptq_vs_vq4_rot_gptq", "VQ8 vs VQ4 (27B)"),
        ], "Primary and secondary contrasts, matched recipe across checkpoints. "
           "The 0.8B rows come from the GPTQ run, not the earlier uncompensated one.",
           "tab:primary"),
    }
    for name, body in files.items():
        if not body:
            print(f"skipped {name} (source run not present)")
            continue
        (OUT / name).write_text(body)
        print(f"wrote {name} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
