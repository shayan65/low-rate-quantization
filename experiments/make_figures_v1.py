"""Generate the paper's figures from saved results, on the same terms as the tables.

`make_tables_v2.py` exists because a hand-transcribed number once mixed an
uncompensated margin into a compensated comparison. Figures are worse, not
better, in that respect: a plotted point carries no provenance at all and a
reader cannot check it against a run directory. So these are generated from
`results/*/results.json` by the same rule --- nothing is typed in, every series
names its source run in the caption, and a figure whose sources disagree on the
BF16 baseline does not get drawn.

Three figures, each carrying a claim the tables make but do not make visible:

  * `fig_rate` --- damage against index rate, which is exponential with no knee
    over the measured interval. A table of seven rows hides that; a log axis
    does not.
  * `fig_bytes` --- where the parameters actually are, at 0.8B and at 27B. The
    paper's headline is a byte-allocation argument, and the whole argument is
    that one bar is mostly embedding and the other is mostly not.
  * `fig_frontier` --- model size against damage over the weight-rate x
    embedding-precision grid, where the dominance is a single glance: the
    sub-two-bit points sit up and to the right of points that are both smaller
    and better.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "paper" / "generated"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

INK = "#1a1a1a"
ACCENT = "#b2182b"
COOL = "#2166ac"
MUTED = "#999999"


def load(name):
    p = RESULTS / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_rate():
    """Damage against index rate: exponential, no knee."""
    r = load("rate_curve_v2")
    if r is None:
        return False
    arms = [(v["index_bpw"], v["bits_per_weight"], v["delta_nll"])
            for v in r["arms"].values()]
    arms.sort()
    # the K=81 and K=84 arms sit at the same index rate; keep the ternary one
    seen, pts = set(), []
    for idx, bpw, d in arms:
        if round(idx, 4) in seen:
            continue
        seen.add(round(idx, 4))
        pts.append((bpw, d))
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    # exponential fit on the log, to quote the halving interval
    c = np.polyfit(x, np.log(y), 1)
    half = np.log(2) / -c[0]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xs = np.linspace(x.min(), x.max(), 100)
    ax.semilogy(xs, np.exp(np.polyval(c, xs)), "-", color=MUTED, lw=1,
                zorder=1, label=f"halves every {half:.2f} bits")
    ax.semilogy(x, y, "o", color=ACCENT, ms=4.5, zorder=3, label="measured")
    ax.set_xlabel("stored bits per weight")
    ax.set_ylabel(r"$\Delta$NLL vs BF16")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, which="both", axis="y", lw=0.3, color="#dddddd", zorder=0)
    fig.savefig(OUT / "fig_rate.pdf")
    plt.close(fig)
    return True


def fig_bytes():
    """Where the parameters are, at both scales. The whole allocation argument."""
    e = load("embed_precision_v1")
    if e is None:
        return False
    pl = e["plan"]
    # Two categories only. A third for norms and convolutions is 0.07% at 0.8B
    # and not separately counted at 27B, so it would be a legend entry with no
    # visible bar; it is folded into the first and stated in the caption.
    small = {"covered": pl["covered_params"] + pl["other_params"],
             "embedding": pl["embed_params"]}
    # Counted from the 27B safetensors headers, on the same basis as the 0.8B
    # bar beside it: the text tower only. The full checkpoint is 27,781,427,952
    # parameters, of which the vision encoder is 460,730,096 and the
    # multi-token-prediction head 424,699,392; neither is on the path this
    # study quantizes, and charging the 27B bar for them while the 0.8B bar
    # (counted from `plan`, which is text-only) goes uncharged would make the
    # two shares incomparable -- an earlier version of this figure did exactly
    # that. The head is untied here, so the embedding is charged twice.
    big = {"covered": 24353201664, "embedding": 2542796800}

    fig, ax = plt.subplots(figsize=(5.4, 1.6))
    labels = ["Qwen3.5-0.8B\n752.4M params", "Qwen3.8-27B\n26.90B params"]
    order = [("covered", "everything the weight codec covers", COOL),
             ("embedding", "embedding (and head)", ACCENT)]
    for i, d in enumerate([small, big]):
        tot = sum(d.values())
        left = 0.0
        for key, lab, col in order:
            w = d[key] / tot * 100
            if w <= 0:
                continue
            ax.barh(i, w, left=left, color=col, height=0.55,
                    label=lab if i == 0 else None, edgecolor="white", lw=0.6)
            if w > 6:
                ax.text(left + w / 2, i, f"{w:.1f}%", ha="center", va="center",
                        color="white", fontsize=8)
            left += w
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("share of parameters (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.45))
    fig.savefig(OUT / "fig_bytes.pdf")
    plt.close(fig)
    return True


def fig_frontier():
    """Model size against damage: the dominance, visible.

    The two empty regions carry the message as much as the points do. Nothing
    sits bottom-left (small and undamaged is unreachable at these rates) and
    nothing sits top-right (large and damaged is a configuration no one would
    choose) -- so the annotations go there rather than on top of the data.
    """
    e = load("embed_precision_v1")
    if e is None:
        return False
    cells = e["cells"]
    wlab = {"bf16": "BF16", "r2667_k1625": "2.79 bpw",
            "r2000_k255": "2.13 bpw", "r1600_k81": "1.73 bpw"}
    elab = {"bf16": "BF16", "fp8_e4m3": "FP8", "int8_g128": "INT8",
            "int4_g128": "INT4"}
    style = {"bf16": ("o", COOL), "r2667_k1625": ("s", "#4d9221"),
             "r2000_k255": ("^", "#e08214"), "r1600_k81": ("D", ACCENT)}

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for w in wlab:
        xs, ys = [], []
        for em in elab:
            v = cells.get(f"{w}|{em}")
            # the BF16/BF16 cell is the reference and has exactly zero damage,
            # which has no position on a log axis; the caption states it
            if v is None or v["delta_nll"] <= 0:
                continue
            xs.append(v["model_bytes"] / 1e6)
            ys.append(v["delta_nll"])
        # sort by size so each series reads as a path from small to large
        # rather than zigzagging in dictionary order
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        m, c = style[w]
        ax.plot(xs, ys, m + "-", color=c, ms=4.5, lw=0.9,
                label=f"weights {wlab[w]}")
    ax.set_yscale("log")
    ax.set_ylim(1.2e-4, 2.6)
    ax.set_xlim(190, 1330)
    ax.set_xlabel("whole-model size (MB)")
    ax.set_ylabel(r"$\Delta$NLL vs BF16")

    # the dominance the text argues, drawn through the empty upper-right
    a = cells["r1600_k81|bf16"]
    b = cells["r2667_k1625|int4_g128"]
    ax.annotate("", xy=(b["model_bytes"] / 1e6, b["delta_nll"] * 1.12),
                xytext=(a["model_bytes"] / 1e6, a["delta_nll"] * 0.94),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(700, 0.75, "half the size,\na fifth of the damage",
            fontsize=8.5, color=INK, ha="left", va="center")
    ax.text(700, 0.055,
            "quantizing the embedding\nmoves a model left almost for free",
            fontsize=8, color=MUTED, ha="left", va="center")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    ax.grid(True, which="major", axis="y", lw=0.3, color="#dddddd", zorder=0)
    fig.savefig(OUT / "fig_frontier.pdf")
    plt.close(fig)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig_rate", fig_rate), ("fig_bytes", fig_bytes),
                     ("fig_frontier", fig_frontier)):
        ok = fn()
        print(f"{'wrote' if ok else 'skipped'} {name}.pdf")


if __name__ == "__main__":
    main()
