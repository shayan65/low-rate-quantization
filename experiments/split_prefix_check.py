"""Report the pilot prefix and the previously unevaluated remainder separately.

The 512-block pilot and the frozen full stream disagree: on the storage
dominance contrast the pilot gave -0.001736 (interval including zero) and the
full stream gives -0.007836 (excluding it), and the g64 contrast changes sign.
A four-fold change in a point estimate is more than sampling noise usually
explains, so the question is whether the first 512 blocks are simply unlike the
rest of the corpus.

The review asked for new text to be reported separately from the prefix as well
as in aggregate; `blocks.json` now stores per-block sums, counts and identifiers
for every arm, so this is pure post-processing with no further GPU time. If the
prefix and the remainder give consistent intervals, the difference is sampling;
if they do not, the prefix is unrepresentative and pilot-based conclusions on
this corpus should not be trusted.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def paired(a, b, ids, keep, reps=5000, seed=7):
    """Paired block bootstrap restricted to the blocks in `keep`."""
    sel = np.array([i for i, bid in enumerate(ids) if bid in keep])
    if sel.size == 0:
        return None
    sa = np.asarray(a["block_sums"])[sel]
    ca = np.asarray(a["block_counts"], dtype=float)[sel]
    sb = np.asarray(b["block_sums"])[sel]
    rng = np.random.default_rng(seed)
    n = sel.size
    idx = rng.integers(0, n, size=(reps, n))
    d = (sa[idx].sum(1) / ca[idx].sum(1)) - (sb[idx].sum(1) / ca[idx].sum(1))
    point = sa.sum() / ca.sum() - sb.sum() / ca.sum()
    return {"delta": float(point), "ci": [float(np.percentile(d, 2.5)),
                                          float(np.percentile(d, 97.5))],
            "blocks": int(n), "targets": int(ca.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../results/qwen38_confirm_v2")
    ap.add_argument("--prefix", type=int, default=512,
                    help="blocks the pilot used; the rest were never evaluated before")
    a = ap.parse_args()
    blocks = json.loads((Path(a.run) / "blocks.json").read_text())

    # every arm shares the same block ordering, keyed by identifier
    any_arm = next(k for k in blocks if k != "bf16")
    all_ids = set(blocks[any_arm]["block_ids"])
    pre = {i for i in all_ids if i < a.prefix}
    rest = {i for i in all_ids if i >= a.prefix}

    contrasts = [
        ("vq8_rot_gptq", "scalar3_rot_gptq", "PRIMARY  VQ8 vs scalar g128"),
        ("vq8_rot_gptq", "scalar3_g64_rot_gptq", "second.  VQ8 vs scalar g64 (more bytes)"),
        ("vq8_rot_gptq", "vq4_rot_gptq", "second.  VQ8 vs VQ4"),
        ("scalar3_g64_rot_gptq", "scalar3_rot_gptq", "context  g64 vs g128"),
    ]
    out = {"run": a.run, "prefix_blocks": a.prefix,
           "n_prefix": len(pre), "n_remainder": len(rest), "contrasts": {}}
    print(f"{'contrast':42s} {'split':10s} {'delta':>10s}  {'95% CI':<26s} {'targets':>8s}")
    print("-" * 102)
    for m, ref, label in contrasts:
        if m not in blocks or ref not in blocks:
            continue
        row = {}
        for nm, keep in (("prefix", pre), ("remainder", rest), ("full", all_ids)):
            r = paired(blocks[m], blocks[ref], blocks[m]["block_ids"], keep)
            if r is None:
                continue
            row[nm] = r
            ex = "excl 0" if (r["ci"][0] > 0) == (r["ci"][1] > 0) else "INCL 0"
            print(f"{label:42s} {nm:10s} {r['delta']:+10.6f}  "
                  f"[{r['ci'][0]:+.6f},{r['ci'][1]:+.6f}] {ex:7s} {r['targets']:8,}")
        out["contrasts"][f"{m}_vs_{ref}"] = row
        print()
    Path(a.run, "prefix_split.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
