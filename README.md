# Low-Rate Quantization of a Hybrid Language Model

Code, saved run artifacts and the full running record for the paper
**"Low-Rate Quantization of a Hybrid Language Model: What Moves Loss, and Where
the Bytes Actually Are"** (Shayan Hemmatiyan).

A controlled post-training weight-quantization study below two bits, on one 0.8B
and one 27B hybrid linear-attention checkpoint. No quantization-aware training,
no new algorithm claimed, and no quantum hardware involved despite the
repository's name — the project began from a quantum-inspired polar weight code,
that line produced a negative result, and it was replaced. The dead end is kept
in the record rather than deleted.

## What the study found

- **Byte allocation dominates codec design at this scale.** Quantizing the tied
  embedding costs `0.000098` NLL per 100 MB saved against `0.020873` for the
  first step of the weight codec and `2.127315` for its last. A model with
  2.792-bit weights and an INT4 embedding is half the size and takes a fifth of
  the damage of one with ternary weights and a BF16 embedding. This follows from
  parameter shares (the embedding is 33.8% of the 0.8B text tower, 9.45% of the
  27B one), so it inverts with scale.
- **Calibration is a lever that costs no bytes.** Damage is minimized when
  calibration and evaluation domains match; a prespecified 50/50 mixture at the
  same token budget cuts worst-domain damage below either pure condition, and a
  half-budget control shows this is coverage of activation directions rather
  than sample count.
- **The codec results hold, and are smaller than both of the above.** A shared
  eight-dimensional codebook beats a learned three-level scalar code at matched
  bytes; GPTQ-style compensation removes 68–69% of the damage while *raising*
  weight MSE; rate dominates compensation and coverage dominates rate.
- **The converted model is served on compressed weights**, reproducing its
  decoded-BF16 loss to `1.6e-05` at `1.92x` less resident memory.

## Layout

| path | contents |
|---|---|
| `paper/main.tex` | the manuscript; builds with `tectonic main.tex` |
| `paper/fig_method.tex` | the method diagram (TikZ, hand-written) |
| `paper/generated/` | every table and figure, generated from `results/` — never hand-written |
| `paper/ternary_proposal.md` | the running record, §§1–23, including withdrawn claims |
| `experiments/` | the 35 runners, kernels and tests the paper rests on |
| `experiments/archive/` | 53 scripts from abandoned directions, kept for the negative results |
| `paper/archive/` | superseded drafts and table fragments |
| `results/` | saved JSON and summaries for every run the paper cites |

## Reproducing the tables and figures

```bash
cd experiments && python make_tables_v2.py && python make_figures_v1.py
```

Both read `results/*/results.json`. The table generator refuses to emit a table
whose sources disagree on the BF16 baseline or the evaluation stream, and every
row and every series names the run it came from. Nothing in either is typed in
by hand.

Building the paper needs only `main.tex`, `fig_method.tex` and `generated/`:

```bash
cd paper && tectonic main.tex
```

## Notes on reading the record

`paper/ternary_proposal.md` is a chronological log, not a summary. Several
sections correct or withdraw earlier ones — the additivity claim was reversed by
a frozen-artifact test, a cancellation explanation was falsified across refits,
a resident-byte count was found to charge FP16 for FP32 tensors, and a
peak-memory comparison was found to be measuring the benchmark harness. Where a
section is superseded it says so and points forward. The paper states the
corrected positions.

Runs targeted a single RTX 3090; the remote workstation address is read from
`$QQ_WORKSTATION` and is not recorded here.

## Licence

Code and artifacts: MIT (see `LICENSE`). The manuscript text is the author's.
