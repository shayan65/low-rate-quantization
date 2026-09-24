# arXiv submission

Everything the arXiv form asks for, in the order it asks for it. Copy the
fields; do not retype them. The abstract in particular runs against arXiv's
1920-character limit — it was 2551 characters before being cut for it, and has
45 to spare now — so change `main.tex` and regenerate the text below rather
than editing it in place:

```bash
./abstract_plain.py
```

That converts the abstract out of `main.tex` rather than transcribing it, and
reports how many characters are left. `--check` fails if it no longer fits, or
if the copy pasted below has drifted from the paper.

## The tarball

```bash
./make_arxiv.sh
```

Writes `arxiv-submission.tar.gz`. The script copies the declared file list into
a scratch directory and builds *there* before writing anything, so a file that
exists in the working tree but is missing from the list fails locally instead of
on arXiv. It refuses to produce a tarball if that build reports an undefined
reference or an over/underfull box.

Contents: `main.tex`, `fig_method.tex`, and `generated/` (14 tables, 3 figures).
The bibliography is a `thebibliography` environment inside `main.tex`, so there
is no `.bbl` or `.bib` to include. No custom class, no fonts, no shell-escape.

## Fields

**Title**

```
Low-Rate Quantization of a Hybrid Language Model: What Moves Loss, and Where the Bytes Actually Are
```

**Authors**

```
Shayan Hemmatiyan
```

**Abstract** (1875 characters)

```
We study post-training weight quantization below two bits in a hybrid linear-attention language model, with storage measured from bytes written. Three findings outrank the codec comparisons that motivated it. First, byte allocation dominates codec design at this scale: quantizing the tied embedding costs 0.000098 NLL per 100 MB saved against 0.020873 and 2.127315 for the codec's first and last steps, so 2.792-bit weights with an INT4 embedding are half the size and a fifth of the damage of ternary weights with a BF16 embedding, and every sub-two-bit configuration we measure is off the whole-model-size frontier. That is about parameter shares (33.8% of this text tower, 9.45% of the 27B one), so it inverts with scale. Second, the calibration distribution is a lever that costs no bytes: a prespecified 50/50 domain mixture at equal token budget cuts worst-domain damage below either pure condition on three disjoint draws; a half-budget control shows the mechanism is direction coverage, not sample count. Third, the codec results hold but are smaller: on 0.8B DeltaNet projections a shared eight-dimensional codebook beats a learned three-level scalar code by 0.036118 NLL [0.033768, 0.038414] at 0.43% more payload bytes, keeping its sign under eight refits; GPTQ-style compensation removes 68-69% of the damage while raising weight MSE; rate then dominates compensation and coverage dominates rate, yet a Hessian-derived ranking built to exploit coverage loses to random selection. The direction reproduces at 27B, below the refit spread. Finally, the converted model is served on compressed weights: all 186 non-embedding matrices become code-and-codebook modules, reproducing decoded-BF16 loss to 1.6e-5 at 1.92x less resident memory and 3.0x cuBLAS latency once the rotation is fused. We claim no new algorithm and separate what is established from what is not.
```

**Comments**

```
20 pages, 4 figures, 14 tables. Code, saved run artifacts and the running record of the study: https://github.com/shayan65/low-rate-quantization
```

**Categories**

- Primary: `cs.LG` (Machine Learning)
- Cross-list: `cs.CL` (Computation and Language)

The account default is `cs.AI`; change it on the form. The work is an empirical
quantization study, which is where `cs.LG` readers look, and it is about a
language model, which is the cross-list.

**License**

Creative Commons Attribution (CC BY 4.0), matching the MIT licence on the code.
The choice is irrevocable per version.

**MSC / ACM class**

Leave blank.

## Before clicking submit

- `make_arxiv.sh` exited cleanly (it will not write the tarball otherwise).
- The PDF arXiv generates from the tarball is the one to check, not `main.pdf`
  from the working tree — they are built by different TeX distributions.
- The repository URL in the Availability section and in the Comments field
  resolves.
