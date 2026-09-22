# Full WikiText-103 launch status

Status: running on shayan@192.168.1.26 (RTX 3090). No full-run results have completed at this snapshot.

Snapshot UTC: 2026-09-12T03:24:58.889560+00:00

Pinned Salesforce/wikitext revision: `b08601e04326c79dfdd32d625aee71d232d685c3`. Subset: `wikitext-103-raw-v1`.

| Split | Source rows | BPE tokens | Next-token targets per complete pass |
|---|---:|---:|---:|
| train | 1,801,350 | 135,350,275 | 135,350,274 |
| validation | 3,760 | 282,592 | 282,591 |
| test | 4,358 | 323,000 | 322,999 |

Train-only 8,192-token ByteLevel BPE; every source row retained with a trailing newline. Full training covers every target once per epoch, shuffled as contiguous context blocks. Full validation and test use every target, including the tail, with context reset every 512 tokens. There is no dataset subsampling and no synthetic training/evaluation corpus in this sweep. Synthetic tensors in implementation assertions are tests, not scientific results.

Grid: 51 jobs, seeds [0, 1, 2], 3 full epochs each, width 384, depth 6, context 512, batch 16. The 12 hybrid methods and five Transformer methods are enumerated in plan.json.

The first run is hybrid/continuous/seed 0. Early runtime suggests roughly 1–2 days for the grid, subject to actual run times. Run-specific status.json and train.log are authoritative; local files are snapshots.

Workstation root: `/home/shayan/quantum_quantization/results/wikitext103_full_v1`. Sweep log: `/home/shayan/quantum_quantization/results/wikitext103_full_v1_sweep.log`. Outer sweep PID at launch: 23935.

Every run retains config.json, the exact training source snapshot/hash, an optimizer resume checkpoint, best validation checkpoint, epoch records with observed target counts, and a decoded packed deployment export. `result.json` and COMPLETE are produced only after full test evaluation. `run_full_sweep.py --collect` verifies coverage and result provenance before reporting completed metrics. No values are imputed for pending or failed runs.

Independent source-row/token/text audit PASSED for every row in all three splits; see data_audit.json. Every stored row was re-encoded and decoded losslessly against the pinned original source. The GPU pipeline passed causal FFT/loop equivalence, finite gradients for all methods, circular nearest-code tests, odd-size bit packing, export reload equivalence, and exact target coverage including partial final blocks.

Scope: full-data weight-codebook study with real uniform-QAT controls. It does not establish quantum advantage or complete SAWB, learned-codebook, temporal calibration, memory-task, or Qwen validation experiments. Earlier sampled-system simulations and partial-validation smoke results are excluded from the full-data summary.

The separate full-pipeline preflight on WikiText-2 completed one full training epoch (4,256,725 targets), full validation (445,727 targets), and full test (504,896 targets) on a 40,614-parameter 3-bit-phase model, with decoded packed export evaluation. Those preflight metrics are excluded from the WikiText-103 results.
