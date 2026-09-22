# Screening interpretation and resource policy

All 12 small jobs completed on real WikiText-103 samples. Each used 2,000 updates and one of three seeds; these are sample validation scores, not full-dataset scores. All exact coefficient/logit checks passed after the encoder repair. The test split was never loaded.

4-bit magnitude plus 4-bit phase loses only 0.49% mean validation perplexity against unquantized per-weight polar coordinates, but is 9.66% worse than real 8-bit. This suggests the quantization penalty is small in this short test; it does not prove the polar architecture or coordinate optimizer is competitive. The real baseline is still insufficiently trained under the declared NLL<6 screening threshold. The exploratory promotion gate failed on both the real8 margin and training-sufficiency condition.

No full experiment is running or automatically queued. The larger polar sweep now refuses to launch by default and also requires a passing screen report. A passing small screen would warrant a second limited validation stage, not hours of GPU training.

Next discriminating low-cost work: compare unquantized Cartesian complex weights with unquantized polar coordinates, and calibrate learning rates on the small validation sample. Keep data blocks/seed lists and runtime caps fixed. This distinguishes coordinate optimization from complex architectural capacity before changing magnitude or phase bit allocation.

Resource record: the screen stopped on an export discrepancy before finishing, preserving checkpoints. The repair reused completed training; one checkpoint had only export/evaluation recovered. Model classes and all non-export helper/function definitions were independently verified identical via AST. Original model/source snapshots, repaired exporter snapshot, per-seed metrics and exact-export audit are retained. A recovered job has no invented original training-time number. Two failed prototype invocations and a short full-pipeline preflight are excluded from the statistical table.
