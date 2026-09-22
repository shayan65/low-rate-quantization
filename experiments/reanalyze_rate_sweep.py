"""Recompute historical summaries without importing torch or running a fit."""

import argparse
import hashlib
import json
from pathlib import Path

from rate_sweep_analysis import DEFAULT_TENSORS, aggregate_rows, evaluate_gates, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/rate_sweep_v1/results.json"))
    parser.add_argument("--out", type=Path, default=Path("results/rate_sweep_v1_reanalysis"))
    parser.add_argument("--expected-tensors", nargs="+", help="MODEL:TENSOR manifest; default is original six tensors")
    args = parser.parse_args()
    source_path = args.input.resolve()
    output_path = (args.out / "results.json").resolve()
    if source_path == output_path:
        parser.error("Reanalysis must not overwrite the original results")
    original = args.input.read_bytes()
    rows = json.loads(original)["rows"]
    expected = [tuple(spec.split(":", 1)) for spec in args.expected_tensors] if args.expected_tensors else DEFAULT_TENSORS
    if any(len(spec) != 2 for spec in expected):
        parser.error("Expected tensor specifications require MODEL:TENSOR")
    gates = evaluate_gates(rows, expected)
    aggregates = aggregate_rows(rows, expected)
    provenance = {
        "source": str(args.input), "source_sha256": hashlib.sha256(original).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).with_name("rate_sweep_analysis.py").read_bytes()).hexdigest(),
        "expected_tensors": expected, "new_fitting_or_gpu_work": False,
        "scope": "Corrected grouping and gates of historical FP32 reconstruction results; no packed-size or LM-quality claim",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"provenance": provenance, "rows": rows, "gates": gates,
                                       "aggregates": aggregates}, indent=2, allow_nan=False) + "\n")
    (args.out / "summary.md").write_text(render(rows, gates, aggregates) +
        f"\nSource: `{args.input}`\n\nSource SHA-256: `{provenance['source_sha256']}`\n")
    print(json.dumps({"out": str(args.out), "coverage": gates["coverage"], "verdict": gates["verdict"]}, indent=2))
    return 0 if gates["coverage"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
