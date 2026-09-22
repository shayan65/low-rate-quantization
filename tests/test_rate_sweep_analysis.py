"""Regression coverage for model collisions and incomplete research gates."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from rate_sweep_analysis import DIMS, LEVELS, MAX_K, aggregate_rows, evaluate_gates, render

MANIFEST = [("model-A", "same.tensor"), ("model-B", "same.tensor")]


def fixture():
    rows = []
    for model, tensor in MANIFEST:
        base = 10.0 if model == "model-A" else 100.0
        for rotation in ("none", "hadamard"):
            for levels in LEVELS:
                for dim in DIMS:
                    if levels ** dim > MAX_K:
                        continue
                    product = base * (0.95 if rotation == "hadamard" else 1.0)
                    margin = 0 if dim == 1 else 0.2
                    if rotation == "hadamard" and dim > 1:
                        margin = (0.3 if model == "model-A" else 0.18) if levels == 3 else 0.1
                    rows.append({"model": model, "tensor": tensor, "rotation": rotation,
                                 "levels": levels, "dim": dim, "codebook": levels ** dim,
                                 "product_grid_mse": product, "free_vq_mse": product * (1 - margin),
                                 "vq_over_product": 1 - margin})
    return rows


class RateSweepAnalysisTest(unittest.TestCase):
    def test_same_tensor_in_different_models_never_cross_matches(self):
        rows = fixture()
        gates = evaluate_gates(rows, MANIFEST)
        self.assertTrue(gates["coverage"]["complete"])
        self.assertEqual(len(gates["H0_rotation_closes"]), 2)
        for item in gates["H0_rotation_closes"]:
            self.assertAlmostEqual(item["fraction_closed_by_rotation"], 0.25)
        margins = {item["model"]: item for item in gates["H1_margins"]}
        self.assertTrue(margins["model-A"]["passes"])
        self.assertFalse(margins["model-B"]["passes"])
        self.assertFalse(gates["H1_passes"])
        report = render(rows, gates, aggregate_rows(rows, MANIFEST))
        self.assertIn("## model-A / same.tensor", report)
        self.assertIn("## model-B / same.tensor", report)

    def test_complete_single_model_can_pass(self):
        rows = [r for r in fixture() if r["model"] == "model-A"]
        self.assertTrue(evaluate_gates(rows, MANIFEST[:1])["H1_passes"])

    def test_missing_four_bit_comparator_fails_closed(self):
        rows = [r for r in fixture() if not (r["model"] == "model-B" and
                r["rotation"] == "hadamard" and r["levels"] == 16 and r["dim"] == 2)]
        gates = evaluate_gates(rows, MANIFEST)
        self.assertFalse(gates["H1_passes"])
        self.assertFalse(gates["coverage"]["complete"])
        self.assertEqual(len(gates["coverage"]["missing"]), 1)

    def test_entire_missing_tensor_fails_closed(self):
        rows = [r for r in fixture() if r["model"] == "model-A"]
        gates = evaluate_gates(rows, MANIFEST)
        self.assertFalse(gates["H1_passes"])
        self.assertEqual(len(gates["coverage"]["missing"]), 24)

    def test_missing_candidate_dimension_fails_closed(self):
        rows = fixture()
        rows = [r for r in rows if not (r["model"] == "model-A" and
                r["rotation"] == "hadamard" and r["levels"] == 3 and r["dim"] == 8)]
        self.assertFalse(evaluate_gates(rows, MANIFEST)["coverage"]["complete"])

    def test_duplicate_comparator_fails_closed(self):
        rows = fixture()
        rows.append(copy.deepcopy(rows[0]))
        gates = evaluate_gates(rows, MANIFEST)
        self.assertFalse(gates["H1_passes"])
        self.assertEqual(len(gates["coverage"]["duplicates"]), 1)
        self.assertEqual(aggregate_rows(rows, MANIFEST), [])

    def test_invalid_ratio_and_nonfinite_measurements_fail_closed(self):
        for value in (float("inf"), float("nan"), -1.0):
            with self.subTest(value=value):
                rows = fixture()
                rows[0]["free_vq_mse"] = value
                self.assertFalse(evaluate_gates(rows, MANIFEST)["coverage"]["complete"])
        rows = fixture()
        rows[0]["vq_over_product"] = 0.1
        self.assertFalse(evaluate_gates(rows, MANIFEST)["coverage"]["complete"])

    def test_undeclared_tensor_and_empty_design_fail_closed(self):
        self.assertFalse(evaluate_gates(fixture(), MANIFEST[:1])["coverage"]["complete"])
        self.assertFalse(evaluate_gates([], [])["H1_passes"])
        self.assertFalse(evaluate_gates([])["coverage"]["complete"])

    def test_reanalysis_cli_uses_only_stdlib_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "original.json"
            source.write_text(json.dumps({"rows": fixture()}))
            before = source.read_bytes()
            destination = Path(tmp) / "reanalysis"
            command = [sys.executable, "-S", str(ROOT / "experiments/reanalyze_rate_sweep.py"),
                       "--input", str(source), "--out", str(destination), "--expected-tensors",
                       "model-A:same.tensor", "model-B:same.tensor"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(before, source.read_bytes())
            result = json.loads((destination / "results.json").read_text())
            self.assertEqual(result["provenance"]["source_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(result["rows"], fixture())


if __name__ == "__main__":
    unittest.main()
