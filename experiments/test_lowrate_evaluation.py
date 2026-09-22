"""Regression tests for evaluation correctness and safeguards; no model downloads."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import run_ternary_task as runner


class EvaluationAccountingTests(unittest.TestCase):
    def test_complete_stream_has_each_target_once_including_partial_tail(self):
        ids = list(range(261285))
        blocks = runner.token_blocks(ids, 128)
        self.assertEqual(len(blocks), 2042)
        self.assertEqual(len(blocks[-1]) - 1, 36)
        self.assertEqual([x for block in blocks for x in block[1:]], ids[1:])
        self.assertEqual(runner.token_blocks([9]), [])
        self.assertEqual(runner.token_blocks([]), [])

    def test_loss_is_token_weighted_and_saved_without_dropping_tail(self):
        row = runner.aggregate_blocks([128.0, 72.0], [128, 36])
        self.assertAlmostEqual(row["nll"], 200 / 164)
        self.assertNotAlmostEqual(row["nll"], (1.0 + 2.0) / 2)
        self.assertEqual(row["target_tokens"], 164)
        self.assertEqual(row["block_loss_sums"], [128.0, 72.0])
        with self.assertRaises(ValueError):
            runner.aggregate_blocks([float("nan")], [1])

    def test_paired_bootstrap_uses_matching_target_denominators(self):
        a = runner.aggregate_blocks([128, 108], [128, 36])
        b = runner.aggregate_blocks([256, 36], [128, 36])
        result = runner.paired_bootstrap(a, b, reps=256)
        self.assertAlmostEqual(result["delta_nll"], (236 - 292) / 164)
        self.assertEqual(result, runner.paired_bootstrap(a, b, reps=256))
        b["block_target_counts"] = [128, 35]
        with self.assertRaises(ValueError):
            runner.paired_bootstrap(a, b, reps=10)

    def test_primary_gate_needs_quality_bytes_and_positive_damage(self):
        bf = {"nll": 3.0}
        ref = {"nll": 3.1, "storage": {"total_bytes": 100}}
        candidate = {"nll": 3.08, "storage": {"total_bytes": 99}}
        comparison = {"ci95": [-0.03, -0.01]}
        gate = runner.primary_gate(candidate, ref, bf, comparison, reportable=True)
        self.assertEqual(gate["status"], "PASS_EXPLORATORY")
        self.assertAlmostEqual(gate["recovered_damage_fraction"], 0.2)
        candidate["storage"]["total_bytes"] = 101
        self.assertEqual(runner.primary_gate(candidate, ref, bf, comparison,
                                             reportable=True)["status"], "STOP")
        ref["nll"] = 2.99
        gate = runner.primary_gate(candidate, ref, bf, comparison, reportable=True)
        self.assertIsNone(gate["recovered_damage_fraction"])
        self.assertFalse(gate["checks"]["reference_damage_positive"])
        self.assertEqual(runner.primary_gate(candidate, ref, bf, comparison,
                                             reportable=False)["status"], "NOT_APPLICABLE_SMOKE")

    def test_full_protocol_has_no_target_cap_and_smoke_never_uses_validation(self):
        full = runner.build_protocol(runner.parser().parse_args([]))
        smoke = runner.build_protocol(runner.parser().parse_args(["--smoke"]))
        self.assertEqual(full["split"], "validation")
        self.assertIsNone(full["target_cap"])
        self.assertTrue(full["reportable"])
        self.assertEqual(full["expected_validation_targets"], 261284)
        self.assertEqual(smoke["split"], "train")
        self.assertEqual(smoke["target_cap"], 2048)
        self.assertFalse(smoke["reportable"])
        self.assertNotEqual(full["protocol_sha256"], smoke["protocol_sha256"])


class ExperimentSafetyTests(unittest.TestCase):
    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            runner.refuse_nonempty_output(root)
            (root / "existing.json").write_text("important")
            with self.assertRaises(ValueError):
                runner.refuse_nonempty_output(root)
            self.assertEqual((root / "existing.json").read_text(), "important")

    def test_lock_prevents_overlapping_runner_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            with runner.run_lock(path):
                with self.assertRaises(RuntimeError):
                    with runner.run_lock(path):
                        self.fail("Second lock should never be acquired")
            with runner.run_lock(path):
                pass

    def test_deadline_interrupts_blocking_work(self):
        with self.assertRaises(TimeoutError):
            with runner.Deadline(0.03):
                time.sleep(1)

    def test_atomic_json_rejects_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            runner.atomic_json(path, {"nll": 3.0})
            with self.assertRaises(ValueError):
                runner.atomic_json(path, {"nll": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"nll": 3.0})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable in local runtime")
class TorchEvaluatorTests(unittest.TestCase):
    def test_actual_cross_entropy_covers_tail_without_padding(self):
        import torch
        from types import SimpleNamespace

        class UniformModel:
            def __call__(self, inputs, use_cache=False):
                return SimpleNamespace(logits=torch.zeros((*inputs.shape, 5)))

        blocks = runner.token_blocks([i % 5 for i in range(168)], 128)
        row = runner.evaluate(UniformModel(), blocks, batch_size=8, device="cpu")
        self.assertEqual(row["block_target_counts"], [128, 39])
        self.assertEqual(row["target_tokens"], 167)
        self.assertAlmostEqual(row["nll"], math.log(5), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
