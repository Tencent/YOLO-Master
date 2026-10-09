"""Dependency-light unit tests for the A3 accuracy-only runner."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_accuracy", ROOT / "scripts/run_a3_cpu_accuracy.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AccuracyRunnerTests(unittest.TestCase):
    def test_digest_is_key_order_independent(self):
        self.assertEqual(MODULE.digest_json({"a": 1, "b": 2}), MODULE.digest_json({"b": 2, "a": 1}))

    def test_comparison_uses_family_fp32(self):
        rows = {
            "fp32": {"status": "success", "accuracy": {"map50_95": 0.5}},
            "fp16": {"status": "success", "accuracy": {"map50_95": 0.49}},
            "broken": {"status": "failed"},
        }
        compared = MODULE.family_comparison(rows)
        self.assertAlmostEqual(compared["fp16"]["map50_95_loss_percentage_points"], 1.0)
        self.assertNotIn("map50_95_loss", compared["broken"])

    def test_csv_contains_success_and_failure(self):
        rows = {
            "moe": {
                "fp32": {"status": "success", "accuracy": {"map50_95": 0.5}, "size_mb": 1.0},
                "fp16": {"status": "failed", "error_type": "RuntimeError", "error": "bad"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "summary.csv"
            MODULE.write_csv(target, rows)
            text = target.read_text(encoding="utf-8-sig")
            self.assertIn("moe,fp32,success", text)
            self.assertIn("moe,fp16,failed", text)


if __name__ == "__main__":
    unittest.main()
