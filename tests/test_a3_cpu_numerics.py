"""Standalone tests: python tests/test_a3_cpu_numerics.py (no pytest plugins)."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_numerics", ROOT / "scripts/run_a3_cpu_numerics.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class NumericsTests(unittest.TestCase):
    def test_runtime_drift_requires_explicit_cli_flag(self):
        source = (ROOT / "scripts/run_a3_cpu_numerics.py").read_text(encoding="utf-8")
        self.assertIn('"--allow-runtime-drift"', source)
        self.assertIn('patch.object(manifest_module, "_runtime_contract"', source)
        self.assertIn("runtime_contract_match=not runtime_drift", source)

    def test_exact_zero(self):
        row = CHECK.array_metrics(np.zeros((2, 3)), np.zeros((2, 3)), 0, 0)
        self.assertTrue(row["allclose"])
        self.assertEqual(row["relative_l2"], 0)

    def test_zero_reference_nonzero_actual(self):
        row = CHECK.array_metrics(np.zeros(2), np.ones(2), 0, 0)
        self.assertFalse(row["allclose"])
        self.assertIsNone(row["relative_l2"])
        json.dumps(row, allow_nan=False)

    def test_nonfinite_is_failure(self):
        for value in (np.nan, np.inf, -np.inf):
            row = CHECK.array_metrics(np.array([value]), np.array([value]), 1, 1)
            self.assertFalse(row["allclose"])
            self.assertEqual(row["actual_nonfinite"], 1)
            json.dumps(row, allow_nan=False)

    def test_no_broadcast_or_zip_truncation(self):
        self.assertFalse(CHECK.array_metrics(np.zeros((2, 1)), np.zeros((2, 3)), 0, 0)["allclose"])
        self.assertFalse(CHECK.compare_outputs([np.zeros(2)], [], 0, 0)["allclose"])

    def test_detection_classes_exact_even_with_large_tolerance(self):
        a = np.zeros((1, 2, 6))
        b = a.copy()
        b[0, 0, 5] = 1
        row = CHECK.compare_outputs([a], [b], 2, 2, end2end=True)
        self.assertFalse(row["allclose"])
        self.assertFalse(row["outputs"][0]["classes_equal"])

    def test_tensor_output_contract(self):
        import torch

        tensor = torch.ones(1, 2, 6)
        self.assertEqual(len(CHECK.tensor_outputs((tensor, {"aux": tensor}))), 1)
        self.assertEqual(len(CHECK.tensor_outputs((tensor, tensor))), 2)
        with self.assertRaises(TypeError):
            CHECK.tensor_outputs({"prediction": tensor})

    def test_cpu_onnx_runtime(self):
        import onnx
        from onnx import TensorProto, helper

        shape = [1, 3, 4, 4]
        graph = helper.make_graph(
            [helper.make_node("Identity", ["image"], ["prediction"])],
            "identity",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, shape)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.onnx"
            onnx.save(model, path)
            inputs = [np.arange(48, dtype=np.float32).reshape(shape)]
            outputs, metadata = CHECK.run_ort(path, inputs, 1)
            self.assertEqual(metadata["providers"], ["CPUExecutionProvider"])
            np.testing.assert_array_equal(outputs[0][0], inputs[0])


if __name__ == "__main__":
    unittest.main()
