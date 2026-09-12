"""Standalone tests: python tests/test_a3_route_consistency.py."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "route_consistency", ROOT / "scripts/run_a3_route_consistency.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class RouteConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = {
            "topk_exact_min": 0.99,
            "topk_ordered_exact_min": 0.99,
            "top1_flip_max": 0.01,
        }

    @staticmethod
    def metrics(**overrides):
        result = {
            "samples": 2,
            "tokens": 100,
            "topk_ordered_exact_rate": 1.0,
            "topk_exact_rate": 1.0,
            "jaccard_rate": 1.0,
            "token_flip_rate": 0.0,
            "top1_flip_rate": 0.0,
            "sample_flip_rate": 0.0,
            "sample_top1_flip_rate": 0.0,
            "probability_mae": 0.0,
            "total_variation": 0.0,
            "jensen_shannon": 0.0,
        }
        result.update(overrides)
        return result

    def test_gate_passes_boundary(self):
        passed, reasons = CHECK.gate_layer(
            self.metrics(topk_exact_rate=0.99, topk_ordered_exact_rate=0.99, top1_flip_rate=0.01),
            self.thresholds,
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_gate_reports_each_failure(self):
        passed, reasons = CHECK.gate_layer(
            self.metrics(topk_exact_rate=0.98, topk_ordered_exact_rate=0.97, top1_flip_rate=0.02),
            self.thresholds,
        )
        self.assertFalse(passed)
        self.assertEqual(len(reasons), 3)

    def test_family_gate_requires_every_layer(self):
        layers = {
            "good": self.metrics(),
            "bad": self.metrics(topk_exact_rate=0.5, tokens=20),
        }
        contracts = {
            "good": {"representation": "sparse_topk"},
            "bad": {"representation": "dense_probabilities"},
        }
        result = CHECK.summarize_layers(layers, contracts, self.thresholds)
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["failed_layers"], ["bad"])
        self.assertEqual(result["representations"], {"sparse_topk": 1, "dense_probabilities": 1})
        self.assertAlmostEqual(result["weighted"]["topk_exact_rate"], (100.0 + 10.0) / 120.0)

    def test_source_has_route_only_claim_and_runtime_drift_flag(self):
        source = (ROOT / "scripts/run_a3_route_consistency.py").read_text(encoding="utf-8")
        self.assertIn('"--allow-runtime-drift"', source)
        self.assertIn("does not train, export, quantize", source)
        self.assertIn("dense-probability rows are a standardized", source.lower())

    def test_structural_fallback_rejects_unrelated_failure(self):
        def unrelated_failure(_source, _target):
            raise RuntimeError("shape inference failed")

        with self.assertRaisesRegex(RuntimeError, "shape inference failed"):
            CHECK.instrument_routing_outputs_with_structural_fallback(
                Path("source.onnx"), Path("target.onnx"), unrelated_failure
            )

    def test_constant_scalar_absent(self):
        class NotAConstant:
            op_type = "Identity"

        self.assertEqual(CHECK._onnx_constant_scalar(NotAConstant(), None), 0)

    def test_real_onnx_route_probe(self):
        import numpy as np
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx is not installed in the local test interpreter")
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Softmax", ["image"], ["route_probs"], name="/router/Softmax", axis=1)],
            "route_probe",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 4, 2, 2])],
            [helper.make_tensor_value_info("route_probs", TensorProto.FLOAT, [1, 4, 2, 2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        from scripts.a3_precision import sensitivity

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fp32 = directory / "fp32.onnx"
            int8 = directory / "int8.onnx"
            onnx.save(model, fp32)
            onnx.save(model, int8)
            with patch.object(
                sensitivity,
                "letterbox_image",
                return_value=np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2),
            ):
                result = sensitivity.measure_onnx_route_drift(
                    fp32,
                    int8,
                    images=[directory / "mock.jpg"],
                    settings={"imgsz": 2, "ort_providers": ["CPUExecutionProvider"]},
                    work_dir=directory / "probes",
                )
        metrics = result["layers"]["/router/Softmax"]
        self.assertEqual(metrics["topk_exact_rate"], 1.0)
        self.assertEqual(metrics["top1_flip_rate"], 0.0)
        self.assertEqual(result["routing_contracts"]["/router/Softmax"]["representation"], "dense_probabilities")


if __name__ == "__main__":
    unittest.main()
