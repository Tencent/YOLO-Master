"""Isolated regression test for legacy tracing of dictionary extra-state."""

import ast
import io
import unittest
from copy import deepcopy
from pathlib import Path

import onnx
import torch


class ExtraStateLayer(torch.nn.Module):
    """Keep non-tensor routing configuration alongside a real parameter."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.config = {"scale": 2.0, "inference_top_k": 2}

    def get_extra_state(self):
        return self.config.copy()

    def set_extra_state(self, state):
        self.config = state.copy()

    def forward(self, x):
        return x * self.weight * self.config["scale"]


class TestOnnxExtraState(unittest.TestCase):
    def test_export_copy_preserves_parameters_configuration_and_output(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts/a3_precision/onnx_backend.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        hook_node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_onnx_tensor_state_hook"
        )
        namespace = {}
        exec(compile(ast.Module(body=[hook_node], type_ignores=[]), str(source_path), "exec"), namespace)
        original = torch.nn.Sequential(ExtraStateLayer()).eval()
        exported = deepcopy(original)
        exported._register_state_dict_hook(namespace["_onnx_tensor_state_hook"])
        exported = deepcopy(exported)  # The Ultralytics exporter copies the model again.
        self.assertIn("0._extra_state", original.state_dict())
        self.assertNotIn("0._extra_state", exported.state_dict())
        self.assertEqual(original[0].get_extra_state(), exported[0].get_extra_state())
        torch.testing.assert_close(original[0].weight, exported[0].weight)
        x = torch.randn(1, 3)
        torch.testing.assert_close(original(x), exported(x))
        result = io.BytesIO()
        torch.onnx.export(exported, x, result, opset_version=17, dynamo=False)
        graph = onnx.load_model_from_string(result.getvalue())
        onnx.checker.check_model(graph)


if __name__ == "__main__":
    unittest.main()
