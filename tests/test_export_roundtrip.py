import pytest
import torch
from torch import nn

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.export_validation import validate_export_roundtrip


@pytest.mark.parametrize(
    ("module", "sample"),
    [
        (MoABlock(32, num_heads=3), torch.randn(1, 32, 8, 8)),
        (OptimizedMOE(32, 32, num_experts=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoTBlock(32, num_heads=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoLoRALayer(nn.Linear(32, 16), r=2, num_experts=2, top_k=1), torch.randn(2, 32)),
    ],
)
def test_mixture_torchscript_roundtrip(module, sample):
    report = validate_export_roundtrip(module, sample, "torchscript")
    assert report["passed"] is True
    assert report["artifact_bytes"] > 0


@pytest.mark.parametrize(
    ("module", "sample"),
    [
        (MoABlock(32, num_heads=3), torch.randn(1, 32, 8, 8)),
        (OptimizedMOE(32, 32, num_experts=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoTBlock(32, num_heads=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoLoRALayer(nn.Linear(32, 16), r=2, num_experts=2, top_k=1), torch.randn(2, 32)),
    ],
)
def test_mixture_onnx_roundtrip(module, sample):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    report = validate_export_roundtrip(module, sample, "onnx")
    assert report["passed"] is True
    assert report["max_abs_error"] <= 1e-4
