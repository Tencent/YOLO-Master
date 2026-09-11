import copy

import pytest
import torch
from torch import nn

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.export_validation import validate_export_roundtrip


def _export_semantics_reference(module: nn.Module) -> nn.Module | None:
    """Return an eager reference matching the module's documented export semantics.

    MoT with ``export_masked=False`` and ``top_k < NUM_EXPERTS`` exports dense
    softmax blending; the roundtrip then compares against the dense eager path.
    The default ``export_masked=True`` path is bit-exact with eager sparse
    dispatch, so no reference override is needed.
    """
    if isinstance(module, MoTBlock) and module.top_k < module.NUM_EXPERTS and not module.router.export_masked:
        reference = copy.deepcopy(module)
        reference.top_k = reference.NUM_EXPERTS
        return reference
    return None


@pytest.mark.parametrize(
    ("module", "sample"),
    [
        (MoABlock(32, num_heads=3), torch.randn(1, 32, 8, 8)),
        (OptimizedMOE(32, 32, num_experts=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoTBlock(32, num_heads=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoTBlock(32, num_heads=2, top_k=1, export_masked=False), torch.randn(1, 32, 8, 8)),
        (MoLoRALayer(nn.Linear(32, 16), r=2, num_experts=2, top_k=1), torch.randn(2, 32)),
    ],
)
def test_mixture_torchscript_roundtrip(module, sample):
    report = validate_export_roundtrip(module, sample, "torchscript", reference=_export_semantics_reference(module))
    assert report["passed"] is True
    assert report["artifact_bytes"] > 0


@pytest.mark.parametrize(
    ("module", "sample"),
    [
        (MoABlock(32, num_heads=3), torch.randn(1, 32, 8, 8)),
        (OptimizedMOE(32, 32, num_experts=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoTBlock(32, num_heads=2, top_k=1), torch.randn(1, 32, 8, 8)),
        (MoTBlock(32, num_heads=2, top_k=1, export_masked=False), torch.randn(1, 32, 8, 8)),
        (MoLoRALayer(nn.Linear(32, 16), r=2, num_experts=2, top_k=1), torch.randn(2, 32)),
    ],
)
def test_mixture_onnx_roundtrip(module, sample):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    report = validate_export_roundtrip(module, sample, "onnx", reference=_export_semantics_reference(module))
    assert report["passed"] is True
    assert report["max_abs_error"] <= 1e-4


def test_mot_export_masked_matches_sparse_eager():
    """MoT with ``export_masked=True`` must export artifacts matching the eager sparse path.

    The masked export path rebuilds renormalized Top-K weights with static
    traceable ops, so the exported graph (dense compute) is numerically equal
    to eager sparse dispatch — no reference override needed.
    """
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=2, top_k=1).eval()
    assert block.router.export_masked is True
    assert block.export_capabilities()["export_router_weights"] == "masked_topk"
    sample = torch.randn(1, 32, 8, 8)
    for fmt in ("torchscript", "onnx"):
        if fmt == "onnx":
            pytest.importorskip("onnx")
            pytest.importorskip("onnxruntime")
        report = validate_export_roundtrip(block, sample, fmt)
        assert report["passed"] is True, f"{fmt}: max_abs_error={report['max_abs_error']}"
        assert report["max_abs_error"] <= 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
