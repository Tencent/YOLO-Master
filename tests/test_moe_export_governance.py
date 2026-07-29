"""Contracts for opt-in MoE pruning and routing-preserved adapter export."""

import inspect

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.modules.moe.analysis import ExpertStats
from ultralytics.nn.modules.moe.gated import AdaptiveGateMoE
from ultralytics.nn.modules.moe.pruning import prune_moe_module
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.export.engine import torch2onnx
from ultralytics.utils.export_preflight import export_preflight


def _routing_preserved_layer_and_inputs():
    """Build a deterministic layer where three samples select three distinct experts."""
    layer = MoLoRALayer(
        torch.nn.Linear(8, 8, bias=False),
        r=1,
        alpha=1,
        num_experts=3,
        top_k=1,
        use_rslora=False,
    ).eval()
    layer._export_mode = "routing_preserved"
    inputs = torch.zeros(3, 8)
    inputs[0, 0] = inputs[1, 1] = inputs[2, 2] = 2.0
    with torch.no_grad():
        layer.base_layer.weight.zero_()
        layer.router.fc[0].weight.zero_()
        layer.router.fc[0].bias.zero_()
        layer.router.fc[0].weight[0, 0] = 1.0
        layer.router.fc[0].weight[1, 1] = 1.0
        layer.router.fc[-1].weight.zero_()
        layer.router.fc[-1].bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
        layer.router.fc[-1].weight[0, 0] = 5.0
        layer.router.fc[-1].weight[1, 1] = 5.0
        for index, expert in enumerate(layer.experts):
            expert.lora_A.weight.zero_()
            expert.lora_B.weight.zero_()
            expert.lora_A.weight[0, index] = 1.0
            expert.lora_B.weight[index, 0] = 1.0
    return layer, inputs


def test_export_governance_defaults_are_opt_in_and_typed():
    cfg = get_cfg(overrides={"mode": "export"})
    assert cfg.pre_export_prune is False
    assert cfg.molora_export_mode == "dynamic"
    assert cfg.moe_prune_calibration_steps == 8

    with pytest.raises(ValueError, match="moe_prune_calibration_steps"):
        get_cfg(overrides={"mode": "export", "moe_prune_calibration_steps": 0})


def test_routing_preserved_preflight_is_explicit_and_format_limited():
    layer = MoLoRALayer(torch.nn.Linear(8, 8), r=2, num_experts=2, top_k=1)
    report = export_preflight(layer, "onnx", routing_preserved=True, strict=True)
    assert report["decisions"][0]["strategy"] == "routing_preserved"
    assert report["decisions"][0]["dense_fallback"] is False

    with pytest.raises(RuntimeError, match="only supported for ONNX/TorchScript"):
        export_preflight(layer, "engine", routing_preserved=True, strict=True)

    layer.export_capabilities = lambda: {"supported": True, "onnx_routing_preserved": False}
    with pytest.raises(RuntimeError, match="does not advertise"):
        export_preflight(layer, "onnx", routing_preserved=True, strict=True)


def test_routing_preserved_onnx_uses_project_wrapper_and_matches_eager(tmp_path):
    """The supported ONNX path must preserve router decisions and output numerics."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    layer, inputs = _routing_preserved_layer_and_inputs()
    artifact = tmp_path / "molora-routing-preserved.onnx"

    with torch.inference_mode():
        expected = layer(inputs).cpu().numpy()
    assert torch.equal(torch.from_numpy(expected).argmax(dim=1), torch.tensor([0, 1, 2]))
    layer._last_routing_stats = None
    layer.last_routing_snapshot = {}
    layer._last_dispatch_stats = {}
    torch2onnx(layer, inputs, artifact, opset=17, input_names=["images"], output_names=["output0"])
    assert layer._last_routing_stats is None
    assert layer.last_routing_snapshot == {}
    assert layer._last_dispatch_stats == {}

    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    observed = session.run(None, {"images": inputs.numpy()})[0]
    assert observed.shape == expected.shape
    assert observed.dtype == expected.dtype
    assert torch.tensor(observed).allclose(torch.tensor(expected), atol=1e-5, rtol=1e-5)


def test_routing_preserved_onnx_dynamo_export_matches_eager(tmp_path):
    """PyTorch's default dynamo exporter must not capture runtime diagnostics."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        pytest.skip("torch.onnx.export does not support the dynamo exporter")

    layer, inputs = _routing_preserved_layer_and_inputs()
    artifact = tmp_path / "molora-routing-preserved-dynamo.onnx"

    with torch.inference_mode():
        expected = layer(inputs).cpu().numpy()
    assert torch.equal(torch.from_numpy(expected).argmax(dim=1), torch.tensor([0, 1, 2]))
    layer._last_routing_stats = None
    layer.last_routing_snapshot = {}
    layer._last_dispatch_stats = {}
    torch.onnx.export(
        layer,
        inputs,
        artifact,
        opset_version=18,
        input_names=["images"],
        output_names=["output0"],
        dynamo=True,
    )
    assert layer._last_routing_stats is None
    assert layer.last_routing_snapshot == {}
    assert layer._last_dispatch_stats == {}

    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    observed = session.run(None, {"images": inputs.numpy()})[0]
    assert observed.shape == expected.shape
    assert observed.dtype == expected.dtype
    assert torch.tensor(observed).allclose(torch.tensor(expected), atol=1e-5, rtol=1e-5)


def test_in_memory_pruning_isolated_and_forwardable():
    model = torch.nn.Sequential(AdaptiveGateMoE(16, 16, num_experts=3, top_k=1, num_groups=4)).eval()
    usage = {
        "0.routing": {
            0: ExpertStats(hits=10),
            1: ExpertStats(hits=1),
            2: ExpertStats(hits=0),
        }
    }
    pruned, plan = prune_moe_module(model, usage, threshold=0.2)

    assert plan == {"0.routing": [0]}
    assert len(model[0].fused_experts.expert_projections) == 3
    assert len(pruned[0].fused_experts.expert_projections) == 1
    with torch.no_grad():
        output = pruned(torch.zeros(1, 16, 8, 8))
    assert output.shape == (1, 16, 8, 8)


def test_exporter_pre_export_prune_records_manifest_without_source_mutation():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(AdaptiveGateMoE(16, 16, num_experts=3, top_k=1, num_groups=4))

        def forward(self, x):
            return self.model(x)

    source = Wrapper().eval()
    exporter = Exporter(
        overrides={
            "format": "torchscript",
            "pre_export_prune": True,
            "moe_prune_calibration_steps": 1,
            "moe_prune_threshold": 0.2,
        }
    )
    exported_copy = exporter._pre_export_prune(source, torch.zeros(1, 16, 8, 8))

    assert len(source.model[0].fused_experts.expert_projections) == 3
    assert exporter.moe_prune_manifest["applied"] is True
    assert exporter.moe_prune_manifest["pruning_plan"]
    assert len(exported_copy.model[0].fused_experts.expert_projections) == 1
