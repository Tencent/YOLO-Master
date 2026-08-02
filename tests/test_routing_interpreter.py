from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.moe.modules import ES_MOE, OptimizedMOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.routing_interpreter import RoutingInterpreter


class ToyRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, 2, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[[[1.0]]], [[[-1.0]]]]))

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        logits = self.proj(x)
        probabilities = logits.softmax(dim=1)
        return (probabilities, logits) if return_logits else probabilities


class ToyRoutedBlock(nn.Module):
    num_experts = 2
    top_k = 2

    def __init__(self) -> None:
        super().__init__()
        self.router = ToyRouter()
        self.experts = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.last_routing_snapshot: dict = {}
        self.register_buffer("expert_scale", torch.tensor([1.0, -1.0]))

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.expert_scale.new_zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, _ = self.router(x, return_logits=True)
        usage = weights.detach().mean(dim=(0, 2, 3))
        self.last_routing_snapshot = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": usage,
            "mean_router_probs": usage,
            "aux_loss": 0.0,
        }
        scale = torch.einsum("behw,e->bhw", weights, self.expert_scale).unsqueeze(1)
        return x * scale


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.routed = ToyRoutedBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.routed(x)


class LegacyRoutedBlock(nn.Module):
    """Pre-protocol routed block without last_routing_snapshot."""

    num_experts = 2
    top_k = 1

    def __init__(self) -> None:
        super().__init__()
        self.routing = ToyRouter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.routing(x)
        return x * (weights[:, :1] - weights[:, 1:2])


class SparseToyRouter(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        weights = x.new_tensor([0.75, 0.25]).expand(batch_size, -1)
        indices = torch.tensor([0, 1], device=x.device).expand(batch_size, -1)
        return weights, indices, {"source": "sparse-toy"}


class SparseToyRoutedBlock(nn.Module):
    num_experts = 3
    top_k = 2

    def __init__(self) -> None:
        super().__init__()
        self.router = SparseToyRouter()
        self.experts = nn.ModuleList([nn.Identity() for _ in range(self.num_experts)])
        self.last_routing_snapshot: dict = {}
        self.register_buffer("expert_scale", torch.tensor([1.0, -1.0, 3.0]))

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.expert_scale.new_zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices, _ = self.router(x)
        scale = (weights * self.expert_scale[indices]).sum(dim=1)
        return x * scale.view(-1, 1, 1, 1)


class TwoLayerToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = ToyRoutedBlock()
        self.right = ToyRoutedBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) + self.right(x)


def test_collect_layer_summaries_uses_leaf_routed_modules():
    model = ToyModel()
    model(torch.ones(2, 1, 4, 5))

    summaries = RoutingInterpreter(model).collect_layer_summaries()

    assert len(summaries) == 1
    assert summaries[0].layer_name == "routed"
    assert summaries[0].module_type == "ToyRoutedBlock"
    assert summaries[0].num_experts == 2
    assert sum(summaries[0].expert_usage) == pytest.approx(1.0)


def test_detect_routing_collapse_reports_all_signals():
    model = ToyModel()
    model.routed.last_routing_snapshot = {
        "num_experts": 2,
        "top_k": 1,
        "expert_usage": torch.tensor([1.0, 0.0]),
        "aux_loss": 0.0,
    }

    report = RoutingInterpreter(model).detect_routing_collapse()["routed"]

    assert report.collapsed is True
    assert report.dominant_expert == 0
    assert report.dominant_share == pytest.approx(1.0)
    assert report.normalized_gini == pytest.approx(1.0)
    assert report.normalized_entropy == pytest.approx(0.0)
    assert report.dead_experts == (1,)


def test_analyze_expert_specialization_aggregates_features():
    model = ToyModel()
    batches = [torch.ones(2, 1, 3, 4), -torch.ones(2, 1, 3, 4)]

    reports = RoutingInterpreter(model).analyze_expert_specialization(batches)
    report = reports["routed"]

    assert report.num_samples == 4
    assert report.dominant_samples == (2, 2)
    assert report.mean_usage[0] == pytest.approx(report.mean_usage[1], abs=1e-6)
    assert report.feature_signatures[0]["activation_mean"] > report.feature_signatures[1]["activation_mean"]
    assert report.feature_signatures[0]["spatial_height"] == pytest.approx(3.0)
    assert report.feature_signatures[0]["spatial_width"] == pytest.approx(4.0)


def test_capture_routing_preserves_spatial_probability_maps():
    model = ToyModel()
    captured = RoutingInterpreter(model).capture_routing(torch.randn(2, 1, 4, 5))

    heatmap = captured["routed"]
    assert heatmap.probabilities.shape == (2, 2, 4, 5)
    assert heatmap.assignments.shape == (2, 4, 5)
    assert torch.allclose(heatmap.probabilities.sum(dim=1), torch.ones(2, 4, 5), atol=1e-6)


def test_capture_routing_supports_legacy_routed_checkpoints():
    interpreter = RoutingInterpreter(LegacyRoutedBlock())
    heatmaps = interpreter.capture_routing(torch.ones(1, 1, 4, 5), layer_name="<root>")
    heatmap = heatmaps["<root>"]
    summary = interpreter.collect_layer_summaries(heatmaps=heatmaps)[0]
    collapse = interpreter.detect_routing_collapse(heatmaps=heatmaps)["<root>"]

    assert heatmap.probabilities.shape == (1, 2, 4, 5)
    assert sum(summary.expert_usage) == pytest.approx(1.0)
    assert collapse.dominant_share > 0.5


def test_heatmap_summaries_only_include_layers_captured_in_current_run():
    interpreter = RoutingInterpreter(TwoLayerToyModel())
    heatmaps = interpreter.capture_routing(torch.ones(1, 1, 4, 5), layer_name="left")
    interpreter.model.left.last_routing_snapshot["expert_usage"] = torch.tensor([0.0, 1.0])

    summaries = interpreter.collect_layer_summaries(heatmaps=heatmaps)
    collapse = interpreter.detect_routing_collapse(heatmaps=heatmaps)

    assert [summary.layer_name for summary in summaries] == ["left"]
    assert summaries[0].expert_usage[0] > summaries[0].expert_usage[1]
    assert list(collapse) == ["left"]


def test_collect_layer_summaries_rejects_malformed_expert_vectors():
    model = ToyModel()
    model.routed.last_routing_snapshot = {
        "num_experts": 2,
        "expert_usage": torch.tensor([0.5, 0.3, 0.2]),
    }

    assert RoutingInterpreter(model).collect_layer_summaries() == []


def test_sparse_topk_capture_and_causal_routing_restore_state():
    model = SparseToyRoutedBlock().train()
    interpreter = RoutingInterpreter(model)
    batch = torch.ones(2, 1, 3, 4)
    natural = model(batch).detach()

    heatmap = interpreter.capture_routing(batch, layer_name="<root>")["<root>"]
    report = interpreter.routing_causal_analysis(batch, "<root>", expert_idx=2)
    restored = model(batch).detach()

    assert heatmap.probabilities.shape == (2, 3)
    assert heatmap.probabilities[0].tolist() == pytest.approx([0.75, 0.25, 0.0])
    assert report.mean_absolute_difference == pytest.approx(2.5)
    assert model.training is True
    assert model.router.training is True
    assert torch.allclose(restored, natural)


def test_sparse_probability_reconstruction_preserves_nontrivial_spatial_axes():
    weights = torch.full((1, 2, 3, 1), 0.5)
    indices = torch.tensor([0, 1]).view(1, 2, 1, 1).expand_as(weights)

    spatial = RoutingInterpreter._router_probabilities((weights, indices, {}), num_experts=3)
    image_level = RoutingInterpreter._router_probabilities(
        (weights[:, :, :1], indices[:, :, :1], {}), num_experts=3
    )

    assert spatial is not None and spatial.shape == (1, 3, 3, 1)
    assert image_level is not None and image_level.shape == (1, 3)


def test_causal_analysis_forces_expert_and_restores_router():
    model = ToyModel().train()
    interpreter = RoutingInterpreter(model)
    batch = torch.ones(2, 1, 4, 5)
    natural = model(batch).detach()

    report_zero = interpreter.routing_causal_analysis(batch, "routed", expert_idx=0)
    report_one = interpreter.routing_causal_analysis(batch, "routed", expert_idx=1)
    restored = model(batch).detach()

    assert report_zero.mean_absolute_difference < report_one.mean_absolute_difference
    assert report_one.mean_absolute_difference > 1.0
    assert model.training is True
    assert model.routed.training is True
    assert torch.allclose(natural, restored)


def test_dataset_collapse_usage_is_weighted_by_samples_not_batches():
    batches = [torch.ones(2, 1, 3, 4), -torch.ones(1, 1, 3, 4)]

    _, _, collapse = RoutingInterpreter(ToyModel()).run_dataset_analysis(batches)

    positive = torch.softmax(torch.tensor([1.0, -1.0]), dim=0)
    negative = positive.flip(0)
    expected = (2 * positive + negative) / 3
    assert collapse["routed"].expert_usage == pytest.approx(expected.tolist())


def test_single_sample_dataset_metrics_have_finite_zero_spread():
    _, differentiation, _ = RoutingInterpreter(ToyModel()).run_dataset_analysis(
        [torch.ones(1, 1, 3, 4)]
    )

    metrics = differentiation["routed"]
    assert metrics.std_kl_divergence == pytest.approx(0.0)
    assert metrics.std_weight_spread == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("module", "batch", "expected_shape"),
    [
        (ES_MOE(in_channels=32, out_channels=32, num_experts=4, top_k=2), torch.randn(1, 32, 4, 5), (1, 4, 4, 5)),
        (MoABlock(24, num_heads=3), torch.randn(1, 24, 4, 5), (1, 3, 4, 5)),
        (
            MoTBlock(16, num_heads=4, top_k=2, window_size=2, n_points=2),
            torch.randn(1, 16, 4, 4),
            (1, 3, 4, 4),
        ),
        (
            MoLoRALayer(nn.Conv2d(8, 8, 1), r=2, alpha=4, num_experts=3, top_k=2),
            torch.randn(2, 8, 4, 5),
            (2, 3),
        ),
    ],
)
def test_capture_routing_supports_real_mixture_families(module, batch, expected_shape):
    heatmap = RoutingInterpreter(module).capture_routing(batch, layer_name="<root>")["<root>"]

    assert tuple(heatmap.probabilities.shape) == expected_shape
    assert torch.allclose(
        heatmap.probabilities.sum(dim=1),
        torch.ones_like(heatmap.probabilities.sum(dim=1)),
        atol=1e-6,
    )


def test_capture_routing_reconstructs_optimized_moe_sparse_topk_probabilities():
    module = OptimizedMOE(16, 16, num_experts=4, top_k=2)
    heatmap = RoutingInterpreter(module).capture_routing(
        torch.randn(2, 16, 4, 5), layer_name="<root>"
    )["<root>"]

    assert heatmap.probabilities.shape == (2, 4)
    assert torch.allclose(heatmap.probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.count_nonzero(heatmap.probabilities, dim=1).tolist() == [2, 2]


def test_visualize_routing_writes_png_and_json_safe_summary(tmp_path):
    interpreter = RoutingInterpreter(ToyModel())
    batch = torch.linspace(-1.0, 1.0, 20).reshape(1, 1, 4, 5)

    heatmaps = interpreter.capture_routing(batch)
    artifacts = interpreter.save_routing_visualizations(heatmaps, tmp_path, input_image=batch)
    payload = {name: heatmap.to_dict() for name, heatmap in heatmaps.items()}

    assert set(artifacts["routed"]) == {
        "assignment_map",
        "confidence_heatmap",
        "dashboard",
        "expert_0_heatmap",
        "expert_1_heatmap",
    }
    assert (tmp_path / "routed_confidence_heatmap.png").stat().st_size > 0
    assert (tmp_path / "routed_expert_0_heatmap.png").stat().st_size > 0
    assert (tmp_path / "routed_assignment_map.png").stat().st_size > 0
    assert (tmp_path / "routed_routing_dashboard.png").stat().st_size > 0
    assert json.loads(json.dumps(payload))["routed"]["probability_shape"] == [1, 2, 4, 5]
    assert payload["routed"]["visualization_type"] == "spatial_heatmap"


def test_global_router_writes_distribution_not_heatmap(tmp_path):
    interpreter = RoutingInterpreter(SparseToyRoutedBlock())
    heatmaps = interpreter.capture_routing(torch.ones(1, 1, 3, 4), layer_name="<root>")

    artifacts = interpreter.save_routing_visualizations(heatmaps, tmp_path)

    assert set(artifacts["<root>"]) == {"distribution"}
    assert artifacts["<root>"]["distribution"].name == "root_routing_distribution.png"
    assert heatmaps["<root>"].to_dict()["visualization_type"] == "global_distribution"
    assert not list(tmp_path.glob("*heatmap*.png"))


def test_load_image_normalizes_uint8_paths(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 2), color=(255, 128, 0)).save(image_path)

    image = RoutingInterpreter._load_image(image_path)

    assert image is not None
    assert image.shape == (2, 2, 3)
    assert image[0, 0].tolist() == pytest.approx([1.0, 128.0 / 255.0, 0.0])


def test_load_image_uses_preprocessed_single_channel_batch_geometry():
    image = RoutingInterpreter._load_image(torch.ones(1, 1, 4, 5))

    assert image is not None
    assert image.shape == (4, 5)


def test_cli_writes_report_and_heatmap(monkeypatch, tmp_path):
    from tools import routing_interpreter as routing_cli

    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    output_dir = tmp_path / "report"
    monkeypatch.setattr(routing_cli, "_load_model", lambda _path, _device, _half: ToyModel())
    monkeypatch.setattr(
        routing_cli,
        "_load_batch",
        lambda _path, _imgsz, device, dtype: torch.ones(1, 1, 4, 5, device=device, dtype=dtype),
    )

    exit_code = routing_cli.main(
        [str(tmp_path / "toy.pt"), str(image_path), "--imgsz", "8", "--output", str(output_dir)]
    )
    report = json.loads((output_dir / "routing_report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (output_dir / "routed_confidence_heatmap.png").stat().st_size > 0
    assert report["visualizations"]["routed"]["confidence_heatmap"].endswith(
        "routed_confidence_heatmap.png"
    )
    assert list(report["heatmaps"]) == ["routed"]
    assert [summary["layer_name"] for summary in report["summaries"]] == ["routed"]
