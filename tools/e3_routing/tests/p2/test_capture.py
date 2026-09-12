import pytest
import torch
from torch import nn

from e3_p2.capture import SpatialRouterCollector, max_output_delta, routing_diagnostics, routing_metric_fields
from e3_p2.runner import _batch_equivalence


class _MoTRouter(nn.Module):
    def forward(self, x, return_logits=True):
        del return_logits
        logits = torch.stack((x[:, 0], -x[:, 0], x[:, 0] * 0), dim=1)
        dense = torch.softmax(logits, dim=1)
        top_weights, indices = torch.topk(dense, 2, dim=1)
        top_weights = top_weights / top_weights.sum(dim=1, keepdim=True)
        weights = torch.zeros_like(dense).scatter(1, indices, top_weights)
        return weights, indices, logits


class TinyMoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _MoTRouter()

    def forward(self, x):
        weights, _, _ = self.router(x, return_logits=True)
        return x * weights[:, :1]


def test_hook_captures_real_spatial_grid_without_changing_output():
    model = TinyMoT().eval()
    value = torch.randn(1, 2, 4, 5)
    reference = model(value)
    collector = SpatialRouterCollector(family="mot", router_class="_MoTRouter")
    assert collector.register(model) == ["router"]
    observed = model(value)
    collector.remove()
    assert max_output_delta(reference, observed) == 0.0
    assert not collector.handles
    assert len(collector.records) == 1
    assert collector.records[0].weights.shape == (1, 3, 4, 5)
    assert collector.records[0].indices.shape == (1, 2, 4, 5)
    assert collector.records[0].validation["indices_in_range"] is True
    assert collector.records[0].validation["selected_mass_error"] < 1e-5


def test_output_equivalence_detects_structure_shape_and_value_changes():
    value = torch.zeros(1, 2)
    assert max_output_delta((value,), (value.clone(),)) == 0.0
    assert max_output_delta((value,), (value + 1,)) == 1.0
    with pytest.raises(ValueError, match="tensor count"):
        max_output_delta((value,), (value, value))
    with pytest.raises(ValueError, match="shape changed"):
        max_output_delta((value,), (torch.zeros(2, 1),))


def test_collector_rejects_non_spatial_router_output():
    class _MoARouter(nn.Module):
        def forward(self, x):
            weights = torch.softmax(x.mean((2, 3)), dim=1)
            return weights, weights

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = _MoARouter()

        def forward(self, x):
            return self.router(x)

    model = Model()
    collector = SpatialRouterCollector(family="moa", router_class="_MoARouter")
    collector.register(model)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(1, 3, 4, 4))
    collector.remove()


def test_routing_metrics_have_fixed_interpretable_scale():
    uniform = torch.full((3, 2, 4), 1 / 3).numpy()
    uniform_fields = routing_metric_fields(uniform)
    assert uniform_fields["normalized_entropy"] == pytest.approx(1.0)
    assert uniform_fields["top1_margin"] == pytest.approx(0.0)

    one_hot = torch.zeros(3, 2, 4).numpy()
    one_hot[1] = 1.0
    diagnostics = routing_diagnostics(one_hot)
    assert diagnostics["normalized_entropy"]["mean"] == pytest.approx(0.0)
    assert diagnostics["top1_margin"]["mean"] == pytest.approx(1.0)
    assert diagnostics["dominant_token_count"] == [0, 8, 0]
    assert diagnostics["active_dominant_experts"] == 1


def test_single_and_batched_router_grids_keep_sample_identity():
    model = TinyMoT().eval()
    tensors = [torch.randn(1, 2, 4, 5) for _ in range(4)]
    records_per_sample = []
    for tensor in tensors:
        collector = SpatialRouterCollector(family="mot", router_class="_MoTRouter")
        collector.register(model)
        model(tensor)
        collector.remove()
        records_per_sample.append(collector.records)
    result = _batch_equivalence(
        model=model,
        family="mot",
        router_class="_MoTRouter",
        tensors=tensors,
        records_per_sample=records_per_sample,
        batch_sizes=[2, 4],
        torch_module=torch,
    )
    assert result["status"] == "PASS"
    assert [item["compared_samples"] for item in result["sizes"]] == [4, 4]
    assert all(item["indices_equal"] for item in result["sizes"])
