"""Regression tests for composable AdaptiveGateMoE router hooks."""

import torch
from torch import nn

from ultralytics.nn.modules.moe import (
    AdaptiveGateMoE,
    DetailGateHook,
    FeatureRefinementHook,
    RouterHook,
    VisualEnhancedAdaptiveGateMoE,
    build_router_hook,
    resolve_router_hooks,
)


def test_router_hook_registry_resolves_and_rejects_invalid_specs():
    assert isinstance(build_router_hook("detail"), DetailGateHook)
    assert isinstance(build_router_hook("feature_refinement"), FeatureRefinementHook)
    assert [hook.name for hook in resolve_router_hooks(["detail", "context", "refine"])] == [
        "detail",
        "context",
        "refine",
    ]

    try:
        resolve_router_hooks(["detail", "detail"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate router hooks should be rejected")


def test_adaptive_gate_moe_can_compose_hooks_without_state_dict_policy_keys():
    model = AdaptiveGateMoE(
        16,
        16,
        num_experts=2,
        top_k=1,
        num_groups=4,
        router_hooks=["detail", "context", "refine"],
    ).eval()

    assert model.router_hook_names == ("detail", "context", "refine")
    assert isinstance(model.detail_gate, nn.Module)
    assert isinstance(model.context_mixer, nn.Module)
    assert isinstance(model.feature_refiner, nn.Module)
    assert not any(key.startswith("router_hooks") for key in model.state_dict())

    with torch.no_grad():
        output = model(torch.zeros(1, 16, 8, 8))
    assert output.shape == (1, 16, 8, 8)


def test_visual_legacy_variant_uses_hooks_and_preserves_historical_keys():
    model = VisualEnhancedAdaptiveGateMoE(16, 16, num_experts=2, top_k=1, num_groups=4).eval()
    assert model.router_hook_names == ("detail", "context", "refine")

    keys = set(model.state_dict())
    assert any(key.startswith("detail_gate.") for key in keys)
    assert any(key.startswith("context_mixer.") for key in keys)
    assert any(key.startswith("feature_refiner.") for key in keys)

    with torch.no_grad():
        output = model(torch.zeros(1, 16, 8, 8))
    assert output.shape == (1, 16, 8, 8)


def test_custom_hook_runs_in_declaration_order():
    events = []

    class RecordingHook(RouterHook):
        def __init__(self, name):
            self.name = name
            self.stages = frozenset({"pre_route"})

        def apply(self, module, stage, value):
            events.append(self.name)
            return value

    model = AdaptiveGateMoE(
        8,
        8,
        num_experts=2,
        top_k=1,
        router_hooks=[RecordingHook("first"), RecordingHook("second")],
    )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 8, 4, 4))
    assert events == ["first", "second"]
