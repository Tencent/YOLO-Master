from __future__ import annotations

import json

import torch

from e3_p0.collector import RoutingCollector, discover_routed_modules


def routed_predicate(module):
    return all(hasattr(module, name) for name in ("num_experts", "top_k", "last_routing_snapshot"))


def _routed_class(name: str, module_path: str):
    class Routed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 2
            self.top_k = 1
            self.last_routing_snapshot = {}

        def forward(self, value):
            self.last_routing_snapshot = {
                "num_experts": 2,
                "top_k": 1,
                "expert_usage": value.new_tensor([0.6, 0.4]),
                "aux_loss": value.new_zeros(()),
            }
            return value + 1

    Routed.__name__ = name
    Routed.__module__ = module_path
    return Routed


FakeMoE = _routed_class("FakeMoE", "ultralytics.nn.modules.moe.fake")
FakeMoT = _routed_class("MoTBlock", "ultralytics.nn.modules.mot.fake")
FakeLatent = _routed_class("LatentMixture", "ultralytics.nn.modules.latent_mixture")


class ThreeFamilyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = FakeMoE()
        self.mot = FakeMoT()
        self.latent = FakeLatent()

    def forward(self, value):
        return self.latent(self.mot(self.moe(value)))


def test_each_family_is_discovered_by_metadata():
    model = ThreeFamilyModel()
    assert [name for name, _ in discover_routed_modules(model, "moe", predicate=routed_predicate)] == ["moe"]
    assert [name for name, _ in discover_routed_modules(model, "mot", predicate=routed_predicate)] == ["mot"]
    assert [name for name, _ in discover_routed_modules(model, "latent", predicate=routed_predicate)] == ["latent"]


def test_hook_does_not_change_output_and_is_removed():
    model = ThreeFamilyModel().eval()
    sample = torch.zeros(1, 3, 2, 2)
    expected = model(sample.clone())
    collector = RoutingCollector("mot", "unit-test")
    names = collector.register(model, predicate=routed_predicate)
    collector.set_context(pass_name="primary", batch_index=0, batch_size=1, sample_indices=[3], input_ids=["abc"])
    observed = model(sample.clone())
    collector.remove()
    assert names == ["mot"]
    assert torch.equal(observed, expected)
    assert len(collector.events) == 1
    assert collector.handles == []
    assert json.dumps(collector.events[0])
    assert collector.events[0]["runtime"]["sample_indices"] == [3]
    assert collector.events[0]["runtime"]["batch_size"] == 1
    assert not any(hasattr(value, "grad_fn") for value in collector.events[0]["source_snapshot"].values())


def test_leaf_policy_prevents_parent_child_double_counting():
    Parent = _routed_class("C2fMoE", "ultralytics.nn.modules.moe.fake")
    parent = Parent()
    parent.child = FakeMoE()
    model = torch.nn.Sequential(parent)
    discovered = discover_routed_modules(model, "moe", predicate=routed_predicate, leaf_only=True)
    assert [name for name, _ in discovered] == ["0.child"]


def test_moe_eval_buffer_fallback_is_explicitly_labelled():
    module = FakeMoE().eval()
    module.register_buffer("expert_usage_counts", torch.tensor([0.8, 0.2]))
    module.register_buffer("load_balancing_loss", torch.tensor(0.36))

    def forward_without_snapshot(value):
        return value

    module.forward = forward_without_snapshot
    model = torch.nn.Sequential(module)
    collector = RoutingCollector("moe", "unit-test")
    collector.register(model, predicate=routed_predicate)
    model(torch.zeros(1, 2))
    collector.remove()
    assert collector.events[0]["source_snapshot"]["diagnostic_transport"] == "official_eval_runtime_buffers"
    assert collector.events[0]["routing"]["expert_load"] == [0.8, 0.2]


def test_nested_moe_router_hook_captures_eval_weights_and_indices():
    class Router(torch.nn.Module):
        def forward(self, value):
            batch = value.shape[0]
            return (
                value.new_tensor([[0.75, 0.25]]).repeat(batch, 1),
                torch.tensor([[2, 0]], dtype=torch.long).repeat(batch, 1),
                {},
            )

    module = FakeMoE().eval()
    module.num_experts = 3
    module.top_k = 2
    module.routing = Router()

    def routed_forward(value):
        module.routing(value)
        return value

    module.forward = routed_forward
    model = torch.nn.Sequential(module)
    collector = RoutingCollector("moe", "unit-test")
    collector.register(model, predicate=routed_predicate)
    model(torch.zeros(1, 2))
    collector.remove()
    event = collector.events[0]
    assert event["source_snapshot"]["diagnostic_transport"] == "nested_router_forward_hook"
    assert event["routing"]["expert_load"] == [0.5, 0.0, 0.5]
    assert event["routing"]["mixing_weights"]["value"] == [0.75, 0.25]
    assert event["aux_loss"]["state"] == "unavailable"
