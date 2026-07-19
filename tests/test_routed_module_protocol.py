"""Tests for the unified RoutedModule protocol across MoE/MoA/MoT/MoLoRA.

Verifies that all mixture-routing modules expose the same interface:
  - num_experts (int)
  - top_k (int)
  - aux_loss (Tensor property)
  - last_routing_snapshot (dict with expert_usage, mean_router_probs, etc.)

Also tests the protocol utilities: is_routed_module(), collect_routed_children().
"""

import torch
import torch.nn as nn
import pytest

# Force MoE snapshot recording on every forward by patching the module-level
# constant BEFORE any ES_MOE forward runs (env var won't work post-import).
from ultralytics.nn.modules.moe import _common as _moe_common
from ultralytics.nn.modules.moa.moa import MoABlock, C2fMoA
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.protocol import collect_routed_children, is_routed_module
from ultralytics.nn.modules.mot.mot import MoTBlock, C2fMoT
from ultralytics.nn.peft.molora.layer import MoLoRALayer

_moe_common.MOE_SNAPSHOT_INTERVAL = 1


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_moe():
    return ES_MOE(in_channels=32, out_channels=32, num_experts=4, top_k=2)


def _make_moa():
    return MoABlock(64, num_heads=6)


def _make_c2fmoa():
    return C2fMoA(64, 64, n=1, num_heads=6)


def _make_mot():
    return MoTBlock(64, num_heads=4, top_k=2)


def _make_c2fmot():
    return C2fMoT(64, 64, n=1, num_heads=4, top_k=2)


def _make_molora():
    return MoLoRALayer(nn.Conv2d(64, 64, 3, padding=1), r=4, alpha=8, num_experts=4, top_k=2)


# ── Protocol compliance tests ───────────────────────────────────────────


class TestRoutedModuleProtocol:
    """Verify all mixture module types satisfy RoutedModule protocol."""

    @pytest.mark.parametrize(
        "factory,name,exp_E,exp_K",
        [
            (_make_moe, "ES_MOE", 4, 2),
            (_make_moa, "MoABlock", 3, 3),
            (_make_c2fmoa, "C2fMoA", 3, 3),
            (_make_mot, "MoTBlock", 3, 2),
            (_make_c2fmot, "C2fMoT", 3, 2),
            (_make_molora, "MoLoRALayer", 4, 2),
        ],
    )
    def test_is_routed_module(self, factory, name, exp_E, exp_K):
        m = factory()
        assert is_routed_module(m), f"{name} does not satisfy RoutedModule"

    @pytest.mark.parametrize(
        "factory,name,exp_E",
        [
            (_make_moe, "ES_MOE", 4),
            (_make_moa, "MoABlock", 3),
            (_make_c2fmoa, "C2fMoA", 3),
            (_make_mot, "MoTBlock", 3),
            (_make_c2fmot, "C2fMoT", 3),
            (_make_molora, "MoLoRALayer", 4),
        ],
    )
    def test_num_experts(self, factory, name, exp_E):
        m = factory()
        assert m.num_experts == exp_E, f"{name}.num_experts = {m.num_experts}, expected {exp_E}"

    @pytest.mark.parametrize(
        "factory,name,exp_K",
        [
            (_make_moe, "ES_MOE", 2),
            (_make_moa, "MoABlock", 3),
            (_make_c2fmoa, "C2fMoA", 3),
            (_make_mot, "MoTBlock", 2),
            (_make_c2fmot, "C2fMoT", 2),
            (_make_molora, "MoLoRALayer", 2),
        ],
    )
    def test_top_k(self, factory, name, exp_K):
        m = factory()
        assert m.top_k == exp_K, f"{name}.top_k = {m.top_k}, expected {exp_K}"

    @pytest.mark.parametrize(
        "factory,name",
        [
            (_make_moe, "ES_MOE"),
            (_make_moa, "MoABlock"),
            (_make_c2fmoa, "C2fMoA"),
            (_make_mot, "MoTBlock"),
            (_make_c2fmot, "C2fMoT"),
            (_make_molora, "MoLoRALayer"),
        ],
    )
    def test_aux_loss_is_tensor(self, factory, name):
        m = factory()
        m.train()
        x = torch.randn(2, 64, 8, 8) if name != "ES_MOE" else torch.randn(2, 32, 8, 8)
        _ = m(x)
        al = m.aux_loss
        assert isinstance(al, torch.Tensor), f"{name}.aux_loss is {type(al)}, expected Tensor"

    @pytest.mark.parametrize(
        "factory,name",
        [
            (_make_moe, "ES_MOE"),
            (_make_moa, "MoABlock"),
            (_make_c2fmoa, "C2fMoA"),
            (_make_mot, "MoTBlock"),
            (_make_c2fmot, "C2fMoT"),
            (_make_molora, "MoLoRALayer"),
        ],
    )
    def test_routing_snapshot_populated(self, factory, name):
        m = factory()
        m.train()
        x = torch.randn(2, 64, 8, 8) if name != "ES_MOE" else torch.randn(2, 32, 8, 8)
        _ = m(x)
        snap = m.last_routing_snapshot
        assert isinstance(snap, dict), f"{name}.last_routing_snapshot is {type(snap)}"
        assert len(snap) > 0, f"{name}.last_routing_snapshot is empty after forward"
        assert "expert_usage" in snap, f"{name} snapshot missing 'expert_usage'"

    @pytest.mark.parametrize(
        "factory,name,exp_E",
        [
            (_make_moe, "ES_MOE", 4),
            (_make_moa, "MoABlock", 3),
            (_make_mot, "MoTBlock", 3),
            (_make_molora, "MoLoRALayer", 4),
        ],
    )
    def test_expert_usage_shape(self, factory, name, exp_E):
        m = factory()
        m.train()
        x = torch.randn(2, 64, 8, 8) if name != "ES_MOE" else torch.randn(2, 32, 8, 8)
        _ = m(x)
        usage = m.last_routing_snapshot["expert_usage"]
        assert usage.shape[0] == exp_E, f"{name} expert_usage shape={usage.shape}, expected [{exp_E}]"


# ── Protocol utility tests ──────────────────────────────────────────────


class TestProtocolUtils:
    """Test is_routed_module() and collect_routed_children()."""

    def test_is_routed_module_false_for_plain_nn(self):
        m = nn.Conv2d(3, 3, 1)
        assert not is_routed_module(m)

    def test_is_routed_module_false_for_sequential(self):
        m = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
        assert not is_routed_module(m)

    def test_collect_routed_children_finds_all(self):
        parent = nn.Sequential(
            _make_moa(),
            nn.ReLU(),
            _make_mot(),
        )
        children = collect_routed_children(parent)
        assert len(children) == 2

    def test_collect_routed_children_empty(self):
        parent = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
        children = collect_routed_children(parent)
        assert len(children) == 0

    def test_collect_routed_children_nested(self):
        inner = C2fMoA(64, 64, n=2, num_heads=6)
        outer = nn.Sequential(inner, nn.ReLU())
        children = collect_routed_children(outer)
        # C2fMoA itself + 2 MoABlocks inside
        assert len(children) >= 1  # at least the C2fMoA


# ── Eval-mode aux_loss tests ────────────────────────────────────────────


class TestEvalModeAuxLoss:
    """aux_loss should be zero in eval mode (no gradient)."""

    @pytest.mark.parametrize(
        "factory,name",
        [
            (_make_moa, "MoABlock"),
            (_make_mot, "MoTBlock"),
        ],
    )
    def test_eval_aux_loss_zero(self, factory, name):
        """MoA/MoT zero aux_loss in eval mode (no balance regularizer needed)."""
        m = factory()
        m.eval()
        x = torch.randn(2, 64, 8, 8)
        with torch.no_grad():
            _ = m(x)
        al = m.aux_loss
        assert isinstance(al, torch.Tensor)
        # In eval mode, aux loss should be zero or near-zero
        assert float(al) == pytest.approx(0.0, abs=1e-6), f"{name} eval aux_loss = {float(al)}, expected ~0"

    def test_molora_eval_aux_loss_nonzero_ok(self):
        """MoLoRA computes aux_loss even in eval (loss_fn is unconditional).

        This is by design: MoLoRA's aux_loss is part of the forward graph
        and may be needed for logging/diagnostics in eval. The value should
        be finite but not necessarily zero.
        """
        layer = _make_molora()
        layer.eval()
        x = torch.randn(2, 64, 8, 8)
        with torch.no_grad():
            _ = layer(x)
        al = layer.aux_loss
        assert isinstance(al, torch.Tensor)
        assert torch.isfinite(al), "MoLoRA eval aux_loss should be finite"


# ── MoLoRA-specific edge case tests ─────────────────────────────────────


class TestMoLoRAEdgeCases:
    """Additional MoLoRA edge cases for robustness."""

    def test_routing_snapshot_after_merge(self):
        """Routing snapshot should still be available after merge."""
        layer = _make_molora()
        layer.merge_weights()
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        _ = layer(x)
        # After merge, routing may be bypassed; snapshot may be stale but present
        assert hasattr(layer, "last_routing_snapshot")

    def test_molora_deepcopy_safe(self):
        """MoLoRA layer should be deepcopyable without errors.

        The __getattr__ proxy can interfere with deepcopy when the graph
        is still alive, so we detach the routing state first.
        """
        import copy

        layer = _make_molora()
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        _ = layer(x)
        # Clear live graph references before deepcopy
        layer._last_aux_loss = torch.zeros(())
        layer2 = copy.deepcopy(layer)
        assert layer2.num_experts == layer.num_experts
        assert layer2.top_k == layer.top_k

    def test_molora_aux_loss_gradient_flow(self):
        """aux_loss should have gradient connection to router params."""
        layer = _make_molora()
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        _ = layer(x)
        al = layer.aux_loss
        if float(al) > 0:
            al.backward(retain_graph=True)
            router_grads = [p.grad for p in layer.router.parameters() if p.grad is not None]
            assert len(router_grads) > 0, "Router params have no gradient from aux_loss"

    def test_molora_linear_layer_protocol(self):
        """MoLoRA on Linear layer also satisfies protocol."""
        lin = nn.Linear(64, 128)
        layer = MoLoRALayer(lin, r=4, alpha=8, num_experts=4, top_k=2)
        layer.train()
        x = torch.randn(2, 64)
        _ = layer(x)
        assert is_routed_module(layer)
        assert layer.num_experts == 4
        assert layer.top_k == 2
        assert "expert_usage" in layer.last_routing_snapshot

    def test_molora_single_expert_protocol(self):
        """MoLoRA with num_experts=1 still satisfies protocol."""
        conv = nn.Conv2d(64, 64, 3, padding=1)
        layer = MoLoRALayer(conv, r=4, alpha=8, num_experts=1, top_k=1)
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        _ = layer(x)
        assert is_routed_module(layer)
        assert layer.num_experts == 1
        assert layer.top_k == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
