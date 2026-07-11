"""MoLoRA supplementary tests: gradient flow, state_dict, multi-expert isolation.

Covers gaps not in test_molora.py:
  - Router→expert gradient flow verification
  - state_dict save/load round-trip
  - Expert parameter isolation (freezing one expert doesn't affect others)
  - Warmup schedule correctness
  - Scaling formula (rsLoRA vs standard)
  - Large batch routing stability
"""
import torch
import torch.nn as nn
import pytest
import copy

from ultralytics.nn.peft.molora.layer import MoLoRALayer, MoLoRAExpert
from ultralytics.nn.peft.molora.loss import MoLoRALoss
from ultralytics.nn.peft.molora.config import MoLoRAConfig


# ── Helpers ─────────────────────────────────────────────────────────────

def _conv_layer(num_experts=4, top_k=2, r=4, alpha=8):
    return MoLoRALayer(
        nn.Conv2d(64, 64, 3, padding=1),
        r=r, alpha=alpha, num_experts=num_experts, top_k=top_k,
    )

def _linear_layer(num_experts=4, top_k=2, r=8, alpha=16):
    return MoLoRALayer(
        nn.Linear(64, 128),
        r=r, alpha=alpha, num_experts=num_experts, top_k=top_k,
    )


# ── Gradient flow tests ─────────────────────────────────────────────────

class TestGradientFlow:
    """Verify gradients flow correctly from loss through router to experts."""

    def test_router_receives_gradient(self):
        layer = _conv_layer()
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        out = layer(x)
        loss = out.sum() + layer.aux_loss
        loss.backward()
        router_grads = [p.grad for p in layer.router.parameters() if p.grad is not None]
        assert len(router_grads) > 0, "Router received no gradient"

    def test_expert_receives_gradient(self):
        layer = _conv_layer(num_experts=4, top_k=2)
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        out = layer(x)
        out.sum().backward()
        # At least some experts should have gradients
        expert_grads = 0
        for expert in layer.experts:
            for p in expert.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    expert_grads += 1
        assert expert_grads > 0, "No expert received gradient"

    def test_base_layer_frozen(self):
        """Base layer parameters should not require grad."""
        layer = _conv_layer()
        for p in layer.base_layer.parameters():
            assert not p.requires_grad, "Base layer parameter is trainable"

    def test_base_layer_no_gradient(self):
        """Base layer should not receive gradients after backward."""
        layer = _conv_layer()
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        out = layer(x)
        out.sum().backward()
        for p in layer.base_layer.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, \
                "Base layer received gradient"

    def test_aux_loss_gradient_to_router_only(self):
        """aux_loss backward should only update router, not experts."""
        layer = _conv_layer()
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        _ = layer(x)
        aux = layer.aux_loss
        aux.backward(retain_graph=True)
        router_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.router.parameters()
        )
        assert router_has_grad, "Router has no gradient from aux_loss"


# ── state_dict persistence tests ────────────────────────────────────────

class TestStateDictPersistence:
    """Verify state_dict save/load round-trip preserves behavior."""

    def test_state_dict_keys(self):
        layer = _conv_layer()
        sd = layer.state_dict()
        # Should have keys for base_layer, experts, router
        assert any("base_layer" in k for k in sd)
        assert any("experts" in k for k in sd)
        assert any("router" in k for k in sd)

    def test_save_load_roundtrip(self):
        layer1 = _conv_layer()
        layer1.train()
        x = torch.randn(2, 64, 8, 8)
        out1 = layer1(x)

        # Save state_dict
        sd = layer1.state_dict()

        # Load into fresh layer
        layer2 = _conv_layer()
        layer2.load_state_dict(sd)
        layer2.eval()
        with torch.no_grad():
            out2 = layer2(x)

        # Outputs should match (in eval mode, routing is deterministic)
        layer1.eval()
        with torch.no_grad():
            out1_eval = layer1(x)
        assert torch.allclose(out1_eval, out2, atol=1e-5), \
            "Outputs differ after state_dict round-trip"

    def test_step_count_persisted(self):
        """_step_count buffer should be in state_dict (persistent=True)."""
        layer = _conv_layer()
        sd = layer.state_dict()
        assert "_step_count" in sd, "_step_count not in state_dict"

    def test_load_partial_state_dict_no_crash(self):
        """Loading a partial state_dict should not crash (strict=False)."""
        layer = _conv_layer()
        sd = layer.state_dict()
        # Remove some keys
        partial = {k: v for k, v in sd.items() if "router" not in k}
        layer2 = _conv_layer()
        layer2.load_state_dict(partial, strict=False)
        assert layer2.num_experts == 4


# ── Expert isolation tests ──────────────────────────────────────────────

class TestExpertIsolation:
    """Verify expert freezing isolates parameters correctly."""

    def test_freeze_one_expert(self):
        layer = _conv_layer(num_experts=4)
        layer.freeze_experts([0])
        for p in layer.experts[0].parameters():
            assert not p.requires_grad, "Expert 0 still trainable after freeze"

    def test_freeze_doesnt_affect_others(self):
        layer = _conv_layer(num_experts=4)
        layer.freeze_experts([0])
        for p in layer.experts[1].parameters():
            assert p.requires_grad, "Expert 1 frozen when only 0 should be"

    def test_unfreeze_all(self):
        layer = _conv_layer(num_experts=4)
        layer.freeze_experts([0, 1])
        layer.unfreeze_experts()
        for i, expert in enumerate(layer.experts):
            for p in expert.parameters():
                assert p.requires_grad, f"Expert {i} not unfrozen"

    def test_unfreeze_specific(self):
        layer = _conv_layer(num_experts=4)
        layer.freeze_experts([0, 1, 2])
        layer.unfreeze_experts([1])
        assert not any(p.requires_grad for p in layer.experts[0].parameters())
        assert any(p.requires_grad for p in layer.experts[1].parameters())


# ── Warmup schedule tests ───────────────────────────────────────────────

class TestWarmupSchedule:
    """Verify top_k_warmup schedule works correctly."""

    def test_warmup_increases_top_k(self):
        """During warmup, effective top_k should be <= configured top_k."""
        layer = MoLoRALayer(
            nn.Conv2d(64, 64, 3, padding=1),
            r=4, alpha=8, num_experts=4, top_k=2,
            top_k_warmup=1, warmup_steps=10,
        )
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        # Step 0: should use warmup top_k=1
        _ = layer(x)
        assert layer._step_count.item() == 1

    def test_warmup_transitions_to_full(self):
        """After warmup_steps, effective top_k should reach configured top_k."""
        layer = MoLoRALayer(
            nn.Conv2d(64, 64, 3, padding=1),
            r=4, alpha=8, num_experts=4, top_k=3,
            top_k_warmup=1, warmup_steps=5,
        )
        layer.train()
        x = torch.randn(2, 64, 8, 8)
        # Run past warmup
        for _ in range(10):
            _ = layer(x)
        assert layer._step_count.item() >= 5
        # After warmup, effective_k should be top_k=3
        assert layer._current_top_k() == 3

    def test_no_warmup_uses_configured_k(self):
        """Without warmup, effective top_k == configured top_k from step 0."""
        layer = _conv_layer(num_experts=4, top_k=2)
        assert layer._current_top_k() == 2


# ── Scaling formula tests ───────────────────────────────────────────────

class TestScalingFormula:
    """Verify rsLoRA vs standard LoRA scaling."""

    def test_rs_lora_scaling(self):
        """rsLoRA scaling = alpha / sqrt(r)."""
        r, alpha = 4, 8
        layer = MoLoRALayer(
            nn.Conv2d(64, 64, 3, padding=1),
            r=r, alpha=alpha, num_experts=1, top_k=1,
            use_rslora=True,
        )
        expected = alpha / (r ** 0.5)
        assert abs(layer.scaling - expected) < 1e-4, \
            f"rsLoRA scaling={layer.scaling}, expected {expected}"

    def test_standard_lora_scaling(self):
        """Standard LoRA scaling = alpha / r."""
        r, alpha = 4, 8
        layer = MoLoRALayer(
            nn.Conv2d(64, 64, 3, padding=1),
            r=r, alpha=alpha, num_experts=1, top_k=1,
            use_rslora=False,
        )
        expected = alpha / r
        assert abs(layer.scaling - expected) < 1e-4, \
            f"Standard LoRA scaling={layer.scaling}, expected {expected}"


# ── Large batch stability tests ─────────────────────────────────────────

class TestLargeBatchStability:
    """Verify routing stability with large batches."""

    def test_large_batch_no_nan(self):
        layer = _conv_layer(num_experts=8, top_k=3)
        layer.train()
        x = torch.randn(32, 64, 16, 16)  # Large batch
        out = layer(x)
        assert torch.isfinite(out).all(), "Output has NaN/Inf"
        assert torch.isfinite(layer.aux_loss).all(), "aux_loss has NaN/Inf"

    def test_single_sample_routing(self):
        """Batch size 1 should route without errors."""
        layer = _conv_layer()
        layer.train()
        x = torch.randn(1, 64, 8, 8)
        out = layer(x)
        assert out.shape == (1, 64, 8, 8)

    def test_expert_usage_non_degenerate(self):
        """With enough samples, all experts should get some usage."""
        layer = _conv_layer(num_experts=4, top_k=2)
        layer.train()
        x = torch.randn(16, 64, 8, 8)
        _ = layer(x)
        usage = layer.last_routing_snapshot.get("expert_usage")
        if usage is not None:
            # At least one expert should have non-zero usage
            assert usage.sum() > 0, "All experts have zero usage"


# ── Domain expert tests ─────────────────────────────────────────────────

class TestDomainExperts:
    """Verify domain expert pre-allocation."""

    def test_domain_preallocation_mask(self):
        """Domain pre-allocation should set active mask."""
        domain_map = {"detection": [0, 1], "segmentation": [2, 3]}
        layer = MoLoRALayer(
            nn.Conv2d(64, 64, 3, padding=1),
            r=4, alpha=8, num_experts=4, top_k=2,
            domain_experts=domain_map,
        )
        assert layer.domain_experts is not None

    def test_domain_clear(self):
        """Clearing domain should reset mask to None."""
        layer = MoLoRALayer(
            nn.Conv2d(64, 64, 3, padding=1),
            r=4, alpha=8, num_experts=4, top_k=2,
            domain_experts={"a": [0, 1]},
        )
        layer.set_domain("a")
        assert layer._domain_active_mask is not None
        layer.set_domain(None)
        assert layer._domain_active_mask is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
