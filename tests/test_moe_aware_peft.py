"""Unit tests for MoE-aware PEFT extensions.

Coverage targets:
  - MoLoRAMoEAwareConfig validation
  - PerExpertRankAllocator (uniform + frequency modes)
  - RouterCalibration forward / shapes / param count
  - MoLoRAMoEAwareLayer integration (forward, calibration, per-expert ranks)
  - build_moe_aware_layer factory

Run:
    python -m pytest tests/test_moe_aware_peft.py -v --tb=short
"""
import math
from typing import List

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.peft.molora.moe_aware import (
    MoLoRAMoEAwareConfig,
    PerExpertRankAllocator,
    RouterCalibration,
    MoLoRAMoEAwareLayer,
    build_moe_aware_layer,
)
from ultralytics.nn.peft.molora.layer import MoLoRAExpert


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestMoLoRAMoEAwareConfig:
    def test_default_values(self):
        cfg = MoLoRAMoEAwareConfig()
        assert cfg.router_calibration is False
        assert cfg.router_calib_rank == 4
        assert cfg.per_expert_rank is False
        assert cfg.rank_allocator_mode == "frequency"
        assert cfg.rank_budget_total == 32
        assert cfg.rank_min == 2

    def test_validation_router_calib_rank(self):
        with pytest.raises(ValueError, match="router_calib_rank"):
            MoLoRAMoEAwareConfig(router_calib_rank=0)

    def test_validation_mode(self):
        with pytest.raises(ValueError, match="rank_allocator_mode"):
            MoLoRAMoEAwareConfig(rank_allocator_mode="invalid")

    def test_validation_budget(self):
        with pytest.raises(ValueError, match="rank_budget_total"):
            MoLoRAMoEAwareConfig(num_experts=4, rank_budget_total=4, rank_min=2)

    def test_inherits_molora(self):
        cfg = MoLoRAMoEAwareConfig(num_experts=4, top_k=2)
        assert cfg.num_experts == 4
        assert cfg.top_k == 2


# ---------------------------------------------------------------------------
# PerExpertRankAllocator
# ---------------------------------------------------------------------------

class TestPerExpertRankAllocator:
    def test_uniform_mode(self):
        alloc = PerExpertRankAllocator(num_experts=4, total_budget=32, min_rank=2, mode="uniform")
        usage = torch.tensor([0.4, 0.3, 0.2, 0.1])
        ranks = alloc.allocate(usage)
        assert len(ranks) == 4
        assert sum(ranks) == 32
        assert all(r >= 2 for r in ranks)

    def test_frequency_mode_basic(self):
        alloc = PerExpertRankAllocator(num_experts=4, total_budget=32, min_rank=2, mode="frequency")
        usage = torch.tensor([0.50, 0.20, 0.20, 0.10])
        ranks = alloc.allocate(usage)
        assert len(ranks) == 4
        assert sum(ranks) == 32
        assert all(r >= 2 for r in ranks)
        # Highest frequency should get highest rank
        assert ranks[0] >= ranks[3]

    def test_frequency_mode_equal(self):
        alloc = PerExpertRankAllocator(num_experts=4, total_budget=32, min_rank=2, mode="frequency")
        usage = torch.ones(4) / 4
        ranks = alloc.allocate(usage)
        assert len(ranks) == 4
        assert sum(ranks) == 32

    def test_invalid_total_budget(self):
        with pytest.raises(ValueError):
            PerExpertRankAllocator(num_experts=4, total_budget=4, min_rank=2)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            PerExpertRankAllocator(num_experts=4, total_budget=32, min_rank=2, mode="unknown")


# ---------------------------------------------------------------------------
# RouterCalibration
# ---------------------------------------------------------------------------

class TestRouterCalibration:
    def test_init(self):
        rc = RouterCalibration(in_channels=64, num_experts=4, r_r=4)
        assert rc.in_channels == 64
        assert rc.num_experts == 4
        assert rc.r_r == 4
        assert rc.lora_A.out_features == 4
        assert rc.lora_B.out_features == 4

    def test_forward_conv2d_input(self):
        rc = RouterCalibration(in_channels=64, num_experts=4, r_r=4)
        x = torch.randn(2, 64, 8, 8)
        router_logits = torch.randn(2, 4)
        # Warm up: one forward-backward so lora_B receives non-zero gradient
        out = rc(x, router_logits)
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            rc.lora_B.weight += 0.1  # perturb B away from zero
        # Now forward should differ from raw logits
        out2 = rc(x, router_logits)
        assert out2.shape == (2, 4)
        assert not torch.allclose(out2, router_logits)

    def test_forward_linear_input(self):
        rc = RouterCalibration(in_channels=64, num_experts=4, r_r=4)
        x = torch.randn(2, 64)
        router_logits = torch.randn(2, 4)
        out = rc(x, router_logits)
        assert out.shape == (2, 4)

    def test_zero_init_b(self):
        rc = RouterCalibration(in_channels=64, num_experts=4, r_r=4)
        # At init, lora_B is zero, so delta should be zero
        x = torch.randn(2, 64, 8, 8)
        router_logits = torch.randn(2, 4)
        with torch.no_grad():
            out = rc(x, router_logits)
        # Since B is zero, output should equal input logits
        assert torch.allclose(out, router_logits, atol=1e-6)

    def test_trainable_params(self):
        rc = RouterCalibration(in_channels=64, num_experts=4, r_r=4)
        trainable = sum(p.numel() for p in rc.parameters() if p.requires_grad)
        expected = 64 * 4 + 4 * 4  # A + B
        assert trainable == expected


# ---------------------------------------------------------------------------
# MoLoRAMoEAwareLayer
# ---------------------------------------------------------------------------

class TestMoLoRAMoEAwareLayer:
    def _make_layer(self, **kwargs):
        base = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        return MoLoRAMoEAwareLayer(base, num_experts=4, top_k=2, **kwargs)

    def test_forward_basic(self):
        layer = self._make_layer()
        x = torch.randn(2, 16, 8, 8)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)
        assert layer._last_routing_stats is not None

    def test_forward_with_calibration(self):
        rc = RouterCalibration(in_channels=16, num_experts=4, r_r=4)
        layer = self._make_layer(router_calibration=rc)
        x = torch.randn(2, 16, 8, 8)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)
        stats = layer._last_routing_stats
        assert stats["calibration_applied"] is True

    def test_forward_without_calibration(self):
        layer = self._make_layer(router_calibration=None)
        x = torch.randn(2, 16, 8, 8)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)
        stats = layer._last_routing_stats
        assert stats["calibration_applied"] is False

    def test_per_expert_ranks(self):
        ranks = [2, 4, 8, 4]
        layer = self._make_layer(expert_ranks=ranks)
        x = torch.randn(2, 16, 8, 8)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)
        assert layer._expert_ranks == ranks
        # Verify each expert has correct rank
        for i, r in enumerate(ranks):
            assert layer.experts[i].r == r

    def test_per_expert_ranks_mismatched_length(self):
        with pytest.raises(ValueError, match="expert_ranks length"):
            self._make_layer(expert_ranks=[2, 4])

    def test_merged_flag(self):
        layer = self._make_layer()
        assert not layer.merged
        layer.merge_weights()
        assert layer.merged
        x = torch.randn(2, 16, 8, 8)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)
        layer.unmerge_weights()
        assert not layer.merged

    def test_base_frozen(self):
        layer = self._make_layer()
        for p in layer.base_layer.parameters():
            assert not p.requires_grad

    def test_experts_trainable(self):
        layer = self._make_layer()
        for expert in layer.experts:
            for p in expert.parameters():
                assert p.requires_grad

    def test_router_calibration_trainable(self):
        rc = RouterCalibration(in_channels=16, num_experts=4, r_r=4)
        layer = self._make_layer(router_calibration=rc)
        for p in layer.router_calibration.parameters():
            assert p.requires_grad

    def test_extra_repr(self):
        layer = self._make_layer()
        repr_str = layer.extra_repr()
        assert "moe_aware=True" in repr_str


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildMoEAwareLayer:
    def test_factory_basic(self):
        base = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        cfg = MoLoRAMoEAwareConfig(
            r=8, alpha=16, num_experts=4, top_k=2,
            router_calibration=False, per_expert_rank=False,
        )
        layer = build_moe_aware_layer(base, cfg)
        assert isinstance(layer, MoLoRAMoEAwareLayer)
        assert layer.router_calibration is None
        assert layer._expert_ranks is None

    def test_factory_with_calibration(self):
        base = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        cfg = MoLoRAMoEAwareConfig(
            r=8, alpha=16, num_experts=4, top_k=2,
            router_calibration=True, router_calib_rank=4,
            per_expert_rank=False,
        )
        layer = build_moe_aware_layer(base, cfg)
        assert isinstance(layer, MoLoRAMoEAwareLayer)
        assert layer.router_calibration is not None
        assert layer.router_calibration.r_r == 4

    def test_factory_with_frequency_ranks(self):
        base = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        cfg = MoLoRAMoEAwareConfig(
            r=8, alpha=16, num_experts=4, top_k=2,
            per_expert_rank=True,
            rank_allocator_mode="frequency",
            rank_budget_total=32, rank_min=2,
        )
        layer = build_moe_aware_layer(base, cfg)
        assert isinstance(layer, MoLoRAMoEAwareLayer)
        assert layer._expert_ranks is not None
        assert len(layer._expert_ranks) == 4
        assert sum(layer._expert_ranks) == 32

    def test_factory_linear_base(self):
        base = nn.Linear(64, 128)
        cfg = MoLoRAMoEAwareConfig(
            r=8, alpha=16, num_experts=2, top_k=1,
            router_calibration=True, router_calib_rank=4,
        )
        layer = build_moe_aware_layer(base, cfg)
        assert isinstance(layer, MoLoRAMoEAwareLayer)
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128)


# ---------------------------------------------------------------------------
# Integration / Compatibility
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_import_from_public_api(self):
        from ultralytics.nn.peft.molora import (
            MoLoRAMoEAwareConfig,
            PerExpertRankAllocator,
            RouterCalibration,
            MoLoRAMoEAwareLayer,
            build_moe_aware_layer,
        )
        assert MoLoRAMoEAwareConfig is not None

    def test_backward_pass(self):
        layer = MoLoRAMoEAwareLayer(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            num_experts=4, top_k=2,
            router_calibration=RouterCalibration(8, 4, r_r=2),
        )
        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # Calibration params should have gradients
        if layer.router_calibration is not None:
            for p in layer.router_calibration.parameters():
                assert p.grad is not None
