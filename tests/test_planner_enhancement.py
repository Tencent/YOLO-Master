"""Tests for PEFT Planner enhancements (audit, cache, unwrap, metadata).

Covers features added in the optimization pass:
  - DecisionAudit serialization and persistence
  - ArchitectureFingerprint WeakKeyDictionary caching
  - _unwrap_model for DDP / DataParallel / compiled models
  - PlacementDecision.to_dict() for metadata propagation
  - Empty-model robust handling
"""

import json
import tempfile
import weakref
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    DecisionAudit,
    PEFTPlanner,
    PlacementDecision,
    _fingerprint_cache,
)


# ============================================================================
# DecisionAudit
# ============================================================================

class TestDecisionAudit:
    """Test the structured audit record."""

    def test_to_dict(self):
        audit = DecisionAudit(
            timestamp="2026-07-05T12:00:00",
            model_name="yolo11s",
            fingerprint={"phi_attn": 0.0, "phi_text": 0.0},
            variant="lora",
            requested_rank=16,
            decision_status="ACCEPT",
            predicted_delta=0.0656,
            target_modules_count=42,
        )
        d = audit.to_dict()
        assert d["decision_status"] == "ACCEPT"
        assert d["predicted_delta"] == pytest.approx(0.0656)
        assert d["target_modules_count"] == 42

    def test_save_and_load(self, tmp_path):
        audit = DecisionAudit(
            timestamp="2026-07-05T12:00:00",
            model_name="test_model",
            fingerprint={"phi_attn": 0.0},
            variant="lora",
            requested_rank=8,
            decision_status="ADAPT",
            recommended_variant="loha",
            recommended_rank=4,
            predicted_delta=0.03,
            safety_overrides={"r": 4},
            target_modules_count=10,
        )
        path = audit.save(audit_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = DecisionAudit.load(path)
        assert loaded.decision_status == "ADAPT"
        assert loaded.recommended_variant == "loha"
        assert loaded.safety_overrides == {"r": 4}

    def test_save_creates_directory(self, tmp_path):
        audit_dir = tmp_path / "nested" / "audit"
        audit = DecisionAudit(
            timestamp="2026-07-05T12:00:00",
            model_name="m",
            fingerprint={},
            variant="lora",
            requested_rank=0,
            decision_status="REFUSE",
            refusal_reason="test",
            target_modules_count=0,
        )
        path = audit.save(audit_dir=audit_dir)
        assert path.exists()


# ============================================================================
# ArchitectureFingerprint caching
# ============================================================================

class TestFingerprintCache:
    """Test the WeakKeyDictionary cache."""

    def test_cache_hit(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        model = _Model()
        fp1 = ArchitectureFingerprint.compute(model)
        fp2 = ArchitectureFingerprint.compute(model)
        assert fp1 is fp2  # same object (cached)

    def test_cache_auto_invalidation_on_gc(self):
        """WeakKeyDictionary drops entries when the model is GC'd."""
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        # Count before
        before = len(_fingerprint_cache)

        # Create model, compute fingerprint, then drop reference
        model = _Model()
        ArchitectureFingerprint.compute(model)
        assert len(_fingerprint_cache) == before + 1

        del model
        # Force gc so WeakKeyDictionary can clean up
        import gc
        gc.collect()

        assert len(_fingerprint_cache) == before

    def test_invalidate_cache_explicit(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        model = _Model()
        ArchitectureFingerprint.compute(model)
        assert model in _fingerprint_cache

        ArchitectureFingerprint.invalidate_cache(model)
        assert model not in _fingerprint_cache

    def test_cache_no_leak_across_different_instances(self):
        """Different model instances should not share cached fingerprints."""
        class _Model(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.conv = nn.Conv2d(3, n, 3)

        m1 = _Model(16)
        m2 = _Model(32)
        fp1 = ArchitectureFingerprint.compute(m1)
        fp2 = ArchitectureFingerprint.compute(m2)
        assert fp1 is not fp2


# ============================================================================
# _unwrap_model
# ============================================================================

class TestUnwrapModel:
    """Test DDP / DataParallel / torch.compile unwrapping."""

    def test_plain_model(self):
        m = nn.Conv2d(3, 16, 3)
        assert ArchitectureFingerprint._unwrap_model(m) is m

    def test_dataparallel(self):
        m = nn.Sequential(nn.Conv2d(3, 16, 3))
        dp = DataParallel(m)
        unwrapped = ArchitectureFingerprint._unwrap_model(dp)
        assert unwrapped is m

    def test_nested_dataparallel(self):
        m = nn.Sequential(nn.Conv2d(3, 16, 3))
        dp = DataParallel(DataParallel(m))
        unwrapped = ArchitectureFingerprint._unwrap_model(dp)
        assert unwrapped is m

    def test_compiled_model(self):
        """torch.compile wraps model in _orig_mod."""
        m = nn.Sequential(nn.Conv2d(3, 16, 3))
        # Mock the compile wrapper by injecting _orig_mod
        compiled = nn.Sequential(nn.Conv2d(3, 16, 3))
        compiled._orig_mod = m
        unwrapped = ArchitectureFingerprint._unwrap_model(compiled)
        assert unwrapped is m

    def test_dataparallel_with_orig_mod(self):
        """DDP + compile: both .module and ._orig_mod."""
        inner = nn.Sequential(nn.Conv2d(3, 16, 3))
        compiled = nn.Sequential(nn.Conv2d(3, 16, 3))
        compiled._orig_mod = inner
        dp = DataParallel(compiled)
        unwrapped = ArchitectureFingerprint._unwrap_model(dp)
        assert unwrapped is inner


# ============================================================================
# PlacementDecision.to_dict
# ============================================================================

class TestPlacementDecisionDict:
    def test_accept_to_dict(self):
        d = PlacementDecision(status="ACCEPT", predicted_delta=0.05)
        dd = d.to_dict()
        assert dd["status"] == "ACCEPT"
        assert dd["predicted_delta"] == pytest.approx(0.05)
        assert dd["target_modules_hint_count"] == 0

    def test_refuse_to_dict(self):
        d = PlacementDecision(
            status="REFUSE",
            refusal_reason="unsafe",
            predicted_delta=-0.1,
            target_modules_hint=["a", "b"],
        )
        dd = d.to_dict()
        assert dd["status"] == "REFUSE"
        assert dd["refusal_reason"] == "unsafe"
        assert dd["target_modules_hint_count"] == 2

    def test_adapt_to_dict(self):
        d = PlacementDecision(
            status="ADAPT",
            recommended_variant="loha",
            recommended_rank=8,
            predicted_delta=0.03,
            safety_overrides={"r": 8},
        )
        dd = d.to_dict()
        assert dd["recommended_variant"] == "loha"
        assert dd["recommended_rank"] == 8
        assert dd["safety_overrides"] == {"r": 8}


# ============================================================================
# Empty model handling
# ============================================================================

class TestEmptyModel:
    """Test that models with no Conv2d / Linear return zero fingerprint."""

    def test_batchnorm_only_model(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(16)

        m = _Model()
        fp = ArchitectureFingerprint.compute(m)
        assert fp.phi_attn == 0.0
        assert fp.phi_dw == 0.0
        assert fp.phi_group == 0.0
        assert fp.phi_linear == 0.0

    def test_relu_only_model(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.ReLU()

        m = _Model()
        fp = ArchitectureFingerprint.compute(m)
        assert fp.phi_attn == 0.0


# ============================================================================
# PEFTPlanner audit integration
# ============================================================================

class TestPlannerAuditIntegration:
    """Test that plan() emits audit records."""

    def test_audit_file_created_on_accept(self, tmp_path):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

        from ultralytics.utils.lora.config import LoRAConfig

        planner = PEFTPlanner(audit_dir=tmp_path)
        config = LoRAConfig(peft_type="lora", r=8)
        decision = planner.plan(_Model(), config)
        assert decision.status == "ACCEPT"

        # Audit should be saved
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["decision_status"] == "ACCEPT"
        assert data["variant"] == "lora"

    def test_audit_file_created_on_refuse(self, tmp_path):
        class _RTDETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.decoder = nn.Module()
                self.decoder.__class__.__name__ = "RTDETRDecoder"

        from ultralytics.utils.lora.config import LoRAConfig

        planner = PEFTPlanner(audit_dir=tmp_path)
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(_RTDETR(), config)
        assert decision.status == "REFUSE"

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["decision_status"] == "REFUSE"
        assert "RT-DETR" in data["refusal_reason"]

    def test_audit_file_created_on_adapt(self, tmp_path):
        class _YOLO12(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.attn = nn.Module()
                self.attn.__class__.__name__ = "AAttn"

        from ultralytics.utils.lora.config import LoRAConfig

        planner = PEFTPlanner(audit_dir=tmp_path)
        config = LoRAConfig(peft_type="dora", r=16)
        decision = planner.plan(_YOLO12(), config)
        assert decision.status == "ADAPT"

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["decision_status"] == "ADAPT"
        assert data["recommended_variant"] == "lora"  # DoRA degraded to LoRA

