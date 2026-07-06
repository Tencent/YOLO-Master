"""Unit tests for the PEFT Planner module.

Tests calibration of regression coefficients (Eq. 1) and hard policy rules
against the paper's experimental data (Table 1, Fig. 4, Table 2).
"""

import pytest
import torch
import torch.nn as nn

from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    LOVODataCollector,
    LOVODataPoint,
    LOVOValidator,
    PEFTPlanner,
    PEFTVariantProfile,
    PlacementDecision,
    RefusalError,
    is_planner_enabled,
)


# =============================================================================
# Helpers
# =============================================================================

class AAttn(nn.Module):
    """Dummy attention module (AAttn-style, no submodules).

    Mirrors the YOLO12 AAttn module for architecture-family detection.
    Using nn.MultiheadAttention internally would double-count attention
    submodules (e.g. ``out_proj``).  This clean module avoids that side-effect.
    """

    def forward(self, x):
        return x


class RTDETRDecoder(nn.Module):
    """Dummy RTDETR decoder module for architecture-family detection."""

    def forward(self, x):
        return x


class MockTextFusion(nn.Module):
    """Dummy text-fusion module."""

    def forward(self, x):
        return x


# =============================================================================
# Mock Models — deterministically build architecture fingerprints
# =============================================================================

def _make_yolo11s_like():
    """YOLO11s-like: no attention, no text-fusion (φ_attn=0, φ_text=0)."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 16, 3, padding=1)
            self.stage1 = nn.Conv2d(16, 32, 3, padding=1)
            self.stage2 = nn.Conv2d(32, 64, 3, padding=1)
            self.head = nn.Linear(64, 80)

    return _Model()


def _make_yolo12s_like():
    """YOLO12s-like: moderate attention ratio (φ_attn ≈ 0.45).

    6 conv + 3 linear = 9 total modules; 4 attention modules → 4/9 ≈ 0.444.
    """

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 16, 3, padding=1)
            self.stage1 = nn.Conv2d(16, 32, 3, padding=1)
            self.stage2 = nn.Conv2d(32, 64, 3, padding=1)
            self.stage3 = nn.Conv2d(64, 64, 3, padding=1)
            self.stage4 = nn.Conv2d(64, 64, 3, padding=1)
            self.stage5 = nn.Conv2d(64, 64, 3, padding=1)
            self.attn1 = AAttn()
            self.attn2 = AAttn()
            self.attn3 = AAttn()
            self.attn4 = AAttn()
            self.head1 = nn.Linear(64, 80)
            self.head2 = nn.Linear(80, 80)
            self.head3 = nn.Linear(80, 80)

    return _Model()


def _make_rtdetr_like():
    """RT-DETR-like: high attention ratio (φ_attn > 0.7).

    3 conv + 1 linear = 4 total modules; 3 attention modules → 3/4 = 0.75.
    """

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.attn1 = RTDETRDecoder()
            self.attn2 = RTDETRDecoder()
            self.attn3 = RTDETRDecoder()
            self.head = nn.Linear(64, 80)

    return _Model()


def _make_yolo_world_like():
    """YOLO-World-like: text-fusion modules (φ_text > 0.05).

    2 conv + 2 linear = 4 total modules; 2 text-fusion → 2/4 = 0.5.
    """

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 16, 3, padding=1)
            self.text_fusion_proj = nn.Linear(16, 32)
            self.fusion_conv = nn.Conv2d(16, 32, 1)
            self.head = nn.Linear(32, 80)

    return _Model()


# =============================================================================
# TestArchitectureFingerprint
# =============================================================================

class TestArchitectureFingerprint:
    """Test architecture fingerprint computation."""

    def test_yolo11s_like(self):
        model = _make_yolo11s_like()
        fp = ArchitectureFingerprint.compute(model)
        assert fp.phi_attn == 0.0
        assert fp.phi_text == 0.0
        assert fp.phi_dw == 0.0
        assert fp.phi_linear == pytest.approx(1 / 4, abs=1e-6)  # 1 linear / 4 modules

    def test_yolo12s_like(self):
        model = _make_yolo12s_like()
        fp = ArchitectureFingerprint.compute(model)
        # 6 conv + 3 linear = 9 total modules; 4 attention modules
        assert fp.phi_attn == pytest.approx(4 / 9, abs=1e-6)
        assert fp.phi_text == 0.0

    def test_rtdetr_like(self):
        model = _make_rtdetr_like()
        fp = ArchitectureFingerprint.compute(model)
        # 3 conv + 1 linear = 4 total; 3 attention
        assert fp.phi_attn == pytest.approx(0.75, abs=1e-6)
        assert fp.phi_text == 0.0

    def test_yolo_world_like(self):
        model = _make_yolo_world_like()
        fp = ArchitectureFingerprint.compute(model)
        # 2 conv + 2 linear = 4 total; text_fusion_proj and fusion_conv names give text
        assert fp.phi_text == pytest.approx(2 / 4, abs=1e-6)

    def test_depthwise_conv(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dw = nn.Conv2d(16, 16, 3, groups=16, padding=1)
                self.pw = nn.Conv2d(16, 32, 1)
                self.head = nn.Linear(32, 10)

        model = _Model()
        fp = ArchitectureFingerprint.compute(model)
        assert fp.phi_dw == 0.5  # 1 dw out of 2 convs


# =============================================================================
# TestPEFTVariantProfile
# =============================================================================

class TestPEFTVariantProfile:
    """Test calibrated variant profiles from Table 1."""

    def test_lora_xi_is_zero(self):
        prof = PEFTVariantProfile.from_variant("lora")
        assert prof.xi == 0.0

    def test_dora_xi_calibrated(self):
        prof = PEFTVariantProfile.from_variant("dora")
        # Calibrated to 0.0050 (YOLO11s DoRA r=16 is slightly above LoRA r=16 at +0.0710)
        assert prof.xi == pytest.approx(0.0050, abs=1e-3)

    def test_hra_xi_positive(self):
        prof = PEFTVariantProfile.from_variant("hra")
        # HRA is best performer on YOLO11s (+0.0848) and YOLO12s (+0.0791)
        assert prof.xi == pytest.approx(0.0152, abs=1e-3)
        assert prof.xi > 0.0

    def test_loha_xi_negative(self):
        prof = PEFTVariantProfile.from_variant("loha")
        # LoHa underperforms on YOLO11s (+0.0359 vs LoRA +0.0710)
        assert prof.xi == pytest.approx(-0.0208, abs=1e-3)
        assert prof.xi < 0.0

    def test_ia3_xi_negative(self):
        prof = PEFTVariantProfile.from_variant("ia3")
        assert prof.xi == pytest.approx(-0.0117, abs=1e-3)

    def test_lokr_xi_near_zero(self):
        prof = PEFTVariantProfile.from_variant("lokr")
        assert prof.xi == pytest.approx(-0.006, abs=1e-3)

    def test_unknown_variant_fallback(self):
        prof = PEFTVariantProfile.from_variant("unknown_variant")
        assert prof.xi == 0.0

    def test_case_insensitive(self):
        assert PEFTVariantProfile.from_variant("LoRa").xi == 0.0
        assert PEFTVariantProfile.from_variant("DORA").xi == pytest.approx(0.0050, abs=1e-3)


# =============================================================================
# TestPEFTPlannerFit
# =============================================================================

class TestPEFTPlannerFit:
    """Test regression fitting against paper Table 1 data."""

    _PAPER_HISTORY = [
        # YOLO11s (φ_attn=0, φ_text=0, φ_dw=0)
        (ArchitectureFingerprint(0.0, 0.0, 0.0), "lora", 0.0710),
        (ArchitectureFingerprint(0.0, 0.0, 0.0), "dora", 0.0710),
        (ArchitectureFingerprint(0.0, 0.0, 0.0), "loha", 0.0359),
        (ArchitectureFingerprint(0.0, 0.0, 0.0), "lokr", 0.0605),
        (ArchitectureFingerprint(0.0, 0.0, 0.0), "ia3", 0.0552),
        (ArchitectureFingerprint(0.0, 0.0, 0.0), "hra", 0.0848),
        # YOLO12s (φ_attn≈0.45, φ_text=0, φ_dw=0)
        (ArchitectureFingerprint(0.45, 0.0, 0.0), "lora", 0.0645),
        (ArchitectureFingerprint(0.45, 0.0, 0.0), "loha", 0.0560),
        (ArchitectureFingerprint(0.45, 0.0, 0.0), "ia3", 0.0548),
        (ArchitectureFingerprint(0.45, 0.0, 0.0), "hra", 0.0791),
    ]

    def test_fit_reproduces_default_coeffs(self):
        """Fitting on the paper data should recover the calibrated defaults."""
        planner = PEFTPlanner()
        planner.fit(self._PAPER_HISTORY)
        assert len(planner._coeffs) == 5
        # Default coeffs are calibrated to (0.067, 0.004, 0.0, 0.0, 1.0)
        assert planner._coeffs[0] == pytest.approx(0.067, abs=0.01)
        assert planner._coeffs[1] == pytest.approx(0.004, abs=0.01)
        assert planner._coeffs[4] == pytest.approx(1.0, abs=0.05)

    def test_fit_predicts_yolo11s_lora(self):
        planner = PEFTPlanner()
        planner.fit(self._PAPER_HISTORY)
        fp = ArchitectureFingerprint(0.0, 0.0, 0.0)
        pred = planner.predict(fp, "lora")
        assert pred == pytest.approx(0.0710, abs=0.01)

    def test_fit_predicts_yolo11s_hra(self):
        planner = PEFTPlanner()
        planner.fit(self._PAPER_HISTORY)
        fp = ArchitectureFingerprint(0.0, 0.0, 0.0)
        pred = planner.predict(fp, "hra")
        assert pred == pytest.approx(0.0848, abs=0.01)

    def test_fit_predicts_yolo12s_lora(self):
        planner = PEFTPlanner()
        planner.fit(self._PAPER_HISTORY)
        fp = ArchitectureFingerprint(0.45, 0.0, 0.0)
        pred = planner.predict(fp, "lora")
        assert pred == pytest.approx(0.0645, abs=0.01)

    def test_fit_predicts_yolo12s_hra(self):
        planner = PEFTPlanner()
        planner.fit(self._PAPER_HISTORY)
        fp = ArchitectureFingerprint(0.45, 0.0, 0.0)
        pred = planner.predict(fp, "hra")
        assert pred == pytest.approx(0.0791, abs=0.01)

    def test_insufficient_data_fallback(self):
        """With fewer than 5 samples, fit() should fall back to defaults."""
        planner = PEFTPlanner()
        planner.fit([])
        assert planner._coeffs == list(planner.DEFAULT_COEFFS)


# =============================================================================
# TestPEFTPlannerPlan
# =============================================================================

class TestPEFTPlannerPlan:
    """Test placement decisions against paper scenarios (Table 1, Fig. 4)."""

    def test_yolo11s_lora_accept(self):
        """YOLO11s + LoRA r=16 → ACCEPT (Table 1: Δ=+0.0710)."""
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(model, config)
        assert decision.status == "ACCEPT"
        assert decision.predicted_delta == pytest.approx(0.071, abs=0.01)
        assert "attn" not in (decision.target_modules_hint or [])

    def test_yolo11s_dora_accept(self):
        """YOLO11s + DoRA r=16 → ACCEPT (Table 1: Δ=+0.0710)."""
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=16, use_dora=True)
        decision = planner.plan(model, config)
        # phi_attn=0 < 0.3, so no DoRA downgrade
        assert decision.status == "ACCEPT"
        assert decision.predicted_delta == pytest.approx(0.071, abs=0.01)

    def test_yolo11s_loha_accept(self):
        """YOLO11s + LoHa → ACCEPT (Table 1: Δ=+0.0359)."""
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="loha", r=16)
        decision = planner.plan(model, config)
        assert decision.status == "ACCEPT"
        assert decision.predicted_delta == pytest.approx(0.036, abs=0.01)

    def test_yolo11s_hra_accept(self):
        """YOLO11s + HRA → ACCEPT (Table 1: Δ=+0.0848)."""
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="hra", r=16)
        decision = planner.plan(model, config)
        assert decision.status == "ACCEPT"
        assert decision.predicted_delta == pytest.approx(0.085, abs=0.01)

    def test_yolo12s_dora_adapt_to_lora(self):
        """YOLO12s + DoRA → ADAPT to LoRA (Fig. 4: 6/7 catastrophe rate)."""
        model = _make_yolo12s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=16, use_dora=True)
        decision = planner.plan(model, config)
        assert decision.status == "ADAPT"
        assert decision.recommended_variant == "lora"
        assert decision.safety_overrides.get("use_dora") is False

    def test_yolo12s_lora_high_rank_adapt(self):
        """YOLO12s + LoRA r=16 → ADAPT rank cap to 8 (attention-rich > 0.3)."""
        model = _make_yolo12s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(model, config)
        assert decision.status == "ADAPT"
        assert decision.recommended_rank == 8
        assert decision.safety_overrides.get("r") == 8

    def test_yolo12s_lora_rank_8_accept(self):
        """YOLO12s + LoRA r=8 → ACCEPT (rank safe, attention enabled by ADAPT)."""
        model = _make_yolo12s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=8, include_attention=False)
        decision = planner.plan(model, config)
        # Attention-rich >0.3 triggers include_attention override → ADAPT
        assert decision.status == "ADAPT"
        assert decision.safety_overrides.get("include_attention") is True
        assert decision.recommended_rank is None

    def test_rtdetr_lora_refuse(self):
        """RT-DETR + LoRA → REFUSE (Fig. 4: 7/7 catastrophe rate)."""
        model = _make_rtdetr_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(model, config)
        assert decision.status == "REFUSE"
        assert "RT-DETR-like" in decision.refusal_reason
        # predicted_delta is now from regression, not hardcoded -0.600.
        # phi_attn=0.75, lora xi=0.0: 0.0656 + 0.0026*0.75 = 0.06755
        assert decision.predicted_delta == pytest.approx(0.0676, abs=0.01)
        assert decision.predicted_delta is not None

    def test_rtdetr_hra_accept(self):
        """RT-DETR + HRA → ACCEPT (regression-dominant: HRA predicts positive)."""
        model = _make_rtdetr_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="hra", r=16)
        decision = planner.plan(model, config)
        # Regression-dominant: HRA is not LoRA-family; regression predicts
        # Δ = 0.0656 + 0.0026*0.75 + 1.0*0.0152 = 0.08275 > threshold.
        # Safety overrides may still trigger rank cap / attention enable on ADAPT.
        assert decision.status in ("ACCEPT", "ADAPT")
        assert decision.predicted_delta == pytest.approx(0.0828, abs=0.01)
        # The key regression-dominant assertion: no blanket IA3 override
        assert decision.recommended_variant != "ia3"

    def test_yolo_world_lora_adapt_regression(self):
        """YOLO-World + LoRA → ADAPT to best compatible variant (regression-dominant)."""
        model = _make_yolo_world_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(model, config)
        assert decision.status == "ADAPT"
        # LoRA does not support text-fusion; regression evaluates compatible
        # variants: IA3 (0.0539) > LoHa (0.0448). IA3 is the best choice.
        assert decision.recommended_variant == "ia3"
        assert decision.safety_overrides.get("variant_adapted") is True

    def test_yolo11s_no_attention_targets_disabled(self):
        """YOLO11s should have attention targets disabled."""
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=8, include_attention=True)
        decision = planner.plan(model, config)
        # Even though config says include_attention=True, planner should override
        assert decision.status == "ADAPT"
        assert decision.safety_overrides.get("include_attention") is False

    def test_yolo12s_attention_enabled(self):
        """YOLO12s should have safe attention enabled."""
        model = _make_yolo12s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="lora", r=8, include_attention=False)
        decision = planner.plan(model, config)
        assert decision.status == "ADAPT"
        assert decision.safety_overrides.get("include_attention") is True

    def test_plan_variant_wrapper(self):
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        decision = planner.plan_variant(model, "lora", 16)
        assert decision.status == "ACCEPT"
        assert decision.predicted_delta == pytest.approx(0.071, abs=0.01)

    def test_lovo_auto_fit_refuses_catastrophic(self):
        """When LOVO collector includes catastrophic data, plan() auto-fits and
        regression catches catastrophic patterns (regression-dominant + LOVO integration)."""
        points = list(TestLOVOEngine._PAPER_HISTORY)
        points.append(
            LOVODataPoint(
                ArchitectureFingerprint(0.85, 0.0, 0.0),
                "lora",
                -0.600,
                model_name="RT-DETR",
            )
        )
        collector = LOVODataCollector(points)
        planner = PEFTPlanner(lovo_collector=collector)
        model = _make_rtdetr_like()
        config = LoRAConfig(peft_type="lora", r=16)
        decision = planner.plan(model, config)
        # With LOVO-fitted coefficients including RT-DETR catastrophic point,
        # regression should predict below threshold, triggering REFUSE.
        assert decision.status == "REFUSE"
        assert decision.predicted_delta < -0.05

    def test_regression_dominant_no_hard_override_for_safe_hra(self):
        """RT-DETR + HRA: regression predicts positive, no blanket IA3 override.
        Safety overrides (rank cap / attention) may still produce ADAPT."""
        model = _make_rtdetr_like()
        planner = PEFTPlanner()
        config = LoRAConfig(peft_type="hra", r=16)
        decision = planner.plan(model, config)
        # The key regression-dominant assertion: no blanket IA3 override
        assert decision.recommended_variant != "ia3"
        # Regression predicts positive
        assert decision.predicted_delta == pytest.approx(0.0828, abs=0.01)


# =============================================================================
# TestPEFTPlannerDetectTargets
# =============================================================================

class TestPEFTPlannerDetectTargets:
    """Test architecture-conditioned target detection."""

    def test_yolo11s_conv_only(self):
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        targets = planner.detect_targets(model)
        assert all("attn" not in t for t in targets)
        assert any("stem" in t for t in targets)
        assert any("stage1" in t for t in targets)

    def test_yolo12s_includes_safe_attention(self):
        model = _make_yolo12s_like()
        planner = PEFTPlanner()
        targets = planner.detect_targets(model)
        assert len(targets) > 0
        # MockAttention has no submodules, so all conv/linear are included
        assert any("stem" in t for t in targets)
        assert any("head" in t for t in targets)

    def test_rtdetr_empty_targets(self):
        model = _make_rtdetr_like()
        planner = PEFTPlanner()
        targets = planner.detect_targets(model)
        # RT-DETR-like: no targets (refuse)
        assert targets == []

    def test_yolo_world_includes_text_fusion(self):
        model = _make_yolo_world_like()
        planner = PEFTPlanner()
        targets = planner.detect_targets(model)
        # text_fusion_proj and fusion_conv should be included
        assert any("text" in t for t in targets)
        assert any("fusion" in t for t in targets)

    def test_config_filter_exclude_modules(self):
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(exclude_modules=["head"])
        targets = planner.detect_targets(model, config)
        assert all("head" not in t for t in targets)

    def test_config_filter_only_backbone(self):
        model = _make_yolo11s_like()
        planner = PEFTPlanner()
        config = LoRAConfig(only_backbone=True)
        targets = planner.detect_targets(model, config)
        assert all("head" not in t for t in targets)


# =============================================================================
# TestPlacementDecision
# =============================================================================

class TestPlacementDecision:
    """Test PlacementDecision dataclass validation."""

    def test_default_accept(self):
        d = PlacementDecision()
        assert d.status == "ACCEPT"
        assert d.safety_overrides == {}

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            PlacementDecision(status="UNKNOWN")

    def test_refusal_error(self):
        err = RefusalError("test refusal")
        assert str(err) == "test refusal"


# =============================================================================
# TestIsPlannerEnabled
# =============================================================================

class TestIsPlannerEnabled:
    """Test is_planner_enabled helper."""

    def test_lora_planner_enabled_true(self):
        class _Config:
            lora_planner_enabled = True

        assert is_planner_enabled(_Config()) is True

    def test_planner_enabled_true(self):
        class _Config:
            planner_enabled = True

        assert is_planner_enabled(_Config()) is True

    def test_both_false(self):
        class _Config:
            lora_planner_enabled = False
            planner_enabled = False

        assert is_planner_enabled(_Config()) is False

    def test_missing_attrs(self):
        class _Config:
            pass

        assert is_planner_enabled(_Config()) is False


# =============================================================================
# TestRegressionMetrics (sanity-check against paper Table 2)
# =============================================================================

class TestRegressionMetrics:
    """Verify that the calibrated model achieves R² close to the paper value."""

    def test_r2_on_paper_data(self):
        """Using the calibrated default coefficients, predictions should be close
        to the paper's measured values (R² ≈ 0.870 on 10 canonical points).
        """
        planner = PEFTPlanner()
        data = [
            (ArchitectureFingerprint(0.0, 0.0, 0.0), "lora", 0.0710),
            (ArchitectureFingerprint(0.0, 0.0, 0.0), "dora", 0.0710),
            (ArchitectureFingerprint(0.0, 0.0, 0.0), "loha", 0.0359),
            (ArchitectureFingerprint(0.0, 0.0, 0.0), "lokr", 0.0605),
            (ArchitectureFingerprint(0.0, 0.0, 0.0), "ia3", 0.0552),
            (ArchitectureFingerprint(0.0, 0.0, 0.0), "hra", 0.0848),
            (ArchitectureFingerprint(0.45, 0.0, 0.0), "lora", 0.0645),
            (ArchitectureFingerprint(0.45, 0.0, 0.0), "loha", 0.0560),
            (ArchitectureFingerprint(0.45, 0.0, 0.0), "ia3", 0.0548),
            (ArchitectureFingerprint(0.45, 0.0, 0.0), "hra", 0.0791),
        ]
        y = []
        y_pred = []
        for fp, var, actual in data:
            pred = planner.predict(fp, var)
            y.append(actual)
            y_pred.append(pred)

        import numpy as np
        y_arr = np.array(y)
        y_pred_arr = np.array(y_pred)
        ss_res = np.sum((y_arr - y_pred_arr) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot
        # Paper reports R²=0.762 on full matrix; we expect ~0.87 on canonical points
        assert r2 >= 0.75, f"R²={r2:.4f} is too low (< 0.75)"
        assert r2 <= 0.95, f"R²={r2:.4f} is suspiciously high (> 0.95)"


# =============================================================================
# TestLOVOEngine
# =============================================================================

class TestLOVOEngine:
    """Test LOVO data collection and validation engine."""

    _PAPER_HISTORY = [
        # YOLO11s (φ_attn=0, φ_text=0, φ_dw=0)
        LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "lora", 0.0710, model_name="YOLO11s"),
        LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "dora", 0.0710, model_name="YOLO11s"),
        LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "loha", 0.0359, model_name="YOLO11s"),
        LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "lokr", 0.0605, model_name="YOLO11s"),
        LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "ia3", 0.0552, model_name="YOLO11s"),
        LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "hra", 0.0848, model_name="YOLO11s"),
        # YOLO12s (φ_attn≈0.45, φ_text=0, φ_dw=0)
        LOVODataPoint(ArchitectureFingerprint(0.45, 0.0, 0.0), "lora", 0.0645, model_name="YOLO12s"),
        LOVODataPoint(ArchitectureFingerprint(0.45, 0.0, 0.0), "loha", 0.0560, model_name="YOLO12s"),
        LOVODataPoint(ArchitectureFingerprint(0.45, 0.0, 0.0), "ia3", 0.0548, model_name="YOLO12s"),
        LOVODataPoint(ArchitectureFingerprint(0.45, 0.0, 0.0), "hra", 0.0791, model_name="YOLO12s"),
    ]

    def test_collector_add_and_len(self):
        collector = LOVODataCollector()
        collector.add(LOVODataPoint(ArchitectureFingerprint(0.0, 0.0, 0.0), "lora", 0.0710))
        assert len(collector) == 1

    def test_collector_to_history(self):
        collector = LOVODataCollector()
        collector.extend(self._PAPER_HISTORY)
        history = collector.to_history()
        assert len(history) == 10
        assert all(isinstance(h, tuple) and len(h) == 3 for h in history)

    def test_collector_summary(self):
        collector = LOVODataCollector()
        collector.extend(self._PAPER_HISTORY)
        summary = collector.summary()
        assert summary["n_total"] == 10
        assert summary["n_variants"] == 6
        assert summary["delta_mAP_min"] == pytest.approx(0.0359, abs=1e-6)

    def test_collector_save_load_roundtrip(self, tmp_path):
        collector = LOVODataCollector()
        collector.extend(self._PAPER_HISTORY)
        path = tmp_path / "lovo_data.json"
        collector.save(path)
        loaded = LOVODataCollector.load(path)
        assert len(loaded) == 10
        assert loaded.data_points[0].variant == "lora"

    def test_lovo_cross_validate_r2(self):
        validator = LOVOValidator()
        collector = LOVODataCollector()
        collector.extend(self._PAPER_HISTORY)
        result = validator.cross_validate(collector.data_points)
        # LOVO R² is naturally lower than full-fit R² because each leave-out
        # fit uses only 9 of 10 points; with only 10 samples and 5 parameters
        # the variance is high.
        assert result.lovo_r2 >= 0.5
        assert result.lovo_mse < 0.01

    def test_lovo_catastrophe_detection(self):
        # Build a collector that includes multiple catastrophic points
        points = list(self._PAPER_HISTORY)
        points.append(
            LOVODataPoint(ArchitectureFingerprint(0.85, 0.0, 0.0), "lora", -0.600, model_name="RT-DETR")
        )
        points.append(
            LOVODataPoint(ArchitectureFingerprint(0.45, 0.0, 0.0), "dora", -0.055, model_name="YOLO12s")
        )
        collector = LOVODataCollector(points)

        # Full-fit evaluation (not LOVO) — the regression should learn the
        # catastrophic pattern when the points are included in training.
        planner = PEFTPlanner()
        planner.fit(collector.to_history())
        pred = planner.predict(ArchitectureFingerprint(0.85, 0.0, 0.0), "lora")
        assert pred < -0.05  # Should predict catastrophic

        # LOVO evaluation — with at least 2 catastrophic points, some folds
        # will include one catastrophic point in training and predict the other.
        validator = LOVOValidator(threshold=-0.05)
        metrics = validator.evaluate_catastrophe_detection(collector)
        assert metrics["recall"] >= 0.5

    def test_lovo_decision_boundary(self):
        validator = LOVOValidator()
        collector = LOVODataCollector()
        collector.extend(self._PAPER_HISTORY)
        metrics = validator.evaluate_decision_boundary(collector)
        assert metrics["accuracy"] >= 0.8

    def test_lovo_full_report(self):
        validator = LOVOValidator()
        collector = LOVODataCollector()
        collector.extend(self._PAPER_HISTORY)
        report = validator.full_report(collector)
        assert "lovo" in report
        assert "catastrophe_detection" in report
        assert "decision_boundary" in report
        assert "summary" in report
        assert report["summary"]["n_samples"] == 10

    def test_lovo_insufficient_data_raises(self):
        validator = LOVOValidator()
        with pytest.raises(ValueError, match="at least 5"):
            validator.cross_validate([])
