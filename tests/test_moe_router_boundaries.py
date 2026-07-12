"""Boundary and error-handling tests for MoE routers.

Tests:
  - 3-D / 5-D input rejection (must be 4-D NCHW)
  - Channel mismatch detection
  - NaN/Inf input detection
  - NaN/Inf logits detection in BaseRouter._process_logits
  - top_k clamping (k > num_experts, k=0)
  - Exception hierarchy: MoERouterError/ShapeMismatchError inherit YOLOMasterError
"""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.routers import (
    UltraEfficientRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    LocalRoutingLayer,
    _validate_router_input,
)
from ultralytics.utils.errors import MoERouterError, ShapeMismatchError, YOLOMasterError


# =============================================================================
# Fixtures
# =============================================================================

IN_CHANNELS = 64
NUM_EXPERTS = 4
TOP_K = 2


@pytest.fixture
def ultra_router():
    return UltraEfficientRouter(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


@pytest.fixture
def spatial_router():
    return EfficientSpatialRouter(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


@pytest.fixture
def adaptive_router():
    return AdaptiveRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


@pytest.fixture
def local_router():
    return LocalRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


def _valid_input(batch=2, channels=IN_CHANNELS, h=16, w=16):
    return torch.randn(batch, channels, h, w)


# =============================================================================
# _validate_router_input unit tests
# =============================================================================

class TestValidateRouterInput:
    def test_valid_4d_input_passes(self):
        x = _valid_input()
        _validate_router_input(x, IN_CHANNELS)  # should not raise

    def test_3d_input_raises(self):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            _validate_router_input(x, IN_CHANNELS)

    def test_5d_input_raises(self):
        x = torch.randn(2, IN_CHANNELS, 16, 16, 1)
        with pytest.raises(MoERouterError, match="4-D"):
            _validate_router_input(x, IN_CHANNELS)

    def test_channel_mismatch_raises_shape_error(self):
        x = torch.randn(2, 32, 16, 16)  # 32 != 64
        with pytest.raises(ShapeMismatchError):
            _validate_router_input(x, IN_CHANNELS)

    def test_nan_input_raises(self):
        x = _valid_input()
        x[0, 0, 0, 0] = float("nan")
        with pytest.raises(MoERouterError, match="NaN"):
            _validate_router_input(x, IN_CHANNELS)

    def test_inf_input_raises(self):
        x = _valid_input()
        x[0, 0, 0, 0] = float("inf")
        with pytest.raises(MoERouterError, match="NaN"):
            _validate_router_input(x, IN_CHANNELS)


# =============================================================================
# UltraEfficientRouter boundary tests
# =============================================================================

class TestUltraEfficientRouterBoundaries:
    def test_valid_forward(self, ultra_router):
        x = _valid_input()
        ultra_router.eval()
        topk_vals, topk_idx, usage, imp, z = ultra_router(x)
        assert topk_vals.shape[0] == 2
        assert topk_idx.shape[0] == 2

    def test_3d_input_raises(self, ultra_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            ultra_router(x)

    def test_channel_mismatch_raises(self, ultra_router):
        x = torch.randn(2, 32, 16, 16)
        with pytest.raises(ShapeMismatchError):
            ultra_router(x)

    def test_nan_input_raises(self, ultra_router):
        x = _valid_input()
        x[0, 0, 0, 0] = float("nan")
        with pytest.raises(MoERouterError, match="NaN"):
            ultra_router(x)

    def test_top_k_exceeds_num_experts_clamped(self, ultra_router):
        ultra_router.eval()
        x = _valid_input()
        topk_vals, topk_idx, _, _, _ = ultra_router(x, top_k=100)
        assert topk_vals.shape[1] <= NUM_EXPERTS

    def test_top_k_zero_clamped_to_one(self, ultra_router):
        ultra_router.eval()
        x = _valid_input()
        topk_vals, topk_idx, _, _, _ = ultra_router(x, top_k=0)
        assert topk_vals.shape[1] >= 1


# =============================================================================
# EfficientSpatialRouter boundary tests
# =============================================================================

class TestEfficientSpatialRouterBoundaries:
    def test_valid_forward(self, spatial_router):
        x = _valid_input()
        spatial_router.eval()
        result = spatial_router(x)
        assert result[0].shape[0] == 2

    def test_3d_input_raises(self, spatial_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            spatial_router(x)

    def test_channel_mismatch_raises(self, spatial_router):
        x = torch.randn(2, 128, 16, 16)
        with pytest.raises(ShapeMismatchError):
            spatial_router(x)


# =============================================================================
# AdaptiveRoutingLayer boundary tests
# =============================================================================

class TestAdaptiveRoutingLayerBoundaries:
    def test_valid_forward(self, adaptive_router):
        x = _valid_input()
        adaptive_router.eval()
        result = adaptive_router(x)
        assert result[0].shape[0] == 2

    def test_3d_input_raises(self, adaptive_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            adaptive_router(x)


# =============================================================================
# LocalRoutingLayer boundary tests
# =============================================================================

class TestLocalRoutingLayerBoundaries:
    def test_valid_forward(self, local_router):
        x = _valid_input()
        local_router.eval()
        result = local_router(x)
        assert result[0].shape[0] == 2

    def test_3d_input_raises(self, local_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            local_router(x)


# =============================================================================
# Exception hierarchy tests
# =============================================================================

class TestExceptionHierarchy:
    def test_moerouter_error_inherits_yolomaster(self):
        assert issubclass(MoERouterError, YOLOMasterError)

    def test_shapemismatch_error_inherits_yolomaster(self):
        assert issubclass(ShapeMismatchError, YOLOMasterError)

    def test_catch_all_with_yolomaster_error(self):
        """Caller can catch all YOLO-Master errors with one except clause."""
        x = torch.randn(2, 32, 16, 16)
        router = UltraEfficientRouter(IN_CHANNELS, NUM_EXPERTS)
        with pytest.raises(YOLOMasterError):
            router(x)
