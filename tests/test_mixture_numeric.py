"""Contract tests for shared routed-module numerical helpers."""

from unittest.mock import patch

import pytest
import torch

from ultralytics.nn.modules import LatentRouter
from ultralytics.nn.modules._numeric import (
    all_reduce_mean,
    clamp_min_for_dtype,
    disabled_autocast,
    fp_clamp_floor,
    stable_normalize,
)
from ultralytics.nn.modules.moa.router import _MoARouter
from ultralytics.nn.modules.moe.gated import DualStreamGateRouter, ZeroCostRouter
from ultralytics.nn.modules.mot.router import _MoTRouter
from ultralytics.nn.modules.moa.moa import all_reduce_mean as moa_all_reduce_mean
from ultralytics.nn.modules.moe.loss import all_reduce_mean as moe_all_reduce_mean
from ultralytics.nn.modules.mot.mot import all_reduce_mean as mot_all_reduce_mean
from ultralytics.nn.modules.moa._constants import ROUTER_LOGIT_LIMIT as MOA_ROUTER_LOGIT_LIMIT
from ultralytics.nn.modules.mot._constants import ROUTER_LOGIT_LIMIT as MOT_ROUTER_LOGIT_LIMIT


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(torch.float32, 1e-6), (torch.float16, 1e-4), (torch.bfloat16, 1e-3)],
)
def test_fp_clamp_floor_is_dtype_aware(dtype, expected):
    assert fp_clamp_floor(1e-6, dtype) == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_clamp_min_for_dtype_keeps_zero_denominator_positive(dtype):
    result = clamp_min_for_dtype(torch.zeros(2, dtype=dtype))
    assert torch.all(result > 0)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_stable_normalize_is_finite_and_preserves_gradients(dtype):
    values = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=dtype, requires_grad=True)
    normalized = stable_normalize(values, dim=1)
    normalized.float().sum().backward()

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(values.grad).all()
    assert torch.allclose(normalized[1].float(), torch.tensor([0.25, 0.75]), atol=2e-3, rtol=2e-3)


def test_all_mixture_namespaces_share_canonical_all_reduce_mean():
    assert moa_all_reduce_mean is all_reduce_mean
    assert moe_all_reduce_mean is all_reduce_mean
    assert mot_all_reduce_mean is all_reduce_mean


def test_disabled_autocast_falls_back_when_device_is_unsupported():
    with patch("torch.amp.autocast_mode.is_autocast_available", return_value=False), patch(
        "torch.autocast", side_effect=AssertionError("unsupported autocast must not be constructed")
    ):
        with disabled_autocast("mps"):
            result = torch.ones(1) + 1

    assert result.item() == 2


def test_disabled_autocast_uses_supported_cpu_context():
    with patch("torch.amp.autocast_mode.is_autocast_available", return_value=True), patch(
        "torch.autocast", wraps=torch.autocast
    ) as autocast:
        with disabled_autocast("cpu"):
            result = torch.ones(1) + 1

    autocast.assert_called_once_with(device_type="cpu", enabled=False)
    assert result.item() == 2


def test_all_reduce_mean_keeps_global_value_and_local_gradient():
    local = torch.tensor([0.2, 0.8], requires_grad=True)

    def add_remote_rank(value, op=None):
        value.add_(torch.tensor([0.4, 0.6]))

    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_world_size", return_value=2), patch(
        "torch.distributed.get_backend", return_value="gloo"
    ), patch("torch.distributed.all_reduce", side_effect=add_remote_rank):
        result = all_reduce_mean(local)
        result.sum().backward()

    assert torch.allclose(result, torch.tensor([0.3, 0.7]))
    assert torch.allclose(local.grad, torch.ones_like(local))


def test_router_logit_limits_remain_module_specific():
    """The shared constants cleanup must not merge semantically distinct router policies."""
    assert MOA_ROUTER_LOGIT_LIMIT == 80.0
    assert MOT_ROUTER_LOGIT_LIMIT == 80.0


def test_moa_router_keeps_fp32_logits_and_activation_dtype_weights():
    router = _MoARouter(8, num_groups=3).eval().half()
    x = torch.randn(2, 8, 4, 4, dtype=torch.bfloat16)

    probs, logits = router(x, return_logits=True)

    assert probs.dtype == x.dtype
    assert logits.dtype == torch.float32
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.float().sum(dim=1), torch.ones_like(probs[:, 0].float()), atol=2e-3)


def test_mot_router_keeps_fp32_logits_and_activation_dtype_weights():
    router = _MoTRouter(8, num_experts=3, top_k=2).eval().half()
    x = torch.randn(2, 8, 4, 4, dtype=torch.bfloat16)

    weights, indices, logits = router(x, return_logits=True)

    assert weights.dtype == x.dtype
    assert logits.dtype == torch.float32
    assert indices.dtype == torch.long
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.float().sum(dim=1), torch.ones_like(weights[:, 0].float()), atol=2e-3)


def test_latent_router_keeps_fp32_routing_contract_after_dtype_conversion():
    router = LatentRouter(8, num_experts=3).eval().half()
    tokens = torch.randn(2, 8, dtype=torch.bfloat16)

    logits, probs = router(tokens)

    assert logits.dtype == torch.float32
    assert probs.dtype == torch.float32
    assert {parameter.dtype for parameter in router.parameters()} == {torch.float32}
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[:, 0]), atol=1e-6)


@pytest.mark.parametrize("router_cls", [DualStreamGateRouter, ZeroCostRouter])
def test_gated_router_keeps_fp32_aux_statistics_and_activation_dtype_weights(router_cls):
    router = router_cls(8, num_experts=3, top_k=2).train().half()
    x = torch.randn(2, 8, 4, 4, dtype=torch.bfloat16)

    weights, indices, stats = router(x)

    assert weights.dtype == x.dtype
    assert indices.dtype == torch.long
    assert stats["router_probs"].dtype == torch.float32
    assert stats["router_logits"].dtype == torch.float32
    assert {parameter.dtype for parameter in router.parameters()} == {torch.float32}
    assert torch.isfinite(weights).all()
