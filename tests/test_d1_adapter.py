"""Multi-scale adapters and equivalent deterministic P3 upsampling."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scripts.d1 import train as p5
from ultralytics.nn.foundation.cache import FeatureCacheReader
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.modules import DINOFeaturePyramidAdapter, LatentMixture, SeparableBilinear2x

SOURCE_NAMES = ("block4", "block8", "block12")


EXPECTED_SHAPES = {
    "p3": (2, 64, 80, 80),
    "p4": (2, 128, 40, 40),
    "p5": (2, 256, 20, 20),
}


def make_features(
    *,
    batch: int = 2,
    channels: int = 384,
    height: int = 40,
    width: int = 40,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(
            batch,
            channels,
            height,
            width,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        for name in SOURCE_NAMES
    }


def test_formal_shapes_keys_and_candidate_order() -> None:
    adapter = DINOFeaturePyramidAdapter().eval()
    features = make_features()

    with torch.no_grad():
        outputs = adapter(features)

    assert tuple(outputs) == ("p3", "p4", "p5")
    assert adapter.source_names == SOURCE_NAMES
    assert adapter.pyramid_names == ("p3", "p4", "p5")
    assert adapter.out_channels == (64, 128, 256)
    assert adapter.strides == (8, 16, 32)
    for level, expected_shape in EXPECTED_SHAPES.items():
        assert len(outputs[level]) == 3
        assert all(tuple(candidate.shape) == expected_shape for candidate in outputs[level])
        for index, name in enumerate(SOURCE_NAMES):
            assert torch.equal(outputs[level][index], adapter.branches[level][name](features[name]))


def test_nine_branches_have_independent_parameters_and_group_norm_only() -> None:
    adapter = DINOFeaturePyramidAdapter()
    parameter_sets = []
    for level in adapter.pyramid_names:
        for name in adapter.source_names:
            parameters = {id(parameter) for parameter in adapter.branches[level][name].parameters()}
            assert parameters
            parameter_sets.append(parameters)

    assert len(parameter_sets) == 9
    assert all(first.isdisjoint(second) for i, first in enumerate(parameter_sets) for second in parameter_sets[i + 1 :])
    assert any(isinstance(module, nn.GroupNorm) for module in adapter.modules())
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in adapter.modules())


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"in_channels": 0}, ValueError),
        ({"in_channels": True}, ValueError),
        ({"source_names": "block4"}, TypeError),
        ({"source_names": ("block4", "block8")}, ValueError),
        ({"source_names": ("block4", "block4", "block12")}, ValueError),
        ({"source_names": ("block.4", "block8", "block12")}, ValueError),
        ({"pyramid_channels": (64, 128)}, ValueError),
        ({"pyramid_channels": (64, 0, 256)}, ValueError),
        ({"norm_groups": 0}, ValueError),
    ],
)
def test_invalid_constructor_arguments_fail_fast(kwargs, exception) -> None:
    with pytest.raises(exception):
        DINOFeaturePyramidAdapter(**kwargs)


def test_feature_keys_must_match_exactly() -> None:
    adapter = DINOFeaturePyramidAdapter()
    missing = make_features()
    missing.pop("block8")
    with pytest.raises(ValueError, match="missing"):
        adapter(missing)

    extra = make_features()
    extra["other"] = extra["block4"]
    with pytest.raises(ValueError, match="unexpected"):
        adapter(extra)


@pytest.mark.parametrize(
    ("mutate", "exception", "message"),
    [
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 40, dtype=torch.float32)), ValueError, "BCHW"),
        (lambda xs: xs.update(block8=torch.zeros(2, 383, 40, 40)), ValueError, "channels"),
        (lambda xs: xs.update(block8=torch.zeros(1, 384, 40, 40)), ValueError, "batch"),
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 38, 40)), ValueError, "spatial"),
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 40, 40, dtype=torch.float64)), ValueError, "dtype"),
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 40, 40, dtype=torch.int64)), TypeError, "floating"),
    ],
)
def test_invalid_feature_tensors_fail_fast(mutate, exception, message) -> None:
    features = make_features()
    mutate(features)
    with pytest.raises(exception, match=message):
        DINOFeaturePyramidAdapter()(features)


def test_odd_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        DINOFeaturePyramidAdapter()(make_features(height=39))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for a device-mismatch tensor")
def test_device_mismatch_is_rejected() -> None:
    features = make_features()
    features["block12"] = features["block12"].cuda()
    with pytest.raises(ValueError, match="device"):
        DINOFeaturePyramidAdapter()(features)


def test_forward_is_deterministic_and_state_dict_round_trips() -> None:
    torch.manual_seed(0)
    source = DINOFeaturePyramidAdapter().eval()
    restored = DINOFeaturePyramidAdapter().eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    features = make_features(batch=1)

    with torch.no_grad():
        first = source(features)
        repeated = source(features)
        loaded = restored(features)

    for level in source.pyramid_names:
        for a, b, c in zip(first[level], repeated[level], loaded[level]):
            assert torch.equal(a, b)
            assert torch.equal(a, c)
            assert torch.isfinite(a).all()


def test_all_branches_and_inputs_receive_gradients() -> None:
    adapter = DINOFeaturePyramidAdapter().train()
    features = make_features(batch=1, requires_grad=True)
    outputs = adapter(features)
    loss = sum(candidate.square().mean() for candidates in outputs.values() for candidate in candidates)
    loss.backward()

    for feature in features.values():
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0
    for level in adapter.pyramid_names:
        for name in adapter.source_names:
            convolution = next(
                module for module in adapter.branches[level][name].modules() if isinstance(module, nn.Conv2d)
            )
            assert convolution.weight.grad is not None
            assert torch.isfinite(convolution.weight.grad).all()
            assert convolution.weight.grad.abs().sum() > 0


def test_candidates_feed_three_single_scale_latent_mixtures() -> None:
    adapter = DINOFeaturePyramidAdapter().train()
    features = make_features(batch=1)
    candidates = adapter(features)
    mixtures = nn.ModuleDict(
        {
            level: LatentMixture([channels] * 3, channels, residual_init=0.01)
            for level, channels in zip(adapter.pyramid_names, adapter.out_channels)
        }
    ).train()

    outputs = {level: mixtures[level](candidates[level]) for level in adapter.pyramid_names}

    assert tuple(outputs["p3"].shape) == (1, 64, 80, 80)
    assert tuple(outputs["p4"].shape) == (1, 128, 40, 40)
    assert tuple(outputs["p5"].shape) == (1, 256, 20, 20)
    sum(output.square().mean() for output in outputs.values()).backward()
    assert all(any(parameter.grad is not None for parameter in mixtures[level].parameters()) for level in outputs)


def test_real_feature_cache_cuda_fp16() -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    if not cache_value:
        pytest.skip("D1_WP2_CACHE is not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    reader = FeatureCacheReader(Path(cache_value))
    sample_id = min(reader.records)
    cached = reader.get(sample_id)
    assert tuple(cached) == SOURCE_NAMES
    assert all(value.dtype == torch.float16 and tuple(value.shape) == (384, 40, 40) for value in cached.values())
    features = {name: value.unsqueeze(0).cuda(non_blocking=True) for name, value in cached.items()}
    adapter = DINOFeaturePyramidAdapter().cuda().train()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = adapter(features)
        loss = sum(candidate.float().square().mean() for values in outputs.values() for candidate in values)
    loss.backward()

    output_dtypes = {candidate.dtype for values in outputs.values() for candidate in values}
    assert len(output_dtypes) == 1
    assert all(candidate.is_floating_point() for values in outputs.values() for candidate in values)
    assert tuple(outputs["p3"][0].shape) == (1, 64, 80, 80)
    assert tuple(outputs["p4"][0].shape) == (1, 128, 40, 40)
    assert tuple(outputs["p5"][0].shape) == (1, 256, 20, 20)
    assert all(torch.isfinite(candidate).all() for values in outputs.values() for candidate in values)


MODE = "separable_bilinear2x"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((1, 1, 1, 1), (2, 3, 1, 7), (1, 3, 5, 1), (2, 3, 5, 7)))
def test_cpu_bilinear_output_and_gradient(dtype, shape):
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    expected = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    actual = SeparableBilinear2x()(z)
    grad = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z, grad)[0], torch.autograd.grad(expected, x, grad)[0], atol=2e-6, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32, torch.float16))
@pytest.mark.parametrize("shape", ((1, 1, 1, 1), (2, 3, 1, 7), (1, 2, 5, 1), (2, 3, 5, 7), (2, 64, 40, 40)))
@pytest.mark.parametrize("layout", ("contiguous", "channels_last", "transposed"))
def test_cuda_deterministic_output_gradient_and_repeat(dtype, shape, layout):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        x = torch.randn(shape, dtype=dtype, device="cuda")
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        elif layout == "transposed":
            x = x.transpose(-2, -1)
        x = x.detach().requires_grad_(True)
        z = x.detach().clone().requires_grad_(True)
        expected = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        actual = SeparableBilinear2x()(z)
        assert torch.equal(expected, actual)
        grad = torch.randn_like(actual)
        a = torch.autograd.grad(expected, x, grad)[0]
        b = torch.autograd.grad(actual, z, grad)[0]
        atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-6, 1e-5)
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        repeated_input = x.detach().clone().requires_grad_(True)
        repeated = SeparableBilinear2x()(repeated_input)
        repeated_grad = torch.autograd.grad(repeated, repeated_input, grad)[0]
        assert torch.equal(actual, repeated) and torch.equal(b, repeated_grad)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_double_backward_and_empty_state():
    module = SeparableBilinear2x()
    x = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))
    assert torch.autograd.gradgradcheck(module, (x,))
    assert not module.state_dict() and not list(module.parameters())


@pytest.mark.parametrize("mode", (None, True, [], "nearest", "typo"))
def test_invalid_p3_modes_fail_closed(mode):
    with pytest.raises(ValueError):
        DINOFeaturePyramidAdapter(p3_upsample_mode=mode)
    with pytest.raises(ValueError):
        p5.model_config("BASE", mode)


@pytest.mark.parametrize("variant", tuple(p5.VARIANTS))
def test_registered_model_preserves_initial_state_and_checkpoint(variant, tmp_path):
    legacy = p5.construct_model(variant)
    fast = p5.construct_model(variant, MODE)
    assert all(isinstance(fast.adapter.branches["p3"][n][-1], SeparableBilinear2x) for n in fast.source_names)
    assert all(isinstance(legacy.adapter.branches["p3"][n][-1], torch.nn.Upsample) for n in legacy.source_names)
    assert set(legacy.state_dict()) == set(fast.state_dict())
    fast.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in fast.parameters()) == p5.VARIANTS[variant]["downstream_parameters"]
    features = {n: torch.randn(1, 384, 4, 4) for n in fast.source_names}
    batch = {
        "features": features,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, items = fast(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.adapter.parameters())
    path = tmp_path / "checkpoint.pt"
    torch.save(fast.checkpoint_payload(), path)
    restored = D1FoundationDetectionModel.from_checkpoint_payload(torch.load(path, weights_only=False)).eval()
    assert restored.config_dict()["adapter"]["p3_upsample_mode"] == MODE
    with torch.no_grad():
        torch.testing.assert_close(fast.eval()(features)[0], restored(features)[0], rtol=0, atol=0)
