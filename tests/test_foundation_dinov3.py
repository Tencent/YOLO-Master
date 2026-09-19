"""DINOv3 Teacher protocol with offline and opt-in local-weight integration tests."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.nn.foundation import DINOv3Teacher


class DummyBackbone(nn.Module):
    def __init__(self, *, output="feature_maps", register_tokens=2, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(patch_size=4, hidden_size=hidden_size, num_register_tokens=register_tokens)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.output = output
        self.calls = []

    def forward(self, pixel_values):
        self.calls.append((pixel_values.shape, pixel_values.dtype, self.training))
        batch, _, height, width = pixel_values.shape
        grid_h, grid_w = height // 4, width // 4
        dense = self.scale * torch.ones(batch, self.config.hidden_size, grid_h, grid_w, device=pixel_values.device)
        if self.output == "feature_maps":
            return SimpleNamespace(feature_maps=(dense / 2, dense), pooler_output=dense.mean(dim=(2, 3)))
        tokens = self.scale * torch.arange(
            1 + self.config.num_register_tokens + grid_h * grid_w,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, -1, 1).expand(batch, -1, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)


class BadBackbone(DummyBackbone):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, pixel_values):
        output = super().forward(pixel_values)
        if self.kind == "nan":
            output.feature_maps = (torch.full_like(output.feature_maps[-1], float("nan")),)
        elif self.kind == "batch":
            output.feature_maps = (output.feature_maps[-1][:1],)
        elif self.kind == "grid":
            output.feature_maps = (output.feature_maps[-1][:, :, :-1, :],)
        elif self.kind == "pooled":
            output.pooler_output = torch.zeros(pixel_values.shape[0], 2)
        elif self.kind == "missing":
            return SimpleNamespace()
        return output


class MultiLayerBackbone(nn.Module):
    def __init__(self, *, output="feature_maps", fault=None, register_tokens=3):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=4,
            hidden_size=8,
            num_register_tokens=register_tokens,
            num_hidden_layers=12,
            stage_names=["stem", *(f"stage{index}" for index in range(1, 13))],
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.output = output
        self.fault = fault
        self.calls = []
        self.output_hidden_states_calls = []

    def forward(self, pixel_values, output_hidden_states=False):
        self.calls.append((pixel_values.shape, pixel_values.dtype, self.training))
        self.output_hidden_states_calls.append(output_hidden_states)
        batch, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.config.patch_size, width // self.config.patch_size
        feature_maps = []
        for stage in self.config.out_features:
            layer = int(stage.removeprefix("stage"))
            if self.output == "tokens":
                prefix = torch.full(
                    (batch, 1 + self.config.num_register_tokens, self.config.hidden_size),
                    -float(layer),
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
                patches = torch.full(
                    (batch, grid_h * grid_w, self.config.hidden_size),
                    float(layer),
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
                feature = self.scale * torch.cat((prefix, patches), dim=1)
            else:
                feature = self.scale * torch.full(
                    (batch, self.config.hidden_size, grid_h, grid_w),
                    float(layer),
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
            feature_maps.append(feature)

        if self.fault == "count":
            feature_maps.pop()
        elif self.fault == "batch":
            feature_maps[0] = feature_maps[0][:1]
        elif self.fault == "grid":
            feature_maps[1] = feature_maps[1][:, :, :-1, :]
        elif self.fault == "channel":
            feature_maps[1] = feature_maps[1][:, :-1, :, :]
        elif self.fault == "nan":
            feature_maps[1] = torch.full_like(feature_maps[1], float("nan"))
        elif self.fault == "inf":
            feature_maps[2] = torch.full_like(feature_maps[2], float("inf"))

        pooled = torch.full(
            (batch, self.config.hidden_size),
            12.0,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        if self.fault == "pooled":
            pooled = pooled[:, :-1]
        output = SimpleNamespace(feature_maps=tuple(feature_maps), pooler_output=pooled)
        if output_hidden_states:
            final_tokens = torch.full(
                (
                    batch,
                    1 + self.config.num_register_tokens + grid_h * grid_w,
                    self.config.hidden_size,
                ),
                12.0,
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            final_tokens[:, 0, :] = 120.0
            output.pooler_output = None
            output.hidden_states = (final_tokens,)
        return output


def test_dummy_teacher_is_frozen_and_always_eval():
    model = DummyBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu")

    assert teacher.training is False
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train(True)
    assert teacher.training is False
    assert model.training is False


def test_teacher_to_updates_device_and_dtype_without_unfreezing():
    model = DummyBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu")

    moved = teacher.to(dtype=torch.float64)

    assert moved is teacher
    assert teacher.device == torch.device("cpu")
    assert teacher.dtype == torch.float64
    assert next(teacher.parameters()).dtype == torch.float64
    assert teacher.preprocess(torch.zeros(1, 3, 4, 4)).dtype == torch.float64
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_feature_maps_output_is_normalized_to_p4_with_metadata():
    model = DummyBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu")
    result = teacher.encode(torch.zeros(2, 3, 5, 6))

    assert result.dense["p4"].shape == (2, 8, 2, 2)
    assert result.pooled.shape == (2, 8)
    assert result.metadata["input_size"] == (5, 6)
    assert result.metadata["padded_size"] == (8, 8)
    assert result.metadata["grid_size"] == (2, 2)
    assert result.metadata["num_register_tokens"] == 2
    assert torch.isfinite(result.dense["p4"]).all()
    assert model.calls[-1][0] == torch.Size((2, 3, 8, 8))


def test_token_sequence_output_respects_register_token_count():
    model = DummyBackbone(output="tokens", register_tokens=3)
    teacher = DINOv3Teacher(model=model, device="cpu")
    result = teacher.encode(torch.zeros(1, 3, 8, 12))

    assert result.dense["p4"].shape == (1, 8, 2, 3)
    assert result.metadata["prefix_tokens"] == 4
    assert result.pooled.shape == (1, 8)


def test_token_sequence_without_prefix_is_supported():
    model = DummyBackbone(output="tokens", register_tokens=2)
    teacher = DINOv3Teacher(model=model, device="cpu")
    tokens = torch.randn(1, 2, 8)
    result = teacher._parse_output(SimpleNamespace(feature_maps=tokens), batch_size=1, spatial_size=(4, 8))
    assert result.dense["p4"].shape == (1, 8, 1, 2)


def test_multilayer_output_uses_public_stages_and_returns_ordered_bchw_features():
    model = MultiLayerBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu", output_layers=(4, 8, 12))
    result = teacher.encode(torch.zeros(2, 3, 8, 12))

    assert model.config.out_features == ["stage4", "stage8", "stage12"]
    assert tuple(result.dense) == ("block4", "block8", "block12")
    for layer, name in zip((4, 8, 12), result.dense):
        assert result.dense[name].shape == (2, 8, 2, 3)
        assert torch.all(result.dense[name] == layer)
    assert torch.all(result.pooled == 12)
    assert result.metadata["output_layers"] == (4, 8, 12)
    assert result.metadata["output_layer_indices"] == (3, 7, 11)
    assert result.metadata["backbone_stages"] == ("stage4", "stage8", "stage12")
    assert result.metadata["feature_names"] == ("block4", "block8", "block12")
    assert result.metadata["grid_size"] == (2, 3)
    assert result.metadata["patch_size"] == 4
    assert result.metadata["hidden_dim"] == 8
    assert result.metadata["num_register_tokens"] == 3
    assert result.metadata["prefix_tokens"] == 4


def test_multilayer_token_outputs_remove_cls_and_register_tokens():
    model = MultiLayerBackbone(output="tokens", register_tokens=4)
    teacher = DINOv3Teacher(model=model, device="cpu", output_layers=(4, 8, 12))
    result = teacher.encode(torch.zeros(1, 3, 8, 8))

    for layer, feature in zip((4, 8, 12), result.dense.values()):
        assert feature.shape == (1, 8, 2, 2)
        assert torch.all(feature == layer)
    assert result.metadata["prefix_tokens"] == 5


def test_pooled_uses_final_layer_when_dense_selection_omits_it():
    model = MultiLayerBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu", output_layers=(4, 8))
    result = teacher.encode(torch.zeros(1, 3, 8, 8))

    assert tuple(result.dense) == ("block4", "block8")
    assert model.output_hidden_states_calls == [True]
    assert torch.all(result.pooled == 120)


@pytest.mark.parametrize(
    "output_layers, exception",
    [
        ("4,8,12", TypeError),
        ({4, 8, 12}, TypeError),
        ([], ValueError),
        ([4, 4], ValueError),
        ([8, 4], ValueError),
        ([0], ValueError),
        ([-1], ValueError),
        ([True], TypeError),
        ([4, "8"], TypeError),
        ([13], ValueError),
    ],
)
def test_invalid_output_layers_fail_fast(output_layers, exception):
    with pytest.raises(exception):
        DINOv3Teacher(model=MultiLayerBackbone(), device="cpu", output_layers=output_layers)


def test_missing_public_backbone_stage_fails_fast():
    model = MultiLayerBackbone()
    model.config.stage_names.remove("stage8")
    with pytest.raises(ValueError, match="requested stages"):
        DINOv3Teacher(model=model, device="cpu", output_layers=(4, 8, 12))


@pytest.mark.parametrize(
    "fault, message",
    [
        ("count", "returned 2 feature maps"),
        ("batch", "batch"),
        ("grid", "grid"),
        ("channel", "channels"),
        ("nan", "NaN or Inf"),
        ("inf", "NaN or Inf"),
        ("pooled", "pooled"),
    ],
)
def test_invalid_multilayer_outputs_fail_fast(fault, message):
    teacher = DINOv3Teacher(
        model=MultiLayerBackbone(fault=fault),
        device="cpu",
        output_layers=(4, 8, 12),
    )
    with pytest.raises(ValueError, match=message):
        teacher.encode(torch.zeros(2, 3, 8, 8))


def test_encode_restores_model_eval_and_inference_mode():
    model = MultiLayerBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu", output_layers=(4, 8, 12))
    model.train(True)
    model.scale.requires_grad_(True)

    result = teacher.encode(torch.zeros(1, 3, 8, 8, requires_grad=True))

    assert model.calls[-1][-1] is False
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(not feature.requires_grad for feature in result.dense.values())
    assert result.pooled.requires_grad is False


def test_model_loader_is_injected_without_transformers_or_network_access():
    model = DummyBackbone()
    calls = []

    def loader(model_id, weights_path):
        calls.append((model_id, weights_path))
        return model

    teacher = DINOv3Teacher(model_id="local-dummy", weights_path="dummy.bin", model_loader=loader, device="cpu")
    assert calls == [("local-dummy", "dummy.bin")]
    assert teacher.model is model


def test_model_loader_with_single_argument_is_supported():
    model = DummyBackbone()
    calls = []

    def loader(model_id):
        calls.append(model_id)
        return model

    teacher = DINOv3Teacher(model_id="single-arg", model_loader=loader, device="cpu")
    assert calls == ["single-arg"]
    assert teacher.model is model


@pytest.mark.parametrize(
    "kind, message",
    [
        ("nan", "NaN or Inf"),
        ("batch", "batch"),
        ("grid", "grid"),
        ("pooled", "pooled"),
        ("missing", "does not contain"),
    ],
)
def test_invalid_backbone_outputs_fail_fast(kind, message):
    teacher = DINOv3Teacher(model=BadBackbone(kind), device="cpu")
    with pytest.raises(ValueError, match=message):
        teacher.encode(torch.zeros(2, 3, 8, 8))


def test_loader_internal_type_error_is_not_retried():
    calls = []

    def loader(model_id, weights_path):
        calls.append((model_id, weights_path))
        raise TypeError("loader failure")

    with pytest.raises(TypeError, match="loader failure"):
        DINOv3Teacher(model_loader=loader, device="cpu")
    assert len(calls) == 1


def test_real_dinov3_vits16_multilayer_640():
    weights_value = os.getenv("D1_DINOV3_WEIGHTS")
    if not weights_value:
        pytest.skip("set D1_DINOV3_WEIGHTS to a local Transformers model directory")
    if not torch.cuda.is_available():
        pytest.skip("DINOv3 real-weight integration test requires CUDA")

    weights = Path(weights_value)
    assert weights.is_dir(), f"D1_DINOV3_WEIGHTS is not a directory: {weights}"
    teacher = DINOv3Teacher(
        weights_path=weights,
        local_files_only=True,
        dtype="fp16",
        device="cuda:0",
        output_layers=(4, 8, 12),
    )
    torch.manual_seed(0)
    images = torch.rand(1, 3, 640, 640)

    first = teacher.encode(images)
    second = teacher.encode(images)

    assert tuple(first.dense) == ("block4", "block8", "block12")
    assert all(feature.shape == (1, 384, 40, 40) for feature in first.dense.values())
    assert first.metadata["output_layers"] == (4, 8, 12)
    assert first.metadata["output_layer_indices"] == (3, 7, 11)
    assert first.metadata["backbone_stages"] == ("stage4", "stage8", "stage12")
    assert first.metadata["feature_names"] == ("block4", "block8", "block12")
    assert first.metadata["prefix_tokens"] == 5
    assert first.metadata["grid_size"] == (40, 40)
    assert first.metadata["hidden_dim"] == 384
    assert teacher.training is False
    assert teacher.model.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(not feature.requires_grad for feature in first.dense.values())
    for name in first.dense:
        torch.testing.assert_close(first.dense[name], second.dense[name], rtol=0, atol=0)
    torch.testing.assert_close(first.pooled, second.pooled, rtol=0, atol=0)
