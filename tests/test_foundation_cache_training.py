"""Offline contracts for training-side Foundation cache consumption (foundation_cache_dir)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.nn.foundation import FoundationFeatures, save_foundation_features
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel, build_foundation_distillation_wrapper
from ultralytics.nn.modules.head import Detect


class TinyStudent(nn.Module):
    """Small Detect graph with a real P4 source and a task-loss test double."""

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, 4, 1), nn.Conv2d(4, 8, 1), nn.Conv2d(8, 16, 1), nn.Conv2d(16, 32, 1)])
        head = Detect(nc=2, ch=(8, 16, 32))
        head.f, head.i = [1, 2, 3], 4
        self.model.append(head)
        self.yaml = {"channels": 3}
        self.stride = torch.tensor([32])
        self.nc = 2
        self.names = {0: "zero", 1: "one"}
        self.args = SimpleNamespace(imgsz=64)
        self.criterion = None

    def forward(self, x, *args, **kwargs):
        outputs = []
        for index, layer in enumerate(self.model):
            if index == self.model[-1].i:
                return layer([outputs[source] for source in layer.f])
            x = layer(x)
            outputs.append(x)
        raise AssertionError("unreachable")

    def loss(self, batch, preds=None):
        return self.model[0].weight.square().mean().reshape(1), torch.ones(3, device=batch["img"].device)


class DummyTeacher(nn.Module):
    """Frozen, offline Foundation Teacher double with an encode call counter."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(10), requires_grad=True)
        self.calls = 0

    def freeze(self):
        self.eval()
        self.anchor.requires_grad_(False)

    def encode(self, images):
        self.calls += 1
        feature = self.anchor.view(1, 10, 1, 1).expand(images.shape[0], 10, 2, 2)
        return FoundationFeatures(dense={"p4": feature}, pooled=feature.mean((2, 3)))


def config(**overrides):
    values = dict(
        foundation_enabled=True,
        foundation_loss_weight=1.0,
        foundation_target_levels=["p4"],
        foundation_multiscale=False,
        foundation_align_dim=4,
        foundation_loss="hybrid",
        foundation_relation_mode="sampled",
        foundation_relation_samples=2,
        imgsz=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_cache(tmp_path, keys=("a", "b"), channels=10):
    cache = tmp_path / "cache"
    for index, key in enumerate(keys):
        feature = torch.full((1, channels, 2, 2), float(index + 1))
        features = FoundationFeatures(dense={"p4": feature}, pooled=feature.mean((2, 3)))
        save_foundation_features(cache / f"{key}.pt", features)
    return cache


def test_cache_only_training_without_online_teacher(tmp_path):
    cache = make_cache(tmp_path)
    student = TinyStudent()
    wrapper = FoundationDistillationModel(student, None, config(foundation_cache_dir=str(cache)))
    assert wrapper.teacher_manager is None
    assert wrapper.cache_dir == cache
    wrapper.train()
    batch = {"img": torch.rand(2, 3, 64, 64), "im_file": ["/data/a.jpg", "/data/b.jpg"]}
    total, items = wrapper(batch)
    assert total.shape == (2,)
    assert total[-1].item() > 0
    total.sum().backward()
    assert student.model[0].weight.grad is not None
    assert wrapper.projector.student_proj[0].weight.grad is not None


def test_cache_is_preferred_over_online_teacher_with_fallback(tmp_path):
    cache = make_cache(tmp_path)
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = FoundationDistillationModel(student, teacher, config(foundation_cache_dir=str(cache)))
    probe_calls = teacher.calls
    wrapper.train()
    total, _ = wrapper({"img": torch.rand(2, 3, 64, 64), "im_file": ["/x/a.jpg", "/x/b.jpg"]})
    assert total[-1].item() > 0
    assert teacher.calls == probe_calls  # batch resolved from cache
    total, _ = wrapper({"img": torch.rand(2, 3, 64, 64), "im_file": ["/x/a.jpg", "/x/missing.jpg"]})
    assert total[-1].item() > 0
    assert teacher.calls == probe_calls + 1  # cache miss falls back to online encode


def test_cache_only_mode_requires_im_file_keys(tmp_path):
    cache = make_cache(tmp_path)
    wrapper = FoundationDistillationModel(TinyStudent(), None, config(foundation_cache_dir=str(cache)))
    wrapper.train()
    with pytest.raises(ValueError, match="im_file"):
        wrapper({"img": torch.rand(2, 3, 64, 64)})
    with pytest.raises(FileNotFoundError, match="missing"):
        wrapper({"img": torch.rand(1, 3, 64, 64), "im_file": ["/x/absent.jpg"]})


def test_builder_supports_cache_only_mode(tmp_path):
    cache = make_cache(tmp_path)
    wrapper = build_foundation_distillation_wrapper(
        TinyStudent(), config(foundation_teacher="none", foundation_cache_dir=str(cache))
    )
    assert isinstance(wrapper, FoundationDistillationModel)
    assert wrapper.teacher_manager is None


def test_cache_dir_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        FoundationDistillationModel(TinyStudent(), None, config(foundation_cache_dir=str(tmp_path / "nope")))
    cache = make_cache(tmp_path)
    with pytest.raises(ValueError, match="single-teacher"):
        FoundationDistillationModel(
            TinyStudent(), DummyTeacher(), config(foundation_teacher="multi", foundation_cache_dir=str(cache))
        )


def test_without_cache_dir_disabled_contract_is_unchanged(tmp_path):
    wrapper = FoundationDistillationModel(TinyStudent(), None, config())
    assert wrapper.teacher_manager is None
    image = torch.rand(1, 3, 64, 64)
    assert wrapper(image) is not None  # transparent student passthrough


class RecordingStudent(TinyStudent):
    """TinyStudent that records every loss() call's batch for response-KD assertions."""

    def __init__(self):
        super().__init__()
        self.loss_calls = []

    def loss(self, batch, preds=None):
        if "bboxes" in batch:
            self.loss_calls.append(
                {
                    "bboxes": batch["bboxes"].detach().clone(),
                    "cls": batch["cls"].detach().clone(),
                    "batch_idx": batch["batch_idx"].detach().clone(),
                }
            )
        return super().loss(batch, preds)


def make_cache_with_response(tmp_path, keys=("a", "b"), channels=10, prompts=("zero", "one")):
    cache = tmp_path / "cache"
    for index, key in enumerate(keys):
        feature = torch.full((1, channels, 2, 2), float(index + 1))
        features = FoundationFeatures(dense={"p4": feature}, pooled=feature.mean((2, 3)))
        generator = torch.Generator().manual_seed(index)
        response = {
            "boxes": torch.rand(len(prompts), 1, 5, 4, generator=generator),
            "logits": torch.randn(len(prompts), 1, 5, generator=generator),
            # First prompt gets high scores on the first two queries; second prompt stays below threshold.
            "scores": torch.tensor(
                [[[0.9, 0.8, 0.1, 0.0, 0.0]] * len(prompts)]
            ).transpose(1, 2)
            if False
            else _response_scores(len(prompts), queries=5, index=index),
            "prompts": prompts,
            "box_format": "cxcywh_norm",
            "processed_size": (64, 64),
            "input_size": (64, 64),
        }
        save_foundation_features(cache / f"{key}.pt", features, response=response)
    return cache


def _response_scores(prompts, queries, index):
    scores = torch.full((prompts, 1, queries), 0.1)
    scores[0, 0, 0] = 0.9 + 0.01 * index
    scores[0, 0, 1] = 0.85 + 0.01 * index
    return scores


def test_response_kd_builds_pseudo_batch_from_cached_responses(tmp_path):
    cache = make_cache_with_response(tmp_path)
    student = RecordingStudent()
    wrapper = FoundationDistillationModel(
        student,
        None,
        config(
            foundation_cache_dir=str(cache),
            foundation_response_distill=True,
            foundation_response_loss_weight=0.5,
            foundation_response_score_threshold=0.5,
        ),
    )
    wrapper.train()
    batch = {"img": torch.rand(2, 3, 64, 64), "im_file": ["/x/a.jpg", "/x/b.jpg"]}
    total, items = wrapper(batch)
    assert total[-1].item() > 0
    total.sum().backward()
    assert student.model[0].weight.grad is not None
    assert wrapper.last_foundation_loss.item() > 0
    metrics = wrapper.foundation_metrics()
    assert metrics["foundation_response_enabled"] == 1.0
    assert metrics["foundation_response_boxes"] == 4.0  # 2 high-score queries × 2 images
    pseudo_calls = [call for call in student.loss_calls if call["bboxes"].shape[0] > 0]
    assert len(pseudo_calls) == 1
    pseudo = pseudo_calls[0]
    assert pseudo["bboxes"].shape == (4, 4)
    assert pseudo["cls"].reshape(-1).unique().tolist() == [0.0]  # prompt "zero" → class id 0
    assert pseudo["batch_idx"].unique().tolist() == [0.0, 1.0]


def test_response_kd_rejects_prompt_mismatch(tmp_path):
    cache = make_cache_with_response(tmp_path, prompts=("person", "dog"))
    wrapper = FoundationDistillationModel(
        RecordingStudent(),
        None,
        config(
            foundation_cache_dir=str(cache),
            foundation_response_distill=True,
            foundation_response_loss_weight=0.5,
        ),
    )
    wrapper.train()
    with pytest.raises(ValueError, match="not in student.names"):
        wrapper({"img": torch.rand(2, 3, 64, 64), "im_file": ["/x/a.jpg", "/x/b.jpg"]})


def test_response_kd_requires_cached_responses(tmp_path):
    cache = tmp_path / "cache"
    for key in ("a", "b"):
        feature = torch.full((1, 10, 2, 2), 1.0)
        features = FoundationFeatures(dense={"p4": feature}, pooled=feature.mean((2, 3)))
        save_foundation_features(cache / f"{key}.pt", features)
    wrapper = FoundationDistillationModel(
        RecordingStudent(),
        None,
        config(
            foundation_cache_dir=str(cache),
            foundation_response_distill=True,
            foundation_response_loss_weight=0.5,
        ),
    )
    wrapper.train()
    with pytest.raises(ValueError, match="re-run offline extraction"):
        wrapper({"img": torch.rand(2, 3, 64, 64), "im_file": ["/x/a.jpg", "/x/b.jpg"]})


def test_response_kd_init_requires_cache_or_detect_teacher():
    with pytest.raises(ValueError, match="foundation_response_distill requires"):
        FoundationDistillationModel(
            RecordingStudent(),
            DummyTeacher(),
            config(foundation_response_distill=True, foundation_response_loss_weight=0.5),
        )


def test_response_kd_threshold_filters_all_boxes_returns_zero_loss(tmp_path):
    cache = make_cache_with_response(tmp_path)
    student = RecordingStudent()
    wrapper = FoundationDistillationModel(
        student,
        None,
        config(
            foundation_cache_dir=str(cache),
            foundation_response_distill=True,
            foundation_response_loss_weight=0.5,
            foundation_response_score_threshold=0.99,
        ),
    )
    wrapper.train()
    total, _ = wrapper({"img": torch.rand(2, 3, 64, 64), "im_file": ["/x/a.jpg", "/x/b.jpg"]})
    assert wrapper.foundation_metrics()["foundation_response_boxes"] == 0.0
    pseudo_calls = [call for call in student.loss_calls if call["bboxes"].shape[0] > 0]
    assert pseudo_calls == []


def test_builder_supports_response_distill_cache_only(tmp_path):
    cache = make_cache_with_response(tmp_path)
    wrapper = build_foundation_distillation_wrapper(
        RecordingStudent(),
        config(
            foundation_teacher="none",
            foundation_cache_dir=str(cache),
            foundation_response_distill=True,
            foundation_response_loss_weight=0.5,
        ),
    )
    assert isinstance(wrapper, FoundationDistillationModel)
    assert wrapper._response_enabled is True
