"""Offline contracts for the F06 Foundation distillation wrapper."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.nn.foundation import FoundationFeatures
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

    def predict(self, x, *args, **kwargs):
        outputs = []
        for index, layer in enumerate(self.model):
            if index == self.model[-1].i:
                return layer([outputs[source] for source in layer.f])
            x = layer(x)
            outputs.append(x)
        raise AssertionError("unreachable")

    def forward(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        # Keep the task loss connected to the student graph while retaining the normal three-item contract.
        return self.model[0].weight.square().mean().reshape(1), torch.ones(3, device=batch["img"].device)

    def init_criterion(self):
        return None

    def set_head_attr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.model[-1], key, value)


class DummyTeacher(nn.Module):
    """Frozen, offline Foundation Teacher double."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(10), requires_grad=True)
        self.calls = 0
        self.freeze_calls = 0

    def freeze(self):
        self.freeze_calls += 1
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
        foundation_cosine_weight=1.0,
        foundation_relation_weight=1.0,
        foundation_relation_mode="sampled",
        foundation_relation_samples=2,
        foundation_foreground_weighting=False,
        foundation_foreground_weight=1.5,
        foundation_boundary_weight=1.0,
        foundation_background_weight=0.25,
        imgsz=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_builder_is_exact_student_noop():
    student = TinyStudent()
    assert build_foundation_distillation_wrapper(student, config(foundation_enabled=False)) is student
    assert build_foundation_distillation_wrapper(student, config(foundation_loss_weight=0.0)) is student


def test_builder_requires_explicit_injected_manager_for_non_transformers_backend():
    with pytest.raises(ValueError, match="does not construct"):
        build_foundation_distillation_wrapper(TinyStudent(), config(foundation_backend="local"))


def test_builder_accepts_injected_siglip2_teacher_manager():
    student, teacher = TinyStudent(), DummyTeacher()
    teacher.name = "siglip2"
    wrapper = build_foundation_distillation_wrapper(
        student,
        config(foundation_teacher="siglip2", foundation_model="google/siglip2-base-patch16-512"),
        teacher_manager=teacher,
    )
    assert isinstance(wrapper, FoundationDistillationModel)
    assert wrapper.checkpoint_metadata()["teacher"] == "siglip2"


def test_builder_accepts_offline_injected_teacher_manager():
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = build_foundation_distillation_wrapper(student, config(), teacher_manager=teacher)
    assert isinstance(wrapper, FoundationDistillationModel)
    assert wrapper.teacher_manager is teacher


def test_training_adds_foundation_loss_and_only_student_projector_receive_gradients():
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = FoundationDistillationModel(student, teacher, config())
    assert wrapper.teacher_manager is teacher
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all("teacher_model" not in key for key in wrapper.state_dict())
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_parameter_ids.isdisjoint(id(parameter) for parameter in teacher.parameters())

    wrapper.train()
    batch = {"img": torch.rand(2, 3, 64, 64)}
    total, items = wrapper(batch)
    assert total.shape == (2,)
    assert items.shape == (4,)
    assert total[-1].item() > 0
    total.sum().backward()
    assert student.model[0].weight.grad is not None
    assert wrapper.projector.student_proj[0].weight.grad is not None
    assert teacher.anchor.grad is None
    assert wrapper.last_foundation_loss.item() == pytest.approx(float(total[-1]), rel=1e-5)


def test_foreground_weighted_training_uses_gt_boxes_and_reports_weighting():
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = FoundationDistillationModel(
        student,
        teacher,
        config(foundation_foreground_weighting=True),
    )
    wrapper.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "batch_idx": torch.tensor([0]),
    }
    total, _ = wrapper(batch)
    assert torch.isfinite(total).all()
    metrics = wrapper.foundation_metrics()
    assert metrics["foundation_foreground_enabled"] == 1.0
    assert metrics["foundation_foreground_mean_weight"] > 0.25


def test_multiscale_wrapper_builds_independent_adapters_and_aggregates_levels():
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = FoundationDistillationModel(
        student,
        teacher,
        config(foundation_multiscale=True, foundation_target_levels=["p3", "p4", "p5"]),
    )
    assert wrapper.multiscale is True
    assert wrapper.target_levels == ("p3", "p4", "p5")
    assert set(wrapper.taps) == {"p3", "p4", "p5"}
    assert set(wrapper.projector) == {"p3", "p4", "p5"}
    calls_before = teacher.calls
    wrapper.train()
    total, _ = wrapper({"img": torch.rand(2, 3, 64, 64)})
    assert total.shape == (2,)
    assert teacher.calls == calls_before + 1
    metrics = wrapper.foundation_metrics()
    assert all(f"foundation_{level}_loss" in metrics for level in ("p3", "p4", "p5"))
    total.sum().backward()
    assert all(wrapper.projector_for(level).student_proj[0].weight.grad is not None for level in ("p3", "p4", "p5"))

    copied = copy.deepcopy(wrapper)
    assert copied.teacher_manager is None
    assert copied.multiscale is True
    assert copied.taps == {}


def test_eval_prediction_and_eval_loss_do_not_call_teacher():
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = FoundationDistillationModel(student, teacher, config())
    dry_run_calls = teacher.calls
    wrapper.eval()
    prediction = wrapper(torch.rand(1, 3, 64, 64))
    assert prediction is not None
    assert teacher.calls == dry_run_calls
    _, items = wrapper.loss({"img": torch.rand(1, 3, 64, 64)})
    assert items.shape == (4,)
    assert teacher.calls == dry_run_calls


def test_proxy_attributes_and_student_only_deployment_copy():
    student, teacher = TinyStudent(), DummyTeacher()
    wrapper = FoundationDistillationModel(student, teacher, config())
    wrapper.nc = 7
    wrapper.names = {i: str(i) for i in range(7)}
    assert student.nc == 7 and student.names[6] == "6"
    deployment = wrapper.deployment_model()
    assert deployment is student
    assert wrapper.tap is None

    # EMA/checkpoint-style deepcopy strips the external teacher and live hook.
    copied = copy.deepcopy(FoundationDistillationModel(TinyStudent(), DummyTeacher(), config()))
    assert copied.teacher_manager is None
    assert copied.tap is None
    assert all("teacher_model" not in key for key in copied.state_dict())


def test_invalid_or_unsupported_f06_options_fail_fast():
    with pytest.raises(ValueError, match="exactly.*p4"):
        FoundationDistillationModel(TinyStudent(), DummyTeacher(), config(foundation_target_levels=["p3"]))
    with pytest.raises(ValueError, match="Unsupported foundation_loss"):
        wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config(foundation_loss="unknown"))
        wrapper({"img": torch.rand(2, 3, 64, 64)})
