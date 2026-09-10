"""Synthetic fail-closed tests for the pre-registered P2-04 Response-Field implementation."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.nn.foundation import (
    RESPONSE_FIELD_CONDITIONS,
    BatchNormBufferSnapshot,
    P4AlignmentProjector,
    apply_response_field_condition_batch,
    build_response_field_paired_view,
    logical_global_batch_index,
    preserve_batchnorm_buffers,
    response_field_condition,
    response_field_kd_loss,
    strict_cosine_kd_loss,
)


@pytest.mark.parametrize("student,teacher", [([1e20, 0.0], [1.0, 0.0]), ([1e20, 1e20], [1e20, 1e20])])
def test_finite_inputs_with_fp32_norm_overflow_fail_closed(student, teacher):
    """Finite inputs must not hide an overflow behind a finite but incorrect loss."""
    student = torch.tensor(student, dtype=torch.float32).reshape(1, 2, 1, 1)
    teacher = torch.tensor(teacher, dtype=torch.float32).reshape(1, 2, 1, 1)
    assert torch.isfinite(student).all() and torch.isfinite(teacher).all()
    with pytest.raises(ValueError, match="FP32 cosine norm"):
        strict_cosine_kd_loss(student, teacher)
    with pytest.raises(ValueError, match="FP32 cosine norm"):
        response_field_kd_loss(torch.zeros_like(student), student, torch.zeros_like(teacher), teacher)


def test_batchnorm_resolved_name_collision_is_rejected_before_snapshot():
    """Distinct BN objects cannot overwrite each other under the same audit name."""
    parent = nn.Module()
    parent.add_module("bn", nn.BatchNorm2d(2))
    other = nn.BatchNorm2d(2)
    with pytest.raises(ValueError, match="duplicate BatchNorm snapshot name") as error:
        BatchNormBufferSnapshot({"a": parent, "a.bn": other})
    assert "root='a', module='bn'" in str(error.value)
    assert "root='a.bn', module='<root>'" in str(error.value)
    assert torch.equal(parent.bn.running_mean, torch.zeros(2))


def test_batchnorm_duplicate_object_under_distinct_names_is_still_rejected():
    """Resolved-name checks do not replace the existing identity guard."""
    bn = nn.BatchNorm2d(2)
    with pytest.raises(ValueError, match="duplicates"):
        BatchNormBufferSnapshot({"a": bn, "b": bn})


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mutate", [False, True])
def test_optional_training_contract_restores_buffers_in_either_mode(training, mutate):
    """False permits train or eval, preserving all buffers and the original mode."""
    bn = nn.BatchNorm2d(2).train(training)
    before = {name: value.clone() for name, value in bn.named_buffers()}
    with preserve_batchnorm_buffers({"bn": bn}, require_training=False) as snapshot:
        if mutate:
            bn.running_mean.add_(7)
            bn.running_var.mul_(3)
            bn.num_batches_tracked.add_(1)
            assert not snapshot.matches()
    assert snapshot.matches() and bn.training is training
    assert all(torch.equal(before[name], value) for name, value in bn.named_buffers())


def test_optional_training_still_rejects_mode_changes_and_restores_buffers():
    """Allowing an initial eval mode does not allow a train/eval jump within the branch."""
    bn = nn.BatchNorm2d(2).eval()
    context = preserve_batchnorm_buffers({"bn": bn}, require_training=False)
    with pytest.raises(RuntimeError, match="training flags changed"), context as snapshot:
        bn.running_mean.add_(7)
        bn.train()
    assert snapshot.matches()


def test_optional_training_restores_buffers_when_branch_raises():
    """Exceptional branch exit also restores buffers without swallowing the branch error."""
    bn = nn.BatchNorm2d(2).eval()
    context = preserve_batchnorm_buffers({"bn": bn}, require_training=False)
    with pytest.raises(RuntimeError, match="branch failed"), context as snapshot:
        bn.running_mean.add_(7)
        raise RuntimeError("branch failed")
    assert snapshot.matches() and not bn.training


class TinyStudent(nn.Module):
    """Small train-mode Student path with one observable BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(images)))


def teacher_feature(images: torch.Tensor) -> torch.Tensor:
    """Return a deterministic frozen four-channel synthetic Teacher feature."""
    return torch.cat((images, images.mean(dim=1, keepdim=True)), dim=1).detach()


def paired_images() -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Build a fixed synthetic clean/perturbed pair."""
    generator = torch.Generator(device="cpu").manual_seed(20260824)
    clean = torch.rand((4, 3, 16, 16), generator=generator)
    perturbed, records = build_response_field_paired_view(
        clean,
        [f"train2017/image_{index:04d}.jpg" for index in range(4)],
        seed=20260824,
        epoch_index=7,
        batch_index_within_epoch=3,
        num_batches_per_epoch=512,
    )
    return clean, perturbed, records


def bn_buffer_state(*modules: nn.Module) -> dict[str, torch.Tensor]:
    """Clone all BatchNorm buffers with stable synthetic names."""
    state = {}
    for root_index, root in enumerate(modules):
        for name, module in root.named_modules():
            if isinstance(module, _BatchNorm):
                prefix = f"{root_index}.{name}"
                for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                    state[f"{prefix}.{buffer_name}"] = getattr(module, buffer_name).detach().clone()
    return state


def test_response_loss_matches_manual_formula_and_detaches_teacher():
    """Response loss is strict token cosine over finite differences, with no Teacher gradient."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    student_clean = torch.randn((2, 4, 3, 3), generator=generator, requires_grad=True)
    student_perturbed = torch.randn((2, 4, 3, 3), generator=generator, requires_grad=True)
    teacher_clean = torch.randn((2, 4, 3, 3), generator=generator, requires_grad=True)
    teacher_perturbed = torch.randn((2, 4, 3, 3), generator=generator, requires_grad=True)
    loss = response_field_kd_loss(student_clean, student_perturbed, teacher_clean, teacher_perturbed)
    expected = (
        1.0
        - F.cosine_similarity(
            student_perturbed - student_clean,
            teacher_perturbed - teacher_clean,
            dim=1,
            eps=1e-6,
        ).mean()
    )
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    loss.backward()
    assert student_clean.grad is not None and torch.isfinite(student_clean.grad).all()
    assert student_perturbed.grad is not None and torch.count_nonzero(student_perturbed.grad)
    assert teacher_clean.grad is None
    assert teacher_perturbed.grad is None


def test_response_and_static_cosine_fail_closed_on_zero_norm_or_nonfinite_input():
    """Zero response vectors and non-finite features cannot be silently skipped."""
    clean = torch.ones((2, 4, 2, 2))
    teacher_clean = torch.zeros_like(clean)
    with pytest.raises(ValueError, match="norm is below"):
        response_field_kd_loss(clean, clean.clone(), teacher_clean, teacher_clean.clone())
    nonfinite = clean.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        strict_cosine_kd_loss(nonfinite, clean)


def test_resume_stable_index_and_paired_digest_ignore_process_local_counter():
    """Logical position, not a process-local counter, determines every paired tensor."""
    clean, _, _ = paired_images()
    paths = [f"train2017/image_{index:04d}.jpg" for index in range(4)]
    uninterrupted_index = logical_global_batch_index(37, 100, 512)
    resumed_index = logical_global_batch_index(37, 100, 512)
    assert uninterrupted_index == resumed_index == 19044
    assert response_field_condition(
        seed=20260824,
        epoch_index=37,
        global_batch_index=uninterrupted_index,
        normalized_image_path=paths[0],
    ) == response_field_condition(
        seed=20260824,
        epoch_index=37,
        global_batch_index=resumed_index,
        normalized_image_path=paths[0],
    )
    uninterrupted, uninterrupted_records = build_response_field_paired_view(
        clean,
        paths,
        seed=20260824,
        epoch_index=37,
        batch_index_within_epoch=100,
        num_batches_per_epoch=512,
    )
    process_local_counter_after_resume = 0
    resumed, resumed_records = build_response_field_paired_view(
        clean,
        paths,
        seed=20260824,
        epoch_index=37,
        batch_index_within_epoch=100,
        num_batches_per_epoch=512,
    )
    assert process_local_counter_after_resume == 0
    assert torch.equal(uninterrupted, resumed)
    assert uninterrupted_records == resumed_records


@pytest.mark.parametrize("arm", ["b", "c"])
def test_static_and_response_perturbed_branches_have_finite_nonzero_gradients(arm: str):
    """Both supervised perturbed branches reach the Student feature and shared projector."""
    torch.manual_seed(4)
    clean, perturbed, _ = paired_images()
    student = TinyStudent().train()
    projector = P4AlignmentProjector(4, 4, align_dim=4, use_norm=True).train()
    clean_feature = student(clean)
    student_clean, teacher_clean = projector(clean_feature, teacher_feature(clean))
    with preserve_batchnorm_buffers({"student": student, "projector": projector}) as snapshot:
        perturbed_feature = student(perturbed)
        perturbed_feature.retain_grad()
        student_perturbed, teacher_perturbed = projector(perturbed_feature, teacher_feature(perturbed))
        assert not snapshot.matches()
    assert snapshot.matches()
    if arm == "b":
        foundation_loss = strict_cosine_kd_loss(student_clean, teacher_clean) + strict_cosine_kd_loss(
            student_perturbed, teacher_perturbed
        )
    else:
        foundation_loss = strict_cosine_kd_loss(student_clean, teacher_clean) + response_field_kd_loss(
            student_clean, student_perturbed, teacher_clean, teacher_perturbed
        )
    foundation_loss.backward()
    projector_gradient = projector.student_proj[0].weight.grad
    assert perturbed_feature.grad is not None and torch.isfinite(perturbed_feature.grad).all()
    assert torch.count_nonzero(perturbed_feature.grad)
    assert projector_gradient is not None and torch.isfinite(projector_gradient).all()
    assert torch.count_nonzero(projector_gradient)


def test_clean_and_perturbed_forwards_use_identical_train_bn_flags_and_restore_buffers():
    """The paired response contains no train/eval jump and leaves only clean BN updates."""
    torch.manual_seed(5)
    clean, perturbed, _ = paired_images()
    student = TinyStudent().train()
    projector = P4AlignmentProjector(4, 4, align_dim=4, use_norm=True).train()
    phase = ["clean"]
    flags: dict[str, dict[str, list[bool]]] = {"clean": {}, "perturbed": {}}
    hooks = []
    for root_name, root in {"student": student, "projector": projector}.items():
        for module_name, module in root.named_modules():
            if isinstance(module, _BatchNorm):
                name = f"{root_name}.{module_name}"
                hooks.append(
                    module.register_forward_pre_hook(
                        lambda current, _inputs, name=name: (
                            flags[phase[0]].setdefault(name, []).append(current.training)
                        )
                    )
                )
    clean_feature = student(clean)
    projector(clean_feature, teacher_feature(clean))
    with preserve_batchnorm_buffers({"student": student, "projector": projector}) as snapshot:
        phase[0] = "perturbed"
        projector(student(perturbed), teacher_feature(perturbed))
        assert not snapshot.matches()
    for hook in hooks:
        hook.remove()
    assert snapshot.matches()
    assert flags["clean"] == flags["perturbed"]
    assert flags["clean"] and all(all(values) for values in flags["clean"].values())


def _run_null_or_clean_reference(
    student: TinyStudent,
    projector: P4AlignmentProjector,
    clean: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    run_null_branch: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Execute one task-only optimizer step with or without the unused Arm-A branch."""
    optimizer = torch.optim.SGD((*student.parameters(), *projector.parameters()), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    clean_feature = student(clean)
    projector(clean_feature, teacher_feature(clean))
    if run_null_branch:
        with preserve_batchnorm_buffers({"student": student, "projector": projector}):
            projector(student(perturbed), teacher_feature(perturbed))
    task_loss = clean_feature.square().mean()
    task_loss.backward()
    gradients = {
        f"student.{name}": parameter.grad.detach().clone()
        for name, parameter in student.named_parameters()
        if parameter.grad is not None
    }
    gradients.update(
        {
            f"projector.{name}": parameter.grad.detach().clone()
            for name, parameter in projector.named_parameters()
            if parameter.grad is not None
        }
    )
    optimizer.step()
    parameters = {f"student.{name}": parameter.detach().clone() for name, parameter in student.named_parameters()}
    parameters.update(
        {f"projector.{name}": parameter.detach().clone() for name, parameter in projector.named_parameters()}
    )
    return gradients, parameters, bn_buffer_state(student, projector)


def test_arm_a_perturbed_branch_has_no_parameter_gradient_optimizer_or_buffer_effect():
    """Arm A is bitwise identical to its clean-task-only matched reference after one batch."""
    torch.manual_seed(8)
    clean, perturbed, _ = paired_images()
    reference_student = TinyStudent().train()
    reference_projector = P4AlignmentProjector(4, 4, align_dim=4, use_norm=True).train()
    null_student = copy.deepcopy(reference_student)
    null_projector = copy.deepcopy(reference_projector)
    reference = _run_null_or_clean_reference(
        reference_student, reference_projector, clean, perturbed, run_null_branch=False
    )
    arm_a = _run_null_or_clean_reference(null_student, null_projector, clean, perturbed, run_null_branch=True)
    for reference_group, arm_a_group in zip(reference, arm_a):
        assert reference_group.keys() == arm_a_group.keys()
        assert all(torch.equal(reference_group[name], arm_a_group[name]) for name in reference_group)


def test_batchnorm_restore_fails_closed_on_buffer_metadata_or_mode_mismatch():
    """A malformed buffer or eval-mode branch is rejected before evidence can be produced."""
    student = TinyStudent().train()
    snapshot = BatchNormBufferSnapshot({"student": student})
    student.bn.running_mean = torch.zeros(3)
    with pytest.raises(RuntimeError, match="metadata changed"):
        snapshot.restore()
    student = TinyStudent().train()
    snapshot = BatchNormBufferSnapshot({"student": student})
    student.bn.running_var = None
    with pytest.raises(TypeError, match="missing during restore"):
        snapshot.restore()
    student = TinyStudent().eval()
    with pytest.raises(RuntimeError, match="must all be in train mode"):
        BatchNormBufferSnapshot({"student": student})
    student = TinyStudent().train()
    with pytest.raises(RuntimeError, match="training flags changed"), preserve_batchnorm_buffers({"student": student}):
        student.eval()


def test_pair_generation_fails_closed_on_ambiguous_position_or_nonportable_path():
    """Malformed logical positions and platform-dependent paths cannot enter a rolling manifest."""
    clean = torch.rand((1, 3, 16, 16))
    with pytest.raises(ValueError, match="below num_batches_per_epoch"):
        build_response_field_paired_view(
            clean,
            ["train2017/image.jpg"],
            seed=1,
            epoch_index=0,
            batch_index_within_epoch=2,
            num_batches_per_epoch=2,
        )
    with pytest.raises(ValueError, match="POSIX-style"):
        build_response_field_paired_view(
            clean,
            [r"train2017\image.jpg"],
            seed=1,
            epoch_index=0,
            batch_index_within_epoch=0,
            num_batches_per_epoch=2,
        )


@pytest.mark.parametrize("family,value,condition_id", RESPONSE_FIELD_CONDITIONS)
def test_exhaustive_calibration_conditions_are_deterministic_and_frozen(family, value, condition_id):
    """Calibration can enumerate all eight conditions without resampling between alpha candidates."""
    clean = torch.rand((2, 3, 16, 16), generator=torch.Generator().manual_seed(9))
    kwargs = {
        "family": family,
        "value": value,
        "condition_id": condition_id,
        "seed": 20260824,
        "epoch_index": 49,
        "batch_index_within_epoch": 3,
        "num_batches_per_epoch": 16,
    }
    first, first_records = apply_response_field_condition_batch(
        clean, ["images/train2017/a.jpg", "images/train2017/b.jpg"], **kwargs
    )
    second, second_records = apply_response_field_condition_batch(
        clean, ["images/train2017/a.jpg", "images/train2017/b.jpg"], **kwargs
    )
    assert torch.equal(first, second)
    assert first_records == second_records
    assert {record["condition_id"] for record in first_records} == {condition_id}
    assert {record["global_batch_index"] for record in first_records} == {49 * 16 + 3}
