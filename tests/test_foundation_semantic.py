"""F13 positive-region semantic distillation contracts."""

import pytest
import torch
from torch import nn

from ultralytics.nn.foundation import (
    FoundationFeatures,
    RegionSemanticProjector,
    positive_region_pool,
    region_image_loss,
    region_text_loss,
    semantic_distillation_loss,
)
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel
from tests.test_foundation_distill_model import TinyStudent, config


def test_positive_region_pool_only_returns_p4_matches_and_gt_classes():
    feature = torch.arange(1 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2)
    # P3=1 token, P4=4 tokens, P5=1 token; only P4 tokens 0 and 3 are positive.
    fg_mask = torch.tensor([[False, True, False, False, True, False]])
    target_gt_idx = torch.tensor([[0, 0, 0, 0, 0, 0]])
    batch = {"cls": torch.tensor([2]), "batch_idx": torch.tensor([0])}
    regions, image_indices, labels, token_indices = positive_region_pool(
        feature, fg_mask, target_gt_idx, batch, feature_shapes=[(1, 1, 1, 1), feature, (1, 8, 1, 1)]
    )
    assert regions.shape == (2, 4)
    assert image_indices.tolist() == [0, 0]
    assert labels.tolist() == [2, 2]
    assert token_indices.tolist() == [0, 3]
    # A P3 positive can be mapped onto a P4 feature grid while retaining its
    # class/positive assignment.
    p3_mask = torch.tensor([[True, False, False, False, False, False]])
    mapped, _, mapped_labels, _ = positive_region_pool(
        feature,
        p3_mask,
        target_gt_idx,
        batch,
        level_index=0,
        source_level_index=1,
        feature_shapes=[(1, 1, 1, 1), feature, (1, 8, 1, 1)],
    )
    assert mapped.shape == (1, 4) and mapped_labels.tolist() == [2]


def test_positive_region_pool_empty_is_graph_connected_zero_compatible():
    feature = torch.randn(1, 4, 2, 2, requires_grad=True)
    batch = {"cls": torch.empty(0), "batch_idx": torch.empty(0, dtype=torch.long)}
    regions, *_ = positive_region_pool(
        feature,
        torch.zeros(1, 6, dtype=torch.bool),
        torch.zeros(1, 6, dtype=torch.long),
        batch,
        feature_shapes=[(1, 1, 1, 1), feature, (1, 8, 1, 1)],
    )
    assert regions.shape == (0, 4)
    (regions.sum() + feature.sum() * 0).backward()
    assert feature.grad is not None


def test_region_losses_and_projector_are_finite_and_differentiable():
    projector = RegionSemanticProjector(4, 6)
    regions = torch.randn(3, 4, requires_grad=True)
    projected = projector(regions)
    labels = torch.tensor([0, 1, 0])
    text = torch.randn(2, 6)
    image = torch.randn(3, 6)
    text_loss = region_text_loss(projected, labels, text)
    image_loss = region_image_loss(projected, image)
    total, _, _ = semantic_distillation_loss(projected, labels, image, text)
    assert torch.isfinite(text_loss) and torch.isfinite(image_loss) and torch.isfinite(total)
    total.backward()
    assert projector.proj[0].weight.grad is not None


class SemanticTeacher(nn.Module):
    name = "siglip2"

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(8))
        self.text_calls = 0

    def freeze(self):
        self.eval()
        self.anchor.requires_grad_(False)

    def encode(self, images):
        dense = self.anchor[:4].view(1, 4, 1, 1).expand(images.shape[0], 4, 2, 2)
        semantic = self.anchor.view(1, 8).expand(images.shape[0], 8)
        return FoundationFeatures(dense={"p4": dense}, semantic=semantic / semantic.norm(dim=-1, keepdim=True))

    def encode_text(self, prompts):
        self.text_calls += 1
        return torch.eye(8)[: len(prompts)]


def test_semantic_projector_and_text_cache_stay_training_only():
    student, teacher = TinyStudent(), SemanticTeacher()
    args = config(
        foundation_loss_weight=0.0,
        foundation_semantic_distill=True,
        foundation_semantic_loss_weight=1.0,
        foundation_semantic_text_weight=1.0,
        foundation_semantic_image_weight=1.0,
    )
    wrapper = FoundationDistillationModel(student, teacher, args)
    assert wrapper.semantic_projector is not None
    assert all("teacher" not in name for name in wrapper.state_dict())
    # Text prototypes are not registered parameters/buffers and are cached after first use.
    wrapper._semantic_text_prototypes()
    wrapper._semantic_text_prototypes()
    assert teacher.text_calls == 1
    assert wrapper.checkpoint_metadata()["semantic_distill"] is True
    assert wrapper.checkpoint_metadata()["semantic_dim"] == 8


def test_semantic_training_uses_positive_regions_and_adds_loss():
    student, teacher = TinyStudent(), SemanticTeacher()

    class Assignment:
        def get_assigned_targets_and_loss(self, preds, batch):
            # TinyStudent has one 64x64 feature map per level in this test; mark
            # the first P4 token positive and map it to GT 0.
            sizes = [item.shape[-2] * item.shape[-1] for item in preds["feats"]]
            fg = torch.zeros(2, sum(sizes), dtype=torch.bool)
            fg[0, sizes[0]] = True
            gt = torch.zeros_like(fg, dtype=torch.long)
            return (fg, gt, None, None, None), torch.zeros(3), torch.zeros(3)

    args = config(
        foundation_loss_weight=0.0,
        foundation_semantic_distill=True,
        foundation_semantic_loss_weight=1.0,
        foundation_semantic_text_weight=1.0,
        foundation_semantic_image_weight=1.0,
    )
    student.criterion = Assignment()
    wrapper = FoundationDistillationModel(student, teacher, args)
    wrapper.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "cls": torch.tensor([0.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "batch_idx": torch.tensor([0]),
    }
    total, _ = wrapper.loss(batch)
    assert total[-1].item() > 0
    assert wrapper.foundation_metrics()["foundation_semantic_regions"] == 1.0
    assert teacher.text_calls == 1


def test_semantic_requires_siglip2_semantic_and_text_interfaces():
    class NoSemanticTeacher(SemanticTeacher):
        def encode(self, images):
            return FoundationFeatures(dense={"p4": torch.zeros(images.shape[0], 4, 2, 2)})

    with pytest.raises(ValueError, match="semantic features"):
        FoundationDistillationModel(
            TinyStudent(),
            NoSemanticTeacher(),
            config(foundation_loss_weight=0.0, foundation_semantic_distill=True, foundation_semantic_loss_weight=1.0),
        )
