"""Offline contracts for the SAM3 (DART) Foundation Teacher adapter."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.foundation import FoundationTeacher, SAM3Teacher


class DummySAM3Backbone(nn.Module):
    def __init__(self, channels=8, seq=4):
        super().__init__()
        self.channels = channels
        self.seq = seq
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.image_calls = 0
        self.text_calls = 0

    def forward_image(self, samples):
        self.image_calls += 1
        batch, _, height, width = samples.shape
        fpn = [self.scale * torch.ones(batch, self.channels, height // s, width // s) for s in (2, 4, 8)]
        pos = [torch.zeros_like(feature) for feature in fpn]
        return {"backbone_fpn": fpn, "vision_pos_enc": pos, "vision_features": fpn[-1]}

    def forward_text(self, captions, device="cpu"):
        self.text_calls += 1
        count = len(captions)
        features = self.scale * torch.ones(self.seq, count, self.channels)
        features[0] += torch.arange(count).unsqueeze(-1) * torch.arange(self.channels)
        mask = torch.zeros(count, self.seq, dtype=torch.bool)
        mask[:, -1] = True  # last token is padding
        return {"language_features": features, "language_mask": mask}


class DummyDecoder(nn.Module):
    def __init__(self, queries=5, channels=8, layers=2):
        super().__init__()
        self.query_embed = nn.Embedding(queries, channels)
        self.bbox_embed = nn.Linear(channels, 4)
        self.layers = layers

    def forward(self, tgt, memory, **kwargs):
        queries, batch, channels = tgt.shape
        hs = torch.zeros(self.layers, queries, batch, channels)
        reference_boxes = torch.full((self.layers, queries, batch, 4), 0.5)
        presence = torch.zeros(self.layers, 1, batch)
        return hs, reference_boxes, presence, None


class DummyTransformer(nn.Module):
    def __init__(self, queries=5, channels=8):
        super().__init__()
        self.decoder = DummyDecoder(queries=queries, channels=channels)

    def encoder(self, *, src, src_pos, prompt, feat_sizes, **kwargs):
        return {
            "memory": src[-1],
            "padding_mask": None,
            "pos_embed": src_pos[-1],
            "level_start_index": torch.zeros(1, dtype=torch.long),
            "spatial_shapes": torch.tensor([list(feat_sizes[-1])]),
            "valid_ratios": torch.ones(prompt.shape[1], 1, 2),
        }


class DummyScoring(nn.Module):
    def forward(self, hs, prompt, prompt_mask):
        layers, batch, queries, _ = hs.shape
        return torch.zeros(layers, batch, queries, 1)


class DummySAM3(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.backbone = DummySAM3Backbone(channels=channels)
        self.transformer = DummyTransformer(channels=channels)
        self.dot_prod_scoring = DummyScoring()
        self.num_feature_levels = 1


def make_teacher(**kwargs):
    return SAM3Teacher(model=DummySAM3(), device="cpu", image_size=16, patch_size=2, **kwargs)


def test_sam3_teacher_satisfies_protocol_and_is_frozen():
    teacher = make_teacher()
    assert isinstance(teacher, FoundationTeacher)
    assert teacher.name == "sam3"
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train(True)
    assert teacher.training is False


def test_sam3_encode_returns_multiscale_dense_features():
    teacher = make_teacher()
    features = teacher.encode(torch.rand(2, 3, 12, 10))
    assert set(features.dense) == {"p2", "p3", "p4"}
    assert features.dense["p2"].shape == (2, 8, 8, 8)
    assert features.dense["p3"].shape == (2, 8, 4, 4)
    assert features.dense["p4"].shape == (2, 8, 2, 2)
    assert features.pooled.shape == (2, 8)
    assert features.metadata["strides"] == {"p2": 2.0, "p3": 4.0, "p4": 8.0}
    assert features.metadata["input_size"] == (12, 10)
    assert features.metadata["processed_size"] == (16, 16)
    assert features.metadata["backend"] == "dart-sam3"
    assert teacher.model.backbone.image_calls == 1


def test_sam3_text_prototypes_are_normalized_and_cached():
    teacher = make_teacher()
    first = teacher.encode_text(["person", "dog"])
    second = teacher.encode_text(["person", "dog"])
    assert torch.equal(first, second)
    assert first.shape == (2, 8)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2))
    assert not torch.allclose(first[0], first[1])
    assert teacher.model.backbone.text_calls == 1
    teacher.clear_text_cache()
    teacher.encode_text(["person", "dog"])
    assert teacher.model.backbone.text_calls == 2


def test_sam3_preprocess_and_prompt_boundaries():
    teacher = make_teacher()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        teacher.preprocess(torch.full((1, 3, 4, 4), 2.0))
    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        teacher.preprocess(torch.rand(3, 4, 4))
    with pytest.raises(ValueError, match="non-empty"):
        teacher.encode_text([""])
    with pytest.raises(ValueError, match="non-empty"):
        teacher.detect(torch.rand(1, 3, 8, 8), [])


def test_sam3_detect_returns_response_channel():
    teacher = make_teacher()
    response = teacher.detect(torch.rand(2, 3, 8, 8), ["person", "dog", "car"])
    assert response["boxes"].shape == (3, 2, 5, 4)
    assert response["logits"].shape == (3, 2, 5)
    assert response["scores"].shape == (3, 2, 5)
    assert response["box_format"] == "cxcywh_norm"
    assert response["prompts"] == ("person", "dog", "car")
    assert torch.all(response["boxes"] >= 0) and torch.all(response["boxes"] <= 1)
    assert torch.all(response["scores"] >= 0) and torch.all(response["scores"] <= 1)
    assert response["processed_size"] == (16, 16)


def test_sam3_detect_tensors_are_writable_outside_inference_mode():
    teacher = make_teacher()
    response = teacher.detect(torch.rand(1, 3, 8, 8), ["person"])
    response["boxes"][0, 0, 0, 0] = 0.0  # must not raise inference-tensor errors
    features = teacher.encode(torch.rand(1, 3, 8, 8))
    features.dense["p4"] += 1.0


def test_sam3_image_size_must_align_with_patch_size():
    with pytest.raises(ValueError, match="multiple of patch_size"):
        SAM3Teacher(model=DummySAM3(), device="cpu", image_size=15, patch_size=2)


def test_sam3_missing_dependency_reports_boundary(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sam3", None)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", None)
    with pytest.raises(ImportError, match="DART"):
        SAM3Teacher(device="cpu")
