"""Offline contracts for Foundation Teacher pre-extraction caching."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.foundation import (
    FoundationFeatures,
    SAM3Teacher,
    extract_foundation_cache,
    load_foundation_batch,
    load_foundation_features,
    save_foundation_features,
)


class DummySAM3Backbone(nn.Module):
    def __init__(self, channels=8, seq=4):
        super().__init__()
        self.channels = channels
        self.seq = seq
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.image_calls = 0

    def forward_image(self, samples):
        self.image_calls += 1
        batch, _, height, width = samples.shape
        fpn = [self.scale * torch.ones(batch, self.channels, height // s, width // s) for s in (2, 4, 8)]
        return {"backbone_fpn": fpn, "vision_pos_enc": [torch.zeros_like(f) for f in fpn]}

    def forward_text(self, captions, device="cpu"):
        count = len(captions)
        features = self.scale * torch.ones(self.seq, count, self.channels)
        features[0] += torch.arange(count).unsqueeze(-1) * torch.arange(self.channels)
        mask = torch.zeros(count, self.seq, dtype=torch.bool)
        mask[:, -1] = True
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
    def __init__(self, channels=8):
        super().__init__()
        self.decoder = DummyDecoder(channels=channels)

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


def make_teacher():
    return SAM3Teacher(model=DummySAM3(), device="cpu", image_size=16, patch_size=2)


def make_samples(count, size=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [(f"img_{index}", torch.rand(3, size, size, generator=generator)) for index in range(count)]


def test_save_load_roundtrip_preserves_features_and_response(tmp_path):
    teacher = make_teacher()
    image = torch.rand(1, 3, 8, 8)
    features, response = teacher.encode_with_response(image, ["person", "dog"])
    path = save_foundation_features(tmp_path / "sample.pt", features, response=response)
    loaded, loaded_response = load_foundation_features(path)
    assert isinstance(loaded, FoundationFeatures)
    assert set(loaded.dense) == set(features.dense)
    for name in features.dense:
        assert loaded.dense[name].shape == features.dense[name].shape
        assert loaded.dense[name].dtype == torch.float32  # restored from fp16 storage
        assert torch.allclose(loaded.dense[name], features.dense[name], atol=1e-2)
    assert torch.allclose(loaded.pooled, features.pooled, atol=1e-2)
    assert loaded.metadata["strides"] == features.metadata["strides"]
    assert loaded_response["boxes"].shape == response["boxes"].shape
    assert loaded_response["prompts"] == ("person", "dog")
    assert loaded_response["box_format"] == "cxcywh_norm"
    stored = torch.load(path, weights_only=False)
    assert stored["dense"]["p4"].dtype == torch.float16
    assert stored["response"]["boxes"].dtype == torch.float16


def test_extract_writes_per_image_files_and_prototypes(tmp_path):
    teacher = make_teacher()
    summary = extract_foundation_cache(
        teacher, make_samples(3), tmp_path, prompts=["person", "dog"], batch_size=2
    )
    assert summary["written"] == ["img_0", "img_1", "img_2"]
    assert summary["skipped"] == []
    assert summary["prototypes"] == tmp_path / "text_prototypes.pt"
    prototypes = torch.load(summary["prototypes"], weights_only=False)
    assert prototypes["prompts"] == ("person", "dog")
    assert prototypes["prototypes"].shape == (2, 8)
    for key in summary["written"]:
        features, response = load_foundation_features(tmp_path / f"{key}.pt")
        assert set(features.dense) == {"p2", "p3", "p4"}
        assert features.dense["p4"].shape == (1, 8, 2, 2)
        assert features.pooled.shape == (1, 8)
        assert response["boxes"].shape == (2, 1, 5, 4)  # (P, B=1, Q, 4)
        assert response["logits"].shape == (2, 1, 5)
    # shared single backbone pass per batch: ceil(3 / 2) = 2 image forwards
    assert teacher.model.backbone.image_calls == 2


def test_extract_skips_existing_files_unless_overwrite(tmp_path):
    teacher = make_teacher()
    samples = make_samples(2)
    extract_foundation_cache(teacher, samples, tmp_path)
    calls = teacher.model.backbone.image_calls
    summary = extract_foundation_cache(teacher, samples, tmp_path)
    assert summary["written"] == [] and summary["skipped"] == ["img_0", "img_1"]
    assert teacher.model.backbone.image_calls == calls
    summary = extract_foundation_cache(teacher, samples, tmp_path, overwrite=True)
    assert summary["written"] == ["img_0", "img_1"]
    assert teacher.model.backbone.image_calls > calls


def test_extract_without_prompts_caches_features_only(tmp_path):
    teacher = make_teacher()
    summary = extract_foundation_cache(teacher, make_samples(1), tmp_path)
    assert summary["prototypes"] is None
    features, response = load_foundation_features(summary["files"][0])
    assert response is None
    assert features.dense["p2"].shape == (1, 8, 8, 8)


def test_extract_validates_keys_and_samples(tmp_path):
    teacher = make_teacher()
    with pytest.raises(ValueError, match="relative"):
        extract_foundation_cache(teacher, [("/abs/key", torch.rand(3, 8, 8))], tmp_path)
    with pytest.raises(ValueError, match=r"\[3,H,W\]"):
        extract_foundation_cache(teacher, [("bad", torch.rand(1, 3, 8, 8))], tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        extract_foundation_cache(teacher, [], tmp_path, batch_size=0)


def test_extract_handles_mixed_image_sizes(tmp_path):
    teacher = make_teacher()
    samples = [("small", torch.rand(3, 8, 8)), ("large", torch.rand(3, 12, 12)), ("small2", torch.rand(3, 8, 8))]
    summary = extract_foundation_cache(teacher, samples, tmp_path, batch_size=4)
    assert summary["written"] == ["small", "large", "small2"]
    features, _ = load_foundation_features(tmp_path / "large.pt")
    assert features.metadata["processed_size"] == (16, 16)


def test_load_restores_requested_dtype(tmp_path):
    teacher = make_teacher()
    features = teacher.encode(torch.rand(1, 3, 8, 8))
    path = save_foundation_features(tmp_path / "s.pt", features, save_dtype=None)
    stored = torch.load(path, weights_only=False)
    assert stored["dense"]["p4"].dtype == torch.float32
    loaded, _ = load_foundation_features(path, dtype=torch.float16)
    assert loaded.dense["p4"].dtype == torch.float16


def _make_response(prompts=("person", "dog"), batch=1, queries=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "boxes": torch.rand(len(prompts), batch, queries, 4, generator=generator),
        "logits": torch.randn(len(prompts), batch, queries, generator=generator),
        "scores": torch.rand(len(prompts), batch, queries, generator=generator),
        "prompts": prompts,
        "box_format": "cxcywh_norm",
        "processed_size": (16, 16),
        "input_size": (8, 8),
    }


def test_load_foundation_batch_with_response(tmp_path):
    teacher = make_teacher()
    prompts = ("person", "dog")
    for index, key in enumerate(("alpha", "beta")):
        features, response = teacher.encode_with_response(
            torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(index)), prompts
        )
        save_foundation_features(tmp_path / f"{key}.pt", features, response=response)
    features, response = load_foundation_batch(tmp_path, ["alpha", "beta"], with_response=True)
    assert features.dense["p4"].shape[0] == 2
    assert response is not None
    assert response["prompts"] == prompts
    assert response["boxes"].shape == (len(prompts), 2, 5, 4)
    assert response["logits"].shape == (len(prompts), 2, 5)
    assert response["scores"].shape == (len(prompts), 2, 5)
    features_only, response_only = load_foundation_batch(tmp_path, ["alpha", "beta"])
    assert isinstance(features_only, FoundationFeatures)
    assert response_only is None


def test_load_foundation_batch_with_response_validates_prompts(tmp_path):
    teacher = make_teacher()
    save_foundation_features(
        tmp_path / "a.pt", teacher.encode(torch.rand(1, 3, 8, 8)), response=_make_response(prompts=("person", "dog"))
    )
    save_foundation_features(
        tmp_path / "b.pt", teacher.encode(torch.rand(1, 3, 8, 8)), response=_make_response(prompts=("car", "dog"))
    )
    with pytest.raises(ValueError, match="prompts"):
        load_foundation_batch(tmp_path, ["a", "b"], with_response=True)


def test_load_foundation_batch_with_response_none_when_any_entry_missing(tmp_path):
    teacher = make_teacher()
    features = teacher.encode(torch.rand(1, 3, 8, 8))
    save_foundation_features(tmp_path / "a.pt", features, response=_make_response())
    save_foundation_features(tmp_path / "b.pt", features)
    loaded_features, response = load_foundation_batch(tmp_path, ["a", "b"], with_response=True)
    assert loaded_features.dense["p4"].shape[0] == 2
    assert response is None


def test_extract_levels_filter_stores_only_requested_levels(tmp_path):
    teacher = make_teacher()
    summary = extract_foundation_cache(teacher, make_samples(2), tmp_path, prompts=["person"], levels=["p4"])
    for key in summary["written"]:
        features, response = load_foundation_features(tmp_path / f"{key}.pt")
        assert set(features.dense) == {"p4"}
        assert features.dense["p4"].shape == (1, 8, 2, 2)
        assert features.pooled.shape == (1, 8)
        assert response["boxes"].shape == (1, 1, 5, 4)


def test_extract_levels_filter_validates_names(tmp_path):
    teacher = make_teacher()
    with pytest.raises(ValueError, match="not produced"):
        extract_foundation_cache(teacher, make_samples(1), tmp_path, levels=["p6"])
    with pytest.raises(ValueError, match="non-empty"):
        extract_foundation_cache(teacher, make_samples(1), tmp_path, levels=[])


def test_load_foundation_batch_concatenates_samples_in_key_order(tmp_path):
    teacher = make_teacher()
    samples = make_samples(3)
    extract_foundation_cache(teacher, samples, tmp_path, levels=["p4"])
    batch, response = load_foundation_batch(tmp_path, ["img_2", "img_0"])
    assert isinstance(batch, FoundationFeatures)
    assert response is None
    assert set(batch.dense) == {"p4"}
    assert batch.dense["p4"].shape == (2, 8, 2, 2)
    assert batch.pooled.shape == (2, 8)
    single, _ = load_foundation_features(tmp_path / "img_2.pt")
    assert torch.allclose(batch.dense["p4"][0], single.dense["p4"][0])


def test_load_foundation_batch_reports_missing_and_inconsistent_entries(tmp_path):
    teacher = make_teacher()
    extract_foundation_cache(teacher, make_samples(1), tmp_path, levels=["p4"])
    with pytest.raises(FileNotFoundError, match="missing"):
        load_foundation_batch(tmp_path, ["img_0", "img_9"])
    with pytest.raises(ValueError, match="non-empty"):
        load_foundation_batch(tmp_path, [])
    extract_foundation_cache(teacher, [("full", torch.rand(3, 8, 8))], tmp_path)
    with pytest.raises(ValueError, match="levels"):
        load_foundation_batch(tmp_path, ["img_0", "full"])
