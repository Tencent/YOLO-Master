"""Tests for reliability-weighted boxes and confirmed-negative prompt sampling."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ultralytics.data.augment import BaseMixTransform, Format, RandomLoadText
from ultralytics.data.dataset import GroundingDataset, YOLODataset
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import v8DetectionLoss


def test_confirmed_negative_is_kept_as_a_negative_prompt():
    """Explicit teacher abstentions are sampled even when random-negative sampling is disabled."""
    labels = {
        "img": np.zeros((16, 16, 3), dtype=np.uint8),
        "cls": np.array([[0, 0.4]], dtype=np.float32),
        "texts": [["car"], ["airship"], ["boat"]],
        "confirmed_negative_texts": ["airship"],
        "instances": Instances(
            bboxes=np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        ),
    }
    loaded = RandomLoadText(neg_samples=(0, 0), max_samples=3, padding=True, padding_value=["pad"])(labels)

    assert loaded["texts"][:2] == ["car", "airship"]
    assert np.allclose(loaded["cls"], [[0.0, 0.4]])

    formatted = Format()(loaded)
    batch = YOLODataset.collate_fn([formatted])
    assert batch["cls"].shape == (1, 1)
    assert batch["gt_weights"].tolist() == pytest.approx([0.4])


def test_anchor_weights_follow_ground_truth_assignment_and_leave_background_neutral():
    """Reliability is aligned by image-local target index, while unassigned anchors retain weight one."""
    loss = v8DetectionLoss.__new__(v8DetectionLoss)
    loss.device = torch.device("cpu")
    batch = {
        "batch_idx": torch.tensor([0, 0, 1]),
        "gt_weights": torch.tensor([0.2, 0.7, 0.9]),
    }
    target_gt_idx = torch.tensor([[0, 1, 1], [0, 1, 0]])
    fg_mask = torch.tensor([[True, True, False], [True, False, False]])

    anchor_weights = loss._get_anchor_weights(batch, 2, 2, target_gt_idx, fg_mask, torch.float32)

    assert np.allclose(anchor_weights.numpy(), [[0.2, 0.7, 1.0], [0.9, 1.0, 1.0]])


def test_anchor_weights_reject_misaligned_metadata():
    loss = v8DetectionLoss.__new__(v8DetectionLoss)
    loss.device = torch.device("cpu")
    batch = {"batch_idx": torch.tensor([0]), "gt_weights": torch.tensor([0.5, 0.7])}

    with pytest.raises(ValueError, match="gt_weights has 2 values for 1 ground-truth boxes"):
        loss._get_anchor_weights(
            batch,
            1,
            1,
            torch.zeros((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
            torch.float32,
        )


def test_mixed_text_remapping_preserves_weights_and_negative_concepts():
    labels = {
        "texts": [["cat"], ["plane"]],
        "cls": np.array([[0, 0.3]], dtype=np.float32),
        "confirmed_negative_texts": ["plane"],
        "mix_labels": [
            {
                "texts": [["dog"], ["boat"]],
                "cls": np.array([[0, 0.8]], dtype=np.float32),
                "confirmed_negative_texts": ["boat"],
            }
        ],
    }

    BaseMixTransform._update_label_text(labels)

    text_ids = {texts[0]: idx for idx, texts in enumerate(labels["texts"])}
    assert labels["texts"] == [["cat"], ["plane"], ["dog"], ["boat"]]
    assert labels["cls"][0, 0] == text_ids["cat"]
    assert labels["cls"][0, 1] == pytest.approx(0.3)
    assert labels["mix_labels"][0]["cls"][0, 0] == text_ids["dog"]
    assert labels["mix_labels"][0]["cls"][0, 1] == pytest.approx(0.8)
    assert set(labels["confirmed_negative_texts"]) == {"plane", "boat"}


def test_grounding_dataset_reads_box_weights_and_negative_only_images(tmp_path):
    from PIL import Image

    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (64, 48)).save(image_root / "positive.jpg")
    Image.new("RGB", (64, 48)).save(image_root / "negative.jpg")
    payload = {
        "images": [
            {
                "id": 0,
                "file_name": "positive.jpg",
                "height": 48,
                "width": 64,
                "caption": "cat. airplane.",
                "negative_captions": ["airplane"],
            },
            {
                "id": 1,
                "file_name": "negative.jpg",
                "height": 48,
                "width": 64,
                "caption": "balloon.",
                "negative_captions": ["balloon"],
            },
        ],
        "annotations": [
            {
                "image_id": 0,
                "bbox": [10, 10, 20, 15],
                "iscrowd": 0,
                "tokens_positive": [[0, 3]],
                "weight": 0.35,
            }
        ],
    }
    json_file = tmp_path / "grounding.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")
    dataset = GroundingDataset.__new__(GroundingDataset)
    dataset.json_file = str(json_file)
    dataset.img_path = str(image_root)
    dataset.im_files = []
    dataset.prefix = ""
    dataset.use_obb = False

    labels = dataset.cache_labels(tmp_path / "labels.cache")["labels"]

    assert len(labels) == 2
    assert labels[0]["cls"].shape == (1, 2)
    assert labels[0]["cls"][0, 1] == pytest.approx(0.35)
    assert labels[0]["texts"] == [["cat"], ["airplane"]]
    assert labels[0]["confirmed_negative_texts"] == ["airplane"]
    assert labels[1]["cls"].shape == (0, 2)
    assert labels[1]["confirmed_negative_texts"] == ["balloon"]
