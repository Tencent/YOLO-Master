"""Image-level leakage guards for S2 training and open-vocabulary evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.ov_distill.class_split import build_class_split, validate_class_split
from scripts.ov_distill.convert import run_convert
from scripts.ov_distill.reliability import assert_disjoint_manifests
from scripts.ov_distill.split_audit import validate_training_eval_split


def _write_manifest(path: Path, images: list[Path]) -> None:
    path.write_text("".join(json.dumps({"image": str(image)}) + "\n" for image in images), encoding="utf-8")


def _write_class_split(path: Path, unseen: list[str] | None = None) -> Path:
    artifact = build_class_split()
    unseen = unseen or [artifact["unseen"][0]]
    artifact["unseen"] = sorted(unseen)
    artifact["seen"] = sorted(set(artifact["class_names"]) - set(unseen))
    artifact["unseen_ratio"] = len(unseen) / len(artifact["class_names"])
    artifact = validate_class_split(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _class_metadata(path: Path, weight_mode: str = "uniform") -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return {
        "provenance_version": 1,
        "class_split": str(path.resolve()),
        "seen_classes": artifact["seen"],
        "unseen_classes": artifact["unseen"],
        "weight_mode": weight_mode,
    }


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "white").save(path)
    return path


def test_disjoint_manifests_resolve_symlink_aliases(tmp_path):
    image = _image(tmp_path / "images" / "train.jpg")
    alias = tmp_path / "alias.jpg"
    alias.symlink_to(image)
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    _write_manifest(train_manifest, [image])
    _write_manifest(eval_manifest, [alias])

    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_manifests(train_manifest, eval_manifest, "training", "evaluation")


def test_training_split_checks_grounding_and_yaml_image_sets(tmp_path):
    train_image = _image(tmp_path / "train" / "images" / "train.jpg")
    eval_image = _image(tmp_path / "eval" / "images" / "eval.jpg")
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    _write_manifest(train_manifest, [train_image])
    _write_manifest(eval_manifest, [eval_image])
    class_split = _write_class_split(tmp_path / "class_split.json")

    grounding_json = tmp_path / "grounding_train.json"
    grounding_json.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "train.jpg"}],
                "annotations": [],
                "metadata": _class_metadata(class_split),
            }
        ),
        encoding="utf-8",
    )
    val_yaml = tmp_path / "eval.yaml"
    val_yaml.write_text(
        "path: " + str((tmp_path / "eval").resolve()) + "\ntrain: images\nval: images\nnames:\n  0: person\n",
        encoding="utf-8",
    )

    stats = validate_training_eval_split(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        grounding_json=grounding_json,
        image_root=train_image.parent,
        class_split=class_split,
        val_yaml=val_yaml,
    )

    assert stats == {
        "train_images": 1,
        "eval_images": 1,
        "grounding_images": 1,
        "image_overlap": 0,
        "class_split": str(class_split.resolve()),
        "training_classes": 0,
        "unseen_classes_in_training": 0,
        "calibration_images": 0,
        "val_yaml_checked": True,
    }


def test_training_split_rejects_yaml_images_missing_from_manifest(tmp_path):
    train_image = _image(tmp_path / "train" / "images" / "train.jpg")
    eval_image = _image(tmp_path / "eval" / "images" / "eval.jpg")
    extra_eval_image = _image(tmp_path / "eval" / "images" / "unlisted.jpg")
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    _write_manifest(train_manifest, [train_image])
    _write_manifest(eval_manifest, [eval_image])
    class_split = _write_class_split(tmp_path / "class_split.json")
    grounding_json = tmp_path / "grounding_train.json"
    grounding_json.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "train.jpg"}],
                "annotations": [],
                "metadata": _class_metadata(class_split),
            }
        ),
        encoding="utf-8",
    )
    val_yaml = tmp_path / "eval.yaml"
    val_yaml.write_text(
        "path: " + str((tmp_path / "eval").resolve()) + "\ntrain: images\nval: images\nnames:\n  0: person\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same images"):
        validate_training_eval_split(
            train_manifest=train_manifest,
            eval_manifest=eval_manifest,
            grounding_json=grounding_json,
            image_root=train_image.parent,
            class_split=class_split,
            val_yaml=val_yaml,
        )
    assert extra_eval_image.exists()


def test_training_split_rejects_unseen_class_in_grounding_caption(tmp_path):
    train_image = _image(tmp_path / "train" / "train.jpg")
    eval_image = _image(tmp_path / "eval" / "eval.jpg")
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    _write_manifest(train_manifest, [train_image])
    _write_manifest(eval_manifest, [eval_image])
    class_split = _write_class_split(tmp_path / "class_split.json", unseen=["cat"])
    grounding_json = tmp_path / "grounding_train.json"
    grounding_json.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "train.jpg", "caption": "cat.", "negative_captions": []}],
                "annotations": [{"image_id": 0, "tokens_positive": [[0, 3]]}],
                "metadata": _class_metadata(class_split),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the seen split"):
        validate_training_eval_split(
            train_manifest=train_manifest,
            eval_manifest=eval_manifest,
            grounding_json=grounding_json,
            image_root=train_image.parent,
            class_split=class_split,
        )


def test_training_split_rejects_calibration_images_in_evaluation_manifest(tmp_path):
    train_image = _image(tmp_path / "train" / "train.jpg")
    eval_image = _image(tmp_path / "eval" / "eval.jpg")
    train_manifest, eval_manifest = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write_manifest(train_manifest, [train_image])
    _write_manifest(eval_manifest, [eval_image])
    calibration_manifest = tmp_path / "calibration.jsonl"
    _write_manifest(calibration_manifest, [eval_image])
    class_split = _write_class_split(tmp_path / "class_split.json")
    metadata = _class_metadata(class_split, weight_mode="calibrated")
    metadata["calibration_manifest"] = str(calibration_manifest.resolve())
    grounding_json = tmp_path / "grounding_train.json"
    grounding_json.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "train.jpg", "caption": "", "negative_captions": []}],
                "annotations": [],
                "metadata": metadata,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calibration manifest overlaps the evaluation"):
        validate_training_eval_split(
            train_manifest=train_manifest,
            eval_manifest=eval_manifest,
            grounding_json=grounding_json,
            image_root=train_image.parent,
            class_split=class_split,
        )


def test_grounding_conversion_filters_unseen_boxes_and_negative_prompts(tmp_path):
    image = _image(tmp_path / "images" / "train.jpg")
    eval_image = _image(tmp_path / "eval" / "eval.jpg")
    train_manifest, eval_manifest = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write_manifest(train_manifest, [image])
    _write_manifest(eval_manifest, [eval_image])
    class_split = _write_class_split(tmp_path / "class_split.json", unseen=["cat"])
    cleaned = tmp_path / "cleaned.jsonl"
    cleaned_rows = [
        {"image": str(image), "image_path": str(image), "query": "car", "box": [2, 2, 12, 12]},
        {"image": str(image), "image_path": str(image), "query": "cat", "box": [14, 2, 24, 12]},
    ]
    cleaned.write_text("".join(json.dumps(row) + "\n" for row in cleaned_rows), encoding="utf-8")
    negatives = tmp_path / "negatives.jsonl"
    negatives.write_text(
        "".join(
            json.dumps({"image": str(image), "image_path": str(image), "query": name}) + "\n" for name in ("cat", "bus")
        ),
        encoding="utf-8",
    )

    stats = run_convert(
        cleaned=cleaned,
        out=tmp_path / "dataset",
        fmt="grounding",
        negatives=negatives,
        include_manifest=train_manifest,
        class_split=class_split,
    )

    payload = json.loads((tmp_path / "dataset" / "grounding_train.json").read_text(encoding="utf-8"))
    audit = validate_training_eval_split(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        grounding_json=tmp_path / "dataset" / "grounding_train.json",
        image_root=image.parent,
        class_split=class_split,
    )
    assert stats["boxes"] == 1
    assert stats["filtered_unseen_boxes"] == 1
    assert stats["filtered_unseen_negatives"] == 1
    assert payload["images"][0]["caption"] == "bus. car."
    assert payload["images"][0]["negative_captions"] == ["bus"]
    assert audit["training_classes"] == 2
    assert audit["image_overlap"] == 0
