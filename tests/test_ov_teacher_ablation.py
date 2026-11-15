"""Matched single-teacher versus multi-teacher student-supervision export tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.ov_distill.ablate_teacher_supervision import prepare_datasets
from scripts.ov_distill.class_split import build_class_split, validate_class_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_split(path: Path) -> Path:
    artifact = build_class_split()
    artifact["unseen"] = ["cat"]
    artifact["seen"] = sorted(set(artifact["class_names"]) - {"cat"})
    artifact["unseen_ratio"] = 1 / len(artifact["class_names"])
    path.write_text(json.dumps(validate_class_split(artifact)), encoding="utf-8")
    return path


def test_prepare_teacher_ablation_exports_matched_audited_datasets(tmp_path):
    image_root = tmp_path / "train_images"
    image_root.mkdir()
    train_images = [image_root / f"train_{index}.jpg" for index in range(3)]
    eval_image = tmp_path / "eval_images" / "eval.jpg"
    for image in [*train_images, eval_image]:
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), "white").save(image)

    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    _write_jsonl(train_manifest, [{"image": str(image)} for image in train_images])
    _write_jsonl(eval_manifest, [{"image": str(eval_image)}])
    class_split = _write_split(tmp_path / "class_split.json")

    single_cleaned = tmp_path / "single_cleaned.jsonl"
    multi_cleaned = tmp_path / "multi_cleaned.jsonl"
    _write_jsonl(
        single_cleaned,
        [
            {"image": str(train_images[0]), "image_path": str(train_images[0]), "query": "car", "box": [1, 1, 15, 15]},
            {"image": str(train_images[1]), "image_path": str(train_images[1]), "query": "bus", "box": [2, 2, 18, 18]},
        ],
    )
    _write_jsonl(
        multi_cleaned,
        [
            {"image": str(train_images[0]), "image_path": str(train_images[0]), "query": "car", "box": [1, 1, 15, 15]},
            {
                "image": str(train_images[0]),
                "image_path": str(train_images[0]),
                "query": "truck",
                "box": [18, 1, 35, 18],
            },
            {"image": str(train_images[1]), "image_path": str(train_images[1]), "query": "bus", "box": [2, 2, 18, 18]},
            {"image": str(train_images[2]), "image_path": str(train_images[2]), "query": "cat", "box": [5, 5, 20, 20]},
        ],
    )
    single_negatives, multi_negatives = tmp_path / "single_negatives.jsonl", tmp_path / "multi_negatives.jsonl"
    _write_jsonl(
        single_negatives,
        [{"image": str(train_images[2]), "image_path": str(train_images[2]), "query": "truck"}],
    )
    _write_jsonl(
        multi_negatives,
        [{"image": str(train_images[2]), "image_path": str(train_images[2]), "query": "airplane"}],
    )

    report = prepare_datasets(
        out=tmp_path / "ablation",
        single_cleaned=single_cleaned,
        multi_cleaned=multi_cleaned,
        single_negatives=single_negatives,
        multi_negatives=multi_negatives,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        image_root=image_root,
        class_split=class_split,
    )

    single = json.loads(Path(report["datasets"]["single_teacher"]["dataset_json"]).read_text(encoding="utf-8"))
    multi = json.loads(Path(report["datasets"]["multi_teacher"]["dataset_json"]).read_text(encoding="utf-8"))
    assert report["protocol"]["same_train_eval_split_for_both_arms"] is True
    assert report["protocol"]["box_weight_mode"] == "uniform"
    assert report["datasets"]["single_teacher"]["boxes"] == 2
    assert report["datasets"]["multi_teacher"]["boxes"] == 3
    assert report["datasets"]["multi_teacher"]["filtered_unseen_boxes"] == 1
    assert report["paired_image_comparison"]["shared_images"] == 3
    assert report["paired_image_comparison"]["single_teacher_boxes_on_shared_images"] == 2
    assert report["paired_image_comparison"]["multi_teacher_boxes_on_shared_images"] == 3
    assert report["datasets"]["single_teacher"]["split_audit"]["image_overlap"] == 0
    assert all(annotation["weight"] == 1.0 for annotation in single["annotations"] + multi["annotations"])
    assert all("cat" not in image["caption"] for image in multi["images"])
    assert report["training_launched"] is False
