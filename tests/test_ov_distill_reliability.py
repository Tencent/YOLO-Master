"""GT reliability calibration and weighted/unweighted conversion tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.ov_distill.class_split import build_class_split
from scripts.ov_distill.convert import run_convert
from scripts.ov_distill.reliability import calibrate_reliability, fit_isotonic, predict_isotonic


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_isotonic_mapping_is_monotonic_and_bounded():
    calibrator = fit_isotonic([0.1, 0.4, 0.8, 0.9], [0, 1, 1, 0])
    predictions = predict_isotonic([0.0, 0.1, 0.2, 0.4, 0.8, 0.9, 1.0], calibrator)
    assert predictions.tolist() == sorted(predictions.tolist())
    assert all(0.0 <= value <= 1.0 for value in predictions)


def test_calibration_rejects_overlapping_image_manifests(tmp_path):
    image = tmp_path / "same.jpg"
    calibration_manifest = tmp_path / "calibration.jsonl"
    training_manifest = tmp_path / "training.jsonl"
    _write_jsonl(calibration_manifest, [{"image": str(image)}])
    _write_jsonl(training_manifest, [{"image": str(image)}])
    with pytest.raises(ValueError, match="overlap"):
        calibrate_reliability(
            cleaned=tmp_path / "not-read.jsonl",
            gt_labels=tmp_path / "labels",
            calibrate_manifest=calibration_manifest,
            train_manifest=training_manifest,
            out=tmp_path / "calibrator.json",
        )


def test_calibration_matches_each_ground_truth_box_once(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cal1, cal2, train = (images / f"{name}.jpg" for name in ("cal1", "cal2", "train"))
    for image in (cal1, cal2, train):
        Image.new("RGB", (100, 100), "white").save(image)
    # COCO class id 0 is person; normalized GT box is [15, 15, 35, 35].
    (labels / "cal1.txt").write_text("0 0.25 0.25 0.2 0.2\n", encoding="utf-8")
    (labels / "cal2.txt").write_text("0 0.25 0.25 0.2 0.2\n", encoding="utf-8")

    def row(image: Path, box: list[float], score: float) -> dict:
        return {
            "image": str(image),
            "image_path": str(image),
            "query": "person",
            "box": box,
            "reliability_score": score,
        }

    cleaned = tmp_path / "cleaned.jsonl"
    _write_jsonl(
        cleaned,
        [
            row(cal1, [15, 15, 35, 35], 0.9),  # TP
            row(cal1, [15, 15, 35, 35], 0.7),  # duplicate: one GT cannot validate two boxes
            row(cal1, [70, 70, 90, 90], 0.1),  # FP
            row(cal2, [15, 15, 35, 35], 0.5),  # TP
            {**row(cal1, [15, 15, 35, 35], 0.8), "query": "cat"},  # excluded by the class holdout
        ],
    )
    calibration_manifest = tmp_path / "calibration.jsonl"
    training_manifest = tmp_path / "training.jsonl"
    _write_jsonl(calibration_manifest, [{"image": str(cal1)}, {"image": str(cal2)}])
    _write_jsonl(training_manifest, [{"image": str(train)}])
    class_split = build_class_split()
    class_split["unseen"] = ["cat"]
    class_split["seen"] = sorted(set(class_split["class_names"]) - {"cat"})
    class_split["unseen_ratio"] = 1 / len(class_split["class_names"])
    class_split_path = tmp_path / "class_split.json"
    class_split_path.write_text(json.dumps(class_split), encoding="utf-8")

    output = tmp_path / "calibrator.json"
    payload = calibrate_reliability(
        cleaned=cleaned,
        gt_labels=labels,
        calibrate_manifest=calibration_manifest,
        train_manifest=training_manifest,
        out=output,
        min_samples=4,
        min_positives=2,
        min_negatives=2,
        class_split=class_split_path,
    )
    stats = payload["statistics"]
    assert stats["boxes"] == 4
    assert stats["true_positives"] == 2
    assert stats["false_positives"] == 2
    assert stats["calibrated"]["brier"] <= stats["raw"]["brier"]
    assert json.loads(output.read_text(encoding="utf-8"))["calibration"]["method"] == "isotonic_pav"
    assert payload["statistics"]["boxes"] == 4
    assert payload["unseen_classes"] == ["cat"]


def test_convert_weight_ablation_uses_same_disjoint_image_set(tmp_path):
    train_image, calibration_image = tmp_path / "train.jpg", tmp_path / "calibration.jpg"
    for image in (train_image, calibration_image):
        Image.new("RGB", (100, 80), "white").save(image)
    cleaned = tmp_path / "cleaned.jsonl"
    _write_jsonl(
        cleaned,
        [
            {
                "image": str(train_image),
                "image_path": str(train_image),
                "query": "person",
                "box": [10, 10, 30, 30],
                "reliability_score": 0.4,
                "supervision_weight": 0.4,
            },
            {
                "image": str(calibration_image),
                "image_path": str(calibration_image),
                "query": "person",
                "box": [20, 20, 40, 40],
                "reliability_score": 0.9,
                "supervision_weight": 0.9,
            },
        ],
    )
    negatives = tmp_path / "negatives.jsonl"
    _write_jsonl(
        negatives,
        [
            {"image": str(train_image), "image_path": str(train_image), "query": "airship"},
            {"image": str(calibration_image), "image_path": str(calibration_image), "query": "airship"},
        ],
    )
    train_manifest = tmp_path / "train_manifest.jsonl"
    _write_jsonl(train_manifest, [{"image": str(train_image)}])
    calibration_manifest = tmp_path / "calibration_manifest.jsonl"
    _write_jsonl(calibration_manifest, [{"image": str(calibration_image)}])
    calibration_map = tmp_path / "mapping.json"
    calibration_map.write_text(
        json.dumps(
            {
                "calibration": {
                    "method": "isotonic_pav",
                    "upper_bounds": [0.5, 1.0],
                    "probabilities": [0.2, 0.8],
                },
                "calibration_manifest": str(calibration_manifest.resolve()),
            }
        ),
        encoding="utf-8",
    )

    common = {
        "cleaned": cleaned,
        "fmt": "grounding",
        "negatives": negatives,
        "include_manifest": train_manifest,
    }
    calibrated_stats = run_convert(
        out=tmp_path / "weighted",
        weight_mode="calibrated",
        calibration_map=calibration_map,
        **common,
    )
    uniform_stats = run_convert(out=tmp_path / "uniform", weight_mode="uniform", **common)
    assert calibrated_stats["images"] == uniform_stats["images"] == 1
    assert calibrated_stats["boxes"] == uniform_stats["boxes"] == 1
    assert calibrated_stats["confirmed_negatives"] == uniform_stats["confirmed_negatives"] == 1
    weighted = json.loads((tmp_path / "weighted/grounding_train.json").read_text(encoding="utf-8"))
    unweighted = json.loads((tmp_path / "uniform/grounding_train.json").read_text(encoding="utf-8"))
    assert weighted["annotations"][0]["weight"] == pytest.approx(0.2)
    assert unweighted["annotations"][0]["weight"] == 1.0
    assert weighted["images"][0]["negative_captions"] == ["airship"]

    with pytest.raises(ValueError, match="overlaps the GT calibration split"):
        run_convert(
            cleaned=cleaned,
            out=tmp_path / "leaky",
            fmt="grounding",
            weight_mode="calibrated",
            calibration_map=calibration_map,
            include_manifest=calibration_manifest,
        )
