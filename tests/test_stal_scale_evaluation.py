import json

import pytest
from PIL import Image

from scripts.stal.evaluate_scale_ap import build_coco_ground_truth, evaluate_scale_ap, resolve_split_images


def test_build_coco_ground_truth_preserves_original_pixel_area(tmp_path):
    """COCO size bins must use original-image pixel areas, not resized training coordinates."""
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "000001.jpg"
    Image.new("RGB", (200, 100)).save(image_path)
    (label_dir / "000001.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")

    coco = build_coco_ground_truth([image_path], {0: "target"})

    assert coco["images"] == [{"id": 1, "file_name": "000001.jpg", "width": 200, "height": 100}]
    assert coco["annotations"][0]["bbox"] == pytest.approx([90.0, 40.0, 20.0, 20.0])
    assert coco["annotations"][0]["area"] == pytest.approx(400.0)
    assert coco["annotations"][0]["category_id"] == 1


def test_resolve_split_images_filters_and_sorts_supported_images(tmp_path):
    """Directory splits should ignore labels and return a deterministic image order."""
    Image.new("RGB", (8, 8)).save(tmp_path / "b.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    resolved = resolve_split_images(str(tmp_path))

    assert [path.name for path in resolved] == ["a.png", "b.jpg"]


def test_evaluate_scale_ap_reports_perfect_small_object_prediction(tmp_path):
    """The external evaluator should expose COCO small AP from an exact synthetic prediction."""
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "000001.jpg"
    Image.new("RGB", (200, 100)).save(image_path)
    (label_dir / "000001.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    annotations_path = tmp_path / "annotations.json"
    predictions_path = tmp_path / "predictions.json"
    annotations_path.write_text(json.dumps(build_coco_ground_truth([image_path], {0: "target"})), encoding="utf-8")
    predictions_path.write_text(
        json.dumps([{"image_id": 1, "category_id": 1, "bbox": [90.0, 40.0, 20.0, 20.0], "score": 0.99}]),
        encoding="utf-8",
    )

    metrics = evaluate_scale_ap(annotations_path, predictions_path)

    assert metrics["AP"] == pytest.approx(1.0)
    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["AP75"] == pytest.approx(1.0)
    assert metrics["APs"] == pytest.approx(1.0)
    assert metrics["AP50s"] == pytest.approx(1.0)
    assert metrics["AR500"] == pytest.approx(1.0)
    assert metrics["ARs500"] == pytest.approx(1.0)


@pytest.mark.parametrize("side,expected", [(31, "APs"), (32, "APm"), (96, "APl")])
def test_visdrone_string_ids_and_exact_area_boundaries(tmp_path, side, expected):
    """String IDs must work and boundary GTs must belong to exactly one size bin."""
    image_id = "0000001_02999_d_0000005"
    bbox = [10, 10, side, side]
    gt = {
        "images": [{"id": image_id, "file_name": image_id + ".jpg", "width": 200, "height": 200}],
        "annotations": [
            {"id": 1, "image_id": image_id, "category_id": 1, "bbox": bbox, "area": side * side, "iscrowd": 0}
        ],
        "categories": [{"id": 1, "name": "target"}],
    }
    annotation_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "pred.json"
    annotation_path.write_text(json.dumps(gt), encoding="utf-8")
    prediction_path.write_text(
        json.dumps([{"image_id": image_id, "category_id": 1, "bbox": bbox, "score": 0.99}]), encoding="utf-8"
    )
    metrics = evaluate_scale_ap(annotation_path, prediction_path)
    assert metrics[expected] == pytest.approx(1.0)
    for key in {"APs", "APm", "APl"} - {expected}:
        assert metrics[key] == -1.0
    assert json.loads(annotation_path.read_text())["images"][0]["id"] == image_id


def evaluate_fixture(tmp_path, predictions, annotations=None):
    """Evaluate explicit geometry so expected AP does not depend on the implementation under test."""
    gt = {
        "images": [{"id": 1, "width": 1000, "height": 1000}],
        "categories": [{"id": 1, "name": "target"}],
        "annotations": annotations
        if annotations is not None
        else [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 0}],
    }
    annotation_path, prediction_path = tmp_path / "gt.json", tmp_path / "pred.json"
    annotation_path.write_text(json.dumps(gt), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    return evaluate_scale_ap(annotation_path, prediction_path)


def test_aps_averages_iou_thresholds_instead_of_ap50(tmp_path):
    """IoU=0.6 passes exactly .50/.55/.60, yielding APs=.3 while AP50s=1."""
    metrics = evaluate_fixture(tmp_path, [{"image_id": 1, "category_id": 1, "bbox": [5, 0, 20, 20], "score": 0.9}])
    assert metrics["APs"] == pytest.approx(0.3)
    assert metrics["AP50s"] == pytest.approx(1)
    assert metrics["AP75"] == pytest.approx(0)
    assert metrics["ARs500"] == pytest.approx(0.3)


def test_max_dets_500_keeps_rank_500_and_drops_rank_501(tmp_path):
    """A sole TP at rank 500 survives; at rank 501 it cannot contribute."""
    false_positive = {"image_id": 1, "category_id": 1, "bbox": [100, 100, 20, 20], "score": 0.9}
    true_positive = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.1}
    metrics = evaluate_fixture(tmp_path, [false_positive] * 499 + [true_positive])
    assert metrics["APs"] == pytest.approx(1 / 500)
    assert metrics["ARs500"] == pytest.approx(1)
    metrics = evaluate_fixture(tmp_path, [false_positive] * 500 + [true_positive])
    assert metrics["APs"] == 0
    assert metrics["ARs500"] == 0


def test_no_detections_is_zero_but_absent_gt_bin_is_unavailable(tmp_path):
    """A failed detector is not the same as an unavailable metric."""
    metrics = evaluate_fixture(tmp_path, [])
    assert metrics["APs"] == metrics["ARs500"] == 0
    assert metrics["APm"] == metrics["APl"] == -1
    assert all(value == -1 for value in evaluate_fixture(tmp_path, [], annotations=[]).values())


@pytest.mark.parametrize(
    "update",
    [{"category_id": 2}, {"bbox": [0, 0, -1, 20]}, {"score": float("nan")}, {"score": -0.1}, {"score": 1.1}],
)
def test_invalid_predictions_cannot_silently_improve_metrics(tmp_path, update):
    """Unknown classes and invalid numbers must not be silently ignored by the external evaluator."""
    prediction = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9}
    with pytest.raises(ValueError):
        evaluate_fixture(tmp_path, [{**prediction, **update}])


@pytest.mark.parametrize("label", ["0.5 .5 .5 .1 .1", "0 .5 .5 nan .1", "0 .5 .5 -.1 .1"])
def test_invalid_yolo_labels_are_rejected(tmp_path, label):
    """Malformed GT must not change class identity or area silently."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image = images / "1.jpg"
    Image.new("RGB", (100, 100)).save(image)
    (labels / "1.txt").write_text(label, encoding="utf-8")
    with pytest.raises(ValueError):
        build_coco_ground_truth([image], {0: "target"})


def test_missing_labels_are_not_silently_counted_as_background(tmp_path):
    """Incomplete validation mirrors must fail rather than inflate recall by deleting GT."""
    images = tmp_path / "images"
    images.mkdir()
    image = images / "1.jpg"
    Image.new("RGB", (100, 100)).save(image)
    with pytest.raises(FileNotFoundError, match="Missing evaluation label"):
        build_coco_ground_truth([image], {0: "target"})
    with pytest.raises(ValueError, match="no images"):
        build_coco_ground_truth([], {0: "target"})


@pytest.mark.parametrize("update", [{"area": 10000}, {"category_id": 2}, {"bbox": [0, 0, -20, -20]}])
def test_external_ground_truth_cannot_silently_change_metric_population(tmp_path, update):
    """Direct JSON evaluation must enforce bbox area and class validity just like the YOLO converter."""
    annotation = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 0}
    with pytest.raises(ValueError):
        evaluate_fixture(tmp_path, [], annotations=[{**annotation, **update}])


def test_duplicate_annotation_ids_are_rejected_before_coco_indexing(tmp_path):
    """COCO's ID index must not overwrite distinct GT records sharing an annotation ID."""
    annotation = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 0}
    with pytest.raises(ValueError, match="Duplicate annotation"):
        evaluate_fixture(tmp_path, [], annotations=[annotation, {**annotation, "bbox": [50, 50, 20, 20]}])
