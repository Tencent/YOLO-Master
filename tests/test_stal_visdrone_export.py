import json

import pytest
from PIL import Image

from scripts.stal.export_visdrone_results import export_visdrone_results


def test_export_visdrone_results_writes_official_rows_and_empty_images(tmp_path):
    """Exporter should preserve all images and emit the official eight-field DET row layout."""
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (20, 10)).save(images / "000001.jpg")
    Image.new("RGB", (20, 10)).save(images / "000002.jpg")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "file_name": "000001.jpg",
                    "category_id": 4,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.875,
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = export_visdrone_results(predictions, images, tmp_path / "results")

    assert summary == {"images": 2, "detections": 1, "empty_images": 1}
    assert (tmp_path / "results" / "000001.txt").read_text(encoding="utf-8") == (
        "1.000,2.000,3.000,4.000,0.87500000,4,-1,-1\n"
    )
    assert (tmp_path / "results" / "000002.txt").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("category", [0, 11, 1.5, float("nan")])
def test_export_visdrone_results_rejects_non_evaluated_categories(tmp_path, category):
    """Ignored-region and others categories must not enter official detector submissions."""
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (20, 10)).save(images / "000001.jpg")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps([{"image_id": 1, "category_id": category, "bbox": [1, 2, 3, 4], "score": 0.5}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="category_id"):
        export_visdrone_results(predictions, images, tmp_path / "results")


@pytest.mark.parametrize("update", [{"bbox": [0, 0, 0, 2]}, {"score": float("inf")}, {"score": -0.1}, {"score": 1.1}])
def test_invalid_export_does_not_write_result_files(tmp_path, update):
    """Reject malformed values before creating a partial official submission."""
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (20, 20)).save(images / "1.jpg")
    predictions = tmp_path / "pred.json"
    prediction = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.9}
    predictions.write_text(json.dumps([{**prediction, **update}]), encoding="utf-8")
    with pytest.raises(ValueError):
        export_visdrone_results(predictions, images, tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("file_name", ["000002.jpg", "missing.jpg"])
def test_export_rejects_conflicting_image_identifiers(tmp_path, file_name):
    """A prediction must not move to another image or hide a bad filename behind its numeric ID."""
    images = tmp_path / "images"
    images.mkdir()
    for name in ("000001.jpg", "000002.jpg"):
        Image.new("RGB", (20, 20)).save(images / name)
    predictions = tmp_path / "pred.json"
    predictions.write_text(
        json.dumps([{"image_id": 1, "file_name": file_name, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.9}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        export_visdrone_results(predictions, images, tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("names", [("1.jpg", "1.png"), ("01.jpg", "001.jpg")])
def test_export_rejects_colliding_image_names(tmp_path, names):
    """Neither output TXT names nor numeric image IDs may alias two images."""
    images = tmp_path / "images"
    images.mkdir()
    for name in names:
        Image.new("RGB", (20, 20)).save(images / name)
    predictions = tmp_path / "pred.json"
    predictions.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        export_visdrone_results(predictions, images, tmp_path / "results")


def test_export_rejects_stale_result_files_without_overwriting(tmp_path):
    """Reusing a directory from another split must not leave a mixed submission behind."""
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (20, 20)).save(images / "1.jpg")
    predictions = tmp_path / "pred.json"
    predictions.write_text("[]", encoding="utf-8")
    output = tmp_path / "results"
    output.mkdir()
    (output / "old.txt").write_text("old run", encoding="utf-8")
    (output / "1.txt").write_text("current run", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected result"):
        export_visdrone_results(predictions, images, output)
    assert (output / "1.txt").read_text() == "current run"
    assert (output / "old.txt").read_text() == "old run"
