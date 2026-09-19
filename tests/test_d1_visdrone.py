"""VisDrone annotation conversion, split isolation, and official evaluation protocol."""

import pytest
from PIL import Image

from scripts.d1.evaluate_visdrone import METRICS, TOOLKIT_COMMIT, export_predictions, validate_official_report
from scripts.d1.prepare_visdrone import convert_annotation, prepare


def test_annotation_flags_and_clipping():
    text = "-5,0,20,10,1,1,0,2\n0,0,20,20,0,4,0,0\n0,0,9,9,1,0,0,0\n0,0,9,9,0,11,0,0\n"
    label, side, count = convert_annotation(text, 100, 50)
    assert label.decode().split() == ["0", "0.0750000000", "0.1000000000", "0.1500000000", "0.2000000000"]
    assert count["training_boxes"] == 1 and count["ignored_boxes"] == 3
    assert count["clipped_boxes"] == 1 and side[0]["original"][-1] == 2
    assert len(side) == 4


@pytest.mark.parametrize(
    "text", ["nan,0,1,1,1,1,0,0", "0,0,-1,1,1,1,0,0", "0,0,1,1,1,1.5,0,0", "0,0,1,1,2,1,0,0", "0,0,1"]
)
def test_bad_annotations_fail(text):
    with pytest.raises(ValueError):
        convert_annotation(text, 100, 100)


def test_empty_and_zero_area():
    assert convert_annotation("", 32, 32)[0] == b""
    labels, side, stats = convert_annotation("100,100,5,5,1,2,0,0", 32, 32)
    assert labels == b"" and stats["dropped_degenerate"] == 1
    assert side[0]["disposition"] == "degenerate_after_clip"


def make_dataset(tmp):
    source = tmp / "original"
    for i, split in enumerate(("train", "val", "test-dev")):
        folder = source / f"VisDrone2019-DET-{split}"
        (folder / "images").mkdir(parents=True)
        (folder / "annotations").mkdir()
        Image.new("RGB", (32, 32), color=(i * 50, 0, 0)).save(folder / "images" / f"id_{i}.jpg")
        (folder / "annotations" / f"id_{i}.txt").write_text("0,0,10,10,1,1,0,0\n")
    return source


def test_prepare_is_stable_preserves_originals(tmp_path):
    source = make_dataset(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "prepared"
    counts = dict.fromkeys(("train", "val", "test-dev"), 1)
    first = prepare(source, target, counts=counts)
    assert first == prepare(source, target, counts=counts, verify=True)
    assert before == {p: p.read_bytes() for p in before}
    assert len((target / "samples.jsonl").read_text().splitlines()) == 3
    assert first["training_ignore_background_mask"] is False
    (target / "labels/visdrone-train/id_0.txt").write_text("corrupt")
    with pytest.raises(ValueError):
        prepare(source, target, counts=counts, verify=True)


def test_export_empty_and_sorted(tmp_path):
    predictions = [
        {"image_id": "a", "category_id": 9, "score": score, "bbox": [1, 2, 3, 4]} for score in (0.2, 0.9, 0.0001)
    ]
    export_predictions(predictions, ["a", "b"], tmp_path)
    assert (tmp_path / "b.txt").read_bytes() == b""
    assert (tmp_path / "a.txt").read_text().splitlines() == ["1,2,3,4,0.9,10,-1,-1", "1,2,3,4,0.2,10,-1,-1"]


@pytest.mark.parametrize(
    "category,score,box",
    [(10, 0.1, [0, 0, 1, 1]), (True, 0.1, [0, 0, 1, 1]), (0, float("nan"), [0, 0, 1, 1]), (0, 0.1, [0, 0, -1, 1])],
)
def test_invalid_export(tmp_path, category, score, box):
    with pytest.raises(ValueError):
        export_predictions([{"image_id": "a", "category_id": category, "score": score, "bbox": box}], ["a"], tmp_path)


def test_official_scale():
    report = {
        "backend": "official-matlab",
        "toolkit_commit": TOOLKIT_COMMIT,
        "metrics_percent": dict.fromkeys(METRICS, 20),
        "metrics": dict.fromkeys(METRICS, 0.2),
    }
    assert validate_official_report(report) == report
    report["metrics"]["AP_all"] = 20
    with pytest.raises(ValueError):
        validate_official_report(report)


def test_same_split_duplicate_bytes_are_preserved(tmp_path):
    source = make_dataset(tmp_path)
    train = source / "VisDrone2019-DET-train"
    (train / "images/id_repeat.jpg").write_bytes((train / "images/id_0.jpg").read_bytes())
    (train / "annotations/id_repeat.txt").write_text("")
    result = prepare(source, tmp_path / "prepared", counts={"train": 2, "val": 1, "test-dev": 1})
    assert result["splits"]["train"]["same_split_duplicate_image_bytes"] == [["id_0", "id_repeat"]]
    assert result["splits"]["train"]["images"] == 2


def test_cross_split_duplicate_bytes_rejected(tmp_path):
    source = make_dataset(tmp_path)
    first = source / "VisDrone2019-DET-train/images/id_0.jpg"
    (source / "VisDrone2019-DET-val/images/id_1.jpg").write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="across official splits"):
        prepare(source, tmp_path / "prepared", counts={"train": 1, "val": 1, "test-dev": 1})


def test_default_preparation_does_not_require_test_dev(tmp_path, monkeypatch):
    import yaml

    from scripts.d1 import prepare_visdrone

    source = make_dataset(tmp_path)
    monkeypatch.setattr(prepare_visdrone, "COUNTS", {"train": 1, "val": 1, "test-dev": 1610})
    output = tmp_path / "prepared"
    report = prepare_visdrone.prepare(source, output)
    assert set(report["splits"]) == {"train", "val"}
    assert "test" not in yaml.safe_load((output / "dataset.yaml").read_text())
