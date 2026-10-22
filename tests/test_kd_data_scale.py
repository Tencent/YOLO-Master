"""Reject confounded data-scale comparisons before performing any training."""

import copy

import pytest

from scripts.compare_kd_data_scale import validate_baseline


def fixture():
    def row(i):
        return {"image": f"{i}.jpg", "image_sha256": f"image{i}", "label_sha256": f"label{i}"}

    old = {"splits": {"train": [row(0), row(1)], "dev": [row(9)]}}
    new = {"splits": {"train": [row(i) for i in range(4)], "dev": [row(9)]}}
    report = {
        "settings": {"batch": 2, "seed": 42},
        "baseline_stable": True,
        "dev": [
            {
                "name": "sparse_kd_long",
                "kd_mode": "plain",
                "training_routing": {"executed_experts_per_sample": 2},
                "history": [{"detection_loss": 0, "examples": 2, "optimizer_steps": 1} for _ in range(6)],
            }
        ],
    }
    return report, old, new


def test_nested_data_scale_accepts_equal_real_budgets_and_unchanged_development():
    report, old, new = fixture()
    result = validate_baseline(report, old, new, {"batch": 2, "seed": 42})
    assert len(result["history"]) == 6


@pytest.mark.parametrize("change", ["seed", "labels", "dev", "steps", "detection", "experts", "complete", "size"])
def test_data_scale_rejects_confounded_comparisons(change):
    report, old, new = fixture()
    if change == "seed":
        report["settings"]["seed"] = 43
    elif change == "labels":
        new["splits"]["train"][0]["label_sha256"] = "modified"
    elif change == "dev":
        new["splits"]["dev"] = copy.deepcopy(new["splits"]["train"][:1])
    elif change == "steps":
        report["dev"][0]["history"][0]["optimizer_steps"] = 2
    elif change == "detection":
        report["dev"][0]["history"][0]["detection_loss"] = 1
    elif change == "experts":
        report["dev"][0]["training_routing"]["executed_experts_per_sample"] = 3
    elif change == "complete":
        report["baseline_stable"] = False
    else:
        new["splits"]["train"].pop()
    with pytest.raises(ValueError):
        validate_baseline(report, old, new, {"batch": 2, "seed": 42})


def test_download_receipts_reuse_only_unchanged_images(tmp_path, monkeypatch):
    import io

    import cv2
    import numpy as np

    from scripts.prepare_sparse_supervised_data import download_verified_image

    encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
    calls = []

    def response(url, timeout):
        calls.append(url)
        body = io.BytesIO(encoded)
        body.headers = {"Content-Length": str(len(encoded))}
        return body

    monkeypatch.setattr("scripts.prepare_sparse_supervised_data.urllib.request.urlopen", response)
    path = tmp_path / "image.jpg"
    download_verified_image("https://example.com/image.jpg", path)
    download_verified_image("https://example.com/image.jpg", path)
    assert len(calls) == 1
    path.write_bytes(b"changed")
    download_verified_image("https://example.com/image.jpg", path)
    assert len(calls) == 2 and path.read_bytes() == encoded


def test_incomplete_download_never_replaces_existing_image(tmp_path, monkeypatch):
    import io

    from scripts.prepare_sparse_supervised_data import download_verified_image

    def response(url, timeout):
        body = io.BytesIO(b"short")
        body.headers = {"Content-Length": "100"}
        return body

    monkeypatch.setattr("scripts.prepare_sparse_supervised_data.urllib.request.urlopen", response)
    path = tmp_path / "image.jpg"
    path.write_bytes(b"old content")
    with pytest.raises(RuntimeError, match="download failed"):
        download_verified_image("https://example.com/image.jpg", path)
    assert path.read_bytes() == b"old content"
    assert not path.with_suffix(".jpg.receipt.json").exists()


def test_prepare_resume_keeps_plan_and_reuses_verified_downloads(tmp_path, monkeypatch):
    import io
    import json
    import zipfile

    import cv2
    import numpy as np

    from scripts import prepare_sparse_supervised_data as preparation

    archive = tmp_path / "labels.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("coco/train2017.txt", "000001.jpg\n000002.jpg\n")
        bundle.writestr("coco/val2017.txt", "000101.jpg\n000102.jpg\n000103.jpg\n000104.jpg\n")
    exclusion = tmp_path / "excluded.json"
    exclusion.write_text(json.dumps({"calibration": [], "evaluation": []}))
    output = tmp_path / "data"
    requests = []

    def response(url, timeout):
        requests.append(url)
        color = int(url.rsplit("/", 1)[1].split(".")[0])
        encoded = cv2.imencode(".jpg", np.full((8, 8, 3), color, dtype=np.uint8))[1].tobytes()
        body = io.BytesIO(encoded)
        body.headers = {"Content-Length": str(len(encoded))}
        return body

    monkeypatch.setattr(preparation.urllib.request, "urlopen", response)
    argv = [
        "prepare",
        "--archive",
        str(archive),
        "--exclude-report",
        str(exclusion),
        "--output",
        str(output),
        "--train-source",
        "train2017",
        "--counts",
        "1",
        "1",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert preparation.main() == 0 and len(requests) == 3
    initial = (output / "manifest.json").read_bytes()
    monkeypatch.setattr("sys.argv", argv + ["--resume"])
    assert preparation.main() == 0 and len(requests) == 3
    assert (output / "manifest.json").read_bytes() == initial
    monkeypatch.setattr("sys.argv", argv[:-1] + ["2", "--resume"])
    with pytest.raises(SystemExit):
        preparation.main()
    assert (output / "manifest.json").read_bytes() == initial and len(requests) == 3
