"""Deterministic extraction, resumable batches, and cache command validation."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.d1 import cache_features as cache
from scripts.d1.cache_features import cache_contract, load_image, make_letterbox, split_paths

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cache_contract_is_derived_from_tracked_manifests():
    value = cache_contract(REPO_ROOT)

    assert value["schema_version"] == "d1-cache-v1"
    assert value["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert value["teacher_weights_sha256"] == "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
    assert value["output_layers"] == [4, 8, 12]
    assert value["feature_names"] == ["block4", "block8", "block12"]
    assert value["dtype"] == "float16"
    assert value["expected_shape"] == [384, 40, 40]


def test_fixed_100_paths_are_sorted_and_stable(tmp_path):
    paths = [f"images/train2017/{i:012d}.jpg" for i in range(1, 101)]
    manifests = tmp_path / "experiments/d1/manifests"
    manifests.mkdir(parents=True)
    (manifests / "coco2017-train2017.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    listing = manifests / "coco2017-train2017.txt"
    first, first_sha256 = split_paths(listing, "train2017", 100)
    second, second_sha256 = split_paths(listing, "train2017", 100)

    assert len(first) == 100
    assert first == sorted(first)
    assert first == second
    assert first_sha256 == second_sha256
    assert first == paths


def test_letterbox_is_deterministic_rgb_chw(tmp_path):
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    letterbox = make_letterbox()

    first = load_image(path, letterbox)
    second = load_image(path, letterbox)

    assert first.shape == (3, 640, 640)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.allclose(first[:, 320, 320], torch.tensor([30, 20, 10]) / 255)
    assert first.min() >= 0 and first.max() <= 1


@pytest.mark.parametrize(
    "entries",
    [
        [],
        ["images/train2017/a.jpg", "images/train2017/a.jpg"],
        ["images/train2017/b.jpg", "images/train2017/a.jpg"],
        ["../a.jpg"],
        ["/tmp/a.jpg"],
        ["images/val2017/a.jpg"],
        ["images/train2017/a.jpg "],
    ],
)
def test_bad_input_lists_are_rejected(tmp_path, entries):
    path = tmp_path / "list.txt"
    path.write_text("\n".join(entries) + ("\n" if entries else ""))
    with pytest.raises(ValueError, match="portable"):
        split_paths(path, "train2017")


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    pytest.importorskip("fcntl")
    calls = []
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"test-weights")
    contract = cache_contract(REPO_ROOT)
    contract["teacher_weights_sha256"] = cache.sha256_file(weights / "model.safetensors")
    contract["expected_shape"] = [1, 2, 2]
    monkeypatch.setattr(cache, "cache_contract", lambda root: contract)
    monkeypatch.setattr(cache, "verify_model", lambda *a, **kw: {})

    class Teacher:
        def __init__(self, **kwargs):
            assert kwargs["output_layers"] == (4, 8, 12)

        def encode(self, images):
            values = images[:, 0, 320, 320]
            calls.append(values.tolist())
            # Deliberately batch-dependent to detect regrouping after a partial commit.
            values = values + values.mean()
            return SimpleNamespace(
                dense={
                    name: (values[:, None, None, None].expand(-1, 1, 2, 2) + i).half()
                    for i, name in enumerate(cache.FEATURE_NAMES)
                }
            )

    monkeypatch.setattr(cache, "DINOv3Teacher", Teacher)

    def make(split):
        data = tmp_path / split / "data"
        images = data / "images" / split
        images.mkdir(parents=True)
        names = []
        for i in (1, 2, 3):
            filename = f"{i:012d}.jpg"
            assert cv2.imwrite(str(images / filename), np.full((40, 60, 3), i * 20, np.uint8))
            names.append(f"images/{split}/{filename}")
        listing = tmp_path / split / "list.txt"
        listing.write_text("\n".join(names) + "\n")
        return cache.parser().parse_args(
            [
                "build",
                "--data-root",
                str(data),
                "--weights-dir",
                str(weights),
                "--samples-file",
                str(listing),
                "--cache-dir",
                str(tmp_path / split / "first" / split),
                "--split",
                split,
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--target-shard-bytes",
                "1",
            ]
        )

    return make, calls


@pytest.mark.parametrize("split", cache.SPLITS)
def test_unified_build_is_repeatable_and_recovers_index(extraction, split):
    make, calls = extraction
    args = make(split)
    first = cache.build(args)
    assert first["new_sample_count"] == 3 and first["verification"]["tensor_count"] == 9
    assert first["forwarded_image_count"] == 3
    args.cache_dir.joinpath("index.json").unlink()
    args.cache_dir.joinpath("samples.jsonl").unlink()
    again = cache.build(args)
    assert again["resumed_sample_count"] == 3 and again["forwarded_image_count"] == 0
    assert len(calls) == 2
    assert first["verification"]["content_sha256"] == again["verification"]["content_sha256"]
    old = args.cache_dir
    args.cache_dir = old.parent.parent / "second" / split
    second = cache.build(args)
    assert cache.compare_feature_caches(old, args.cache_dir)["identical"]
    assert second["selected_paths_sha256"] == first["selected_paths_sha256"]


def test_partial_batch_resume_replays_original_context(extraction, monkeypatch):
    make, calls = extraction
    args = make("train2017")
    original = cache.FeatureCacheWriter

    class InterruptedWriter(original):
        def add(self, **kwargs):
            if self.committed_sample_count == 1:
                raise RuntimeError("simulated interruption")
            result = super().add(**kwargs)
            self.flush()
            return result

    monkeypatch.setattr(cache, "FeatureCacheWriter", InterruptedWriter)
    with pytest.raises(RuntimeError, match="interruption"):
        cache.build(args)
    monkeypatch.setattr(cache, "FeatureCacheWriter", original)
    report = cache.build(args)
    assert report["new_sample_count"] == 2 and report["resumed_sample_count"] == 1
    assert report["forwarded_image_count"] == 3
    assert calls[0] == calls[1] and len(calls[1]) == 2
    resumed = args.cache_dir
    args.cache_dir = resumed.parent.parent / "reference" / args.split
    cache.build(args)
    assert cache.compare_feature_caches(resumed, args.cache_dir)["identical"]


@pytest.mark.parametrize("change", ("batch", "image", "list", "part", "weights"))
def test_build_rejects_changed_inputs(extraction, change):
    make, calls = extraction
    args = make("train2017")
    cache.build(args)
    if change == "batch":
        args.batch_size = 1
    elif change == "image":
        next((args.data_root / "images/train2017").glob("*.jpg")).write_bytes(b"changed")
    elif change == "list":
        args.limit = 2
    elif change == "part":
        (args.cache_dir / ".unexpected.part").write_bytes(b"partial")
    else:
        (args.weights_dir / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(ValueError):
        cache.build(args)
    assert len(calls) == 2


def test_concurrent_writer_is_rejected(extraction):
    import fcntl

    make, calls = extraction
    args = make("train2017")
    args.cache_dir.mkdir(parents=True)
    with (args.cache_dir / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            cache.build(args)
    assert calls == []


def test_legacy_cache_is_readable_but_cannot_be_rebatched(extraction):
    make, _ = extraction
    args = make("visdrone-val")
    cache.build(args)
    (args.cache_dir / "build.json").unlink()
    assert cache.verify_feature_cache(args.cache_dir)["sample_count"] == 3
    with pytest.raises(ValueError, match="Legacy cache"):
        cache.build(args)


def test_immutable_artifact_checks_before_symlink_write(tmp_path):
    from scripts.d1.artifacts import immutable

    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        immutable(alias / "file.json", b"{}")
    assert not list(outside.iterdir())
