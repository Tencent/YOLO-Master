from pathlib import Path

import pytest
from PIL import Image

from ultralytics.data.foundation_cache import (
    CACHE_SCHEMA_VERSION,
    CacheIdentity,
    load_letterboxed_tensor,
    select_images,
    stable_sample_id,
    validate_manifest,
)


def test_stable_sample_id_is_path_and_content_bound(tmp_path: Path):
    image = tmp_path / "images" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image-a")
    first = stable_sample_id(image, tmp_path)
    assert first == stable_sample_id(image, tmp_path)
    image.write_bytes(b"image-b")
    assert first != stable_sample_id(image, tmp_path)


def test_manifest_rejects_preprocess_mismatch():
    identity = CacheIdentity("model", "revision", (3, 7, 11), 640, "abc")
    manifest = {"schema_version": CACHE_SCHEMA_VERSION, "identity": identity.to_dict(), "samples": []}
    incompatible = CacheIdentity("model", "revision", (3, 7, 11), 640, "different")
    with pytest.raises(ValueError, match="preprocess_fingerprint"):
        validate_manifest(manifest, incompatible)


def test_manifest_rejects_duplicate_samples():
    identity = CacheIdentity("model", "revision", (3, 7, 11), 640, "abc")
    sample = {"sample_id": "same"}
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "samples": [sample, sample.copy()],
    }
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_manifest(manifest, identity)


def test_select_images_uses_sorted_stable_ids(tmp_path: Path):
    images = []
    for name in ("z.jpg", "a.jpg", "m.jpg"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        images.append(path)
    selected = select_images(images, root=tmp_path, limit=2)
    assert [x.sample_id for x in selected] == sorted(x.sample_id for x in selected)
    assert len(selected) == 2


def test_letterbox_is_fixed_square():
    image = Image.new("RGB", (80, 40), color="white")
    tensor, meta = load_letterboxed_tensor(image, imgsz=64)
    assert tensor.shape == (3, 64, 64)
    assert meta["original_shape"] == [40, 80]
    assert meta["resized_shape"] == [32, 64]
    assert meta["pad"] == [0, 16]
