"""Safetensors and NPY integrity, recovery, and cached tensor consumption."""

from __future__ import annotations

import copy
import json
import os
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data._utils.pin_memory import pin_memory

from ultralytics.data.d1_cache import D1FeatureCacheDataset, D1TrainingBatch
from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.foundation import cache as cache_module
from ultralytics.nn.foundation.cache import (
    FeatureCacheReader,
    FeatureCacheWriter,
    build_cache_key,
    compare_feature_caches,
    sha256_bytes,
    sha256_file,
    verify_feature_cache,
)
from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader, open_feature_cache, validate_npy_evidence
from ultralytics.utils import DEFAULT_CFG_DICT


def contract(**updates):
    value = {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": ["block4", "block8", "block12"],
        "dtype": "float16",
        "expected_shape": [2, 2, 2],
    }
    value.update(updates)
    return value


def features(value):
    return {
        name: torch.full((2, 2, 2), float(value + index), dtype=torch.float32)
        for index, name in enumerate(("block4", "block8", "block12"))
    }


def image_sha(value):
    return sha256_bytes(f"image-{value}".encode())


def add_sample(writer, value):
    return writer.add(
        sample_id=f"train2017/{value:012d}",
        split="train2017",
        image_path=f"images/train2017/{value:012d}.jpg",
        image_sha256=image_sha(value),
        features=features(value),
    )


def build_cache(path: Path, values=(1, 2)):
    with FeatureCacheWriter(path, split="train2017", contract=contract(), target_shard_bytes=60) as writer:
        for value in values:
            assert add_sample(writer, value)


def test_cache_key_changes_with_every_content_identity_field():
    base = {
        "image_sha256": "1" * 64,
        "preprocessing_sha256": "2" * 64,
        "teacher_weights_sha256": "3" * 64,
        "output_layers": (4, 8, 12),
        "dtype": "float16",
    }
    baseline = build_cache_key(**base)
    for name, changed in (
        ("image_sha256", "4" * 64),
        ("preprocessing_sha256", "5" * 64),
        ("teacher_weights_sha256", "6" * 64),
        ("output_layers", (4, 12)),
    ):
        candidate = dict(base)
        candidate[name] = changed
        assert build_cache_key(**candidate) != baseline


@pytest.mark.parametrize(
    "updates, exception",
    [
        ({"dtype": "float32"}, ValueError),
        ({"output_layers": []}, ValueError),
        ({"output_layers": [8, 4]}, ValueError),
        ({"feature_names": ["block4"]}, ValueError),
        ({"expected_shape": [2, 2]}, ValueError),
        ({"teacher_weights_sha256": "bad"}, ValueError),
    ],
)
def test_invalid_cache_contract_fails_fast(tmp_path, updates, exception):
    with pytest.raises(exception):
        FeatureCacheWriter(tmp_path, split="train2017", contract=contract(**updates))


def test_sharded_roundtrip_index_and_full_verification(tmp_path):
    build_cache(tmp_path)

    report = verify_feature_cache(tmp_path)
    reader = FeatureCacheReader(tmp_path)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))

    assert report["sample_count"] == 2
    assert report["shard_count"] == 2
    assert report["tensor_count"] == 6
    assert report["part_files"] == []
    assert index["split_counts"] == {"train2017": 2}
    assert index["content_sha256"] == report["content_sha256"]
    assert all(not Path(record["image_path"]).is_absolute() for record in reader.records.values())
    loaded = reader.get("train2017/000000000001")
    assert tuple(loaded) == ("block4", "block8", "block12")
    assert loaded["block4"].dtype == torch.float16
    assert torch.all(loaded["block4"] == 1)
    assert torch.all(loaded["block12"] == 3)


def test_resume_skips_committed_sample_and_appends_next_shard(tmp_path):
    build_cache(tmp_path, values=(1,))
    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)

    assert add_sample(writer, 1) is False
    assert add_sample(writer, 2) is True
    writer.close()

    report = verify_feature_cache(tmp_path)
    assert report["sample_count"] == 2
    assert report["shard_count"] == 2


def test_rank_shard_prefix_is_unique_and_resumes(tmp_path):
    writer = FeatureCacheWriter(
        tmp_path,
        split="train2017",
        contract=contract(),
        target_shard_bytes=60,
        shard_prefix="train2017-r03",
    )
    assert add_sample(writer, 1)
    writer.close()
    resumed = FeatureCacheWriter(
        tmp_path,
        split="train2017",
        contract=contract(),
        target_shard_bytes=60,
        shard_prefix="train2017-r03",
    )
    assert add_sample(resumed, 1) is False
    assert add_sample(resumed, 2)
    resumed.close()

    assert [path.name for path in sorted(tmp_path.glob("*.safetensors"))] == [
        "train2017-r03-00000.safetensors",
        "train2017-r03-00001.safetensors",
    ]
    assert verify_feature_cache(tmp_path)["sample_count"] == 2


@pytest.mark.parametrize("prefix", ("rank00", "../train2017-r00", "train2017/r00", ""))
def test_invalid_shard_prefix_fails_fast(tmp_path, prefix):
    with pytest.raises(ValueError, match="shard_prefix"):
        FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), shard_prefix=prefix)


def test_missing_index_is_rebuilt_from_committed_shard_headers(tmp_path):
    build_cache(tmp_path)
    expected = verify_feature_cache(tmp_path)["content_sha256"]
    (tmp_path / "index.json").unlink()
    (tmp_path / "samples.jsonl").unlink()

    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)
    writer.close()

    assert verify_feature_cache(tmp_path)["content_sha256"] == expected


def test_two_independent_builds_have_identical_content(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    build_cache(first)
    build_cache(second)

    comparison = compare_feature_caches(first, second)

    assert comparison["identical"] is True
    assert comparison["sample_count"] == 2
    assert comparison["first_shards"] == comparison["second_shards"] == 2


@pytest.mark.parametrize(
    "bad_features, message",
    [
        ({"block4": torch.zeros(2, 2, 2)}, "ordered"),
        (
            {
                "block4": torch.zeros(2, 2, 2),
                "block8": torch.zeros(2, 2, 2),
                "block12": torch.zeros(2, 2, 1),
            },
            "shape",
        ),
        (
            {
                "block4": torch.zeros(2, 2, 2),
                "block8": torch.full((2, 2, 2), float("nan")),
                "block12": torch.zeros(2, 2, 2),
            },
            "NaN or Inf",
        ),
    ],
)
def test_invalid_features_fail_before_commit(tmp_path, bad_features, message):
    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)
    with pytest.raises(ValueError, match=message):
        writer.add(
            sample_id="train2017/000000000001",
            split="train2017",
            image_path="images/train2017/000000000001.jpg",
            image_sha256=image_sha(1),
            features=bad_features,
        )


def test_corrupt_shard_fails_checksum_verification(tmp_path):
    build_cache(tmp_path, values=(1,))
    shard = next(tmp_path.glob("*.safetensors"))
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="shard checksum mismatch"):
        verify_feature_cache(tmp_path)


def test_failed_shard_save_preserves_part_file(tmp_path, monkeypatch):
    safe_open, _save_file = cache_module._safetensors_api()

    def fail_after_partial_write(_tensors, filename, metadata):
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cache_module, "_safetensors_api", lambda: (safe_open, fail_after_partial_write))
    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)
    add_sample(writer, 1)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        writer.flush()

    assert (tmp_path / ".train2017-00000.safetensors.part").read_bytes() == b"partial"
    assert not list(tmp_path.glob("*.safetensors"))


def test_reader_reuses_and_evicts_shard_handles(tmp_path, monkeypatch):
    build_cache(tmp_path)
    safe_open, save_file = cache_module._safetensors_api()
    calls = []

    def counted_safe_open(*args, **kwargs):
        calls.append(Path(args[0]).name)
        return safe_open(*args, **kwargs)

    monkeypatch.setattr(cache_module, "_safetensors_api", lambda: (counted_safe_open, save_file))
    reader = FeatureCacheReader(tmp_path, max_open_shards=1)
    reader.get("train2017/000000000001")
    reader.get("train2017/000000000001")
    reader.get("train2017/000000000002")
    reader.get("train2017/000000000001")
    reader.close()

    assert calls == [
        "train2017-00000.safetensors",
        "train2017-00001.safetensors",
        "train2017-00000.safetensors",
    ]


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_reader_rejects_invalid_handle_cache_size(tmp_path, value):
    with pytest.raises(ValueError, match="max_open_shards"):
        FeatureCacheReader(tmp_path, max_open_shards=value)


NAMES = ["block4", "block8", "block12"]


@pytest.fixture
def converted(tmp_path):
    split = "train2017"
    source = tmp_path / "source"
    root = tmp_path / "npy"
    target = root / split
    target.mkdir(parents=True)
    contract = {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "feature_names": NAMES,
        "output_layers": [4, 8, 12],
        "expected_shape": [384, 40, 40],
        "dtype": "float16",
    }
    data = tmp_path / "coco"
    (data / "images" / split).mkdir(parents=True)
    (data / "labels" / split).mkdir(parents=True)
    with FeatureCacheWriter(source, split=split, contract=contract) as writer:
        for i in (1, 2):
            sid = f"{split}/{i:012d}"
            image_path = data / "images" / f"{sid}.jpg"
            assert cv2.imwrite(str(image_path), np.full((80, 120, 3), i, dtype=np.uint8))
            (data / "labels" / f"{sid}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
            writer.add(
                sample_id=sid,
                split=split,
                image_path=f"images/{sid}.jpg",
                image_sha256=sha256_file(image_path),
                features={name: torch.full((384, 40, 40), i + j, dtype=torch.float16) for j, name in enumerate(NAMES)},
            )
    reader = FeatureCacheReader(source)
    provenance = root / "provenance" / split
    provenance.mkdir(parents=True)
    for name in ("index.json", "samples.jsonl"):
        shutil.copyfile(source / name, provenance / name)
    records = []
    for sid, original in reader.records.items():
        array = np.stack([reader.get(sid)[name].numpy() for name in NAMES])
        path = root / f"{sid}.npy"
        np.save(path, array, allow_pickle=False)
        record = copy.deepcopy(original)
        record["source_shard"] = record.pop("shard")
        record.update(npy_path=f"{sid}.npy", npy_bytes=path.stat().st_size, npy_sha256=sha256_file(path))
        records.append(record)
    manifest = root / f"{split}-samples.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    index = {
        "schema_version": "d1-npy-cache-v1",
        "split": split,
        "sample_count": len(records),
        "shape": [3, 384, 40, 40],
        "dtype": "float16",
        "feature_names": NAMES,
        "contract": reader.contract,
        "contract_sha256": reader.index["contract_sha256"],
        "source_index_sha256": sha256_file(source / "index.json"),
        "source_content_sha256": reader.index["content_sha256"],
        "samples_manifest": manifest.name,
        "samples_manifest_sha256": sha256_file(manifest),
        "npy_bytes": sum(r["npy_bytes"] for r in records),
    }
    (root / f"{split}-index.json").write_text(json.dumps(index))
    (root / "receipts").mkdir()
    for shard in reader.index["shards"]:
        selected = [r for r in records if r["source_shard"] == shard["filename"]]
        receipt = {
            "schema_version": "d1-npy-cache-v1",
            "state": "VERIFIED",
            "source_index_sha256": index["source_index_sha256"],
            "source_shard": shard,
            "sample_count": len(selected),
            "verified_tensor_count": len(selected) * 3,
            "files": [
                {"sample_id": r["sample_id"], "path": r["npy_path"], "bytes": r["npy_bytes"], "sha256": r["npy_sha256"]}
                for r in selected
            ],
        }
        (root / "receipts" / f"{shard['filename']}.json").write_text(json.dumps(receipt))
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "d1-npy-cache-v1",
                "status": "COMPLETED",
                "source_indices_sha256": {split: index["source_index_sha256"]},
            }
        )
    )
    return source, target, data, summary


def test_reader_is_bitwise_equal_to_safetensors_and_picklable(converted):
    source, target, _, summary = converted
    original = FeatureCacheReader(source)
    reader = pickle.loads(pickle.dumps(open_feature_cache(target)))
    assert isinstance(reader, NpyFeatureCacheReader)
    assert isinstance(open_feature_cache(source), FeatureCacheReader)
    for sid in reader.records:
        assert tuple(reader.get(sid)) == tuple(NAMES)
        reader.verify_sample(sid)
        for name in NAMES:
            assert torch.equal(reader.get(sid)[name], original.get(sid)[name])
            assert not reader.get(sid)[name].requires_grad
    evidence = validate_npy_evidence(target, summary, "train2017", 2)
    assert evidence["sample_count"] == 2
    assert len(evidence["preflight_hashed_samples"]) == 2


@pytest.mark.parametrize("kind", ["missing", "part", "receipt", "source", "checksum", "summary"])
def test_preflight_fails_closed(converted, kind):
    _, target, _, summary = converted
    path = next(target.glob("*.npy"))
    if kind == "missing":
        path.unlink()
    elif kind == "part":
        (target / "unexpected.part").touch()
    elif kind == "receipt":
        p = next((target.parent / "receipts").glob("*.json"))
        doc = json.loads(p.read_text())
        doc["files"].pop()
        p.write_text(json.dumps(doc))
    elif kind == "source":
        p = target.parent / "provenance/train2017/index.json"
        p.write_text(p.read_text() + " ")
    elif kind == "checksum":
        with path.open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(b"\x00\x00")
    else:
        summary.write_text('{"status":"FAILED"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_npy_evidence(target, summary, "train2017", 2)


@pytest.mark.parametrize("kind", ["path", "duplicate", "order", "split", "hash", "shape"])
def test_reader_rejects_invalid_manifest(converted, kind):
    _, target, _, _ = converted
    index_path = target.parent / "train2017-index.json"
    index = json.loads(index_path.read_text())
    manifest = target.parent / index["samples_manifest"]
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if kind == "path":
        records[0]["npy_path"] = "../outside.npy"
    elif kind == "duplicate":
        records[1] = records[0]
    elif kind == "order":
        records.reverse()
    elif kind == "split":
        records[0]["split"] = "val2017"
    elif kind == "hash":
        index["contract_sha256"] = "f" * 64
    else:
        index["shape"] = [3, 384, 20, 20]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    index["samples_manifest_sha256"] = sha256_file(manifest)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        open_feature_cache(target)


def test_reader_rejects_wrong_dtype_and_nan(converted):
    _, target, _, _ = converted
    reader = NpyFeatureCacheReader(target)
    sid = next(iter(reader.records))
    path = target.parent / reader.records[sid]["npy_path"]
    np.save(path, np.zeros((3, 384, 40, 40), dtype=np.float32))
    reader.records[sid]["npy_bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="dtype"):
        reader.get(sid)
    np.save(path, np.full((3, 384, 40, 40), np.nan, dtype=np.float16))
    reader.records[sid].update(npy_bytes=path.stat().st_size, npy_sha256=sha256_file(path))
    with pytest.raises(ValueError, match="finite"):
        reader.verify_sample(sid)


def test_dataset_and_pin_memory_keep_one_feature_copy(converted, monkeypatch):
    source, target, data, _ = converted
    kwargs = {
        "img_path": str(data / "images/train2017"),
        "data": {"names": {i: str(i) for i in range(80)}, "nc": 80},
        "hyp": SimpleNamespace(**DEFAULT_CFG_DICT),
        "batch_size": 2,
    }
    old = D1FeatureCacheDataset(cache_dir=source, **kwargs)
    new = D1FeatureCacheDataset(cache_dir=target, **kwargs)
    batch = new.collate_fn([new[0], new[1]])
    reference = old.collate_fn([old[0], old[1]])
    calls = []

    def fake_pin(tensor, device=None):
        calls.append(tensor)
        return tensor.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    pinned = pickle.loads(pickle.dumps(pin_memory(batch)))
    assert isinstance(pinned, D1TrainingBatch)
    assert "img" in pinned and pinned["img"] is pinned.get("features")
    assert "img" not in tuple(pinned.keys())
    assert len([t for t in calls if t.ndim == 4]) == 3
    for name in NAMES:
        assert torch.equal(pinned["features"][name], reference["features"][name])
    for key in ("cls", "bboxes", "batch_idx"):
        assert torch.equal(pinned[key], reference[key])
    assert torch.equal(pinned["bboxes"], reference["bboxes"])


def test_resume_does_not_reset_restored_amp_scaler(monkeypatch):
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.amp = True
    trainer.amp_init_scale = 16
    trainer.resume = "checkpoint.pt"
    restored = object()
    monkeypatch.setattr(DetectionTrainer, "_setup_train", lambda self: setattr(self, "scaler", restored))
    trainer._setup_train()
    assert trainer.scaler is restored


@pytest.mark.skipif(not os.environ.get("D1_NPY_CACHE"), reason="local NPY cache integration is opt-in")
def test_real_npy_model_backward_and_checkpoint(tmp_path):
    from ultralytics.nn import D1FoundationDetectionModel
    from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer

    root = Path(os.environ["D1_NPY_CACHE"])
    reader = NpyFeatureCacheReader(root)
    sid = next(iter(reader.records))
    reader.verify_sample(sid)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = D1FoundationDetectionModel().to(device).train()
    inputs = {k: v.unsqueeze(0).to(device=device, dtype=torch.float32) for k, v in reader.get(sid).items()}
    batch = {
        "img": inputs,
        "features": inputs,
        "batch_idx": torch.zeros(1, device=device),
        "cls": torch.zeros(1, 1, device=device),
        "bboxes": torch.full((1, 4), 0.5, device=device),
    }
    with torch.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
        loss, _ = model.loss(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    assert not any("teacher" in k.lower() for k in model.state_dict())
    state = tmp_path / "model.pt"
    torch.save(model.state_dict(), state)
    restored = D1FoundationDetectionModel().to(device)
    initialize_mixture_loss_ema_buffer(restored)
    restored.load_state_dict(torch.load(state, map_location=device, weights_only=True), strict=True)


def test_npy_conversion_preserves_source_and_rejects_corruption(tmp_path):
    import torch

    from scripts.d1.convert_npy import convert_preserving_source
    from ultralytics.nn.foundation.cache import FeatureCacheWriter
    from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader

    split = "visdrone-train"
    source, out = tmp_path / split, tmp_path / "npy"
    contract = {
        "model_id": "test",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": ["block4", "block8", "block12"],
        "dtype": "float16",
        "expected_shape": [384, 40, 40],
    }
    writer = FeatureCacheWriter(source, split=split, contract=contract, shard_prefix=split + "-r00")
    feature = {
        name: torch.full((384, 40, 40), float(i), dtype=torch.float16)
        for i, name in enumerate(contract["feature_names"])
    }
    writer.add(
        sample_id=split + "/original_id",
        split=split,
        image_path="images/" + split + "/original_id.jpg",
        image_sha256="c" * 64,
        features=feature,
    )
    writer.close()
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    first = convert_preserving_source(source, out)
    assert first == convert_preserving_source(source, out)
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    reader = NpyFeatureCacheReader(out / split)
    reader.verify_sample(split + "/original_id")
    for name, value in reader.get(split + "/original_id").items():
        assert torch.equal(value, feature[name])
    path = out / split / "original_id.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        convert_preserving_source(source, out)
