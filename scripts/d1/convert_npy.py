"""Lossless, non-destructive conversion of verified D1 safetensors to per-image NPY."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

from scripts.d1.artifacts import encoded, file_sha, immutable
from ultralytics.nn.foundation.cache import FeatureCacheReader, sha256_bytes
from ultralytics.nn.foundation.npy_cache import NPY_SCHEMA_VERSION, NpyFeatureCacheReader

NAMES = FEATURE_NAMES = ("block4", "block8", "block12")
SHAPE = (384, 40, 40)
sha = sha256_bytes


def safe_path(root, relative):
    root = Path(root).absolute()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path escapes the selected directory")
    target = root / relative
    for item in (target, *target.parents):
        if item.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {item}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Resolved path escapes the selected directory")
    return target


def validate_array(value, record, shape=SHAPE):
    if value.shape != (3, *shape) or value.dtype != np.dtype("float16"):
        raise ValueError("NPY shape/dtype mismatch")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite features")
    if set(record["tensors"]) != set(NAMES):
        raise ValueError("Feature names mismatch")
    for i, name in enumerate(NAMES):
        info = record["tensors"][name]
        if info["shape"] != list(shape) or info["dtype"] != "float16" or info["nbytes"] != value[i].nbytes:
            raise ValueError("Source tensor metadata mismatch")
        if sha(value[i].tobytes(order="C")) != info["sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {record['sample_id']} {name}")


def check_npy(path, record, shape=SHAPE):
    path = safe_path(Path(path).parent, Path(path).name)
    value = np.load(path, allow_pickle=False)
    validate_array(value, record, shape)
    return {
        "sample_id": record["sample_id"],
        "path": record["sample_id"] + ".npy",
        "bytes": path.stat().st_size,
        "sha256": file_sha(path),
    }


def convert_preserving_source(source, output):
    """Verify every tensor before committing converted files; never retire the source."""
    source, output = Path(source), Path(output)
    reader = FeatureCacheReader(source)
    split = source.name
    if split not in ("visdrone-train", "visdrone-val", "train2017", "val2017"):
        raise ValueError("Expected a COCO or VisDrone split directory")
    source_sha = file_sha(source / "index.json")
    for filename in ("index.json", "samples.jsonl"):
        immutable(output / "provenance" / split / filename, (source / filename).read_bytes())
    records, total_bytes = [], 0
    for shard in reader.index["shards"]:
        shard_path = source / shard["filename"]
        if file_sha(shard_path) != shard["sha256"]:
            raise ValueError("Source shard checksum failed")
        members = sorted(
            (r for r in reader.records.values() if r["shard"] == shard["filename"]), key=lambda r: r["sample_id"]
        )
        checked = []
        with safe_open(shard_path, framework="numpy") as handle:
            for record in members:
                # The generic reader enforces portable split/image paths on readback.
                if record["sample_id"] != f"{split}/{Path(record['image_path']).stem}":
                    raise ValueError("Unsafe NPY identity")
                dest = safe_path(output, record["sample_id"] + ".npy")
                if not dest.exists():
                    array = np.stack([handle.get_tensor(record["tensors"][n]["key"]) for n in FEATURE_NAMES])
                    validate_array(array, record)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".part")
                    if tmp.is_symlink():
                        raise ValueError("Unsafe temporary NPY")
                    with tmp.open("wb") as stream:
                        np.save(stream, array, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    check_npy(tmp, record)
                    os.replace(tmp, dest)
                checked.append(check_npy(dest, record))
                item = dict(record)
                item["source_shard"] = item.pop("shard")
                item.update(
                    npy_path=checked[-1]["path"], npy_sha256=checked[-1]["sha256"], npy_bytes=checked[-1]["bytes"]
                )
                total_bytes += item["npy_bytes"]
                records.append(item)
        receipt = {
            "schema_version": NPY_SCHEMA_VERSION,
            "state": "VERIFIED",
            "source_index_sha256": source_sha,
            "source_shard": shard,
            "sample_count": len(members),
            "verified_tensor_count": len(members) * 3,
            "files": checked,
        }
        immutable(output / "receipts" / f"{shard['filename']}.json", encoded(receipt))
    records.sort(key=lambda r: r["sample_id"])
    data = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in records)
    immutable(output / f"{split}-samples.jsonl", data)
    index = {
        "schema_version": NPY_SCHEMA_VERSION,
        "sample_count": len(records),
        "split": split,
        "contract": reader.contract,
        "contract_sha256": reader.index["contract_sha256"],
        "source_content_sha256": reader.index["content_sha256"],
        "source_index_sha256": source_sha,
        "feature_names": list(FEATURE_NAMES),
        "dtype": "float16",
        "shape": [3, 384, 40, 40],
        "samples_manifest": f"{split}-samples.jsonl",
        "samples_manifest_sha256": sha256_bytes(data),
        "npy_bytes": total_bytes,
    }
    immutable(output / f"{split}-index.json", encoded(index))
    converted = NpyFeatureCacheReader(output / split)
    for sid in converted.records:
        converted.verify_sample(sid)
    return index
