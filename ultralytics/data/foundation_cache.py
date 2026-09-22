"""Deterministic cache contracts for frozen foundation-model features."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheIdentity:
    """Fields that make two feature caches compatible."""

    model_id: str
    revision: str
    layers: tuple[int, ...]
    imgsz: int
    preprocess_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible identity."""
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CacheIdentity":
        """Build an identity from a manifest mapping."""
        return cls(
            model_id=str(payload["model_id"]),
            revision=str(payload["revision"]),
            layers=tuple(int(x) for x in payload["layers"]),
            imgsz=int(payload["imgsz"]),
            preprocess_fingerprint=str(payload["preprocess_fingerprint"]),
        )


@dataclass(frozen=True)
class SelectedImage:
    """An image selected deterministically for cache generation."""

    path: Path
    relative_path: str
    image_sha256: str
    sample_id: str


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


image_sha256 = file_sha256


def stable_sample_id(path: str | Path, root: str | Path) -> str:
    """Return an ID bound to the normalized relative path and image content."""
    image = Path(path).resolve()
    relative = image.relative_to(Path(root).resolve()).as_posix()
    material = f"{relative}\0{image_sha256(image)}".encode()
    return hashlib.sha256(material).hexdigest()[:24]


def select_images(paths: Iterable[str | Path], *, root: str | Path, limit: int) -> list[SelectedImage]:
    """Select the first ``limit`` images after sorting by stable sample ID."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    dataset_root = Path(root).resolve()
    selected = []
    for value in paths:
        path = Path(value).resolve()
        relative = path.relative_to(dataset_root).as_posix()
        digest = image_sha256(path)
        sample_id = hashlib.sha256(f"{relative}\0{digest}".encode()).hexdigest()[:24]
        selected.append(SelectedImage(path, relative, digest, sample_id))
    selected.sort(key=lambda item: item.sample_id)
    return selected[:limit]


def preprocessing_fingerprint(payload: dict[str, Any]) -> str:
    """Hash a preprocessing mapping using canonical JSON serialization."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_letterboxed_tensor(image: Image.Image | str | Path, *, imgsz: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load an RGB image as a deterministic square letterboxed CHW float tensor."""
    if imgsz <= 0:
        raise ValueError(f"imgsz must be positive, got {imgsz}")
    source = Image.open(image) if isinstance(image, (str, Path)) else image
    source = source.convert("RGB")
    width, height = source.size
    scale = min(imgsz / height, imgsz / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_left = (imgsz - resized_width) // 2
    pad_top = (imgsz - resized_height) // 2
    canvas = Image.new("RGB", (imgsz, imgsz), color=(114, 114, 114))
    canvas.paste(resized, (pad_left, pad_top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    metadata = {
        "original_shape": [height, width],
        "resized_shape": [resized_height, resized_width],
        "scale": scale,
        "pad": [pad_left, pad_top],
    }
    return tensor, metadata


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON through an fsynced temporary file and atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def validate_manifest(
    manifest: dict[str, Any],
    expected_identity: CacheIdentity | None = None,
    *,
    cache_root: str | Path | None = None,
    verify_files: bool = False,
) -> None:
    """Validate cache identity, sample uniqueness, and optional file integrity."""
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {CACHE_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    identity_payload = manifest.get("identity")
    if not isinstance(identity_payload, dict):
        raise ValueError("manifest identity must be a mapping")
    actual_identity = CacheIdentity.from_dict(identity_payload)
    if expected_identity is not None:
        for field in ("model_id", "revision", "layers", "imgsz", "preprocess_fingerprint"):
            expected = getattr(expected_identity, field)
            actual = getattr(actual_identity, field)
            if actual != expected:
                raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest samples must be a list")
    seen = set()
    root = Path(cache_root).resolve() if cache_root is not None else None
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be a mapping")
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"sample {index} has no sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if verify_files:
            if root is None:
                raise ValueError("cache_root is required when verify_files=True")
            relative = sample.get("cache_path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"sample {sample_id} has no cache_path")
            cache_file = (root / relative).resolve()
            cache_file.relative_to(root)
            if not cache_file.is_file():
                raise ValueError(f"sample {sample_id} cache file missing: {cache_file}")
            expected_hash = str(sample.get("cache_sha256", ""))
            actual_hash = file_sha256(cache_file)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"sample {sample_id} cache_sha256 mismatch: expected {expected_hash}, got {actual_hash}"
                )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheIdentity",
    "SelectedImage",
    "atomic_write_json",
    "file_sha256",
    "image_sha256",
    "load_letterboxed_tensor",
    "preprocessing_fingerprint",
    "select_images",
    "stable_sample_id",
    "validate_manifest",
]
