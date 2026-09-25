"""Shared, dependency-light helpers for the A3 precision harness."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_set_digest(files: list[Path], root: Path | None = None) -> str:
    """Hash an ordered file selection, including names, sizes, and contents."""
    digest = hashlib.sha256()
    for item in files:
        try:
            display = item.relative_to(root).as_posix() if root is not None else str(item)
        except ValueError:
            display = str(item)
        digest.update(f"{display}\t{item.stat().st_size}\t{sha256_file(item)}\n".encode())
    return digest.hexdigest()


def json_dump(path: str | Path, payload: Any) -> Path:
    """Write deterministic UTF-8 JSON and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def expand_env(value: str, *, strict: bool = True) -> str:
    """Expand ``${NAME}`` and ``${NAME:-default}`` without shell evaluation."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name)
        if resolved:
            return resolved
        if default is not None:
            return default
        if strict:
            raise ValueError(f"required environment variable {name!r} is not set")
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def collect_images(source: str | Path, *, limit: int | None = None, seed: int | None = None) -> list[Path]:
    """Collect a stable image list from a directory, image, or text file."""
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        images = [path]
    elif path.is_file():
        images = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            item = Path(raw)
            if not item.is_absolute():
                item = path.parent / item
            images.append(item.resolve())
    elif path.is_dir():
        images = sorted(item.resolve() for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(path)
    missing = [str(item) for item in images if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"image list contains {len(missing)} missing files; first={missing[0]}")
    if seed is not None:
        generator = random.Random(seed)
        generator.shuffle(images)
    return images[:limit] if limit is not None and limit >= 0 else images


def letterbox_image(path: str | Path, imgsz: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load one image using the deployment letterbox convention, returning NCHW RGB."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    height, width = image.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    left, top = (imgsz - new_width) // 2, (imgsz - new_height) // 2
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas[top : top + new_height, left : left + new_width] = resized
    tensor = canvas[:, :, ::-1].astype(dtype) / np.array(255.0, dtype=dtype)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def percentile(values: list[float], q: float) -> float:
    """Return a finite percentile, or zero for an empty input."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0
