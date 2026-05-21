from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Any

from runtime.cli.contract import json_safe
from runtime.cli.normalize import normalize_value, parse_bool, resolved_path

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
def read_image_list_file(path: Path, root: Path | None = None) -> list[Path]:
    images: list[Path] = []
    base = root or path.parent
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        if is_image_file(candidate):
            images.append(candidate)
    return images
def expand_image_reference(value: Any, root: Path | None = None) -> list[Path]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        images: list[Path] = []
        for item in value:
            images.extend(expand_image_reference(item, root=root))
        return images
    text = str(value)
    if text.startswith(("http://", "https://", "data:image/")):
        return []
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = ((root or REPO_ROOT) / candidate).resolve()
    if candidate.is_dir():
        return sorted(path.resolve() for path in candidate.rglob("*") if is_image_file(path))
    if candidate.is_file():
        if candidate.suffix.lower() == ".txt":
            return read_image_list_file(candidate, root=root)
        if is_image_file(candidate):
            return [candidate.resolve()]
    return []
def normalize_dataset_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except Exception:
                continue
        return normalized
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}
def load_dataset_yaml(data_ref: Any) -> tuple[Path, dict[str, Any]]:
    import yaml

    if data_ref in (None, ""):
        raise ValueError("`inputs.data` or `params.data` is required when no image `source` is provided.")
    normalized = normalize_value(data_ref)
    data_path = Path(str(normalized))
    if not data_path.is_absolute():
        data_path = resolved_path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_ref}")
    loaded = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")
    return data_path, loaded
def dataset_settings_dir() -> Path | None:
    try:
        settings = dict(importlib.import_module("ultralytics.utils").SETTINGS)
    except Exception:
        return None
    value = settings.get("datasets_dir")
    return Path(str(value)).expanduser().resolve() if value else None
def resolve_dataset_root(data_path: Path, dataset_cfg: dict[str, Any]) -> Path:
    path_value = dataset_cfg.get("path")
    if path_value in (None, ""):
        return data_path.parent.resolve()
    candidate_path = Path(str(path_value)).expanduser()
    if candidate_path.is_absolute():
        return candidate_path.resolve()

    candidates = [
        (data_path.parent / candidate_path).resolve(),
        (REPO_ROOT / candidate_path).resolve(),
    ]
    settings_dir = dataset_settings_dir()
    if settings_dir is not None:
        candidates.append((settings_dir / candidate_path).resolve())
    candidates.append((REPO_ROOT.parent / "datasets" / candidate_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
def dataset_split_spec(dataset_cfg: dict[str, Any], requested_split: str) -> tuple[str, Any]:
    for split in (requested_split, "val", "train", "test"):
        value = dataset_cfg.get(split)
        if value not in (None, ""):
            return split, value
    raise ValueError(f"Dataset YAML has no usable split for `{requested_split}`.")
def collect_dataset_images(data_ref: Any, split: str) -> tuple[list[Path], dict[str, Any], dict[int, str]]:
    data_path, dataset_cfg = load_dataset_yaml(data_ref)
    root = resolve_dataset_root(data_path, dataset_cfg)
    actual_split, spec = dataset_split_spec(dataset_cfg, split)
    images = expand_image_reference(spec, root=root)
    names = normalize_dataset_names(dataset_cfg.get("names", {}))
    dataset_info = {
        "data": str(data_path),
        "root": str(root),
        "split": actual_split,
        "requested_split": split,
        "source": json_safe(spec),
        "names_count": len(names),
    }
    return images, dataset_info, names
def dedupe_images(images: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for image in images:
        key = str(image.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(image.resolve())
    return unique
def select_image_sample(
    images: list[Path],
    *,
    limit: int | None,
    offset: int = 0,
    stride: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[Path]:
    selected = list(images)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if offset > 0:
        selected = selected[offset:]
    if stride > 1:
        selected = selected[::stride]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected
def collect_multimodal_evaluation_images(
    request: dict[str, Any],
    evaluate_params: dict[str, Any],
    yolo_params: dict[str, Any],
    multimodal_params: dict[str, Any],
) -> tuple[list[Path], dict[str, Any], dict[int, str]]:
    split = str(evaluate_params.get("split", "val"))
    source_ref = request["inputs"].get("source") or yolo_params.pop("source", None) or multimodal_params.get("source")
    data_ref = request["inputs"].get("data") or evaluate_params.get("data") or yolo_params.pop("data", None)
    if source_ref not in (None, ""):
        images = expand_image_reference(source_ref)
        dataset_info = {"source": json_safe(source_ref), "split": None, "root": None, "names_count": 0}
        names: dict[int, str] = {}
    else:
        images, dataset_info, names = collect_dataset_images(data_ref, split)
    images = dedupe_images(images)
    limit_raw = evaluate_params.get("limit", evaluate_params.get("max_images", 5))
    limit = int(limit_raw) if limit_raw not in (None, "") else 5
    limit_value = None if limit <= 0 else limit
    seed_raw = evaluate_params.get("seed")
    seed = int(seed_raw) if seed_raw not in (None, "") else None
    sample = select_image_sample(
        images,
        limit=limit_value,
        offset=int(evaluate_params.get("offset", 0)),
        stride=max(1, int(evaluate_params.get("stride", 1))),
        shuffle=parse_bool(evaluate_params.get("shuffle"), False),
        seed=seed,
    )
    dataset_info["images_total"] = len(images)
    dataset_info["sample_count"] = len(sample)
    dataset_info["sample_limit"] = limit_value
    dataset_info["sample_offset"] = int(evaluate_params.get("offset", 0))
    dataset_info["sample_stride"] = max(1, int(evaluate_params.get("stride", 1)))
    if not sample:
        raise ValueError("No local images were found for multimodal evaluation.")
    return sample, dataset_info, names
def label_path_for_image(image_path: Path) -> Path:
    parts = image_path.resolve().parts
    if "images" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")
def read_ground_truth_summary(image_path: Path, names: dict[int, str], max_objects: int = 30) -> dict[str, Any]:
    label_path = label_path_for_image(image_path)
    summary: dict[str, Any] = {"path": str(label_path), "exists": label_path.exists(), "objects": 0, "labels": [], "label_counts": {}}
    if not label_path.exists():
        return summary
    labels: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(raw_line)
        if parsed is None:
            continue
        class_id, xywhn, segment = parsed
        label = names.get(class_id, str(class_id))
        counts[label] = counts.get(label, 0) + 1
        item = {"class_id": class_id, "label": label}
        if len(segment) >= 6:
            item["segment_points"] = len(segment) // 2
        else:
            item["xywhn"] = [round(float(value), 6) for value in xywhn[:4]]
        labels.append(item)
    summary["objects"] = len(labels)
    summary["labels"] = labels[:max_objects]
    summary["label_counts"] = counts
    if len(labels) > max_objects:
        summary["truncated"] = len(labels) - max_objects
    return summary
def parse_label_line(raw_line: str) -> tuple[int, list[float], list[float]] | None:
    parts = raw_line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except Exception:
        return None
    return class_id, coords[:4], coords[4:]
