"""Create and validate reproducible seen/unseen class splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

from scripts.ov_distill.make_coco128_manifest import COCO_NAMES


def build_class_split(unseen_ratio: float = 0.25, seed: int = 0) -> dict:
    """Build a deterministic COCO-80 seen/unseen split artifact."""
    if not 0 < unseen_ratio < 1:
        raise ValueError(f"unseen_ratio must be between 0 and 1, got {unseen_ratio}")
    names = sorted(COCO_NAMES)
    rng = random.Random(seed)
    n_unseen = max(1, round(len(names) * unseen_ratio))
    unseen = sorted(rng.sample(names, n_unseen))
    unseen_set = set(unseen)
    artifact = {
        "schema_version": 1,
        "dataset": "COCO-80",
        "seed": int(seed),
        "unseen_ratio": float(unseen_ratio),
        "class_names": list(COCO_NAMES),
        "seen": [name for name in names if name not in unseen_set],
        "unseen": unseen,
    }
    return validate_class_split(artifact)


def validate_class_split(artifact: dict) -> dict:
    """Validate split completeness, disjointness and compatibility with COCO-80."""
    if artifact.get("schema_version") != 1:
        raise ValueError(f"unsupported class split schema_version: {artifact.get('schema_version')}")
    expected = set(COCO_NAMES)
    declared = artifact.get("class_names")
    seen = artifact.get("seen")
    unseen = artifact.get("unseen")
    if not isinstance(declared, list) or set(declared) != expected or len(declared) != len(expected):
        raise ValueError("class split class_names must contain each COCO-80 name exactly once")
    if not isinstance(seen, list) or not isinstance(unseen, list) or not seen or not unseen:
        raise ValueError("class split must contain non-empty seen and unseen lists")
    if len(set(seen)) != len(seen) or len(set(unseen)) != len(unseen):
        raise ValueError("class split contains duplicate names")
    if set(seen) & set(unseen):
        raise ValueError("seen and unseen class lists overlap")
    if set(seen) | set(unseen) != expected:
        raise ValueError("seen and unseen classes must partition COCO-80 exactly")
    return {
        **artifact,
        "class_names": list(declared),
        "seen": list(seen),
        "unseen": list(unseen),
    }


def load_class_split(path: Path) -> dict:
    """Load and validate a class split JSON artifact."""
    path = Path(path)
    return validate_class_split(json.loads(path.read_text(encoding="utf-8")))


def write_class_split(path: Path, unseen_ratio: float = 0.25, seed: int = 0) -> dict:
    """Write a reproducible class split artifact and return its payload."""
    artifact = build_class_split(unseen_ratio=unseen_ratio, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact
