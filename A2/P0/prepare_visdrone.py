#!/usr/bin/env python3
"""Download and verify VisDrone2019-DET, then write a local absolute-path dataset YAML.

Triggers the official auto-download in ``ultralytics/cfg/datasets/VisDrone.yaml`` (via
``check_det_dataset``), which fetches train/val/test-dev from the Ultralytics GitHub mirror and
converts the raw VisDrone annotations to YOLO-format labels. Then verifies the split counts and
writes ``datasets/VisDrone.yaml`` with an absolute ``path`` for the training scripts.

The data lands in ``DATASETS_DIR`` (``D:\\yolo\\datasets`` per the user's settings.json), not the
repo's ``datasets/`` dir. This script is idempotent: if the val split already exists it skips the
download and only re-verifies + re-writes the local YAML.

Usage:
    python A2/P0/prepare_visdrone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Ensure this repo's ``ultralytics`` is imported when ``main`` does its local import (a stale
# editable install elsewhere would shadow it).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_YAML = REPO_ROOT / "datasets" / "VisDrone.yaml"

# Expected split sizes from the VisDrone2019-DET config (test-dev only; test-challenge is skipped).
EXPECTED = {"train": 6471, "val": 548, "test": 1610}


def _count_images(root: Path, split: str) -> tuple[int, int]:
    """Return (num_images, num_labels) for a split, tolerant of a missing directory."""
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    return n_img, n_lbl


def verify(root: Path) -> bool:
    """Verify split counts against expectations and print a per-split status table."""
    print(f"Verifying VisDrone2019-DET at {root}:")
    ok = True
    for split, expected in EXPECTED.items():
        n_img, n_lbl = _count_images(root, split)
        good = n_img == expected and n_lbl == expected
        ok &= good
        status = "OK" if good else "MISMATCH"
        print(f"  {split:6s} images={n_img:>5}/{expected}  labels={n_lbl:>5}/{expected}  {status}")
    return ok


def write_local_yaml(root: Path, names: dict[int, str]) -> Path:
    """Write ``datasets/VisDrone.yaml`` with an absolute path (mirrors the A2 smoke approach)."""
    LOCAL_YAML.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"path: {root}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines.extend(f"  {i}: {name}" for i, name in names.items())
    LOCAL_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LOCAL_YAML


def main() -> None:
    from ultralytics.data.utils import check_det_dataset

    print("Resolving VisDrone.yaml (triggers download + conversion if missing)...")
    data = check_det_dataset("VisDrone.yaml")  # idempotent: skips download when val already present
    root = Path(data["path"])

    if not verify(root):
        print(
            "\nMismatch detected. If a partial/stale extraction exists under the dataset dir, "
            "delete it and re-run (Ultralytics' unzip silently skips non-empty targets)."
        )
        raise SystemExit(1)

    yaml_path = write_local_yaml(root, data["names"])
    print(f"\nWrote local dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
