#!/usr/bin/env python3
"""Download Hugging Face datasets for YOLO-Master experiments.

Default target is a YOLO-formatted VisDrone mirror that can be used for the
MoE/MoA/MoT ablation experiments. The script is adapted from the shared
`/jpfs/huangyidan3/DPO/dataset/download_hf_dataset.py` downloader, but all
paths and repo IDs are CLI arguments instead of hard-coded values.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="banu4prasad/VisDrone-Dataset", help="Hugging Face dataset repo ID.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("/jpfs/huangyidan3/datasets/VisDrone-HF"),
        help="Directory where the dataset snapshot will be stored.",
    )
    parser.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--allow-patterns", nargs="*", help="Optional file patterns to download.")
    parser.add_argument("--ignore-patterns", nargs="*", help="Optional file patterns to skip.")
    parser.add_argument("--no-force", action="store_true", help="Do not force re-download existing files.")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent HF downloads.")
    parser.add_argument("--retries", type=int, default=5, help="Retry attempts for transient mirror failures.")
    parser.add_argument("--no-validate", action="store_true", help="Skip YOLO VisDrone train/val completeness checks.")
    return parser.parse_args()


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def validate_visdrone_snapshot(root: Path) -> None:
    checks = {
        "train images": (root / "VisDrone2019-DET-train" / "images", 1000),
        "train labels": (root / "VisDrone2019-DET-train" / "labels", 1000),
        "val images": (root / "VisDrone2019-DET-val" / "images", 100),
        "val labels": (root / "VisDrone2019-DET-val" / "labels", 100),
    }
    errors = []
    for name, (path, minimum) in checks.items():
        found = count_files(path)
        if found < minimum:
            errors.append(f"{name}: found {found}, expected >= {minimum} at {path}")
    if errors:
        raise RuntimeError("HF snapshot is incomplete for YOLO training:\n  " + "\n  ".join(errors))


def main() -> int:
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    print(f"准备从 Hugging Face Hub 下载数据集: {args.repo_id}")
    print(f"将要保存到本地目录: {args.local_dir}")
    print(f"endpoint: {args.endpoint}")

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=str(args.local_dir),
                endpoint=args.endpoint,
                force_download=not args.no_force,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                max_workers=args.max_workers,
            )
            if not args.no_validate:
                validate_visdrone_snapshot(args.local_dir)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"\n下载失败 attempt={attempt}/{args.retries}: {exc}")
            if attempt < args.retries:
                time.sleep(min(60, 5 * attempt))
    if last_error is not None:
        raise last_error

    print("\n数据集下载完成。目录内容:")
    for path in sorted(args.local_dir.iterdir()):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
