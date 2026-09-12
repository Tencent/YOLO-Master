#!/usr/bin/env python3
"""Verify existing D1 COCO/Teacher inputs and generate canonical lists; never download data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.d1.artifacts import encoded, immutable
from scripts.d1.artifacts import file_sha as sha256_file

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "experiments/d1/manifests"
TEACHER = json.loads((MANIFESTS / "dinov3-vits16.json").read_text())
MODEL_ID = TEACHER["model_id"]
MODEL_REVISION = TEACHER["source_revision"]
MODEL_FILES = {name: (info["bytes"], info["sha256"]) for name, info in TEACHER["files"].items()}
EXPECTED_MODEL_CONFIG = TEACHER["config"]
EXPECTED_SPLITS = {"train2017": 118287, "val2017": 5000}


def build_split_list(coco_root, split):
    paths = sorted((Path(coco_root) / "images" / split).glob("*.jpg"))
    relative = [path.relative_to(coco_root).as_posix() for path in paths]
    if len(relative) != EXPECTED_SPLITS[split] or len(relative) != len(set(relative)):
        raise ValueError(f"{split}: wrong image count or duplicate paths")
    return relative


def assert_disjoint_splits(train, val):
    overlap = {Path(p).name for p in train} & {Path(p).name for p in val}
    if overlap:
        raise ValueError(f"COCO train2017 and val2017 overlap: {sorted(overlap)[:5]}")


def write_lines(path, lines):
    data = ("\n".join(lines) + "\n").encode()
    immutable(path, data)
    return hashlib.sha256(data).hexdigest()


def validated_splits(coco_root, repo):
    expected = json.loads((repo / "experiments/d1/manifests/coco2017-splits.json").read_text())["splits"]
    lists = {split: build_split_list(coco_root, split) for split in EXPECTED_SPLITS}
    assert_disjoint_splits(lists["train2017"], lists["val2017"])
    for split, lines in lists.items():
        digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
        if len(lines) != expected[split]["count"] or digest != expected[split]["sha256"]:
            raise ValueError(f"{split}: generated paths do not match the published split contract")
    return lists


def materialize_splits(coco_root, repo, output=None):
    """Validate both canonical splits before publishing either list; retain the old Python API."""
    lists = validated_splits(coco_root, repo)
    output = output or repo / "experiments/d1/manifests"
    for split, lines in lists.items():
        write_lines(output / f"coco2017-{split}.txt", lines)
    return lists


def verify_model(teacher_root, load_model=True):
    files = {}
    for name, (size, digest) in MODEL_FILES.items():
        path = teacher_root / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"Teacher file failed size/SHA256 validation: {name}")
        files[name] = {"bytes": size, "sha256": digest}
    config = json.loads((teacher_root / "config.json").read_text())
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"Teacher architecture mismatch: {key}")
    if load_model:
        from transformers import DINOv3ViTBackbone

        model = DINOv3ViTBackbone.from_pretrained(teacher_root, local_files_only=True)
        if any(getattr(model.config, key) != value for key, value in EXPECTED_MODEL_CONFIG.items()):
            raise ValueError("Loaded Teacher architecture mismatch")
        del model
    return {"model_id": MODEL_ID, "files": files, "model_loaded": load_model}


def verify_labels(coco_root, lists, expected):
    """Check canonical label counts and membership; image/tensor checks occur during extraction."""
    counts = {}
    for split, images in lists.items():
        labels = list((coco_root / "labels" / split).glob("*.txt"))
        counts[split] = len(labels)
        if counts[split] != expected[split] or not {p.stem for p in labels} <= {Path(p).stem for p in images}:
            raise ValueError(f"{split}: label count or image membership mismatch")
        if not (coco_root / "annotations" / f"instances_{split}.json").is_file():
            raise FileNotFoundError(f"{split}: official detection annotations are required")
    return counts


def verify_contract(repo):
    contract = json.loads((repo / "experiments/d1/manifests/experiment-contract.json").read_text())
    if contract["teacher"]["model_id"] != MODEL_ID:
        raise ValueError("Contract teacher model changed")
    expected = [{"implementation_index": block - 1, "name": f"block{block}", "ordinal": block} for block in (4, 8, 12)]
    if contract["features"]["output_blocks"] != expected or contract["features"]["grid_size"] != [40, 40]:
        raise ValueError("Contract block/grid specification changed")


def verify_inputs(coco_root, teacher_root, output, *, repo=ROOT, load_model=True, archives_dir=None):
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Write verification output outside the source repository")
    verify_contract(repo)
    lists = validated_splits(coco_root, repo)
    published = json.loads((repo / "experiments/d1/manifests/coco2017-splits.json").read_text())
    labels = verify_labels(coco_root, lists, published["labels"])
    teacher = verify_model(teacher_root, load_model)
    if archives_dir is not None:
        for name, info in published["archives"].items():
            path = archives_dir / name
            if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
                raise ValueError(f"Source archive failed size/SHA256 validation: {name}")
    report = {
        "schema_version": "d1-input-verification-v1",
        "splits": published["splits"],
        "labels": labels,
        "teacher": teacher,
        "source_archives_verified": archives_dir is not None,
        "scope": "canonical image paths, label count/membership, annotation presence, Teacher size/SHA256/architecture",
    }
    for split, lines in lists.items():
        write_lines(output / f"coco2017-{split}.txt", lines)
    immutable(output / "verification.json", encoded(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--archives-dir", type=Path, help="Optionally verify the four published source archives")
    parser.add_argument("--lists-only", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true", help="Test-only: still verify file hashes")
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.repo.resolve()):
        parser.error("--output must be outside the source repository")
    if args.lists_only:
        verify_contract(args.repo)
        materialize_splits(args.coco_root, args.repo, args.output)
    else:
        if args.weights_dir is None:
            parser.error("--weights-dir is required unless --lists-only is set")
        print(
            json.dumps(
                verify_inputs(
                    args.coco_root,
                    args.weights_dir,
                    args.output,
                    repo=args.repo,
                    load_model=not args.skip_model_load,
                    archives_dir=args.archives_dir,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
