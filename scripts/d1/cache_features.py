#!/usr/bin/env python3
"""Build either dataset's D1 cache from an explicit image list; verify, compare or convert it."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.d1.artifacts import encoded, immutable, write_json
from scripts.d1.prepare_coco import verify_model
from ultralytics.data.augment import LetterBox
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation.cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_TARGET_SHARD_BYTES,
    FeatureCacheReader,
    FeatureCacheWriter,
    canonical_json_bytes,
    compare_feature_caches,
    sha256_bytes,
    sha256_file,
    verify_feature_cache,
)

DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
OUTPUT_LAYERS = (4, 8, 12)
FEATURE_NAMES = ("block4", "block8", "block12")
EXPECTED_SHAPE = (384, 40, 40)
SPLITS = ("train2017", "val2017", "visdrone-train", "visdrone-val")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(repo_root):
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()


def cache_contract(repo_root):
    manifests = repo_root / "experiments/d1/manifests"
    experiment = load_json(manifests / "experiment-contract.json")
    teacher = load_json(manifests / "dinov3-vits16.json")
    if experiment["cache"]["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("Experiment contract and cache implementation schema versions differ")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_id": teacher["model_id"],
        "teacher_weights_sha256": teacher["files"]["model.safetensors"]["sha256"],
        "preprocessing_sha256": sha256_bytes(canonical_json_bytes(experiment["input"])),
        "output_layers": list(OUTPUT_LAYERS),
        "feature_names": list(FEATURE_NAMES),
        "dtype": experiment["cache"]["dtype"],
        "expected_shape": list(EXPECTED_SHAPE),
    }


def split_paths(path, split, limit=None):
    """Lists use root-relative images/SPLIT/ID.jpg paths for both datasets."""
    if split not in SPLITS or limit is not None and limit <= 0:
        raise ValueError("Unsupported split or nonpositive limit")
    entries = path.read_text(encoding="utf-8").splitlines()
    pattern = rf"images/{re.escape(split)}/[A-Za-z0-9][A-Za-z0-9_-]*\.jpg"
    if not entries or entries != sorted(set(entries)) or any(not re.fullmatch(pattern, p) for p in entries):
        raise ValueError("Image list must be nonempty, sorted, unique and use portable paths for the selected split")
    selected = entries if limit is None else entries[:limit]
    return selected, sha256_bytes(("\n".join(selected) + "\n").encode())


def make_letterbox():
    return LetterBox(
        new_shape=(640, 640),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        stride=32,
        padding_value=114,
        interpolation=cv2.INTER_LINEAR,
    )


def load_image(path, letterbox):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    image = letterbox(image=image)
    if image.shape != (640, 640, 3):
        raise ValueError(f"Unexpected letterbox shape: {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float().div_(255.0)


def normalize_device(value):
    return f"cuda:{value}" if value.isdigit() else value


def sample_inventory(data_root, split, paths):
    result = []
    for relative in paths:
        image = data_root / relative
        if not image.resolve().is_relative_to(data_root.resolve()) or not image.is_file():
            raise ValueError(f"Missing image or path outside data root: {relative}")
        result.append(
            {
                "sample_id": f"{split}/{image.stem}",
                "split": split,
                "image_path": relative,
                "image_sha256": sha256_file(image),
            }
        )
    return result


def benchmark_reader(cache_dir, sample_ids):
    reader = FeatureCacheReader(cache_dir)
    start, tensor_bytes = time.perf_counter(), 0
    try:
        for sid in sample_ids:
            tensor_bytes += sum(t.numel() * t.element_size() for t in reader.get(sid).values())
    finally:
        reader.close()
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "tensor_bytes": tensor_bytes,
        "mib_per_second": tensor_bytes / 1024**2 / elapsed if elapsed else None,
    }


def build(args):
    """One process owns each output directory; orchestration belongs outside this reusable entry."""
    import fcntl

    if args.batch_size <= 0 or args.target_shard_bytes <= 0:
        raise ValueError("Batch and shard size must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Cache extraction is single-process; do not launch this entry with torchrun")
    root = args.cache_dir.resolve()
    if root.is_relative_to(args.repo_root.resolve()) or root in (args.data_root.resolve(), args.weights_dir.resolve()):
        raise ValueError("Cache output must be outside the source repo and separate from input roots")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list(root.glob("*.part")):
            raise ValueError("Uncommitted .part files remain; inspect them before retrying")
        return _build(args, root)


def _build(args, root):
    data_root, weights = args.data_root.resolve(), args.weights_dir.resolve()
    contract = cache_contract(args.repo_root)
    paths, paths_sha = split_paths(args.samples_file, args.split, args.limit)
    samples = sample_inventory(data_root, args.split, paths)
    verify_model(weights, load_model=False)
    if sha256_file(weights / "model.safetensors") != contract["teacher_weights_sha256"]:
        raise ValueError("Teacher weights do not match the cache contract")
    device = normalize_device(args.device)
    identity = {
        "schema_version": "d1-list-extraction-v1",
        "extractor_sha256": sha256_file(Path(__file__)),
        "teacher_code_sha256": sha256_file(args.repo_root / "ultralytics/nn/foundation/teachers/dinov3.py"),
        "contract": contract,
        "split": args.split,
        "paths_sha256": paths_sha,
        "images_sha256": sha256_bytes(canonical_json_bytes(samples)),
        "batch_size": args.batch_size,
        "target_shard_bytes": args.target_shard_bytes,
        "device": device,
        "seed": 0,
        "deterministic": True,
        "tf32": False,
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
    }
    if not (root / "build.json").exists() and (list(root.glob("*.safetensors")) or (root / "index.json").exists()):
        raise ValueError("Legacy cache has no batch identity; verify/read it or resume with its original extractor")
    immutable(root / "build.json", encoded(identity))
    writer = FeatureCacheWriter(root, split=args.split, contract=contract, target_shard_bytes=args.target_shard_bytes)
    verify_feature_cache(root)
    reader = FeatureCacheReader(root)
    if set(reader.records) - {s["sample_id"] for s in samples}:
        raise ValueError("Cache contains samples outside the selected list")
    pending = [not writer.is_cached(s["sample_id"], s["image_sha256"]) for s in samples]
    resumed, forwarded, peak_gpu_bytes = len(samples) - sum(pending), 0, 0
    start = time.perf_counter()
    if any(pending):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_device(torch.device(device))
            torch.cuda.reset_peak_memory_stats()
        teacher = DINOv3Teacher(
            model_id=DEFAULT_MODEL_ID,
            weights_path=weights,
            local_files_only=True,
            dtype="fp16",
            device=device,
            output_layers=OUTPUT_LAYERS,
        )
        letterbox = make_letterbox()
        for offset in range(0, len(samples), args.batch_size):
            batch = samples[offset : offset + args.batch_size]
            missing = pending[offset : offset + args.batch_size]
            if not any(missing):
                continue
            # Replay the original full batch, including its already-committed members.
            images = torch.stack([load_image(data_root / s["image_path"], letterbox) for s in batch])
            features = teacher.encode(images).dense
            if tuple(features) != FEATURE_NAMES:
                raise ValueError("Teacher feature names mismatch")
            if any(
                tuple(t.shape) != (len(batch), *contract["expected_shape"]) or not torch.isfinite(t).all()
                for t in features.values()
            ):
                raise ValueError("Teacher batch shape or finite-value check failed")
            forwarded += len(batch)
            for index, sample in enumerate(batch):
                if not missing[index]:
                    old = reader.get(sample["sample_id"])
                    if any(
                        not torch.equal(old[name], features[name][index].cpu().to(torch.float16))
                        for name in FEATURE_NAMES
                    ):
                        raise ValueError("Resumed batch does not reproduce committed FP16 features")
            for index, sample in enumerate(batch):
                if missing[index]:
                    writer.add(**sample, features={name: features[name][index] for name in FEATURE_NAMES})
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_gpu_bytes = torch.cuda.max_memory_allocated()
        del teacher
    writer.close()
    reader.close()
    elapsed = time.perf_counter() - start
    verification = verify_feature_cache(root)
    if verification["sample_count"] != len(samples) or verification["part_files"]:
        raise ValueError("Incomplete cache or unexpected temporary files")
    report = {
        "schema_version": "d1-list-build-report-v1",
        "code_commit": git_commit(args.repo_root),
        "split": args.split,
        "selected_sample_count": len(samples),
        "selected_paths_sha256": paths_sha,
        "new_sample_count": sum(pending),
        "resumed_sample_count": resumed,
        "forwarded_image_count": forwarded,
        "batch_size": args.batch_size,
        "contract": contract,
        "verification": verification,
        "metrics": {
            "extraction_seconds": elapsed,
            "peak_gpu_bytes": peak_gpu_bytes,
            "new_images_per_second": sum(pending) / elapsed if elapsed else None,
        },
        "serialization_validation": "FP16 tensor SHA256 verified after safetensors reload",
    }
    if args.benchmark_read:
        report["metrics"]["read"] = benchmark_reader(root, [s["sample_id"] for s in samples])
    return report


def verify(args):
    return {"verification": verify_feature_cache(args.cache_dir, full_tensor_hash=not args.metadata_only)}


def compare(args):
    return {"comparison": compare_feature_caches(args.cache_dir, args.other_cache_dir)}


def convert_npy(args):
    from scripts.d1.convert_npy import convert_preserving_source

    return convert_preserving_source(args.cache_dir, args.output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="single-process extraction from a sorted image list")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    for name in ("data-root", "weights-dir", "samples-file", "cache-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--target-shard-bytes", type=int, default=DEFAULT_TARGET_SHARD_BYTES)
    p.add_argument("--benchmark-read", action="store_true", help="Optional extra full read; not training throughput")
    p.set_defaults(handler=build)
    p = sub.add_parser("verify")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--metadata-only", action="store_true")
    p.set_defaults(handler=verify)
    p = sub.add_parser("compare")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--other-cache-dir", type=Path, required=True)
    p.set_defaults(handler=compare)
    p = sub.add_parser("to-npy", help="lossless conversion that preserves source shards")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(handler=convert_npy)
    for command in sub.choices.values():
        command.add_argument("--report", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = args.handler(args)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
