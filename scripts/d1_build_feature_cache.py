#!/usr/bin/env python3
"""Build and validate deterministic DINOv3 feature caches for D1 admission."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from ultralytics.data.foundation_cache import (
    CACHE_SCHEMA_VERSION,
    CacheIdentity,
    atomic_write_json,
    file_sha256,
    load_letterboxed_tensor,
    preprocessing_fingerprint,
    select_images,
    validate_manifest,
)
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.foundation.preprocessing import DINOV3_IMAGE_MEAN, DINOV3_IMAGE_STD, prepare_image_tensor


PREPROCESS_VERSION = "d1-letterbox-rgb-bilinear-pad114-v1"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class DINOv3MultiLevelExtractor:
    """Frozen Transformers DINOv3 backbone exposing selected BCHW feature maps."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        layers: tuple[int, ...],
        device: str,
        dtype: str,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import DINOv3ViTBackbone
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError("Install the foundation dependencies with: pip install -e '.[foundation]'") from exc
        if dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r}; choose from {sorted(DTYPES)}")
        self.layers = tuple(layers)
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        self.model = DINOv3ViTBackbone.from_pretrained(
            model_id,
            revision=revision,
            out_indices=list(self.layers),
            local_files_only=local_files_only,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.requires_grad_(False).eval()
        config = self.model.config
        self.patch_size = int(config.patch_size)
        self.hidden_size = int(config.hidden_size)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("DINOv3 extractor has trainable parameters")

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract exactly one finite feature map for every configured layer."""
        pixels = prepare_image_tensor(
            images.to(self.device),
            patch_size=self.patch_size,
            mean=DINOV3_IMAGE_MEAN,
            std=DINOV3_IMAGE_STD,
        ).to(self.dtype)
        with torch.inference_mode():
            output = self.model(pixel_values=pixels)
        feature_maps = getattr(output, "feature_maps", None)
        if not isinstance(feature_maps, (tuple, list)) or len(feature_maps) != len(self.layers):
            count = len(feature_maps) if isinstance(feature_maps, (tuple, list)) else 0
            raise RuntimeError(f"DINOv3 returned {count} feature maps for requested layers {self.layers}")
        result = {}
        for layer, feature in zip(self.layers, feature_maps):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise RuntimeError(f"layer {layer} must be BCHW, got {getattr(feature, 'shape', None)}")
            if not torch.isfinite(feature).all():
                raise RuntimeError(f"layer {layer} contains NaN or Inf")
            result[f"layer_{layer}"] = feature
        return result


def _atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write safetensors through a temporary file and atomic rename."""
    try:
        from safetensors.torch import save_file
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("safetensors is required; install it with: python -m pip install safetensors") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_resource_report(
    *, samples: list[dict[str, Any]], peak_vram_bytes: int, read_seconds: float
) -> dict[str, Any]:
    """Summarize the resource evidence required by the admission check."""
    total_bytes = sum(int(sample["bytes"]) for sample in samples)
    extraction_seconds = sum(float(sample["extraction_seconds"]) for sample in samples)
    count = len(samples)
    return {
        "sample_count": count,
        "cache_total_bytes": total_bytes,
        "cache_bytes_per_sample": total_bytes / count if count else 0.0,
        "extraction_seconds": extraction_seconds,
        "extraction_seconds_per_sample": extraction_seconds / count if count else 0.0,
        "peak_vram_bytes": int(peak_vram_bytes),
        "sequential_read_method": "python-buffered-full-file-read-warm-cache",
        "sequential_read_seconds": float(read_seconds),
        "sequential_read_mb_s": total_bytes / (1024**2) / read_seconds if read_seconds > 0 else 0.0,
    }


def build_cache(
    *,
    image_paths: Iterable[str | Path],
    dataset_root: str | Path,
    output_dir: str | Path,
    extractor: Any,
    identity: CacheIdentity,
    limit: int,
    overwrite: bool = False,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic per-image cache and return its manifest."""
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"cache already exists: {manifest_path}; pass --overwrite to rebuild")
    selected = select_images(image_paths, root=dataset_root, limit=limit)
    if len(selected) < limit:
        raise ValueError(f"requested {limit} images, but dataset split contains only {len(selected)}")
    output.mkdir(parents=True, exist_ok=True)
    features_dir = output / "features"
    features_dir.mkdir(exist_ok=True)
    if torch.cuda.is_available() and str(getattr(extractor, "device", "")).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(getattr(extractor, "device", None))

    samples = []
    resource_samples = []
    for index, item in enumerate(selected, start=1):
        image, letterbox = load_letterboxed_tensor(item.path, imgsz=identity.imgsz)
        started = time.perf_counter()
        extracted = extractor(image.unsqueeze(0))
        tensors = {}
        shapes = {}
        for name, value in extracted.items():
            if value.shape[0] != 1:
                raise RuntimeError(f"sample {item.sample_id} feature {name} has batch {value.shape[0]}, expected 1")
            cached = value[0].detach().to(device="cpu", dtype=torch.float16).contiguous()
            if not torch.isfinite(cached).all():
                raise RuntimeError(f"sample {item.sample_id} feature {name} contains NaN or Inf")
            tensors[name] = cached
            shapes[name] = list(cached.shape)
        expected_names = {f"layer_{layer}" for layer in identity.layers}
        if set(tensors) != expected_names:
            raise RuntimeError(
                f"sample {item.sample_id} feature names mismatch: "
                f"expected {sorted(expected_names)}, got {sorted(tensors)}"
            )
        cache_file = features_dir / f"{item.sample_id}.safetensors"
        _atomic_save_safetensors(cache_file, tensors)
        elapsed = time.perf_counter() - started
        relative_cache = cache_file.relative_to(output).as_posix()
        sample_record = {
            "index": index,
            "sample_id": item.sample_id,
            "image_path": item.relative_path,
            "image_sha256": item.image_sha256,
            "cache_path": relative_cache,
            "cache_sha256": file_sha256(cache_file),
            "dtype": "float16",
            "shapes": shapes,
            "letterbox": letterbox,
            "bytes": cache_file.stat().st_size,
        }
        samples.append(sample_record)
        resource_samples.append({"bytes": sample_record["bytes"], "extraction_seconds": elapsed})
        print(f"[{index}/{limit}] {item.relative_path} -> {relative_cache}", flush=True)

    peak_vram = 0
    if torch.cuda.is_available() and str(getattr(extractor, "device", "")).startswith("cuda"):
        peak_vram = torch.cuda.max_memory_allocated(getattr(extractor, "device", None))
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "preprocess": {
            "version": PREPROCESS_VERSION,
            "color": "RGB",
            "resize": "bilinear",
            "letterbox_value": 114,
            "image_mean": list(DINOV3_IMAGE_MEAN),
            "image_std": list(DINOV3_IMAGE_STD),
        },
        "dataset_root": str(Path(dataset_root).resolve()),
        "samples": samples,
    }
    atomic_write_json(manifest_path, manifest)
    validate_manifest(manifest, identity, cache_root=output, verify_files=True)
    read_seconds = benchmark_sequential_read(output, samples)
    resource_report = build_resource_report(
        samples=resource_samples, peak_vram_bytes=peak_vram, read_seconds=read_seconds
    )
    atomic_write_json(output / "resource-report.json", resource_report)
    interface_report = {
        "cache_input": {"shape": [3, identity.imgsz, identity.imgsz], "dtype": "float32", "range": [0.0, 1.0]},
        "cached_features": samples[0]["shapes"],
        "cached_dtype": "float16",
        "planned_detector_pyramid": {
            "p3": [64, identity.imgsz // 8, identity.imgsz // 8],
            "p4": [128, identity.imgsz // 16, identity.imgsz // 16],
            "p5": [256, identity.imgsz // 32, identity.imgsz // 32],
        },
    }
    atomic_write_json(output / "interface-dimensions.json", interface_report)
    command = shlex.join(argv if argv is not None else sys.argv)
    (output / "reproduce-command.txt").write_text(command + "\n", encoding="utf-8")
    return manifest


def benchmark_sequential_read(output: Path, samples: list[dict[str, Any]]) -> float:
    """Read every cache file fully and return warm-cache elapsed seconds."""
    started = time.perf_counter()
    total_bytes = 0
    for sample in samples:
        cache_file = output / sample["cache_path"]
        with cache_file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                total_bytes += len(block)
    elapsed = time.perf_counter() - started
    expected_bytes = sum(int(sample["bytes"]) for sample in samples)
    if total_bytes != expected_bytes:
        raise RuntimeError(
            f"sequential read byte mismatch: expected {expected_bytes}, got {total_bytes}"
        )
    return elapsed


def collect_image_paths(value: str | list[str]) -> list[Path]:
    """Collect supported images from resolved YOLO split paths."""
    roots = value if isinstance(value, list) else [value]
    images = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_dir():
            images.extend(path for path in root.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS)
        elif root.is_file() and root.suffix.lower() == ".txt":
            for line in root.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute():
                    path = (root.parent / path).resolve()
                if path.suffix[1:].lower() in IMG_FORMATS:
                    images.append(path)
        elif root.is_file() and root.suffix[1:].lower() in IMG_FORMATS:
            images.append(root)
        else:
            raise FileNotFoundError(f"unsupported or missing dataset split path: {root}")
    unique = sorted({path.resolve() for path in images}, key=lambda path: path.as_posix())
    if not unique:
        raise FileNotFoundError(f"no supported images found in split paths: {roots}")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="YOLO detection dataset YAML")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="Tooony133/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--revision", help="Immutable Hugging Face model revision")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 7, 11])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if args.validate_only:
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, CacheIdentity.from_dict(manifest["identity"]), cache_root=output, verify_files=True)
        print(f"D1 cache validation: PASS ({len(manifest['samples'])} samples)")
        return
    if not args.dataset:
        raise ValueError("--dataset is required when building a cache")
    if not args.revision:
        raise ValueError("--revision is required and must identify immutable model weights")
    layers = tuple(args.layers)
    preprocess = {
        "version": PREPROCESS_VERSION,
        "imgsz": args.imgsz,
        "layers": list(layers),
        "mean": list(DINOV3_IMAGE_MEAN),
        "std": list(DINOV3_IMAGE_STD),
        "patch_multiple_padding": True,
    }
    identity = CacheIdentity(args.model, args.revision, layers, args.imgsz, preprocessing_fingerprint(preprocess))
    data = check_det_dataset(args.dataset, split=args.split)
    image_paths = collect_image_paths(data[args.split])
    extractor = DINOv3MultiLevelExtractor(
        args.model,
        revision=args.revision,
        layers=layers,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    manifest = build_cache(
        image_paths=image_paths,
        dataset_root=data["path"],
        output_dir=output,
        extractor=extractor,
        identity=identity,
        limit=args.limit,
        overwrite=args.overwrite,
        argv=sys.argv,
    )
    report = json.loads((output / "resource-report.json").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"D1 cache build: PASS ({len(manifest['samples'])} samples)")


if __name__ == "__main__":
    main()
