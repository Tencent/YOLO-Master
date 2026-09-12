"""Offline Foundation Teacher pre-extraction.

Pre-computes frozen-teacher outputs (dense features, pooled/semantic vectors, text prototypes, and the optional
detection response channel) to per-image ``.pt`` files so heavy teachers (e.g. the 3.3GB SAM3 ViT-H) never enter
the training loop. The module stays dependency-free: image decoding is only imported inside the CLI path.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch

from .protocol import FoundationFeatures


CACHE_VERSION = 1
IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")


def _to_storage(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    tensor = tensor.detach().cpu().contiguous()
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor


def _restore(tensor: torch.Tensor, device: Any, dtype: torch.dtype | None) -> torch.Tensor:
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype or torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def save_foundation_features(
    path: str | Path,
    features: FoundationFeatures,
    *,
    response: Mapping[str, Any] | None = None,
    save_dtype: torch.dtype | None = torch.float16,
) -> Path:
    """Serialize teacher outputs for one image to ``path`` (dense/pooled/semantic stored as ``save_dtype``)."""
    if not isinstance(features, FoundationFeatures):
        raise TypeError(f"features must be FoundationFeatures, got {type(features).__name__}.")
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "dense": {name: _to_storage(feature, save_dtype) for name, feature in features.dense.items()},
        "pooled": None if features.pooled is None else _to_storage(features.pooled, save_dtype),
        "semantic": None if features.semantic is None else _to_storage(features.semantic, save_dtype),
        "metadata": dict(features.metadata),
    }
    if response is not None:
        payload["response"] = {
            name: _to_storage(value, save_dtype) if isinstance(value, torch.Tensor) else value
            for name, value in response.items()
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_foundation_features(
    path: str | Path,
    *,
    device: Any = None,
    dtype: torch.dtype | None = torch.float32,
) -> tuple[FoundationFeatures, dict[str, Any] | None]:
    """Load one cached sample; floating tensors are restored to ``dtype`` (default fp32) on ``device``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "dense" not in payload:
        raise ValueError(f"Foundation cache '{path}' does not contain a dense feature payload.")
    features = FoundationFeatures(
        dense={name: _restore(feature, device, dtype) for name, feature in payload["dense"].items()},
        pooled=None if payload.get("pooled") is None else _restore(payload["pooled"], device, dtype),
        semantic=None if payload.get("semantic") is None else _restore(payload["semantic"], device, dtype),
        metadata=dict(payload.get("metadata") or {}),
    )
    response = payload.get("response")
    if response is not None:
        response = {
            name: _restore(value, device, dtype) if isinstance(value, torch.Tensor) else value
            for name, value in response.items()
        }
    return features, response


def _load_response_only(
    path: str | Path,
    device: Any,
    dtype: torch.dtype | None,
) -> tuple[FoundationFeatures, dict[str, Any] | None]:
    """Load only the response payload from a cache file, skipping dense/pooled/semantic features."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Foundation cache '{path}' is not a mapping.")
    response = payload.get("response")
    if response is not None:
        response = {
            name: _restore(value, device, dtype) if isinstance(value, torch.Tensor) else value
            for name, value in response.items()
        }
    return _empty_foundation(), response


def _empty_foundation() -> FoundationFeatures:
    """Return a FoundationFeatures with empty dense dict and None pooled/semantic."""
    return FoundationFeatures(dense={}, pooled=None, semantic=None, metadata={})


def load_foundation_batch(
    cache_dir: str | Path,
    keys: Sequence[str],
    *,
    device: Any = None,
    dtype: torch.dtype | None = torch.float32,
    with_response: bool = False,
    response_only: bool = False,
) -> tuple[FoundationFeatures, dict[str, Any] | None]:
    """Load ``cache_dir/<key>.pt`` samples and concatenate them into one batched :class:`FoundationFeatures`.

    Returns ``(features, response)``.  ``response`` is ``None`` when any entry lacks a cached detection response or
    when ``with_response`` is ``False``; with ``with_response=True`` per-sample responses are concatenated along
    the batch dimension and validated for prompt consistency.

    When ``response_only=True``, dense/pooled/semantic features are skipped (stored as empty/None) and only the
    response payload is loaded.  This avoids loading large feature tensors from disk when only the detection
    response channel is needed (e.g. response-only distillation).
    """
    if not keys:
        raise ValueError("keys must be a non-empty sequence of cache keys.")
    cache_dir = Path(cache_dir)
    samples: list[FoundationFeatures] = []
    responses: list[dict[str, Any] | None] = []
    for key in keys:
        path = cache_dir / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Foundation cache entry '{path}' is missing; re-run offline extraction.")
        if response_only:
            features, response = _load_response_only(path, device, dtype)
        else:
            features, response = load_foundation_features(path, device=device, dtype=dtype)
        samples.append(features)
        responses.append(response)
    if response_only:
        return _empty_foundation(), _merge_responses(keys, responses) if with_response else None
    first = samples[0]
    level_names = tuple(first.dense)
    for key, sample in zip(keys[1:], samples[1:]):
        if tuple(sample.dense) != level_names:
            raise ValueError(
                f"cache entry '{key}' has levels {tuple(sample.dense)} but expected {level_names}; "
                "the cache mixes incompatible extractions."
            )
        if (sample.pooled is None) != (first.pooled is None) or (sample.semantic is None) != (first.semantic is None):
            raise ValueError(f"cache entry '{key}' has an inconsistent pooled/semantic layout.")
    features = FoundationFeatures(
        dense={name: torch.cat([sample.dense[name] for sample in samples]) for name in level_names},
        pooled=None if first.pooled is None else torch.cat([sample.pooled for sample in samples]),
        semantic=None if first.semantic is None else torch.cat([sample.semantic for sample in samples]),
        metadata=dict(first.metadata),
    )
    if not with_response:
        return features, None
    return features, _merge_responses(keys, responses)


def _normalize_prompts(prompts: Any) -> tuple[str, ...]:
    """Normalize prompts to a flat tuple of individual strings for comparison."""
    if not prompts:
        return ()
    tokens: list[str] = []
    for item in prompts:
        tokens.extend(str(item).split())
    return tuple(tokens)


def _merge_responses(keys: Sequence[str], responses: Sequence[dict[str, Any] | None]) -> dict[str, Any] | None:
    if any(response is None for response in responses):
        return None
    prompts = _normalize_prompts(responses[0].get("prompts"))
    for key, response in zip(keys[1:], responses[1:]):
        if _normalize_prompts(response.get("prompts")) != prompts:
            raise ValueError(f"cache entry '{key}' has response prompts {response.get('prompts')}; expected {prompts}.")
    merged: dict[str, Any] = {}
    for name, value in responses[0].items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == 1:
            tensors = [response[name] for response in responses]
            if value.ndim >= 3:
                max_prompts = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    if t.shape[0] < max_prompts:
                        pad_size = max_prompts - t.shape[0]
                        pad_shape = list(t.shape)
                        pad_shape[0] = pad_size
                        padding = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                        padded.append(torch.cat([t, padding], dim=0))
                    else:
                        padded.append(t)
                merged[name] = torch.cat(padded, dim=1)
            else:
                merged[name] = torch.cat(tensors, dim=1)
        else:
            merged[name] = value
    return merged


def _slice_features(features: FoundationFeatures, index: int) -> FoundationFeatures:
    return FoundationFeatures(
        dense={name: feature[index : index + 1] for name, feature in features.dense.items()},
        pooled=None if features.pooled is None else features.pooled[index : index + 1],
        semantic=None if features.semantic is None else features.semantic[index : index + 1],
        metadata=dict(features.metadata),
    )


def _filter_levels(features: FoundationFeatures, levels: tuple[str, ...]) -> FoundationFeatures:
    missing = [level for level in levels if level not in features.dense]
    if missing:
        raise ValueError(f"levels {missing} not produced by the teacher; available: {sorted(features.dense)}.")
    return FoundationFeatures(
        dense={level: features.dense[level] for level in levels},
        pooled=features.pooled,
        semantic=features.semantic,
        metadata=dict(features.metadata),
    )


def _slice_response(response: Mapping[str, Any], index: int, batch: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for name, value in response.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == batch:
            sliced[name] = value[:, index : index + 1]
        else:
            sliced[name] = value
    return sliced


def extract_foundation_cache(
    teacher: Any,
    samples: Iterable[tuple[str, torch.Tensor]],
    output_dir: str | Path,
    *,
    prompts: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
    batch_size: int = 1,
    save_dtype: torch.dtype | None = torch.float16,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pre-extract teacher outputs for ``(key, image[3,H,W])`` samples into ``output_dir/<key>.pt`` files.

    When ``prompts`` are given, text prototypes are stored once in ``text_prototypes.pt`` and, if the teacher
    exposes a detection response channel, per-image responses are cached alongside the dense features via a
    single shared backbone pass (``encode_with_response``) when available. ``levels`` restricts which dense
    pyramid levels are stored (e.g. ``["p4"]`` cuts SAM3 cache size from ~53MB to ~3MB per image).

    Returns:
        dict: summary with ``written``, ``skipped``, and ``files`` lists plus the prototype path when saved.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = None if prompts is None else tuple(str(prompt) for prompt in prompts)
    levels = None if levels is None else tuple(str(level).lower() for level in levels)
    if levels is not None and not levels:
        raise ValueError("levels must be None or a non-empty sequence of level names.")
    summary: dict[str, Any] = {"written": [], "skipped": [], "files": [], "prototypes": None}

    if prompts and callable(getattr(teacher, "encode_text", None)):
        prototypes = teacher.encode_text(prompts)
        prototype_path = output_dir / "text_prototypes.pt"
        torch.save(
            {"version": CACHE_VERSION, "prompts": prompts, "prototypes": _to_storage(prototypes, save_dtype)},
            prototype_path,
        )
        summary["prototypes"] = prototype_path

    use_response = bool(prompts) and callable(getattr(teacher, "detect", None))
    shared_pass = use_response and callable(getattr(teacher, "encode_with_response", None))

    def flush(batch: list[tuple[str, torch.Tensor]]) -> None:
        if not batch:
            return
        keys = [key for key, _ in batch]
        images = torch.stack([image for _, image in batch])
        if shared_pass:
            features, response = teacher.encode_with_response(images, prompts)
        else:
            features = teacher.encode(images)
            response = teacher.detect(images, prompts) if use_response else None
        if levels is not None:
            features = _filter_levels(features, levels)
        for index, key in enumerate(keys):
            path = output_dir / f"{key}.pt"
            save_foundation_features(
                path,
                _slice_features(features, index),
                response=None if response is None else _slice_response(response, index, len(keys)),
                save_dtype=save_dtype,
            )
            summary["written"].append(key)
            summary["files"].append(path)

    pending: list[tuple[str, torch.Tensor]] = []
    for key, image in samples:
        key = str(key)
        portable_paths = (PurePosixPath(key), PureWindowsPath(key))
        if not key or any(path.anchor or ".." in path.parts for path in portable_paths):
            raise ValueError(f"cache key {key!r} must be a relative, non-empty path fragment.")
        if not overwrite and (output_dir / f"{key}.pt").exists():
            summary["skipped"].append(key)
            continue
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"sample '{key}' must be a [3,H,W] tensor, got {getattr(image, 'shape', None)}.")
        if pending and pending[-1][1].shape != image.shape:
            flush(pending)
            pending = []
        pending.append((key, image))
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
    flush(pending)
    return summary


def iter_image_samples(source: str | Path, *, imgsz: int | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, [3,H,W] float tensor in [0,1])`` samples from an image file or directory (CLI helper)."""
    import numpy as np
    from PIL import Image

    source = Path(source)
    if source.is_file():
        files = [source]
        root = source.parent
    elif source.is_dir():
        files = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        root = source
    else:
        raise FileNotFoundError(f"source '{source}' is neither an image file nor a directory.")
    if not files:
        raise FileNotFoundError(f"source '{source}' contains no images ({', '.join(IMAGE_SUFFIXES)}).")
    for path in files:
        image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1).float().div(255.0)
        if imgsz is not None and tuple(image.shape[-2:]) != (imgsz, imgsz):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(imgsz, imgsz), mode="bilinear", align_corners=False, antialias=True
            ).squeeze(0)
        yield path.relative_to(root).with_suffix("").as_posix(), image


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI: pre-extract SAM3 teacher features/responses for a directory of images."""
    import argparse

    parser = argparse.ArgumentParser(description="Offline Foundation Teacher feature pre-extraction.")
    parser.add_argument("--teacher", choices=["sam3"], default="sam3")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--output", required=True, help="Output cache directory.")
    parser.add_argument("--checkpoint", default=None, help="Local teacher checkpoint (e.g. sam3.pt).")
    parser.add_argument("--dart-repo", default=None, help="Local DART repository root for the sam3 package.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Class prompts for prototypes + responses.")
    parser.add_argument("--levels", nargs="*", default=None, help="Dense levels to store (e.g. p4); default all.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None, help="Teacher square input resolution.")
    parser.add_argument("--imgsz", type=int, default=None, help="Pre-resize loaded images to imgsz×imgsz.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    from .teachers import SAM3Teacher

    teacher_kwargs: dict[str, Any] = {"checkpoint": args.checkpoint, "dart_repo": args.dart_repo, "device": args.device}
    if args.image_size is not None:
        teacher_kwargs["image_size"] = args.image_size
    teacher = SAM3Teacher(**teacher_kwargs)
    summary = extract_foundation_cache(
        teacher,
        iter_image_samples(args.source, imgsz=args.imgsz),
        args.output,
        prompts=args.prompts,
        levels=args.levels,
        batch_size=args.batch_size,
        save_dtype=torch.float16 if args.save_dtype == "fp16" else torch.float32,
        overwrite=args.overwrite,
    )
    print(
        f"foundation cache: {len(summary['written'])} written, {len(summary['skipped'])} skipped -> {args.output}"
        + (f" (prototypes: {summary['prototypes']})" if summary["prototypes"] else "")
    )
    return summary


if __name__ == "__main__":
    main()


__all__ = [
    "CACHE_VERSION",
    "extract_foundation_cache",
    "iter_image_samples",
    "load_foundation_batch",
    "load_foundation_features",
    "save_foundation_features",
]
