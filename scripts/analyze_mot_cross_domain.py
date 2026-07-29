#!/usr/bin/env python3
"""Audit MoT routing across multiple domains with one immutable checkpoint.

This script addresses a common experimental confound: comparing routing traces
from models trained on different datasets. Every domain in one invocation is
evaluated with the exact same model object and checkpoint SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

EXPERT_NAMES = ("LocalConvTransformer", "WindowTransformer", "DeformableTransformer")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PERTURBATIONS = ("hflip", "brightness_low", "brightness_high")

DETAIL_FIELDS = (
    "domain",
    "perturbation",
    "image_id",
    "sample_fingerprint",
    "cluster_id",
    "layer",
    "expert_id",
    "expert",
    "top1_tokens",
    "total_tokens",
    "top1_share",
    "mean_weight",
    "mean_probability",
    "routing_entropy",
    "normalized_entropy",
    "effective_experts",
    "top1_margin",
)


@dataclass(frozen=True)
class DomainSpec:
    """Named image domain used by the routing audit."""

    name: str
    path: Path


def stable_seed(seed: int, *parts: str) -> int:
    """Derive a process-independent uint32 seed from text labels."""
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return (seed ^ int.from_bytes(digest[:4], "little")) % (2**32)


def parse_domain_specs(values: list[str], root: Path = ROOT) -> list[DomainSpec]:
    """Parse repeated ``NAME=PATH`` arguments and reject ambiguous names."""
    specs: list[DomainSpec] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid domain {value!r}; expected NAME=PATH")
        name, raw_path = (part.strip() for part in value.split("=", 1))
        if not name or not raw_path:
            raise ValueError(f"invalid domain {value!r}; NAME and PATH must both be non-empty")
        normalized = name.casefold()
        if normalized in seen:
            raise ValueError(f"duplicate domain name: {name}")
        seen.add(normalized)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        specs.append(DomainSpec(name=name, path=path.resolve()))
    if not specs:
        raise ValueError("at least one --domain NAME=PATH is required")
    return specs


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading large checkpoints into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(path: Path) -> list[Path]:
    """Discover supported images below a domain root in deterministic order."""
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"domain path does not exist or is not an image directory: {path}")
    images = [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise FileNotFoundError(f"no supported images found under domain path: {path}")
    return sorted(images)


def choose_domain_samples(
    domains: list[DomainSpec],
    max_images: int,
    seed: int,
    equalize: bool,
) -> dict[str, list[Path]]:
    """Select deterministic samples and optionally enforce equal domain sizes."""
    available = {domain.name: discover_images(domain.path) for domain in domains}
    target = min(len(paths) for paths in available.values()) if equalize else None
    if max_images > 0:
        target = min(target, max_images) if target is not None else max_images

    selected: dict[str, list[Path]] = {}
    for domain in domains:
        paths = available[domain.name]
        count = min(len(paths), target) if target is not None else len(paths)
        if count == len(paths):
            chosen = paths
        else:
            rng = np.random.default_rng(stable_seed(seed, domain.name))
            indices = np.sort(rng.choice(len(paths), size=count, replace=False))
            chosen = [paths[int(index)] for index in indices]
        selected[domain.name] = chosen
    return selected


def normalize_image_array(array: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB integer or floating scientific images to float RGB.

    医疗 TIFF 常为 16-bit 灰度图。这里使用有限像素的 0.5/99.5 分位数做稳健归一化，
    避免少量极亮像素压缩主体对比度；该步骤只用于路由审计，不修改原始数据。
    """
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[..., None]
    elif data.ndim == 3 and data.shape[-1] not in {1, 2, 3, 4} and data.shape[0] in {1, 3, 4}:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image array, got shape {data.shape}")
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    elif data.shape[-1] == 2:
        data = np.repeat(data[..., :1], 3, axis=-1)
    else:
        data = data[..., :3]

    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0

    numeric = data.astype(np.float32)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.zeros(numeric.shape, dtype=np.float32)
    low, high = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(numeric.shape, dtype=np.float32)
    numeric = np.nan_to_num(numeric, nan=low, posinf=high, neginf=low)
    return np.clip((numeric - low) / (high - low), 0.0, 1.0).astype(np.float32)


def load_image_tensor(path: Path, imgsz: int) -> torch.Tensor:
    """Load an image and apply aspect-ratio-preserving square letterboxing."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        array = normalize_image_array(np.asarray(image))
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    _, height, width = tensor.shape
    scale = min(imgsz / max(width, 1), imgsz / max(height, 1))
    new_height = max(1, min(imgsz, round(height * scale)))
    new_width = max(1, min(imgsz, round(width * scale)))
    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    canvas = torch.full((3, imgsz, imgsz), 114.0 / 255.0, dtype=torch.float32)
    top = (imgsz - new_height) // 2
    left = (imgsz - new_width) // 2
    canvas[:, top : top + new_height, left : left + new_width] = resized
    return canvas


def normalize_torch_device(device: str) -> str:
    """Normalize CLI device aliases while retaining a CPU fallback."""
    if not device:
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def load_model(model_path: Path, device: str, nc: int) -> torch.nn.Module:
    """Load a YOLO checkpoint or YAML config as a raw detection module."""
    if model_path.suffix.lower() in {".pt", ".pth"}:
        from ultralytics import YOLO

        model = YOLO(str(model_path)).model
    else:
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(str(model_path), ch=3, nc=nc, verbose=False)
    return model.to(torch.device(device)).eval()


class RouterCollector:
    """Collect per-image routing summaries from every MoT router."""

    def __init__(self, model: torch.nn.Module):
        from ultralytics.nn.modules.mot import MoTBlock

        self.records: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, MoTBlock):
                self.handles.append(module.router.register_forward_hook(self._make_hook(name)))
        if not self.handles:
            raise ValueError("the supplied model contains no MoTBlock modules")

    def _make_hook(self, layer_name: str):
        def hook(module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            weights = output[0].detach().float()
            if weights.ndim != 4:
                return
            if len(output) >= 3:
                logits = output[2].detach().float()
                temperature = max(float(module.temperature), torch.finfo(torch.float32).tiny)
                probabilities = torch.softmax(logits / temperature, dim=1)
            else:
                probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

            batch, experts = probabilities.shape[:2]
            image_ids = self.context.get("image_ids", [])
            sample_fingerprints = self.context.get("sample_fingerprints", [])
            cluster_ids = self.context.get("cluster_ids", [])
            top1 = probabilities.argmax(dim=1)
            entropy_map = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
            entropy = entropy_map.mean(dim=(1, 2))
            normalized_entropy = entropy / math.log(max(experts, 2))
            effective_experts = entropy.exp()
            sorted_probs = probabilities.topk(min(2, experts), dim=1).values
            margin = sorted_probs[:, 0] - sorted_probs[:, 1] if experts > 1 else sorted_probs[:, 0] * 0

            for batch_index in range(batch):
                image_id = image_ids[batch_index] if batch_index < len(image_ids) else f"sample_{batch_index}"
                sample_fingerprint = (
                    sample_fingerprints[batch_index]
                    if batch_index < len(sample_fingerprints)
                    else stable_fallback_fingerprint(self.context.get("domain", "unknown"), str(image_id))
                )
                cluster_id = (
                    cluster_ids[batch_index]
                    if batch_index < len(cluster_ids)
                    else sample_fingerprint
                )
                token_count = int(top1[batch_index].numel())
                counts = torch.bincount(top1[batch_index].reshape(-1).long(), minlength=experts).float()
                routed_mean = weights[batch_index].mean(dim=(1, 2))
                probability_mean = probabilities[batch_index].mean(dim=(1, 2))
                for expert_id in range(experts):
                    expert_name = EXPERT_NAMES[expert_id] if expert_id < len(EXPERT_NAMES) else f"Expert{expert_id}"
                    self.records.append(
                        {
                            "domain": self.context.get("domain", "unknown"),
                            "perturbation": self.context.get("perturbation", "base"),
                            "image_id": str(image_id),
                            "sample_fingerprint": sample_fingerprint,
                            "cluster_id": str(cluster_id),
                            "layer": layer_name,
                            "expert_id": expert_id,
                            "expert": expert_name,
                            "top1_tokens": int(counts[expert_id].item()),
                            "total_tokens": token_count,
                            "top1_share": float(counts[expert_id].item() / max(token_count, 1)),
                            "mean_weight": float(routed_mean[expert_id].item()),
                            "mean_probability": float(probability_mean[expert_id].item()),
                            "routing_entropy": float(entropy[batch_index].item()),
                            "normalized_entropy": float(normalized_entropy[batch_index].item()),
                            "effective_experts": float(effective_experts[batch_index].item()),
                            "top1_margin": float(margin[batch_index].mean().item()),
                        }
                    )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def apply_perturbation(tensor: torch.Tensor, perturbation: str) -> torch.Tensor:
    """Apply a deterministic, label-preserving routing robustness probe."""
    if perturbation == "base":
        return tensor
    if perturbation == "hflip":
        return tensor.flip(-1)
    if perturbation == "brightness_low":
        return (tensor * 0.80).clamp(0.0, 1.0)
    if perturbation == "brightness_high":
        return (tensor * 1.20).clamp(0.0, 1.0)
    raise ValueError(f"unsupported perturbation: {perturbation}")


def relative_image_id(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def stable_fallback_fingerprint(domain: str, image_id: str) -> str:
    """Return a non-content identity when sample hashing is explicitly disabled."""
    return hashlib.sha256(f"{domain}\0{image_id}".encode()).hexdigest()


def resolve_cluster_id(
    image_id: str,
    sample_fingerprint: str,
    cluster_pattern: re.Pattern[str] | None,
) -> str:
    """Map an image to an optional repeated-measures cluster such as a video sequence."""
    if cluster_pattern is None:
        return sample_fingerprint
    match = cluster_pattern.search(image_id)
    if match is None:
        return sample_fingerprint
    if "cluster" in match.groupdict():
        cluster_id = match.group("cluster")
    elif match.lastindex:
        cluster_id = match.group(1)
    else:
        cluster_id = match.group(0)
    return cluster_id or sample_fingerprint


def collect_routing(
    model: torch.nn.Module,
    domains: list[DomainSpec],
    selected: dict[str, list[Path]],
    sample_fingerprints: dict[tuple[str, str], str],
    sample_clusters: dict[tuple[str, str], str],
    device: str,
    imgsz: int,
    batch_size: int,
    perturbations: list[str],
) -> list[dict[str, Any]]:
    """Run all domains through one model instance and collect router traces."""
    collector = RouterCollector(model)
    try:
        with torch.inference_mode():
            for domain in domains:
                paths = selected[domain.name]
                for start in range(0, len(paths), batch_size):
                    batch_paths = paths[start : start + batch_size]
                    tensor = torch.stack([load_image_tensor(path, imgsz) for path in batch_paths]).to(device)
                    image_ids = [relative_image_id(path, domain.path) for path in batch_paths]
                    fingerprints = [
                        sample_fingerprints[(domain.name, image_id)]
                        for image_id in image_ids
                    ]
                    cluster_ids = [sample_clusters[(domain.name, image_id)] for image_id in image_ids]
                    for perturbation in ("base", *perturbations):
                        collector.context = {
                            "domain": domain.name,
                            "perturbation": perturbation,
                            "image_ids": image_ids,
                            "sample_fingerprints": fingerprints,
                            "cluster_ids": cluster_ids,
                        }
                        _ = model(apply_perturbation(tensor, perturbation))
    finally:
        collector.close()
    return collector.records


def mean_std(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan")
    return float(data.mean()), float(data.std(ddof=1)) if data.size > 1 else 0.0


def image_level_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average layer traces per image to keep images, not tokens, as statistical units."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in records:
        if row["perturbation"] != "base":
            continue
        key = (row["domain"], row["image_id"], row["expert"])
        grouped.setdefault(key, []).append(row)
    output = []
    metrics = (
        "top1_share",
        "mean_weight",
        "mean_probability",
        "normalized_entropy",
        "effective_experts",
        "top1_margin",
    )
    for (domain, image_id, expert), items in sorted(grouped.items()):
        sample_fingerprint = str(
            items[0].get("sample_fingerprint") or stable_fallback_fingerprint(domain, image_id)
        )
        cluster_id = str(items[0].get("cluster_id") or sample_fingerprint)
        output.append(
            {
                "domain": domain,
                "image_id": image_id,
                "sample_fingerprint": sample_fingerprint,
                "cluster_id": cluster_id,
                "expert": expert,
                **{metric: float(np.mean([item[metric] for item in items])) for metric in metrics},
            }
        )
    return output


def statistical_unit_records(
    records: list[dict[str, Any]],
    cluster_aware: bool,
) -> list[dict[str, Any]]:
    """Aggregate images within repeated-measures clusters when requested."""
    image_rows = image_level_records(records)
    if not cluster_aware:
        return [
            {
                **row,
                "unit_id": row["sample_fingerprint"],
                "n_images_in_unit": 1,
            }
            for row in image_rows
        ]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in image_rows:
        grouped.setdefault((row["domain"], row["cluster_id"], row["expert"]), []).append(row)
    metrics = (
        "top1_share",
        "mean_weight",
        "mean_probability",
        "normalized_entropy",
        "effective_experts",
        "top1_margin",
    )
    output = []
    for (domain, cluster_id, expert), items in sorted(grouped.items()):
        output.append(
            {
                "domain": domain,
                "cluster_id": cluster_id,
                "unit_id": cluster_id,
                "expert": expert,
                "n_images_in_unit": len(items),
                **{metric: float(np.mean([item[metric] for item in items])) for metric in metrics},
            }
        )
    return output


def summarize_domains(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize routing by domain, layer, and expert with image-level variance."""
    base = [row for row in records if row["perturbation"] == "base"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in base:
        grouped.setdefault((row["domain"], row["layer"], row["expert"]), []).append(row)

    rows: list[dict[str, Any]] = []
    metrics = ("top1_share", "mean_probability", "normalized_entropy", "effective_experts", "top1_margin")
    for (domain, layer, expert), items in sorted(grouped.items()):
        row: dict[str, Any] = {"domain": domain, "layer": layer, "expert": expert, "n_images": len(items)}
        for metric in metrics:
            metric_mean, metric_std = mean_std([item[metric] for item in items])
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_std"] = metric_std
        rows.append(row)

    all_layers = image_level_records(base)
    grouped_all: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_layers:
        grouped_all.setdefault((row["domain"], row["expert"]), []).append(row)
    for (domain, expert), items in sorted(grouped_all.items()):
        row = {"domain": domain, "layer": "all", "expert": expert, "n_images": len(items)}
        for metric in metrics:
            metric_mean, metric_std = mean_std([item[metric] for item in items])
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_std"] = metric_std
        rows.append(row)
    return rows


def bootstrap_mean_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a 95% CI for ``mean(B) - mean(A)``."""
    if values_a.size == 0 or values_b.size == 0 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample_a = rng.choice(values_a, size=values_a.size, replace=True)
        sample_b = rng.choice(values_b, size=values_b.size, replace=True)
        differences[index] = sample_b.mean() - sample_a.mean()
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def permutation_p_value_two_sided(
    values_a: np.ndarray,
    values_b: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    """Return a two-sided randomization-test p-value for a mean difference."""
    if values_a.size == 0 or values_b.size == 0 or permutations <= 0:
        return float("nan")
    observed = abs(float(values_b.mean() - values_a.mean()))
    combined = np.concatenate((values_a, values_b))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        shuffled = rng.permutation(combined)
        difference = abs(float(shuffled[values_a.size :].mean() - shuffled[: values_a.size].mean()))
        hits += difference >= observed
    return float((hits + 1) / (permutations + 1))


def bootstrap_paired_mean_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a paired 95% CI for ``mean(B - A)``."""
    if values_a.size == 0 or values_a.size != values_b.size or samples <= 0:
        return float("nan"), float("nan")
    differences = values_b - values_a
    rng = np.random.default_rng(seed)
    sampled = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled[index] = rng.choice(differences, size=differences.size, replace=True).mean()
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def paired_permutation_p_value_two_sided(
    values_a: np.ndarray,
    values_b: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    """Return a paired sign-flip randomization p-value."""
    if values_a.size == 0 or values_a.size != values_b.size or permutations <= 0:
        return float("nan")
    differences = values_b - values_a
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=differences.size)
        hits += abs(float((differences * signs).mean())) >= observed
    return float((hits + 1) / (permutations + 1))


def hedges_g(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return small-sample-corrected effect size for ``B - A``."""
    if values_a.size < 2 or values_b.size < 2:
        return float("nan")
    degrees = values_a.size + values_b.size - 2
    pooled_variance = (
        (values_a.size - 1) * values_a.var(ddof=1) + (values_b.size - 1) * values_b.var(ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        return 0.0 if np.isclose(values_a.mean(), values_b.mean()) else float("inf")
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
    return float(correction * (values_b.mean() - values_a.mean()) / math.sqrt(pooled_variance))


def paired_hedges_g(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return bias-corrected standardized mean change for paired units."""
    if values_a.size < 2 or values_a.size != values_b.size:
        return float("nan")
    differences = values_b - values_a
    spread = differences.std(ddof=1)
    if spread <= 0:
        return 0.0 if np.isclose(differences.mean(), 0.0) else float("inf")
    correction = 1.0 - 3.0 / (4.0 * (differences.size - 1) - 1.0)
    return float(correction * differences.mean() / spread)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Control false discovery rate across all domain/expert comparisons."""
    output = [float("nan")] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    if not finite:
        return output
    ordered = sorted(finite, key=lambda item: item[1])
    count = len(ordered)
    running = 1.0
    for rank_from_end in range(count - 1, -1, -1):
        original_index, value = ordered[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, value * count / rank)
        output[original_index] = min(running, 1.0)
    return output


def pairwise_statistics(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
    permutations: int,
    seed: int,
    alpha: float,
    cluster_aware: bool = False,
) -> list[dict[str, Any]]:
    """Compare routing with image or sequence-cluster units and overlap guards."""
    image_rows = image_level_records(records)
    unit_rows = statistical_unit_records(records, cluster_aware=cluster_aware)
    image_samples: dict[tuple[str, str, str], set[str]] = {}
    for row in image_rows:
        for metric in ("top1_share", "mean_probability"):
            image_samples.setdefault((row["domain"], row["expert"], metric), set()).add(row["sample_fingerprint"])
        if row["expert"] == EXPERT_NAMES[0]:
            image_samples.setdefault((row["domain"], "all_experts", "normalized_entropy"), set()).add(
                row["sample_fingerprint"]
            )

    observations: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in unit_rows:
        unit_id = row["unit_id"]
        for metric in ("top1_share", "mean_probability"):
            observations.setdefault((row["domain"], row["expert"], metric), {})[unit_id] = row[metric]
        if row["expert"] == EXPERT_NAMES[0]:
            observations.setdefault((row["domain"], "all_experts", "normalized_entropy"), {})[unit_id] = row[
                "normalized_entropy"
            ]

    rows: list[dict[str, Any]] = []
    domains = list(dict.fromkeys(row["domain"] for row in records if row["perturbation"] == "base"))
    targets = [(expert, metric) for expert in EXPERT_NAMES for metric in ("top1_share", "mean_probability")]
    targets.append(("all_experts", "normalized_entropy"))
    for domain_a, domain_b in itertools.combinations(domains, 2):
        for expert, metric in targets:
            units_a = observations.get((domain_a, expert, metric), {})
            units_b = observations.get((domain_b, expert, metric), {})
            fingerprints_a = image_samples.get((domain_a, expert, metric), set())
            fingerprints_b = image_samples.get((domain_b, expert, metric), set())
            shared_samples = fingerprints_a.intersection(fingerprints_b)
            shared_units = set(units_a).intersection(units_b) if cluster_aware else set()
            comparison_seed = stable_seed(seed, domain_a, domain_b, expert, metric)
            invalid_reason = ""
            if shared_samples:
                values_a = np.asarray(list(units_a.values()), dtype=np.float64)
                values_b = np.asarray(list(units_b.values()), dtype=np.float64)
                comparison_design = "invalid_shared_images"
                invalid_reason = "shared_samples_require_distinct_images"
                comparison_valid = False
            elif cluster_aware and shared_units:
                paired_ids = sorted(shared_units)
                values_a = np.asarray([units_a[unit_id] for unit_id in paired_ids], dtype=np.float64)
                values_b = np.asarray([units_b[unit_id] for unit_id in paired_ids], dtype=np.float64)
                comparison_design = "paired_sequence_clusters"
                comparison_valid = values_a.size >= 2
                if not comparison_valid:
                    invalid_reason = "at_least_two_paired_clusters_required"
            else:
                values_a = np.asarray(list(units_a.values()), dtype=np.float64)
                values_b = np.asarray(list(units_b.values()), dtype=np.float64)
                comparison_design = "independent_clusters" if cluster_aware else "independent_images"
                comparison_valid = values_a.size >= 2 and values_b.size >= 2
                if not comparison_valid:
                    invalid_reason = "at_least_two_units_per_group_required"

            if comparison_valid and comparison_design == "paired_sequence_clusters":
                ci_low, ci_high = bootstrap_paired_mean_diff_ci(
                    values_a, values_b, bootstrap_samples, comparison_seed
                )
                p_value = paired_permutation_p_value_two_sided(values_a, values_b, permutations, comparison_seed)
                effect_size = paired_hedges_g(values_a, values_b)
                effect_size_type = "paired_hedges_g"
            elif comparison_valid:
                ci_low, ci_high = bootstrap_mean_diff_ci(values_a, values_b, bootstrap_samples, comparison_seed)
                p_value = permutation_p_value_two_sided(values_a, values_b, permutations, comparison_seed)
                effect_size = hedges_g(values_a, values_b)
                effect_size_type = "independent_hedges_g"
            else:
                ci_low, ci_high = float("nan"), float("nan")
                p_value = float("nan")
                effect_size = float("nan")
                effect_size_type = ""
            rows.append(
                {
                    "domain_a": domain_a,
                    "domain_b": domain_b,
                    "expert": expert,
                    "metric": metric,
                    "analysis_unit": "sequence_cluster" if cluster_aware else "image",
                    "comparison_design": comparison_design,
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "n_images_a": len(fingerprints_a),
                    "n_images_b": len(fingerprints_b),
                    "n_clusters_a": len(units_a) if cluster_aware else 0,
                    "n_clusters_b": len(units_b) if cluster_aware else 0,
                    "n_paired_clusters": len(shared_units) if cluster_aware and not shared_samples else 0,
                    "n_shared": len(shared_samples),
                    "shared_fraction_min": (
                        len(shared_samples) / min(len(fingerprints_a), len(fingerprints_b))
                        if fingerprints_a and fingerprints_b
                        else float("nan")
                    ),
                    "comparison_valid": comparison_valid,
                    "invalid_reason": invalid_reason,
                    "mean_a": float(values_a.mean()) if values_a.size else float("nan"),
                    "mean_b": float(values_b.mean()) if values_b.size else float("nan"),
                    "mean_diff_b_minus_a": (
                        float(values_b.mean() - values_a.mean()) if values_a.size and values_b.size else float("nan")
                    ),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "hedges_g": effect_size,
                    "effect_size_type": effect_size_type,
                    "permutation_p_value_two_sided": p_value,
                }
            )

    q_values = benjamini_hochberg([row["permutation_p_value_two_sided"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["fdr_q_value"] = q_value
        ci_excludes_zero = bool(
            np.isfinite(row["bootstrap_ci95_low"])
            and np.isfinite(row["bootstrap_ci95_high"])
            and (row["bootstrap_ci95_low"] > 0.0 or row["bootstrap_ci95_high"] < 0.0)
        )
        row["ci_excludes_zero"] = ci_excludes_zero
        row["significant_after_fdr"] = bool(
            row["comparison_valid"] and np.isfinite(q_value) and q_value <= alpha and ci_excludes_zero
        )
        row["alpha"] = alpha
    return rows


def sample_overlap_summary(
    records: list[dict[str, Any]],
    cluster_aware: bool = False,
) -> list[dict[str, Any]]:
    """Summarize shared image identities so domain-comparison assumptions are auditable."""
    samples: dict[str, set[str]] = {}
    clusters: dict[str, set[str]] = {}
    for row in image_level_records(records):
        samples.setdefault(row["domain"], set()).add(row["sample_fingerprint"])
        clusters.setdefault(row["domain"], set()).add(row["cluster_id"])
    rows = []
    for domain_a, domain_b in itertools.combinations(sorted(samples), 2):
        shared = samples[domain_a].intersection(samples[domain_b])
        denominator = min(len(samples[domain_a]), len(samples[domain_b]))
        row = {
            "domain_a": domain_a,
            "domain_b": domain_b,
            "n_a": len(samples[domain_a]),
            "n_b": len(samples[domain_b]),
            "n_shared": len(shared),
            "shared_fraction_min": len(shared) / denominator if denominator else float("nan"),
            "independent_sample_test_valid": not shared,
        }
        if cluster_aware:
            shared_clusters = clusters[domain_a].intersection(clusters[domain_b])
            cluster_denominator = min(len(clusters[domain_a]), len(clusters[domain_b]))
            row.update(
                {
                    "n_clusters_a": len(clusters[domain_a]),
                    "n_clusters_b": len(clusters[domain_b]),
                    "n_shared_clusters": len(shared_clusters),
                    "shared_cluster_fraction_min": (
                        len(shared_clusters) / cluster_denominator if cluster_denominator else float("nan")
                    ),
                    "independent_sample_test_valid": not shared and not shared_clusters,
                    "paired_cluster_test_available": bool(not shared and shared_clusters),
                }
            )
        rows.append(row)
    return rows


def jensen_shannon_divergence(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return base-2 Jensen-Shannon divergence in the closed interval [0, 1]."""
    eps = np.finfo(np.float64).tiny
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    values_a = values_a / max(float(values_a.sum()), eps)
    values_b = values_b / max(float(values_b.sum()), eps)
    midpoint = 0.5 * (values_a + values_b)
    kl_a = np.sum(values_a * np.log2(np.clip(values_a / np.clip(midpoint, eps, None), eps, None)))
    kl_b = np.sum(values_b * np.log2(np.clip(values_b / np.clip(midpoint, eps, None), eps, None)))
    return float(np.clip(0.5 * (kl_a + kl_b), 0.0, 1.0))


def robustness_statistics(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure distribution drift under deterministic label-preserving perturbations."""
    vectors: dict[tuple[str, str, str, str], dict[int, float]] = {}
    entropies: dict[tuple[str, str, str, str], float] = {}
    for row in records:
        key = (row["domain"], row["image_id"], row["layer"], row["perturbation"])
        vectors.setdefault(key, {})[int(row["expert_id"])] = float(row["mean_probability"])
        entropies[key] = float(row["normalized_entropy"])

    detailed: list[dict[str, Any]] = []
    for key, base_values in sorted(vectors.items()):
        domain, image_id, layer, perturbation = key
        if perturbation != "base":
            continue
        base = np.asarray([base_values.get(index, 0.0) for index in range(len(EXPERT_NAMES))])
        for probe in PERTURBATIONS:
            probe_key = (domain, image_id, layer, probe)
            if probe_key not in vectors:
                continue
            changed_values = vectors[probe_key]
            changed = np.asarray([changed_values.get(index, 0.0) for index in range(len(EXPERT_NAMES))])
            detailed.append(
                {
                    "domain": domain,
                    "image_id": image_id,
                    "layer": layer,
                    "perturbation": probe,
                    "jensen_shannon_divergence": jensen_shannon_divergence(base, changed),
                    "l1_distance": float(np.abs(base - changed).sum()),
                    "dominant_expert_agreement": int(base.argmax() == changed.argmax()),
                    "normalized_entropy_delta": float(entropies[probe_key] - entropies[key]),
                }
            )

    image_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in detailed:
        image_groups.setdefault((row["domain"], row["image_id"], row["perturbation"]), []).append(row)
    image_rows = []
    metrics = (
        "jensen_shannon_divergence",
        "l1_distance",
        "dominant_expert_agreement",
        "normalized_entropy_delta",
    )
    for (domain, image_id, perturbation), items in sorted(image_groups.items()):
        image_rows.append(
            {
                "domain": domain,
                "image_id": image_id,
                "perturbation": perturbation,
                **{metric: float(np.mean([item[metric] for item in items])) for metric in metrics},
            }
        )

    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in image_rows:
        summary_groups.setdefault((row["domain"], row["perturbation"]), []).append(row)
    summary = []
    for (domain, perturbation), items in sorted(summary_groups.items()):
        js_values = np.asarray([item["jensen_shannon_divergence"] for item in items])
        summary.append(
            {
                "domain": domain,
                "perturbation": perturbation,
                "n_images": len(items),
                "jensen_shannon_divergence_mean": float(js_values.mean()),
                "jensen_shannon_divergence_p95": float(np.quantile(js_values, 0.95)),
                "l1_distance_mean": float(np.mean([item["l1_distance"] for item in items])),
                "dominant_expert_agreement_rate": float(np.mean([item["dominant_expert_agreement"] for item in items])),
                "normalized_entropy_delta_mean": float(np.mean([item["normalized_entropy_delta"] for item in items])),
            }
        )
    return detailed, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...] | None = None) -> None:
    """Write a stable UTF-8 CSV, including a header for empty result sets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_routing_heatmaps(summary: list[dict[str, Any]], output: Path) -> None:
    """Plot aggregate and layer-resolved probability/top-1 routing heatmaps."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    aggregate_rows = [row for row in summary if row["layer"] == "all"]
    layer_rows = [row for row in summary if row["layer"] != "all"]
    if not aggregate_rows:
        return
    frame = pd.DataFrame(aggregate_rows)
    pivot = frame.pivot(index="domain", columns="expert", values="mean_probability_mean")
    pivot = pivot.reindex(columns=list(EXPERT_NAMES))
    uniform = 1.0 / len(EXPERT_NAMES)
    deviation = pivot - uniform
    limit = max(float(np.nanmax(np.abs(deviation.to_numpy()))), 1e-6)
    figure, axis = plt.subplots(figsize=(9, max(3.0, 0.65 * len(pivot.index))))
    sns.heatmap(deviation, annot=True, fmt="+.4f", cmap="vlag", center=0.0, vmin=-limit, vmax=limit, ax=axis)
    axis.set_title("Mean routing probability deviation from uniform")
    axis.set_xlabel("Transformer expert")
    axis.set_ylabel("Image domain")
    figure.tight_layout()
    figure.savefig(output / "routing_probability_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    if not layer_rows:
        return
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame["domain_layer"] = layer_frame["domain"] + " | " + layer_frame["layer"]
    probability = layer_frame.pivot(
        index="domain_layer", columns="expert", values="mean_probability_mean"
    ).reindex(columns=list(EXPERT_NAMES))
    probability_deviation = probability - uniform
    layer_limit = max(float(np.nanmax(np.abs(probability_deviation.to_numpy()))), 1e-6)
    height = max(5.0, 0.34 * len(probability.index))

    figure, axis = plt.subplots(figsize=(10, height))
    sns.heatmap(
        probability_deviation,
        annot=True,
        fmt="+.3f",
        cmap="vlag",
        center=0.0,
        vmin=-layer_limit,
        vmax=layer_limit,
        ax=axis,
    )
    axis.set_title("Layer-resolved routing probability deviation from uniform")
    axis.set_xlabel("Transformer expert")
    axis.set_ylabel("Domain | MoT layer")
    figure.tight_layout()
    figure.savefig(output / "routing_layer_probability_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    top1 = layer_frame.pivot(index="domain_layer", columns="expert", values="top1_share_mean").reindex(
        columns=list(EXPERT_NAMES)
    )
    figure, axis = plt.subplots(figsize=(10, height))
    sns.heatmap(top1, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axis)
    axis.set_title("Layer-resolved expert top-1 activation share")
    axis.set_xlabel("Transformer expert")
    axis.set_ylabel("Domain | MoT layer")
    figure.tight_layout()
    figure.savefig(output / "routing_layer_top1_share_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_recommendations(
    path: Path,
    summary: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
) -> None:
    """Generate data-backed Chinese observations without making accuracy claims."""
    lines = [
        "# 同检查点路由审计：自动生成观察",
        "",
        "> 这些结论只描述路由行为，不等同于检测精度或临床有效性；最终建议需与 mAP、延迟共同审阅。",
        "",
        "## 场景观察",
        "",
    ]
    all_layer = [row for row in summary if row["layer"] == "all"]
    for domain in sorted({row["domain"] for row in all_layer}):
        candidates = [row for row in all_layer if row["domain"] == domain]
        if not candidates:
            continue
        dominant = max(candidates, key=lambda row: row["mean_probability_mean"])
        lines.append(
            f"- `{domain}`：平均概率最高的专家为 **{dominant['expert']}**"
            f"（{dominant['mean_probability_mean']:.4f}，n={dominant['n_images']}）。"
        )

    finite_pairwise = [
        row
        for row in pairwise
        if row["expert"] != "all_experts"
        and row.get("comparison_valid", True)
        and np.isfinite(row["mean_diff_b_minus_a"])
    ]
    stable_pairwise = [row for row in finite_pairwise if row.get("significant_after_fdr", False)]
    if stable_pairwise:
        strongest = max(stable_pairwise, key=lambda row: abs(row["hedges_g"]) if np.isfinite(row["hedges_g"]) else -1)
        lines.extend(
            [
                "",
                "## 通过复合显著性判据的最大差异",
                "",
                (
                    f"- `{strongest['domain_a']}` → `{strongest['domain_b']}` 的 "
                    f"**{strongest['expert']} / {strongest['metric']}** 差值为 "
                    f"{strongest['mean_diff_b_minus_a']:+.4f}，95% bootstrap CI "
                    f"[{strongest['bootstrap_ci95_low']:+.4f}, {strongest['bootstrap_ci95_high']:+.4f}]，"
                    f"FDR q={strongest['fdr_q_value']:.4g}，设计为 "
                    f"`{strongest['comparison_design']}`。"
                ),
            ]
        )
    elif finite_pairwise:
        lines.extend(
            [
                "",
                "## 推断结论",
                "",
                "- 没有专家差异同时满足 FDR q ≤ alpha 且 bootstrap CI 不跨 0；仅保留描述统计。",
            ]
        )

    if robustness:
        most_stable = min(robustness, key=lambda row: row["jensen_shannon_divergence_mean"])
        least_stable = max(robustness, key=lambda row: row["jensen_shannon_divergence_mean"])
        lines.extend(
            [
                "",
                "## 稳定性观察",
                "",
                (
                    f"- 最稳定组合是 `{most_stable['domain']}` / `{most_stable['perturbation']}`："
                    f"平均 JSD={most_stable['jensen_shannon_divergence_mean']:.5f}，"
                    f"主导专家一致率={most_stable['dominant_expert_agreement_rate']:.2%}。"
                ),
                (
                    f"- 漂移最大组合是 `{least_stable['domain']}` / `{least_stable['perturbation']}`："
                    f"平均 JSD={least_stable['jensen_shannon_divergence_mean']:.5f}，"
                    f"P95={least_stable['jensen_shannon_divergence_p95']:.5f}。"
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    """Return whether tracked or untracked source changes are present."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="One MoT checkpoint or YAML used for every domain.")
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named image domain. Repeat for each domain.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--nc", type=int, default=80, help="Class count used only when --model is a YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--equalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--perturbations",
        nargs="*",
        choices=PERTURBATIONS,
        default=list(PERTURBATIONS),
        help="Deterministic robustness probes. Passing the flag with no values disables all probes.",
    )
    parser.add_argument(
        "--no-perturbations",
        action="store_true",
        help="Explicitly disable robustness probes (preferred over an empty --perturbations argument).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cluster-regex",
        default=None,
        help=(
            "Optional regex applied to image_id. Its named 'cluster' group or first capture group identifies "
            "repeated-measures units such as VisDrone video sequences."
        ),
    )
    parser.add_argument("--hash-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mot_cross_domain/routing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.imgsz <= 0:
        raise SystemExit("--batch and --imgsz must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be between 0 and 1")
    if args.bootstrap_samples <= 0 or args.permutations <= 0:
        raise SystemExit("--bootstrap-samples and --permutations must be positive")
    if args.no_perturbations:
        args.perturbations = []
    try:
        cluster_pattern = re.compile(args.cluster_regex) if args.cluster_regex else None
    except re.error as error:
        raise SystemExit(f"invalid --cluster-regex: {error}") from error

    domains = parse_domain_specs(args.domain)
    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model file not found: {model_path}")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    selected = choose_domain_samples(domains, args.max_images, args.seed, args.equalize)
    sample_rows = []
    sample_fingerprints: dict[tuple[str, str], str] = {}
    sample_clusters: dict[tuple[str, str], str] = {}
    for domain in domains:
        for path in selected[domain.name]:
            image_id = relative_image_id(path, domain.path)
            content_sha256 = sha256_file(path) if args.hash_samples else ""
            fingerprint = content_sha256 or hashlib.sha256(str(path.resolve()).encode()).hexdigest()
            cluster_id = resolve_cluster_id(image_id, fingerprint, cluster_pattern)
            sample_fingerprints[(domain.name, image_id)] = fingerprint
            sample_clusters[(domain.name, image_id)] = cluster_id
            sample_rows.append(
                {
                    "domain": domain.name,
                    "image_id": image_id,
                    "cluster_id": cluster_id,
                    "size_bytes": path.stat().st_size,
                    "sha256": content_sha256,
                    "sample_fingerprint": fingerprint,
                }
            )

    device = normalize_torch_device(args.device)
    model = load_model(model_path, device=device, nc=args.nc)
    records = collect_routing(
        model,
        domains,
        selected,
        sample_fingerprints,
        sample_clusters,
        device=device,
        imgsz=args.imgsz,
        batch_size=args.batch,
        perturbations=args.perturbations,
    )
    summary = summarize_domains(records)
    pairwise = pairwise_statistics(
        records,
        args.bootstrap_samples,
        args.permutations,
        args.seed,
        args.alpha,
        cluster_aware=cluster_pattern is not None,
    )
    sample_overlap = sample_overlap_summary(records, cluster_aware=cluster_pattern is not None)
    robustness_detailed, robustness_summary = robustness_statistics(records)

    write_csv(output / "routing_detailed.csv", records, DETAIL_FIELDS)
    write_csv(output / "domain_summary.csv", summary)
    write_csv(output / "pairwise_statistics.csv", pairwise)
    write_csv(output / "sample_overlap.csv", sample_overlap)
    write_csv(output / "robustness_detailed.csv", robustness_detailed)
    write_csv(output / "robustness_summary.csv", robustness_summary)

    write_csv(output / "sample_manifest.csv", sample_rows)
    write_recommendations(output / "recommendations_zh.md", summary, pairwise, robustness_summary)
    if args.plots:
        plot_routing_heatmaps(summary, output)

    model_digest = sha256_file(model_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "command": [sys.executable, *sys.argv],
        "model": str(model_path),
        "model_sha256": model_digest,
        "same_checkpoint_for_all_domains": True,
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "equalized_sample_count": args.equalize,
        "max_images": args.max_images,
        "perturbations": args.perturbations,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "alpha": args.alpha,
        "analysis_unit": "sequence_cluster" if cluster_pattern is not None else "image",
        "cluster_regex": args.cluster_regex,
        "overlap_policy": (
            "Exact shared images invalidate inference. Shared sequence clusters use paired cluster bootstrap and "
            "sign-flip permutation tests when --cluster-regex is set."
        ),
        "preprocessing": "robust percentile normalization + RGB conversion + aspect-ratio-preserving letterbox",
        "domains": [
            {
                "name": domain.name,
                "path": str(domain.path),
                "selected_images": len(selected[domain.name]),
            }
            for domain in domains
        ],
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[routing] checkpoint sha256={model_digest}")
    print(f"[routing] collected {len(records)} expert rows from {len(domains)} domains")
    print(f"[routing] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
