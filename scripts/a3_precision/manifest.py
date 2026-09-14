"""Configuration loading and immutable asset locking for precision experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from .common import collect_images, expand_env, file_set_digest, json_dump, sha256_file


FIVE_FAMILIES = ("moe", "moa", "mot", "molora", "latent")


@dataclass(frozen=True)
class ModelSpec:
    """One locked routed-model family."""

    family: str
    checkpoint: Path
    config: Path
    adapter: Path | None = None
    enabled: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    """Calibration and validation dataset inputs."""

    data_yaml: Path
    calibration_images: Path
    validation_images: Path
    validation_labels: Path
    calibration_samples: int
    validation_samples: int


@dataclass(frozen=True)
class HarnessConfig:
    """Resolved experiment config plus its source provenance."""

    source: Path
    raw: dict[str, Any]
    output_dir: Path
    models: tuple[ModelSpec, ...]
    dataset: DatasetSpec


def _resolve_path(value: Any, *, repo_root: Path, strict_env: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a non-empty path string, got {value!r}")
    expanded = expand_env(value, strict=strict_env)
    if expanded.startswith("repo://"):
        return (repo_root / expanded[len("repo://") :]).resolve()
    path = Path(expanded).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path, *, strict_env: bool = True) -> HarnessConfig:
    """Load YAML, expand environment paths, and enforce the five-family contract."""
    source = Path(path).expanduser().resolve()
    # Expand the complete YAML rather than only paths: device strings and
    # sample counts are cloud-profile inputs too. Expansion is textual but
    # does not invoke a shell.
    raw = yaml.safe_load(expand_env(source.read_text(encoding="utf-8"), strict=strict_env))
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("precision harness config must be a schema_version=1 mapping")
    repo_root = Path(__file__).resolve().parents[2]
    experiment = raw.get("experiment") or {}
    dataset_raw = raw.get("dataset") or {}
    model_rows = raw.get("models")
    if not isinstance(model_rows, list):
        raise ValueError("config.models must be a list")
    models = []
    for row in model_rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid model row: {row!r}")
        family = str(row.get("family", "")).lower()
        adapter = row.get("adapter")
        models.append(
            ModelSpec(
                family=family,
                checkpoint=_resolve_path(row.get("checkpoint"), repo_root=repo_root, strict_env=strict_env),
                config=_resolve_path(row.get("config"), repo_root=repo_root, strict_env=strict_env),
                adapter=_resolve_path(adapter, repo_root=repo_root, strict_env=strict_env) if adapter else None,
                enabled=bool(row.get("enabled", True)),
            )
        )
    enabled_families = tuple(model.family for model in models if model.enabled)
    if len(enabled_families) != len(set(enabled_families)):
        raise ValueError(f"duplicate enabled model families: {enabled_families}")
    if set(enabled_families) != set(FIVE_FAMILIES):
        raise ValueError(f"enabled model families must be exactly {FIVE_FAMILIES}, got {enabled_families}")
    dataset = DatasetSpec(
        data_yaml=_resolve_path(dataset_raw.get("data_yaml"), repo_root=repo_root, strict_env=strict_env),
        calibration_images=_resolve_path(
            dataset_raw.get("calibration_images"), repo_root=repo_root, strict_env=strict_env
        ),
        validation_images=_resolve_path(
            dataset_raw.get("validation_images"), repo_root=repo_root, strict_env=strict_env
        ),
        validation_labels=_resolve_path(
            dataset_raw.get("validation_labels"), repo_root=repo_root, strict_env=strict_env
        ),
        calibration_samples=int(dataset_raw.get("calibration_samples", 300)),
        validation_samples=int(dataset_raw.get("validation_samples", 548)),
    )
    output_dir = _resolve_path(
        experiment.get("output_dir", "runs/a3_five_family_precision"),
        repo_root=repo_root,
        strict_env=strict_env,
    )
    return HarnessConfig(source=source, raw=raw, output_dir=output_dir, models=tuple(models), dataset=dataset)


def _git_state(repo_root: Path) -> dict[str, Any]:
    def invoke(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, timeout=15
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = invoke("status", "--porcelain")
    return {
        "commit": invoke("rev-parse", "HEAD"),
        "branch": invoke("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _runtime_contract() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "onnx", "onnxruntime", "onnxruntime-gpu", "opencv-python", "torch", "ultralytics"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    contract: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        contract["torch_runtime"] = {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "cuda_available": cuda_available,
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError:
        contract["torch_runtime"] = None
    try:
        import onnxruntime as ort

        contract["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        contract["onnxruntime_providers"] = None
    return contract


def _hash_path(path: Path, *, ignored_suffixes: tuple[str, ...] = ()) -> tuple[str, int, str]:
    if path.is_file():
        return sha256_file(path), path.stat().st_size, "file"
    digest = hashlib.sha256()
    size = 0
    for item in sorted(
        child for child in path.rglob("*") if child.is_file() and child.suffix.lower() not in ignored_suffixes
    ):
        relative = item.relative_to(path).as_posix()
        item_hash = sha256_file(item)
        item_size = item.stat().st_size
        digest.update(f"{relative}\t{item_size}\t{item_hash}\n".encode())
        size += item_size
    return digest.hexdigest(), size, "directory"


def _asset(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    ignored_suffixes = (".cache",) if role == "validation_labels" else ()
    digest, size, kind = _hash_path(path, ignored_suffixes=ignored_suffixes)
    return {
        "role": role,
        "path": str(path),
        "kind": kind,
        "size_bytes": size,
        "sha256": digest,
        "ignored_suffixes": list(ignored_suffixes),
    }


def _resolved_contract(config: HarnessConfig) -> dict[str, Any]:
    """Return the environment-expanded settings whose identity must not drift."""
    return {
        "raw": config.raw,
        "output_dir": str(config.output_dir),
        "models": [
            {
                "family": spec.family,
                "checkpoint": str(spec.checkpoint),
                "config": str(spec.config),
                "adapter": str(spec.adapter) if spec.adapter is not None else None,
                "enabled": spec.enabled,
            }
            for spec in config.models
        ],
        "dataset": {
            "data_yaml": str(config.dataset.data_yaml),
            "calibration_images": str(config.dataset.calibration_images),
            "validation_images": str(config.dataset.validation_images),
            "validation_labels": str(config.dataset.validation_labels),
            "calibration_samples": config.dataset.calibration_samples,
            "validation_samples": config.dataset.validation_samples,
        },
    }


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _model_contract(spec: ModelSpec) -> dict[str, Any]:
    """Load a checkpoint and prove that its live architecture matches its declared family."""
    from ultralytics import YOLO

    from .routing import routed_leaf_modules

    yolo = YOLO(str(spec.checkpoint))
    if spec.adapter is not None and not yolo.load_adapters(spec.adapter):
        raise RuntimeError(f"failed to load {spec.family} adapter from {spec.adapter}")
    model = yolo.model
    module_counts = Counter(type(module).__name__ for module in model.modules())
    family_patterns = {
        "moe": re.compile(r"(?:^|_)moe|moe(?:$|_)", re.IGNORECASE),
        "moa": re.compile(r"moa", re.IGNORECASE),
        "mot": re.compile(r"mot", re.IGNORECASE),
        "molora": re.compile(r"molora", re.IGNORECASE),
        "latent": re.compile(r"latent", re.IGNORECASE),
    }
    matched_types = {
        name: count for name, count in sorted(module_counts.items()) if family_patterns[spec.family].search(name)
    }
    routed = routed_leaf_modules(model, family=spec.family)
    if not matched_types or not routed:
        raise RuntimeError(
            f"{spec.family} checkpoint family contract failed: "
            f"matched_types={matched_types}, routed_layers={list(routed)}"
        )

    config_payload = yaml.safe_load(spec.config.read_text(encoding="utf-8")) or {}
    configured_nc = config_payload.get("nc") if isinstance(config_payload, dict) else None
    names = getattr(model, "names", None)
    checkpoint_nc = len(names) if isinstance(names, (dict, list, tuple)) else getattr(model, "nc", None)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_architecture = getattr(model, "yaml", {})
    checkpoint = getattr(yolo, "ckpt", None)
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    return {
        "declared_family": spec.family,
        "matched_module_types": matched_types,
        "routed_layer_count": len(routed),
        "routed_layers": sorted(routed),
        "parameter_count": int(parameters),
        "checkpoint_architecture_sha256": _json_digest(checkpoint_architecture),
        "checkpoint_training": {
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "epochs_requested": train_args.get("epochs") if isinstance(train_args, dict) else None,
            "data": train_args.get("data") if isinstance(train_args, dict) else None,
        },
        "config_nc": int(configured_nc) if configured_nc is not None else None,
        "checkpoint_nc": int(checkpoint_nc) if checkpoint_nc is not None else None,
    }


def build_lock(config: HarnessConfig) -> dict[str, Any]:
    """Build a strict lock with hashes for every model, config, and input list."""
    repo_root = Path(__file__).resolve().parents[2]
    inspect_checkpoints = bool(config.raw.get("locking", {}).get("inspect_checkpoints", False))
    data_payload = yaml.safe_load(config.dataset.data_yaml.read_text(encoding="utf-8")) or {}
    dataset_names = data_payload.get("names") if isinstance(data_payload, dict) else None
    dataset_nc = (
        len(dataset_names)
        if isinstance(dataset_names, (dict, list, tuple))
        else data_payload.get("nc")
        if isinstance(data_payload, dict)
        else None
    )
    model_locks = []
    for spec in config.models:
        assets = [_asset(spec.config, "model_config"), _asset(spec.checkpoint, "checkpoint")]
        if spec.adapter is not None:
            assets.append(_asset(spec.adapter, "adapter"))
        contract = _model_contract(spec) if inspect_checkpoints and spec.enabled else None
        if (
            contract is not None
            and dataset_nc is not None
            and contract["checkpoint_nc"] is not None
            and int(contract["checkpoint_nc"]) != int(dataset_nc)
        ):
            raise RuntimeError(
                f"{spec.family} checkpoint has {contract['checkpoint_nc']} classes, "
                f"but the locked dataset has {dataset_nc}"
            )
        model_locks.append(
            {
                "family": spec.family,
                "enabled": spec.enabled,
                "assets": assets,
                "model_contract": contract,
            }
        )
    calibration = collect_images(
        config.dataset.calibration_images,
        limit=config.dataset.calibration_samples,
        seed=int(config.raw.get("experiment", {}).get("seed", 42)),
    )
    validation = collect_images(config.dataset.validation_images, limit=config.dataset.validation_samples)
    if not calibration or not validation:
        raise ValueError("calibration and validation image sets must both be non-empty")
    if config.dataset.calibration_images.resolve() == config.dataset.validation_images.resolve():
        raise ValueError("calibration_images and validation_images must be different sources to prevent leakage")
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _asset(config.source, "harness_config"),
        "resolved_contract": {
            "sha256": _json_digest(_resolved_contract(config)),
            "value": _resolved_contract(config),
        },
        "repository": {"root": str(repo_root), **_git_state(repo_root)},
        "environment": {
            "python": sys.version,
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "contract": _runtime_contract(),
        },
        "models": model_locks,
        "dataset": {
            "data_yaml": _asset(config.dataset.data_yaml, "data_yaml"),
            "class_count": int(dataset_nc) if dataset_nc is not None else None,
            "validation_labels": _asset(config.dataset.validation_labels, "validation_labels"),
            "calibration": {
                "source": str(config.dataset.calibration_images),
                "count": len(calibration),
                "selection_digest": file_set_digest(calibration, config.dataset.calibration_images),
                "images": [str(path) for path in calibration],
            },
            "validation": {
                "source": str(config.dataset.validation_images),
                "labels": str(config.dataset.validation_labels),
                "count": len(validation),
                "selection_digest": file_set_digest(validation, config.dataset.validation_images),
                "images": [str(path) for path in validation],
            },
        },
    }
    return lock


def write_lock(config: HarnessConfig, path: str | Path | None = None) -> Path:
    """Write a new immutable-input lock."""
    return json_dump(path or config.output_dir / "experiment.lock.json", build_lock(config))


def verify_lock(config: HarnessConfig, path: str | Path | None = None) -> dict[str, Any]:
    """Re-hash locked assets and reject any drift before expensive cloud work."""
    lock_path = Path(path or config.output_dir / "experiment.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    failures = []
    resolved_digest = _json_digest(_resolved_contract(config))
    if resolved_digest != lock.get("resolved_contract", {}).get("sha256"):
        failures.append(
            {
                "field": "resolved_contract",
                "reason": "environment_or_setting_drift",
                "expected": lock.get("resolved_contract", {}).get("sha256"),
                "actual": resolved_digest,
            }
        )
    current_runtime = _runtime_contract()
    if current_runtime != lock.get("environment", {}).get("contract"):
        failures.append(
            {
                "field": "environment.contract",
                "reason": "runtime_environment_drift",
                "expected": lock.get("environment", {}).get("contract"),
                "actual": current_runtime,
            }
        )
    locked_models = {row["family"]: {asset["role"]: asset["path"] for asset in row["assets"]} for row in lock["models"]}
    for spec in config.models:
        expected = locked_models.get(spec.family, {})
        current = {"model_config": str(spec.config), "checkpoint": str(spec.checkpoint)}
        if spec.adapter is not None:
            current["adapter"] = str(spec.adapter)
        if expected != current:
            failures.append(
                {"family": spec.family, "reason": "resolved_asset_path_drift", "expected": expected, "actual": current}
            )
    for key, actual in (
        ("data_yaml", str(config.dataset.data_yaml)),
        ("calibration_source", str(config.dataset.calibration_images)),
        ("validation_source", str(config.dataset.validation_images)),
        ("validation_labels", str(config.dataset.validation_labels)),
    ):
        expected = {
            "data_yaml": lock["dataset"]["data_yaml"]["path"],
            "calibration_source": lock["dataset"]["calibration"]["source"],
            "validation_source": lock["dataset"]["validation"]["source"],
            "validation_labels": lock["dataset"]["validation_labels"]["path"],
        }[key]
        if actual != expected:
            failures.append({"field": key, "reason": "resolved_path_drift", "expected": expected, "actual": actual})
    assets = [lock["config"], lock["dataset"]["data_yaml"], lock["dataset"]["validation_labels"]]
    for model in lock["models"]:
        assets.extend(model["assets"])
    for asset in assets:
        path_obj = Path(asset["path"])
        if not path_obj.exists():
            failures.append({"path": str(path_obj), "reason": "missing"})
            continue
        ignored_suffixes = (".cache",) if asset.get("role") == "validation_labels" else ()
        actual, _size, _kind = _hash_path(path_obj, ignored_suffixes=ignored_suffixes)
        if actual != asset["sha256"]:
            failures.append(
                {"path": str(path_obj), "reason": "sha256_mismatch", "expected": asset["sha256"], "actual": actual}
            )
    for split in ("calibration", "validation"):
        info = lock["dataset"][split]
        images = [Path(path) for path in info["images"]]
        if any(not path.is_file() for path in images):
            failures.append({"split": split, "reason": "image_missing"})
            continue
        actual = file_set_digest(images, Path(info["source"]))
        if actual != info["selection_digest"]:
            failures.append({"split": split, "reason": "selection_digest_mismatch"})
    if failures:
        raise RuntimeError(f"experiment lock verification failed: {json.dumps(failures, ensure_ascii=False)}")
    return lock


def model_lock(lock: dict[str, Any], family: str) -> dict[str, Any]:
    """Return the lock row for one family."""
    return next(row for row in lock["models"] if row["family"] == family)
