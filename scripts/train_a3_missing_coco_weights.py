#!/usr/bin/env python3
"""Train, resume, verify, and register the four missing A3 COCO weights.

This runner intentionally launches one family per process. Formal GPU runs can
use a quick directory gate after the full CPU preflight has already passed, so
starting a run does not repeatedly enumerate the complete COCO tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MISSING_FAMILIES = ("moa", "mot", "molora", "latent")
ALL_FAMILIES = ("moe", *MISSING_FAMILIES)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


@dataclass(frozen=True)
class FamilySpec:
    name: str
    config: Path
    registered_weight: Path
    minimum_transfer_ratio: float
    trainer_overrides: dict[str, Any]


@dataclass(frozen=True)
class TrainingContract:
    source: Path
    coco_root: Path
    moe_weight: Path
    initializer_weight: Path
    output_root: Path
    registry_root: Path
    dataset: dict[str, int]
    training: dict[str, Any]
    families: dict[str, FamilySpec]


def _expand(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name)
        if resolved:
            return resolved
        if default is not None:
            return default
        raise ValueError(f"required environment variable {name!r} is not set")

    return ENV_PATTERN.sub(replace, value)


def _resolve_path(value: str, *, repo_root: Path) -> Path:
    expanded = _expand(str(value))
    if expanded.startswith("repo://"):
        return (repo_root / expanded.removeprefix("repo://")).resolve()
    return Path(expanded).expanduser().resolve()


def load_contract(path: str | Path) -> TrainingContract:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported training contract schema: {payload.get('schema_version')!r}")

    paths = payload.get("paths") or {}
    dataset = payload.get("dataset") or {}
    training = payload.get("training") or {}
    raw_models = payload.get("models") or {}
    if set(raw_models) != set(MISSING_FAMILIES):
        raise ValueError(f"models must be exactly {MISSING_FAMILIES}, got {tuple(raw_models)}")

    registry_root = _resolve_path(paths["registry_root"], repo_root=REPO_ROOT)
    families = {}
    for name in MISSING_FAMILIES:
        row = raw_models[name] or {}
        families[name] = FamilySpec(
            name=name,
            config=_resolve_path(row["config"], repo_root=REPO_ROOT),
            registered_weight=(registry_root / row["registered_weight"]).resolve(),
            minimum_transfer_ratio=float(row["minimum_transfer_ratio"]),
            trainer_overrides=dict(row.get("trainer_overrides") or {}),
        )

    return TrainingContract(
        source=source,
        coco_root=_resolve_path(paths["coco_root"], repo_root=REPO_ROOT),
        moe_weight=_resolve_path(paths["moe_weight"], repo_root=REPO_ROOT),
        initializer_weight=_resolve_path(paths["initializer_weight"], repo_root=REPO_ROOT),
        output_root=_resolve_path(paths["output_root"], repo_root=REPO_ROOT),
        registry_root=registry_root,
        dataset={key: int(value) for key, value in dataset.items()},
        training=dict(training),
        families=families,
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _image_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir())


def _label_count(path: Path) -> int:
    return sum(item.is_file() and item.suffix.lower() == ".txt" for item in path.iterdir()) if path.is_dir() else 0


def verify_dataset(contract: TrainingContract) -> dict[str, Any]:
    roots = {
        "train_images": contract.coco_root / "images" / "train2017",
        "val_images": contract.coco_root / "images" / "val2017",
        "train_labels": contract.coco_root / "labels" / "train2017",
        "val_labels": contract.coco_root / "labels" / "val2017",
    }
    counts = {
        "train_images": _image_count(roots["train_images"]),
        "val_images": _image_count(roots["val_images"]),
        "train_labels": _label_count(roots["train_labels"]),
        "val_labels": _label_count(roots["val_labels"]),
    }
    expected = contract.dataset
    errors = []
    if counts["train_images"] != expected["expected_train_images"]:
        errors.append(f"train images: expected {expected['expected_train_images']}, got {counts['train_images']}")
    if counts["val_images"] != expected["expected_val_images"]:
        errors.append(f"val images: expected {expected['expected_val_images']}, got {counts['val_images']}")
    if counts["train_labels"] < expected["minimum_train_labels"]:
        errors.append(f"train labels: expected at least {expected['minimum_train_labels']}, got {counts['train_labels']}")
    if counts["val_labels"] < expected["minimum_val_labels"]:
        errors.append(f"val labels: expected at least {expected['minimum_val_labels']}, got {counts['val_labels']}")
    if errors:
        raise RuntimeError("COCO dataset gate failed:\n- " + "\n- ".join(errors))
    return {"root": str(contract.coco_root), "paths": {key: str(value) for key, value in roots.items()}, **counts}


def quick_verify_dataset(contract: TrainingContract) -> dict[str, Any]:
    """Check the already-preflighted COCO layout without scanning 200k files."""
    required = {
        "train_images": contract.coco_root / "images" / "train2017",
        "val_images": contract.coco_root / "images" / "val2017",
        "train_labels": contract.coco_root / "labels" / "train2017",
        "val_labels": contract.coco_root / "labels" / "val2017",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_dir()]
    if missing:
        raise RuntimeError("COCO quick gate failed; missing directories:\n- " + "\n- ".join(missing))
    return {
        "root": str(contract.coco_root),
        "mode": "quick",
        "paths": {name: str(path) for name, path in required.items()},
    }


def materialize_dataset_yaml(contract: TrainingContract) -> Path:
    canonical = REPO_ROOT / "ultralytics" / "cfg" / "datasets" / "coco.yaml"
    names = (yaml.safe_load(canonical.read_text(encoding="utf-8")) or {}).get("names")
    if not isinstance(names, dict) or len(names) != 80:
        raise RuntimeError(f"canonical COCO class metadata is invalid: {canonical}")
    target = contract.output_root / "a3_coco_local.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "path": str(contract.coco_root),
                "train": "images/train2017",
                "val": "images/val2017",
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return target


def _family_module_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _matches_family(module_name: str, family: str) -> bool:
    lowered = module_name.lower()
    if family == "moe":
        return "moe" in lowered and all(token not in lowered for token in ("moa", "mot", "molora", "latent"))
    return family in lowered


def inspect_checkpoint(path: Path, expected_family: str | None) -> dict[str, Any]:
    from ultralytics import YOLO

    if not path.is_file():
        raise FileNotFoundError(path)
    yolo = YOLO(str(path))
    model = yolo.model
    names = getattr(model, "names", None)
    nc = len(names) if isinstance(names, (dict, list, tuple)) else getattr(model, "nc", None)
    counts = _family_module_counts(model)
    matched = (
        {name: count for name, count in counts.items() if _matches_family(name, expected_family)}
        if expected_family is not None
        else {}
    )
    if int(nc or 0) != 80:
        raise RuntimeError(f"{path} is not an 80-class COCO checkpoint (nc={nc!r})")
    if expected_family is not None and not matched:
        raise RuntimeError(f"{path} does not contain {expected_family} modules")
    checkpoint = getattr(yolo, "ckpt", None)
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    epochs_requested = train_args.get("epochs") if isinstance(train_args, dict) else None
    data = train_args.get("data") if isinstance(train_args, dict) else None
    try:
        trained_epochs = int(epochs_requested)
    except (TypeError, ValueError):
        trained_epochs = 0
    if trained_epochs <= 0:
        raise RuntimeError(f"{path} has no positive training provenance (train_args.epochs={epochs_requested!r})")
    if not data or "coco" not in str(data).lower():
        raise RuntimeError(f"{path} does not record COCO training data (train_args.data={data!r})")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_mib": round(path.stat().st_size / 1024**2, 3),
        "nc": int(nc),
        "family": expected_family or "initializer",
        "matched_modules": matched,
        "epochs_requested": trained_epochs,
        "data": data,
    }


def materialize_checkpoint_yaml(checkpoint: Path, target: Path) -> Path:
    """Write the exact embedded architecture used by a checkpoint for later locking."""
    from ultralytics import YOLO

    model_yaml = getattr(YOLO(str(checkpoint)).model, "yaml", None)
    if not isinstance(model_yaml, dict) or not model_yaml:
        raise RuntimeError(f"checkpoint has no embedded model YAML: {checkpoint}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(model_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def _tensor_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    import torch

    return {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }


def _flatten_tensors(value: Any):
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)


def preflight_family(contract: TrainingContract, family: str, *, forward: bool) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    spec = contract.families[family]
    if not spec.config.is_file():
        raise FileNotFoundError(spec.config)

    if family == "molora":
        from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model

        yolo = YOLO(str(contract.initializer_weight))
        config = MoLoRAConfig.from_args(None, **spec.trainer_overrides)
        yolo.model = get_peft_molora_model(yolo.model, config)
        matched_tensors = total_tensors = len(_tensor_shapes(yolo.model))
        transfer_ratio = 1.0
    else:
        source_model = YOLO(str(contract.initializer_weight)).model
        source_shapes = _tensor_shapes(source_model)
        yolo = YOLO(str(spec.config))
        target_shapes = _tensor_shapes(yolo.model)
        matched_tensors = sum(
            name in source_shapes and source_shapes[name] == shape for name, shape in target_shapes.items()
        )
        total_tensors = len(target_shapes)
        transfer_ratio = matched_tensors / total_tensors
        if transfer_ratio < spec.minimum_transfer_ratio:
            raise RuntimeError(
                f"{family} initializer transfer ratio {transfer_ratio:.1%} is below "
                f"the required {spec.minimum_transfer_ratio:.1%}"
            )
        yolo.load(str(contract.initializer_weight))

    model = yolo.model.cpu().eval()
    counts = _family_module_counts(model)
    matched = {name: count for name, count in counts.items() if _matches_family(name, family)}
    if not matched:
        raise RuntimeError(f"preflight model for {family} contains no matching family modules")

    forward_tensors = 0
    if forward:
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64, dtype=torch.float32))
        tensors = list(_flatten_tensors(output))
        if not tensors or not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise RuntimeError(f"{family} synthetic CPU forward produced missing or non-finite outputs")
        forward_tensors = len(tensors)

    return {
        "family": family,
        "config": str(spec.config),
        "matched_modules": matched,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "initializer": str(contract.initializer_weight),
        "matched_transfer_tensors": matched_tensors,
        "total_target_tensors": total_tensors,
        "transfer_ratio": transfer_ratio,
        "minimum_transfer_ratio": spec.minimum_transfer_ratio,
        "cpu_forward": bool(forward),
        "forward_tensor_count": forward_tensors,
    }


def preflight(contract: TrainingContract, families: tuple[str, ...], *, forward: bool) -> dict[str, Any]:
    dataset = verify_dataset(contract)
    moe = inspect_checkpoint(contract.moe_weight, "moe")
    initializer = inspect_checkpoint(contract.initializer_weight, None)
    moe_config = materialize_checkpoint_yaml(
        contract.moe_weight, contract.output_root / "moe_checkpoint_architecture.yaml"
    )
    architectures = [preflight_family(contract, family, forward=forward) for family in families]
    return {
        "status": "passed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "moe_checkpoint": moe,
        "moe_checkpoint_config": str(moe_config),
        "initializer_checkpoint": initializer,
        "architectures": architectures,
    }


def _run_dir(contract: TrainingContract, family: str) -> Path:
    return contract.output_root / "runs" / family


def _resumable(path: Path) -> bool:
    if not path.is_file():
        return False
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return (
        isinstance(checkpoint, dict)
        and int(checkpoint.get("epoch", -1)) >= 0
        and checkpoint.get("optimizer") is not None
    )


def _register(source: Path, target: Path, *, force: bool) -> dict[str, Any]:
    source_digest = sha256_file(source)
    if target.is_file():
        target_digest = sha256_file(target)
        if target_digest == source_digest:
            return {"path": str(target), "sha256": target_digest, "status": "already_registered"}
        if not force:
            raise FileExistsError(
                f"refusing to replace a different registered checkpoint: {target}; "
                "inspect it first or pass --force-register"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != source_digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint copy verification failed: {source} -> {target}")
    os.replace(temporary, target)
    return {"path": str(target), "sha256": source_digest, "status": "registered"}


def _training_kwargs(contract: TrainingContract, family: str, args: argparse.Namespace, dataset_yaml: Path):
    values = dict(contract.training)
    for key in ("epochs", "batch", "workers", "device", "fraction"):
        override = getattr(args, key)
        if override is not None:
            values[key] = override
    values.update(
        {
            "data": str(dataset_yaml),
            "project": str(contract.output_root / "runs"),
            "name": family,
            "exist_ok": True,
            "pretrained": True,
            "val": True,
            "save": True,
            "patience": int(values["epochs"]) + 1,
        }
    )
    values.update(contract.families[family].trainer_overrides)
    return values


def train_family(contract: TrainingContract, family: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires a CUDA GPU; use --mode preflight on CPU")
    requested_device = str(args.device or contract.training.get("device", "cuda:0"))
    if not requested_device.startswith("cuda") and not requested_device.isdigit():
        raise ValueError(f"formal training device must be CUDA, got {requested_device!r}")

    if args.skip_dataset_scan:
        quick_verify_dataset(contract)
    else:
        verify_dataset(contract)
    inspect_checkpoint(contract.initializer_weight, None)
    dataset_yaml = materialize_dataset_yaml(contract)
    run_dir = _run_dir(contract, family)
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"

    if args.resume != "never" and _resumable(last):
        print(f"[{family}] resuming interrupted run: {last}", flush=True)
        model = YOLO(str(last))
        model.train(resume=str(last), device=requested_device)
    elif args.resume == "required":
        raise FileNotFoundError(f"no resumable checkpoint for {family}: {last}")
    elif best.is_file():
        evidence = inspect_checkpoint(best, family)
        registration = _register(best, contract.families[family].registered_weight, force=args.force_register)
        return {"status": "completed_existing", "checkpoint": evidence, "registration": registration}
    else:
        if last.exists() and args.resume == "never":
            raise FileExistsError(f"run directory already contains {last}; use --resume auto or choose a clean output root")
        spec = contract.families[family]
        if family == "molora":
            model = YOLO(str(contract.initializer_weight))
        else:
            model = YOLO(str(spec.config))
            model.load(str(contract.initializer_weight))
        kwargs = _training_kwargs(contract, family, args, dataset_yaml)
        print(f"[{family}] starting one-GPU training with: {json.dumps(kwargs, ensure_ascii=False)}", flush=True)
        model.train(**kwargs)

    candidate = best if best.is_file() else last
    evidence = inspect_checkpoint(candidate, family)
    registration = _register(candidate, contract.families[family].registered_weight, force=args.force_register)
    result = {
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "training_contract": str(contract.source),
        "training_contract_sha256": sha256_file(contract.source),
        "initializer_checkpoint_sha256": sha256_file(contract.initializer_weight),
        "checkpoint": evidence,
        "registration": registration,
    }
    evidence_path = run_dir / "a3_training_evidence.json"
    evidence_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{family}] registered: {registration['path']}", flush=True)
    return result


def verify_registry(contract: TrainingContract) -> dict[str, Any]:
    rows = {"moe": inspect_checkpoint(contract.moe_weight, "moe")}
    missing = []
    for family in MISSING_FAMILIES:
        path = contract.families[family].registered_weight
        if not path.is_file():
            missing.append(str(path))
            continue
        rows[family] = inspect_checkpoint(path, family)
    return {"status": "passed" if not missing else "incomplete", "models": rows, "missing": missing}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "a3_missing_coco_weights.yaml",
    )
    parser.add_argument("--mode", choices=("preflight", "train", "verify"), required=True)
    parser.add_argument("--family", choices=(*MISSING_FAMILIES, "all"), default="all")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--device")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--skip-forward", action="store_true", help="skip synthetic CPU forward in preflight")
    parser.add_argument(
        "--skip-dataset-scan",
        action="store_true",
        help="use a quick directory gate; only after the full CPU preflight has passed",
    )
    parser.add_argument("--force-register", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and args.family == "all":
        parser.error("training requires exactly one --family; launch families sequentially")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch is not None and args.batch == 0:
        parser.error("--batch cannot be zero")
    if args.fraction is not None and not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    return args


def main() -> int:
    args = parse_args()
    contract = load_contract(args.config)
    selected = MISSING_FAMILIES if args.family == "all" else (args.family,)
    if args.mode == "preflight":
        result = preflight(contract, selected, forward=not args.skip_forward)
    elif args.mode == "train":
        result = train_family(contract, selected[0], args)
    else:
        result = verify_registry(contract)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"passed", "completed", "completed_existing"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
