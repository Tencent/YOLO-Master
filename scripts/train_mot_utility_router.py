#!/usr/bin/env python3
"""Train one frozen-detector MoT router from a causal detection utility matrix.

The detector and Transformer experts remain frozen. Image features at the
requested MoTBlock input are extracted once, then only the router is optimized
against soft utility targets. Train/validation separation is by video sequence,
not by frame, to prevent near-duplicate leakage.

Example:
    python scripts/train_mot_utility_router.py \
      --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
      --data ultralytics/cfg/datasets/VisDrone.yaml \
      --matrix runs/mot_detection_utility/train_l14_m0_512/detection_utility_matrix.csv \
      --split train \
      --layer model.14.m.0 \
      --device 0 \
      --output runs/mot_utility_router/l14_m0
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mot_cross_domain import load_model, normalize_torch_device, write_csv
from scripts.build_mot_detection_utility import build_validation_loader, resolve_mot_layer


@dataclass(frozen=True)
class UtilityRecord:
    """One image's causal expert utility supervision."""

    image_id: str
    sequence_id: str
    target: torch.Tensor
    anchor: torch.Tensor
    forced_losses: torch.Tensor


@dataclass(frozen=True)
class FeatureCache:
    """Frozen MoT inputs and aligned utility labels."""

    image_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    features: torch.Tensor
    targets: torch.Tensor
    anchors: torch.Tensor
    forced_losses: torch.Tensor


class TargetInputProbe:
    """Capture the input feature map received by one MoTBlock."""

    def __init__(self, block: nn.Module):
        self.feature: torch.Tensor | None = None
        self.handle = block.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("MoTBlock did not receive a tensor input")
        self.feature = inputs[0].detach()

    def clear(self) -> None:
        self.feature = None

    def close(self) -> None:
        self.handle.remove()


def read_utility_matrix(path: Path) -> tuple[list[UtilityRecord], tuple[str, ...]]:
    """Load and validate a detection utility matrix."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"utility matrix is empty: {path}")

    expert_names = []
    expert_id = 0
    while f"expert_{expert_id}_total" in rows[0]:
        expert_names.append(str(rows[0][f"expert_{expert_id}_name"]))
        expert_id += 1
    if len(expert_names) < 2:
        raise ValueError("utility matrix must contain at least two experts")

    records = []
    seen = set()
    for row in rows:
        image_id = Path(str(row["image_id"])).name
        if image_id in seen:
            raise ValueError(f"duplicate image id in utility matrix: {image_id}")
        seen.add(image_id)
        target = torch.tensor(
            [float(row[f"expert_{index}_target_probability"]) for index in range(len(expert_names))],
            dtype=torch.float32,
        )
        anchor = torch.tensor(
            [float(row[f"expert_{index}_router_probability"]) for index in range(len(expert_names))],
            dtype=torch.float32,
        )
        forced_losses = torch.tensor(
            [float(row[f"expert_{index}_total"]) for index in range(len(expert_names))],
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(torch.cat((target, anchor, forced_losses))).all()):
            raise ValueError(f"non-finite utility values for {image_id}")
        if not torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-4):
            raise ValueError(f"utility targets do not sum to one for {image_id}")
        if not torch.isclose(anchor.sum(), torch.tensor(1.0), atol=1e-4):
            raise ValueError(f"anchor probabilities do not sum to one for {image_id}")
        records.append(
            UtilityRecord(
                image_id=image_id,
                sequence_id=str(row["sequence_id"]),
                target=target,
                anchor=anchor,
                forced_losses=forced_losses,
            )
        )
    return records, tuple(expert_names)


def split_sequence_indices(
    sequence_ids: tuple[str, ...] | list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], tuple[str, ...]]:
    """Split records by sequence so adjacent frames cannot cross the boundary."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    unique = sorted(set(sequence_ids))
    if len(unique) < 2:
        raise ValueError("at least two video sequences are required")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique))
    validation_count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation_sequences = tuple(sorted(shuffled[:validation_count]))
    validation_set = set(validation_sequences)
    train_indices = [index for index, sequence in enumerate(sequence_ids) if sequence not in validation_set]
    validation_indices = [index for index, sequence in enumerate(sequence_ids) if sequence in validation_set]
    return train_indices, validation_indices, validation_sequences


def utility_probabilities(router: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Return image-level probabilities from spatial router logits."""
    logits = router._compute_logits(features)
    temperature = torch.as_tensor(router.temperature, device=logits.device, dtype=logits.dtype).clamp_min(1e-6)
    probabilities = torch.softmax(logits / temperature, dim=1)
    if probabilities.ndim > 2:
        probabilities = probabilities.mean(dim=tuple(range(2, probabilities.ndim)))
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def utility_importance(forced_losses: torch.Tensor, power: float) -> torch.Tensor:
    """Weight supervision by causal utility span without letting outliers dominate."""
    if power < 0:
        raise ValueError("importance power must be non-negative")
    spans = forced_losses.max(dim=1).values - forced_losses.min(dim=1).values
    if power == 0:
        return torch.ones_like(spans)
    median = spans.detach().median().clamp_min(1e-8)
    return (spans / median).clamp(0.25, 4.0).pow(power)


def utility_objective(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    anchors: torch.Tensor,
    forced_losses: torch.Tensor,
    *,
    anchor_weight: float,
    importance_power: float,
) -> torch.Tensor:
    """Compute utility-weighted soft-label cross entropy."""
    if not 0 <= anchor_weight < 1:
        raise ValueError("anchor_weight must be in [0, 1)")
    blended = (1.0 - anchor_weight) * targets + anchor_weight * anchors
    per_sample = -(blended * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    importance = utility_importance(forced_losses, importance_power)
    return (per_sample * importance).sum() / importance.sum().clamp_min(1e-8)


def routing_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    forced_losses: torch.Tensor,
) -> dict[str, float]:
    """Measure soft-target fit and hard decision detection regret."""
    selected = probabilities.argmax(dim=1)
    oracle = forced_losses.argmin(dim=1)
    selected_losses = forced_losses.gather(1, selected.unsqueeze(1)).squeeze(1)
    oracle_losses = forced_losses.min(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    entropy /= math.log(probabilities.shape[1])
    return {
        "cross_entropy": float((-(targets * probabilities.clamp_min(1e-8).log()).sum(dim=1)).mean()),
        "oracle_accuracy": float((selected == oracle).float().mean()),
        "mean_regret": float((selected_losses - oracle_losses).mean()),
        "p95_regret": float(torch.quantile(selected_losses - oracle_losses, 0.95)),
        "mean_normalized_entropy": float(entropy.mean()),
    }


def extract_feature_cache(
    model: nn.Module,
    block: nn.Module,
    records: list[UtilityRecord],
    data: Path,
    split: str,
    *,
    device: torch.device,
    imgsz: int,
    workers: int,
) -> FeatureCache:
    """Run the frozen detector once per matrix image and cache target-layer inputs."""
    validator, base_loader = build_validation_loader(
        model,
        data,
        split,
        device=device,
        imgsz=imgsz,
        workers=workers,
    )
    dataset = base_loader.dataset
    dataset_index = {Path(path).name: index for index, path in enumerate(dataset.im_files)}
    missing = sorted(record.image_id for record in records if record.image_id not in dataset_index)
    if missing:
        raise ValueError(f"{len(missing)} matrix images are absent from dataset split {split}: {missing[:3]}")
    subset_indices = [dataset_index[record.image_id] for record in records]
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=dataset.collate_fn,
    )
    probe = TargetInputProbe(block)
    features = []
    seen_ids = []
    try:
        model.eval()
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                batch = validator.preprocess(batch)
                probe.clear()
                _ = model(batch["img"])
                if probe.feature is None:
                    raise RuntimeError("target MoT input hook did not run")
                image_id = Path(str(batch["im_file"][0])).name
                expected = records[index].image_id
                if image_id != expected:
                    raise RuntimeError(f"feature/utility alignment changed: expected {expected}, got {image_id}")
                features.append(probe.feature[0].to(device="cpu", dtype=torch.float16))
                seen_ids.append(image_id)
                if (index + 1) % 100 == 0 or index + 1 == len(records):
                    print(f"[utility-router] cached {index + 1}/{len(records)} feature maps")
    finally:
        probe.close()
        close = getattr(base_loader, "close", None)
        if callable(close):
            close()
    shapes = {tuple(feature.shape) for feature in features}
    if len(shapes) != 1:
        raise ValueError(f"cached feature shapes differ; use rect=False and a fixed image size: {sorted(shapes)}")
    return FeatureCache(
        image_ids=tuple(seen_ids),
        sequence_ids=tuple(record.sequence_id for record in records),
        features=torch.stack(features),
        targets=torch.stack([record.target for record in records]),
        anchors=torch.stack([record.anchor for record in records]),
        forced_losses=torch.stack([record.forced_losses for record in records]),
    )


def subset_tensors(cache: FeatureCache, indices: list[int], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Move one cache subset to the training device."""
    index = torch.tensor(indices, dtype=torch.long)
    return (
        cache.features[index].to(device=device, dtype=torch.float32),
        cache.targets[index].to(device),
        cache.anchors[index].to(device),
        cache.forced_losses[index].to(device),
    )


def evaluate_router(
    router: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
) -> tuple[dict[str, float], torch.Tensor]:
    """Evaluate a router without retaining feature-sized activations."""
    features, targets, _anchors, forced_losses = tensors
    probabilities = []
    router.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            probabilities.append(utility_probabilities(router, features[start : start + batch_size]).cpu())
    all_probabilities = torch.cat(probabilities)
    return routing_metrics(all_probabilities, targets.cpu(), forced_losses.cpu()), all_probabilities


def file_sha256(path: Path) -> str:
    """Return a reproducible file identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_matrix_manifest(matrix: Path, model_sha: str, layer_name: str, split: str) -> None:
    """Reject accidental checkpoint, layer, or split mismatches."""
    path = matrix.parent / "experiment_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model_sha256": model_sha,
        "layer": layer_name,
        "split": split,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        detail = ", ".join(f"{key}: {manifest.get(key)!r} != {expected[key]!r}" for key in mismatches)
        raise ValueError(f"utility matrix manifest mismatch: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--importance-power", type=float, default=0.5)
    parser.add_argument("--enable-scene-head", action="store_true")
    parser.add_argument("--scene-hidden-dim", type=int, default=12)
    parser.add_argument("--enable-global-utility-head", action="store_true")
    parser.add_argument("--utility-hidden-dim", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.workers < 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--imgsz, --epochs and --batch-size must be positive; --workers must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("--learning-rate must be positive and --weight-decay non-negative")
    if not 0 <= args.anchor_weight < 1 or args.importance_power < 0:
        raise SystemExit("--anchor-weight must be in [0,1) and --importance-power non-negative")
    if args.scene_hidden_dim <= 0 or args.utility_hidden_dim <= 0:
        raise SystemExit("--scene-hidden-dim and --utility-hidden-dim must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    matrix_path = args.matrix.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = normalize_torch_device(args.device)
    device = torch.device(device_name)
    model = load_model(model_path, device_name, nc=10)
    layer_name, block = resolve_mot_layer(model, args.layer)
    records, expert_names = read_utility_matrix(matrix_path)
    if len(expert_names) != block.num_experts:
        raise SystemExit(f"matrix has {len(expert_names)} experts but {layer_name} has {block.num_experts}")
    model_sha = file_sha256(model_path)
    verify_matrix_manifest(matrix_path, model_sha, layer_name, args.split)
    if args.enable_scene_head:
        block.router.enable_scene_aware(args.scene_hidden_dim)
    if args.enable_global_utility_head:
        block.router.enable_global_utility_head(args.utility_hidden_dim)

    # 检测器和专家始终冻结；只允许目标层 router 更新。
    model.requires_grad_(False)
    block.router.requires_grad_(True)
    cache = extract_feature_cache(
        model,
        block,
        records,
        data_path,
        args.split,
        device=device,
        imgsz=args.imgsz,
        workers=args.workers,
    )
    train_indices, validation_indices, validation_sequences = split_sequence_indices(
        cache.sequence_ids,
        args.validation_fraction,
        args.seed,
    )
    train_tensors = subset_tensors(cache, train_indices, device)
    validation_tensors = subset_tensors(cache, validation_indices, device)
    baseline_train, baseline_probabilities = evaluate_router(block.router, train_tensors, args.batch_size)
    baseline_validation, _ = evaluate_router(block.router, validation_tensors, args.batch_size)
    anchor_fidelity_mae = float((baseline_probabilities - train_tensors[2].cpu()).abs().mean())

    optimizer = torch.optim.AdamW(
        block.router.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    best_state = copy.deepcopy(block.router.state_dict())
    best_epoch = 0
    best_key = (baseline_validation["mean_regret"], baseline_validation["cross_entropy"])
    history = []
    stale_epochs = 0
    train_features, train_targets, train_anchors, train_losses = train_tensors
    for epoch in range(1, args.epochs + 1):
        block.router.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        epoch_loss = 0.0
        sample_count = 0
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size].to(device)
            probabilities = utility_probabilities(block.router, train_features[batch_indices])
            loss = utility_objective(
                probabilities,
                train_targets[batch_indices],
                train_anchors[batch_indices],
                train_losses[batch_indices],
                anchor_weight=args.anchor_weight,
                importance_power=args.importance_power,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.router.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch_indices)
            sample_count += len(batch_indices)
        scheduler.step()
        train_metrics, _ = evaluate_router(block.router, train_tensors, args.batch_size)
        validation_metrics, _ = evaluate_router(block.router, validation_tensors, args.batch_size)
        current_key = (validation_metrics["mean_regret"], validation_metrics["cross_entropy"])
        improved = current_key < best_key
        if improved:
            best_key = current_key
            best_epoch = epoch
            best_state = copy.deepcopy(block.router.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "optimization_loss": epoch_loss / max(sample_count, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "best": improved,
            }
        )
        print(
            f"[utility-router] epoch={epoch:03d} "
            f"train_regret={train_metrics['mean_regret']:.6f} "
            f"val_regret={validation_metrics['mean_regret']:.6f} "
            f"val_acc={validation_metrics['oracle_accuracy']:.3f}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[utility-router] early stop after {stale_epochs} stale epochs")
            break

    block.router.load_state_dict(best_state)
    final_train, _ = evaluate_router(block.router, train_tensors, args.batch_size)
    final_validation, _ = evaluate_router(block.router, validation_tensors, args.batch_size)
    bundle = {
        "format": "yolo-master-mot-utility-router-v1",
        "base_checkpoint_sha256": model_sha,
        "layer_name": layer_name,
        "expert_names": expert_names,
        "router_state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "best_epoch": best_epoch,
        "baseline_train_metrics": baseline_train,
        "baseline_validation_metrics": baseline_validation,
        "final_train_metrics": final_train,
        "final_validation_metrics": final_validation,
        "config": {
            "image_size": args.imgsz,
            "anchor_weight": args.anchor_weight,
            "importance_power": args.importance_power,
            "validation_fraction": args.validation_fraction,
            "scene_aware": bool(block.router.scene_aware),
            "scene_hidden_dim": block.router.scene_hidden_dim,
            "global_utility_head": getattr(block.router, "utility_projector", None) is not None,
            "utility_hidden_dim": (
                int(block.router.utility_projector[0].out_features)
                if getattr(block.router, "utility_projector", None) is not None
                else None
            ),
            "seed": args.seed,
        },
    }
    torch.save(bundle, output / "utility_router.pt")
    write_csv(output / "training_history.csv", history)
    report = {
        "layer": layer_name,
        "matrix_sha256": file_sha256(matrix_path),
        "base_checkpoint_sha256": model_sha,
        "images": len(records),
        "train_images": len(train_indices),
        "validation_images": len(validation_indices),
        "train_sequences": len({cache.sequence_ids[index] for index in train_indices}),
        "validation_sequences": len(validation_sequences),
        "sequence_overlap": False,
        "validation_sequence_ids": validation_sequences,
        "trainable_parameters": sum(parameter.numel() for parameter in block.router.parameters()),
        "frozen_detector_and_experts": True,
        "scene_aware": bool(block.router.scene_aware),
        "scene_hidden_dim": block.router.scene_hidden_dim,
        "global_utility_head": getattr(block.router, "utility_projector", None) is not None,
        "anchor_fidelity_mae": anchor_fidelity_mae,
        "best_epoch": best_epoch,
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "final_train": final_train,
        "final_validation": final_validation,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[utility-router] best_epoch={best_epoch} "
        f"validation_regret={baseline_validation['mean_regret']:.6f}"
        f"->{final_validation['mean_regret']:.6f}"
    )
    print(f"[utility-router] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
