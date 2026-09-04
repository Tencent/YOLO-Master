#!/usr/bin/env python3
"""Train and evaluate the D1 LatentMixture detector from offline DINOv3 features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from ultralytics.data.foundation_cache import atomic_write_json, validate_manifest
from ultralytics.data.preloaded_feature_cache import PreloadedFeatureBatchLoader
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.mixture_loss import _collect_mixture_aux_loss
from ultralytics.nn.modules import Detect, LatentMixture
from ultralytics.utils.loss import E2EDetectLoss
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import xywh2xyxy


TEACHER_TOKENS = ("teacher", "dinov3", "foundation_model", "transformers")
RESULT_FIELDS = (
    "epoch",
    "train/total_loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/mixture_aux_loss",
    "val/total_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr",
    "epoch_seconds",
    "peak_vram_bytes",
)


def seed_everything(seed: int) -> None:
    """Seed all local generators and request deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def split_sample_indices(
    sample_count: int,
    train_count: int,
    val_count: int,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic data split independently of the training seed."""
    if min(
        sample_count,
        train_count,
        val_count,
    ) <= 0:
        raise ValueError(
            "sample-count, train-count, and "
            "val-count must be positive"
        )

    required = train_count + val_count

    if sample_count < required:
        raise ValueError(
            f"cache has {sample_count} samples, "
            f"but split needs {required}"
        )

    order = list(range(sample_count))
    random.Random(split_seed).shuffle(order)

    train_indices = sorted(
        order[:train_count]
    )
    val_indices = sorted(
        order[train_count:required]
    )

    return train_indices, val_indices


def label_path(dataset_root: Path, image_relative_path: str) -> Path:
    """Resolve the YOLO label corresponding to a manifest image path."""
    image = Path(image_relative_path)
    parts = list(image.parts)
    try:
        image_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"image path does not contain an images directory: {image}") from exc
    parts[image_index] = "labels"
    return (dataset_root / Path(*parts)).with_suffix(".txt")


def transform_letterbox_xywhn(boxes: torch.Tensor, letterbox: dict[str, Any], imgsz: int) -> torch.Tensor:
    """Map normalized source-image xywh labels into normalized cached-letterbox coordinates."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    height, width = (float(value) for value in letterbox["original_shape"])
    resized_height, resized_width = (float(value) for value in letterbox["resized_shape"])
    scale_x, scale_y = resized_width / width, resized_height / height
    pad_left, pad_top = (float(value) for value in letterbox["pad"])
    result = boxes.clone().float()
    result[:, 0] = (boxes[:, 0] * width * scale_x + pad_left) / imgsz
    result[:, 1] = (boxes[:, 1] * height * scale_y + pad_top) / imgsz
    result[:, 2] = boxes[:, 2] * width * scale_x / imgsz
    result[:, 3] = boxes[:, 3] * height * scale_y / imgsz
    return result.clamp_(0.0, 1.0)


def read_yolo_labels(path: Path, letterbox: dict[str, Any], imgsz: int, nc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one YOLO label file and apply the exact cache letterbox transform."""
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return torch.empty((0, 1), dtype=torch.float32), torch.empty((0, 4), dtype=torch.float32)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(fields)}")
        rows.append([float(value) for value in fields])
    labels = torch.tensor(rows, dtype=torch.float32)
    classes = labels[:, :1]
    if not torch.all((classes >= 0) & (classes < nc)):
        raise ValueError(f"class index outside [0, {nc}) in {path}")
    boxes = transform_letterbox_xywhn(labels[:, 1:5], letterbox, imgsz)
    return classes, boxes


class CachedDetectionDataset(Dataset):
    """Offline DINO feature tensors paired with the original YOLO annotations."""

    def __init__(self, cache_root: Path, manifest: dict[str, Any], indices: list[int], nc: int) -> None:
        self.cache_root = cache_root
        self.dataset_root = Path(manifest["dataset_root"])
        self.samples = [manifest["samples"][index] for index in indices]
        self.layers = tuple(int(value) for value in manifest["identity"]["layers"])
        self.imgsz = int(manifest["identity"]["imgsz"])
        self.nc = nc

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        cache_file = self.cache_root / sample["cache_path"]
        tensors = load_file(str(cache_file), device="cpu")
        features = tuple(tensors[f"layer_{layer}"] for layer in self.layers)
        expected = tuple(tuple(sample["shapes"][f"layer_{layer}"]) for layer in self.layers)
        actual = tuple(tuple(feature.shape) for feature in features)
        if actual != expected:
            raise RuntimeError(f"feature shape mismatch for {sample['sample_id']}: {actual} != {expected}")
        classes, boxes = read_yolo_labels(
            label_path(self.dataset_root, sample["image_path"]), sample["letterbox"], self.imgsz, self.nc
        )
        return {
            "features": features,
            "cls": classes,
            "bboxes": boxes,
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
        }


def collate_cached(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack cached levels and concatenate targets in Ultralytics loss format."""
    levels = len(batch[0]["features"])
    features = tuple(torch.stack([sample["features"][level] for sample in batch]) for level in range(levels))
    classes, boxes, batch_indices = [], [], []
    for index, sample in enumerate(batch):
        count = len(sample["cls"])
        classes.append(sample["cls"])
        boxes.append(sample["bboxes"])
        batch_indices.append(torch.full((count,), index, dtype=torch.long))
    return {
        "features": features,
        "cls": torch.cat(classes, dim=0),
        "bboxes": torch.cat(boxes, dim=0),
        "batch_idx": torch.cat(batch_indices, dim=0),
        "targets": [(sample["cls"], sample["bboxes"]) for sample in batch],
        "sample_ids": [sample["sample_id"] for sample in batch],
        "image_paths": [sample["image_path"] for sample in batch],
    }


class CachedLatentDetector(nn.Module):
    """Trainable LatentMixture pyramid and YOLO26 Detect head; contains no teacher."""

    def __init__(self, *, in_channels: tuple[int, ...], nc: int, imgsz: int, epochs: int) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"P0 expects three cached levels, got {in_channels}")
        if imgsz % 32:
            raise ValueError(f"imgsz must be divisible by 32, got {imgsz}")
        self.imgsz = imgsz
        self.p3 = LatentMixture(in_channels, 64, num_experts=4, residual_init=0.01)
        self.p4 = LatentMixture(in_channels, 128, num_experts=4, residual_init=0.01)
        self.p5 = LatentMixture(in_channels, 256, num_experts=4, residual_init=0.01)
        self.detect = Detect(nc=nc, reg_max=1, end2end=True, ch=(64, 128, 256))
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.bias_init()
        # Ultralytics loss only indexes ``model[-1]``; keep a plain list here so
        # the same modules are not registered twice in state_dict().
        self.model = [self.p3, self.p4, self.p5, self.detect]
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=epochs)
        self.class_weights = None

    def pyramid(self, raw: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Project the three cached teacher levels into P3/P4/P5."""
        sizes = ((self.imgsz // 8,) * 2, (self.imgsz // 16,) * 2, (self.imgsz // 32,) * 2)
        mixtures = (self.p3, self.p4, self.p5)
        output = []
        for size, mixture in zip(sizes, mixtures):
            resized = [
                feature
                if tuple(feature.shape[-2:]) == size
                else F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
                for feature in raw
            ]
            output.append(mixture(resized))
        return output

    def training_predictions(self, raw: tuple[torch.Tensor, ...]) -> dict[str, dict[str, torch.Tensor]]:
        """Return raw one-to-many and one-to-one tensors regardless of eval mode."""
        features = self.pyramid(raw)
        one2many = self.detect.forward_head(features, **self.detect.one2many)
        detached = [feature.detach() for feature in features]
        one2one = self.detect.forward_head(detached, **self.detect.one2one)
        return {"one2many": one2many, "one2one": one2one}

    def forward(self, raw: tuple[torch.Tensor, ...]) -> Any:
        return self.detect(self.pyramid(raw))


def audit_model(model: nn.Module) -> dict[str, Any]:
    """Enforce the P0 trainability and teacher-exclusion contracts."""
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    allowed_prefixes = ("p3.", "p4.", "p5.", "detect.")
    unexpected = [name for name in trainable if not name.startswith(allowed_prefixes)]
    teacher_names = [
        name for name, _ in model.named_parameters() if any(token in name.lower() for token in TEACHER_TOKENS)
    ]
    if unexpected or teacher_names:
        raise RuntimeError(f"trainability audit failed: unexpected={unexpected[:10]}, teacher={teacher_names[:10]}")
    return {
        "status": "PASS",
        "teacher_loaded": False,
        "allowed_trainable_prefixes": ["p3", "p4", "p5", "detect"],
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "unexpected_trainable": unexpected,
        "teacher_parameters": teacher_names,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """Move features and loss targets to the selected device."""
    features = tuple(
        feature.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        for feature in batch["features"]
    )
    targets = {
        "cls": batch["cls"].to(device, non_blocking=True),
        "bboxes": batch["bboxes"].to(device, non_blocking=True),
        "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
    }
    return features, targets


def match_predictions(predictions: torch.Tensor, target_cls: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Greedily match predictions to targets across COCO IoU thresholds."""
    thresholds = torch.linspace(0.5, 0.95, 10, device=predictions.device)
    correct = torch.zeros((len(predictions), len(thresholds)), dtype=torch.bool, device=predictions.device)
    if len(predictions) == 0 or len(target_cls) == 0:
        return correct
    iou = box_iou(target_boxes, predictions[:, :4])
    iou *= target_cls.view(-1, 1) == predictions[:, 5].view(1, -1)
    for column, threshold in enumerate(thresholds):
        matches = torch.nonzero(iou >= threshold, as_tuple=False)
        if matches.numel() == 0:
            continue
        match_iou = iou[matches[:, 0], matches[:, 1]]
        order = torch.argsort(match_iou, descending=True)
        matches = matches[order]
        used_targets, used_predictions = set(), set()
        for target_index, prediction_index in matches.tolist():
            if target_index in used_targets or prediction_index in used_predictions:
                continue
            used_targets.add(target_index)
            used_predictions.add(prediction_index)
            correct[prediction_index, column] = True
    return correct


@torch.inference_mode()
def validate(
    model: CachedLatentDetector,
    loader: DataLoader,
    criterion: E2EDetectLoss,
    device: torch.device,
    names: dict[int, str],
    conf: float,
) -> tuple[dict[str, float], list[float]]:
    """Compute true YOLO detection loss and decoded COCO-style box metrics."""
    model.eval()
    metrics = DetMetrics(names=names)
    loss_sum = torch.zeros(3, device=device)
    sample_count = 0
    image_index = 0
    for batch in loader:
        features, targets = move_batch(batch, device)
        raw_predictions = model.training_predictions(features)
        loss_vector, loss_items = criterion(raw_predictions, targets)
        loss_sum += loss_items * len(batch["sample_ids"])
        sample_count += len(batch["sample_ids"])

        decoded, _ = model(features)
        for local_index, (classes_cpu, boxes_cpu) in enumerate(batch["targets"]):
            predictions = decoded[local_index]
            predictions = predictions[predictions[:, 4] >= conf]
            classes = classes_cpu.view(-1).to(device)
            boxes = boxes_cpu.to(device) * model.imgsz
            boxes = xywh2xyxy(boxes)
            correct = match_predictions(predictions, classes, boxes)
            count = len(classes)
            metrics.update_stats(
                {
                    "tp": correct.cpu().numpy(),
                    "conf": predictions[:, 4].float().cpu().numpy(),
                    "pred_cls": predictions[:, 5].float().cpu().numpy(),
                    "target_cls": classes.float().cpu().numpy(),
                    "target_img": np.full(count, image_index, dtype=np.int64),
                    "im_name": batch["image_paths"][local_index],
                }
            )
            image_index += 1
    metrics.process(plot=False)
    results = {key: float(value) for key, value in metrics.results_dict.items() if key in metrics.keys}
    mean_loss = (loss_sum / max(sample_count, 1)).tolist()
    if not all(math.isfinite(value) for value in (*results.values(), *mean_loss)):
        raise RuntimeError(f"non-finite validation result: metrics={results}, loss={mean_loss}")
    return results, mean_loss


def save_checkpoint(path: Path, model: nn.Module, epoch: int, row: dict[str, Any]) -> None:
    """Save only student state and JSON-safe evidence; never serialize a teacher."""
    state = model.state_dict()
    leaked = [name for name in state if any(token in name.lower() for token in TEACHER_TOKENS)]
    if leaked:
        raise RuntimeError(f"teacher key leaked into checkpoint: {leaked[:10]}")
    torch.save({"epoch": epoch, "model": state, "metrics": row, "teacher_loaded": False}, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "dataset split seed; defaults to "
            "--seed for backward compatibility"
        ),
    )
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixture-aux-weight", type=float, default=0.1)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--project", type=Path, default=Path("runs/d1/p0"))
    parser.add_argument("--name", default="cached-latent-coco128-s0")
    parser.add_argument(
        "--cache-residency",
        choices=("stream", "cpu", "gpu"),
        default="stream",
        help=(
            "stream features per sample, preload into "
            "host memory, or keep the FP16 cache on GPU"
        ),
    )
    parser.add_argument("--verify-cache-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch, args.train_count, args.val_count) <= 0:
        raise ValueError("epochs, batch, train-count, and val-count must be positive")
    seed_everything(args.seed)
    split_seed = (
        args.seed
        if args.split_seed is None
        else args.split_seed
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    cache_root = args.cache.resolve()
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, cache_root=cache_root, verify_files=args.verify_cache_files)
    required = args.train_count + args.val_count
    if len(manifest["samples"]) < required:
        raise ValueError(f"cache has {len(manifest['samples'])} samples, but split needs {required}")
    data = check_det_dataset(args.dataset, autodownload=False)
    name_values = data["names"]
    names = (
        {int(key): str(value) for key, value in name_values.items()}
        if isinstance(name_values, dict)
        else {index: str(value) for index, value in enumerate(name_values)}
    )
    nc = len(names)

    train_indices, val_indices = (
        split_sample_indices(
            sample_count=len(
                manifest["samples"]
            ),
            train_count=args.train_count,
            val_count=args.val_count,
            split_seed=split_seed,
        )
    )
    output = (args.project / args.name).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    weights = output / "weights"
    weights.mkdir(parents=True, exist_ok=True)

    train_dataset = CachedDetectionDataset(cache_root, manifest, train_indices, nc)
    val_dataset = CachedDetectionDataset(cache_root, manifest, val_indices, nc)
    generator = torch.Generator().manual_seed(args.seed)
    preload_seconds = 0.0
    preloaded_bytes = 0

    if args.cache_residency == "stream":
        loader_options = dict(
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_cached,
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_options,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_options,
        )
    else:
        if args.workers != 0:
            raise ValueError(
                "preloaded cache residency requires --workers 0"
            )
        if (
            args.cache_residency == "gpu"
            and device.type != "cuda"
        ):
            raise ValueError(
                "GPU cache residency requires a CUDA device"
            )

        storage_device = (
            device
            if args.cache_residency == "gpu"
            else torch.device("cpu")
        )
        preload_started = time.perf_counter()

        train_loader = PreloadedFeatureBatchLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            generator=generator,
            storage_device=storage_device,
        )
        val_loader = PreloadedFeatureBatchLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            generator=None,
            storage_device=storage_device,
        )

        preload_seconds = (
            time.perf_counter() - preload_started
        )
        preloaded_bytes = (
            train_loader.preloaded_bytes
            + val_loader.preloaded_bytes
        )

    first = manifest["samples"][0]
    layers = tuple(int(value) for value in manifest["identity"]["layers"])
    in_channels = tuple(int(first["shapes"][f"layer_{layer}"][0]) for layer in layers)
    imgsz = int(manifest["identity"]["imgsz"])
    model = CachedLatentDetector(in_channels=in_channels, nc=nc, imgsz=imgsz, epochs=args.epochs).to(device)
    audit = audit_model(model)
    criterion = E2EDetectLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config = vars(args).copy()
    config.update(
        cache=str(cache_root),
        project=str(args.project),
        output=str(output),
        imgsz=imgsz,
        layers=list(layers),
        in_channels=list(in_channels),
        nc=nc,
        split_seed=split_seed,
        cache_identity=manifest["identity"],
    )
    atomic_write_json(output / "args.json", config)
    atomic_write_json(output / "trainable-audit.json", audit)
    atomic_write_json(
        output / "split.json",
        {
            "seed": split_seed,
            "training_seed": args.seed,
            "split_seed": split_seed,
            "train_sample_ids": [
                manifest["samples"][index][
                    "sample_id"
                ]
                for index in train_indices
            ],
            "val_sample_ids": [
                manifest["samples"][index][
                    "sample_id"
                ]
                for index in val_indices
            ],
        },
    )
    results_path = output / "results.csv"
    best_fitness = -math.inf
    total_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_sum = 0.0
            loss_sum = torch.zeros(3, device=device)
            aux_sum = 0.0
            sample_count = 0
            for batch in train_loader:
                features, targets = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model.training_predictions(features)
                loss_vector, loss_items = criterion(predictions, targets)
                aux_loss = _collect_mixture_aux_loss(model, device, latent_gain=1.0, aux_budget=3.0)
                total_loss = loss_vector.sum() + args.mixture_aux_weight * aux_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                batch_size = len(batch["sample_ids"])
                # Native YOLO loss is already multiplied by batch size.
                total_sum += float(total_loss.detach())
                loss_sum += loss_items * batch_size
                aux_sum += float(aux_loss.detach()) * batch_size
                sample_count += batch_size

            validation, val_loss = validate(model, val_loader, criterion, device, names, args.conf)
            peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            train_loss = (loss_sum / sample_count).tolist()
            row = {
                "epoch": epoch,
                "train/total_loss": total_sum / sample_count,
                "train/box_loss": train_loss[0],
                "train/cls_loss": train_loss[1],
                "train/dfl_loss": train_loss[2],
                "train/mixture_aux_loss": aux_sum / sample_count,
                "val/total_loss": sum(val_loss),
                "val/box_loss": val_loss[0],
                "val/cls_loss": val_loss[1],
                "val/dfl_loss": val_loss[2],
                **validation,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "peak_vram_bytes": peak_vram,
            }
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise RuntimeError(f"non-finite result row: {row}")
            writer.writerow(row)
            stream.flush()
            save_checkpoint(weights / "last.pt", model, epoch, row)
            fitness = float(row["metrics/mAP50-95(B)"])
            if fitness >= best_fitness:
                best_fitness = fitness
                save_checkpoint(weights / "best.pt", model, epoch, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()

    final_row = row
    training_elapsed = time.perf_counter() - total_started
    summary = {
        "status": "PASS",
        "p0_contract": {
            "offline_cache": True,
            "teacher_loaded_during_training": False,
            "trainable_components": ["LatentMixture-P3", "LatentMixture-P4", "LatentMixture-P5", "Detect"],
            "true_yolo_labels": True,
            "true_detection_loss": True,
            "validation_metrics": True,
            "student_only_checkpoints": True,
        },
        "output": str(output),
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "training_seed": args.seed,
        "split_seed": split_seed,
        "cache_residency": args.cache_residency,
        "preload_seconds": preload_seconds,
        "preloaded_bytes": preloaded_bytes,
        "elapsed_seconds": training_elapsed,
        "elapsed_seconds_with_preload": (
            training_elapsed + preload_seconds
        ),
        "gpu_hours_training": (
            training_elapsed / 3600
            if device.type == "cuda"
            else 0.0
        ),
        "peak_vram_bytes": int(final_row["peak_vram_bytes"]),
        "best_map50_95": best_fitness,
        "final": final_row,
        "trainability_audit": audit,
    }
    atomic_write_json(output / "p0-summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("D1 offline-cache P0: PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"D1 offline-cache P0: FAIL: {exc}", file=sys.stderr, flush=True)
        raise
