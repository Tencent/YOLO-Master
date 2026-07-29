#!/usr/bin/env python3
"""Build an image-level detection utility matrix for one MoT routing layer.

For each image, the script measures the natural-routing detection loss and the
counterfactual loss obtained by forcing every token through each expert. The
resulting expert losses define an oracle expert, router regret, and soft utility
targets suitable for utility-router supervision.

Each counterfactual changes one requested MoT layer while every other routed
layer retains its natural decision. This gives a layer-local intervention even
for a full multi-MoT checkpoint.

Example:
    python scripts/build_mot_detection_utility.py \
      --model runs/mot_cross_domain/training/v10_mot_p5/weights/best.pt \
      --data ultralytics/cfg/datasets/VisDrone.yaml \
      --device 0 \
      --max-images 128 \
      --output runs/mot_detection_utility
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mot_cross_domain import EXPERT_NAMES, load_model, normalize_torch_device, write_csv
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.utils.routing_interpreter import RoutingInterpreter

LOSS_NAMES = ("box", "cls", "dfl")


@dataclass(frozen=True)
class DetectionLoss:
    """Native detection loss components, excluding routed auxiliary losses."""

    box: float
    cls: float
    dfl: float
    routing_aux: float

    @property
    def total(self) -> float:
        return self.box + self.cls + self.dfl


class RouterProbabilityProbe:
    """Capture the dense natural-router probability distribution."""

    def __init__(self, layer_name: str, block: MoTBlock):
        self.layer_name = layer_name
        self.probabilities: torch.Tensor | None = None
        self.handle = block.router.register_forward_hook(self._capture)

    def _capture(self, module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) < 3 or not isinstance(output[2], torch.Tensor):
            raise RuntimeError(f"{self.layer_name} router must return weights, indices, and logits")
        logits = output[2].detach().float()
        temperature = max(float(module.temperature), torch.finfo(torch.float32).tiny)
        probabilities = torch.softmax(logits / temperature, dim=1)
        reduce_dims = tuple(range(2, probabilities.ndim))
        self.probabilities = probabilities.mean(dim=reduce_dims).cpu() if reduce_dims else probabilities.cpu()

    def clear(self) -> None:
        self.probabilities = None

    def close(self) -> None:
        self.handle.remove()


def detection_loss_from_items(items: torch.Tensor | list[Any] | tuple[Any, ...]) -> DetectionLoss:
    """Normalize Ultralytics loss items while excluding mixture auxiliary loss."""
    values = torch.as_tensor(items).detach().float().reshape(-1).cpu()
    if values.numel() < len(LOSS_NAMES):
        raise ValueError(f"expected at least three detection losses, got {values.numel()}")
    routing_aux = float(values[3:].sum()) if values.numel() > 3 else 0.0
    return DetectionLoss(
        box=float(values[0]),
        cls=float(values[1]),
        dfl=float(values[2]),
        routing_aux=routing_aux,
    )


def soft_utility_targets(losses: list[float] | np.ndarray, temperature: float) -> np.ndarray:
    """Convert lower-is-better expert losses into a stable probability target."""
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("expert losses must be a finite one-dimensional array")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("utility temperature must be positive and finite")
    logits = -(values - values.min()) / temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def summarize_labels(batch: dict[str, Any], imgsz: int) -> dict[str, float | int]:
    """Describe one image's target density and scale without exposing local paths."""
    boxes = batch["bboxes"].detach().float().reshape(-1, 4)
    classes = batch["cls"].detach().long().reshape(-1)
    areas = boxes[:, 2] * boxes[:, 3] if boxes.numel() else boxes.new_zeros((0,))
    small_threshold = (32.0 / imgsz) ** 2
    large_threshold = (96.0 / imgsz) ** 2
    return {
        "num_targets": int(classes.numel()),
        "num_classes": int(classes.unique().numel()),
        "mean_normalized_area": float(areas.mean()) if areas.numel() else 0.0,
        "small_target_share": float((areas < small_threshold).float().mean()) if areas.numel() else 0.0,
        "large_target_share": float((areas >= large_threshold).float().mean()) if areas.numel() else 0.0,
    }


def select_indices(dataset_size: int, max_images: int, seed: int) -> set[int]:
    """Select a reproducible subset without privileging early video sequences."""
    if max_images <= 0 or max_images >= dataset_size:
        return set(range(dataset_size))
    rng = np.random.default_rng(seed)
    return {int(index) for index in rng.choice(dataset_size, size=max_images, replace=False)}


def resolve_mot_layer(model: torch.nn.Module, requested: str | None) -> tuple[str, MoTBlock]:
    """Resolve one leaf MoTBlock; require an explicit name for multi-layer models."""
    layers = {name: module for name, module in model.named_modules() if isinstance(module, MoTBlock)}
    if requested is not None:
        if requested not in layers:
            choices = ", ".join(sorted(layers)) or "<none>"
            raise ValueError(f"MoT layer {requested!r} was not found; choices: {choices}")
        return requested, layers[requested]
    if len(layers) != 1:
        choices = ", ".join(sorted(layers)) or "<none>"
        raise ValueError(f"--layer is required when the model has {len(layers)} MoTBlocks: {choices}")
    return next(iter(layers.items()))


def prepare_model_for_detection_loss(model: torch.nn.Module) -> None:
    """Restore complete training hyperparameters omitted by inference checkpoints."""
    overrides = dict(model.args) if isinstance(getattr(model, "args", None), dict) else {}
    model.args = get_cfg(overrides=overrides)
    model.criterion = None


def build_validation_loader(
    model: torch.nn.Module,
    data: Path,
    split: str,
    *,
    device: torch.device,
    imgsz: int,
    workers: int,
):
    """Build the standard Ultralytics validation loader with batch size one."""
    validator = DetectionValidator(
        args={
            "data": str(data),
            "split": split,
            "imgsz": imgsz,
            "batch": 1,
            "workers": workers,
            "device": str(device),
            "rect": False,
            "plots": False,
        }
    )
    validator.data = check_det_dataset(str(data), split=split)
    validator.device = device
    validator.stride = int(model.stride.max())
    validator.training = False
    return validator, validator.get_dataloader(validator.data[split], 1)


def image_identity(batch: dict[str, Any]) -> tuple[str, str]:
    """Return public image and sequence identifiers."""
    image_id = Path(str(batch["im_file"][0])).name
    sequence = Path(image_id).stem.split("_", 1)[0]
    return image_id, sequence


def evaluate_loss(model: torch.nn.Module, batch: dict[str, Any]) -> DetectionLoss:
    """Evaluate native detection loss for one already-preprocessed image."""
    _, items = model.loss(batch)
    return detection_loss_from_items(items)


def utility_row(
    model: torch.nn.Module,
    interpreter: RoutingInterpreter,
    probe: RouterProbabilityProbe,
    layer_name: str,
    batch: dict[str, Any],
    *,
    imgsz: int,
    temperature: float,
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """Measure natural and forced-expert losses for one image."""
    probe.clear()
    natural = evaluate_loss(model, batch)
    if probe.probabilities is None or probe.probabilities.shape != (1, len(expert_names)):
        shape = None if probe.probabilities is None else tuple(probe.probabilities.shape)
        raise RuntimeError(f"natural router probability shape mismatch: {shape}")
    router_probabilities = probe.probabilities[0].numpy()
    selected_expert = int(router_probabilities.argmax())

    forced = []
    for expert_id in range(len(expert_names)):
        with interpreter.force_expert(layer_name, expert_id):
            forced.append(evaluate_loss(model, batch))

    expert_totals = np.asarray([loss.total for loss in forced], dtype=np.float64)
    targets = soft_utility_targets(expert_totals, temperature)
    oracle_expert = int(expert_totals.argmin())
    image_id, current_sequence = image_identity(batch)
    row: dict[str, Any] = {
        "image_id": image_id,
        "sequence_id": current_sequence,
        **summarize_labels(batch, imgsz),
        **{f"natural_{name}": getattr(natural, name) for name in LOSS_NAMES},
        "natural_total": natural.total,
        "natural_routing_aux": natural.routing_aux,
        "router_selected_expert": selected_expert,
        "router_selected_expert_name": expert_names[selected_expert],
        "oracle_expert": oracle_expert,
        "oracle_expert_name": expert_names[oracle_expert],
        "router_matches_oracle": selected_expert == oracle_expert,
        "router_regret": float(expert_totals[selected_expert] - expert_totals[oracle_expert]),
        "oracle_gain_over_natural": float(natural.total - expert_totals[oracle_expert]),
        "selected_gain_over_natural": float(natural.total - expert_totals[selected_expert]),
    }
    for expert_id, (expert_name, loss) in enumerate(zip(expert_names, forced)):
        prefix = f"expert_{expert_id}"
        row[f"{prefix}_name"] = expert_name
        for loss_name in LOSS_NAMES:
            row[f"{prefix}_{loss_name}"] = getattr(loss, loss_name)
        row[f"{prefix}_total"] = loss.total
        row[f"{prefix}_gain_over_natural"] = natural.total - loss.total
        row[f"{prefix}_target_probability"] = float(targets[expert_id])
        row[f"{prefix}_router_probability"] = float(router_probabilities[expert_id])
    return row


def summarize_matrix(rows: list[dict[str, Any]], expert_names: tuple[str, ...]) -> dict[str, Any]:
    """Aggregate utility-oracle quality and regret."""
    regrets = np.asarray([row["router_regret"] for row in rows], dtype=np.float64)
    oracle_gains = np.asarray([row["oracle_gain_over_natural"] for row in rows], dtype=np.float64)
    selected_gains = np.asarray([row["selected_gain_over_natural"] for row in rows], dtype=np.float64)
    forced_losses = np.asarray(
        [[row[f"expert_{expert_id}_total"] for expert_id in range(len(expert_names))] for row in rows],
        dtype=np.float64,
    )
    sorted_losses = np.sort(forced_losses, axis=1)
    utility_spans = sorted_losses[:, -1] - sorted_losses[:, 0]
    oracle_margins = sorted_losses[:, 1] - sorted_losses[:, 0]
    target_probabilities = np.asarray(
        [[row[f"expert_{expert_id}_target_probability"] for expert_id in range(len(expert_names))] for row in rows],
        dtype=np.float64,
    )
    target_entropy = -(target_probabilities * np.log(target_probabilities.clip(min=1e-12))).sum(axis=1)
    target_entropy /= math.log(len(expert_names))
    summary: dict[str, Any] = {
        "images": len(rows),
        "sequences": len({row["sequence_id"] for row in rows}),
        "router_oracle_accuracy": float(np.mean([row["router_matches_oracle"] for row in rows])),
        "mean_router_regret": float(regrets.mean()),
        "p95_router_regret": float(np.quantile(regrets, 0.95)),
        "mean_oracle_gain_over_natural": float(oracle_gains.mean()),
        "mean_selected_gain_over_natural": float(selected_gains.mean()),
        "natural_better_than_every_forced_expert_share": float(np.mean(oracle_gains < 0)),
        "median_utility_span": float(np.median(utility_spans)),
        "median_oracle_margin": float(np.median(oracle_margins)),
        "low_signal_share_below_1e_4": float(np.mean(utility_spans < 1e-4)),
        "mean_normalized_target_entropy": float(target_entropy.mean()),
    }
    for expert_id, expert_name in enumerate(expert_names):
        summary[f"oracle_share_{expert_id}_{expert_name}"] = float(
            np.mean([row["oracle_expert"] == expert_id for row in rows])
        )
        summary[f"router_share_{expert_id}_{expert_name}"] = float(
            np.mean([row["router_selected_expert"] == expert_id for row in rows])
        )
    return summary


def file_sha256(path: Path) -> str:
    """Return a reproducible checkpoint identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--utility-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.workers < 0:
        raise SystemExit("--imgsz must be positive and --workers must be non-negative")
    if args.utility_temperature <= 0 or not math.isfinite(args.utility_temperature):
        raise SystemExit("--utility-temperature must be positive and finite")

    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = normalize_torch_device(args.device)
    device = torch.device(device_name)
    model = load_model(model_path, device_name, nc=10)
    prepare_model_for_detection_loss(model)
    layer_name, block = resolve_mot_layer(model, args.layer)
    expert_names = tuple(
        EXPERT_NAMES[index] if index < len(EXPERT_NAMES) else f"Expert{index}" for index in range(block.num_experts)
    )
    validator, dataloader = build_validation_loader(
        model,
        data_path,
        args.split,
        device=device,
        imgsz=args.imgsz,
        workers=args.workers,
    )
    selected_indices = select_indices(len(dataloader.dataset), args.max_images, args.seed)
    interpreter = RoutingInterpreter(model)
    probe = RouterProbabilityProbe(layer_name, block)
    rows = []
    try:
        model.eval()
        with torch.inference_mode():
            for index, batch in enumerate(dataloader):
                if index not in selected_indices:
                    continue
                batch = validator.preprocess(batch)
                rows.append(
                    utility_row(
                        model,
                        interpreter,
                        probe,
                        layer_name,
                        batch,
                        imgsz=args.imgsz,
                        temperature=args.utility_temperature,
                        expert_names=expert_names,
                    )
                )
                if len(rows) % 25 == 0 or len(rows) == len(selected_indices):
                    print(f"[utility] processed {len(rows)}/{len(selected_indices)} images")
    finally:
        probe.close()

    if not rows:
        raise SystemExit("no images were evaluated")
    matrix_path = output / "detection_utility_matrix.csv"
    summary = summarize_matrix(rows, expert_names)
    manifest = {
        "protocol": "single-layer forced-expert detection-loss counterfactual",
        "model_sha256": file_sha256(model_path),
        "checkpoint_name": model_path.name,
        "data_config_name": data_path.name,
        "split": args.split,
        "layer": layer_name,
        "expert_names": expert_names,
        "loss_definition": "box + cls + dfl; routed auxiliary loss excluded",
        "utility_target": "softmax(-(forced_loss - min_forced_loss) / temperature)",
        "utility_temperature": args.utility_temperature,
        "image_size": args.imgsz,
        "seed": args.seed,
        "selected_images": len(rows),
    }
    write_csv(matrix_path, rows)
    (output / "utility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[utility] layer={layer_name} images={len(rows)} "
        f"oracle_accuracy={summary['router_oracle_accuracy']:.3f} "
        f"mean_regret={summary['mean_router_regret']:.6f}"
    )
    print(f"[utility] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
