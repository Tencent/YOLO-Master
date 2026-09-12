"""Observe unused candidate quality during a short, unchanged pure TAL training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import ClassVar

import torch

from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner


def candidate_counts(mask_pos, quality, metric, legal, small, eps):
    """Count uncovered GTs with free candidates; never change assignment inputs."""
    uncovered = small & ~mask_pos.bool().any(-1)
    free = ~mask_pos.bool().any(-2)
    free_legal = legal.bool() & free.unsqueeze(1)
    has_free = free_legal.any(-1)
    best_quality = quality.masked_fill(~free_legal, -torch.inf).amax(-1)
    best_metric = metric.masked_fill(~free_legal, -torch.inf).amax(-1)
    eligible = uncovered & has_free
    suppressed = eligible & (best_metric <= eps)
    counts = {
        "small_gt": small.sum(),
        "uncovered": uncovered.sum(),
        "uncovered_without_legal": (uncovered & ~legal.bool().any(-1)).sum(),
        "uncovered_with_free_legal": eligible.sum(),
        "free_positive_ciou": (eligible & (best_quality > 0)).sum(),
        "free_alignment_above_eps": (eligible & (best_metric > eps)).sum(),
        "positive_ciou_but_alignment_below_eps": (suppressed & (best_quality > 0)).sum(),
    }
    for threshold in (0.01, 0.03, 0.05, 0.1, 0.2):
        counts[f"suppressed_ciou_ge_{threshold}"] = (suppressed & (best_quality >= threshold)).sum()
    return {key: int(value.item()) for key, value in counts.items()}


def outside_capacity_counts(boxes, anchors, without_legal, foreground):
    """Bound IoU for boxes decoded from nonnegative distances at existing feature points.

    The tightest enclosing rectangle of a GT and a point maximizes IoU over rectangles
    containing that point. This ignores DFL's finite range, so it is an optimistic bound.
    """
    counts = Counter()
    for batch_idx in range(boxes.shape[0]):
        selected = boxes[batch_idx, without_legal[batch_idx]]
        free = ~foreground[batch_idx].bool()
        for chunk in selected.split(128):
            if not chunk.numel():
                continue
            lower = torch.minimum(chunk[:, None, :2], anchors[None])
            upper = torch.maximum(chunk[:, None, 2:], anchors[None])
            area = (chunk[:, 2:] - chunk[:, :2]).prod(-1)
            bound = area[:, None] / (upper - lower).prod(-1).clamp_min(1e-12)
            best = bound.amax(-1)
            best_free = bound.masked_fill(~free[None], 0).amax(-1)
            counts["outside_gt"] += len(chunk)
            for threshold in (0.5, 0.75, 0.95):
                counts[f"outside_any_iou_cap_ge_{threshold}"] += int((best >= threshold).sum())
                counts[f"outside_free_iou_cap_ge_{threshold}"] += int((best_free >= threshold).sum())
    return dict(counts)


class DiagnosticAssigner(TaskAlignedAssigner):
    """Collect counts without changing masks, target scores, or optimization."""

    records: ClassVar[list] = []
    active_epoch = None

    def get_pos_mask(self, *args, **kwargs):
        """Retain candidate tensors only until the conflict-resolution call."""
        result = super().get_pos_mask(*args, **kwargs)
        self._diagnostic_context = None
        if self.active_epoch is not None:
            if self.stal_candidate_mode != "pure" or self.stal_nwd_weight or self.stal_zero_positive_rescue:
                raise ValueError("Candidate diagnostics require pure TAL with NWD and rescue disabled.")
            boxes, valid = args[3], args[5]
            image_size = torch.as_tensor(kwargs["image_size"], device=boxes.device, dtype=boxes.dtype)
            small = self.small_target_mask(boxes, image_size) & valid.squeeze(-1).bool()
            self._diagnostic_context = (result[3], small, boxes, args[4])
        return result

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Observe final pure assignments before target-score normalization."""
        result = super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes, align_metric)
        if self._diagnostic_context is not None:
            legal, small, boxes, anchors = self._diagnostic_context
            counts = candidate_counts(result[2], overlaps, align_metric, legal, small, self.eps)
            without_legal = small & ~legal.bool().any(-1)
            counts.update(outside_capacity_counts(boxes, anchors, without_legal, result[1]))
            self.records.append({"epoch": self.active_epoch, **counts})
            self._diagnostic_context = None
        return result


def main():
    """Run one bounded diagnostic and save arguments, batch counts, and totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    if args.epochs < 1 or not 0 < args.fraction <= 1:
        raise ValueError("epochs must be positive and fraction must be in (0, 1].")

    import ultralytics.utils.loss as loss_module

    original = loss_module.TaskAlignedAssigner
    DiagnosticAssigner.records = []
    signatures = []
    yolo = YOLO(args.model)

    def start_epoch(trainer):
        DiagnosticAssigner.active_epoch = int(trainer.epoch)

    def end_epoch(trainer):
        DiagnosticAssigner.active_epoch = None

    def record_batch(trainer):
        batch = trainer.batch
        digest = hashlib.sha256()
        for key in ("img", "bboxes", "cls", "batch_idx"):
            digest.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
        signatures.append({"files": batch["im_file"], "sha256": digest.hexdigest()})

    yolo.add_callback("on_train_epoch_start", start_epoch)
    yolo.add_callback("on_train_epoch_end", end_epoch)
    yolo.add_callback("on_train_batch_start", record_batch)
    try:
        loss_module.TaskAlignedAssigner = DiagnosticAssigner
        yolo.train(
            data=args.data,
            epochs=args.epochs,
            fraction=args.fraction,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            pretrained=False,
            optimizer="MuSGD",
            amp=False,
            seed=0,
            deterministic=True,
            mosaic=0,
            close_mosaic=0,
            val=False,
            save=False,
            plots=False,
            patience=0,
            project=str(args.output.parent),
            name=args.output.name,
            stal_candidate_mode="pure",
            stal_enabled=False,
            stal_zero_positive_rescue=False,
            stal_nwd_weight=0,
        )
    finally:
        loss_module.TaskAlignedAssigner = original
        DiagnosticAssigner.active_epoch = None
    totals = Counter()
    for record in DiagnosticAssigner.records:
        totals.update({key: value for key, value in record.items() if key != "epoch"})
    if not DiagnosticAssigner.records:
        raise RuntimeError("No training assignments observed; diagnostic is invalid.")
    output = {
        "protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "totals": dict(totals),
        "batches": DiagnosticAssigner.records,
        "batch_signatures": signatures,
        "interpretation": "Training-only candidate counts; no claim of AP improvement or official VisDrone evaluation.",
    }
    (args.output / "candidate-diagnostics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": dict(totals)}, indent=2))


if __name__ == "__main__":
    main()
