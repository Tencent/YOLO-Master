#!/usr/bin/env python3
"""Compare short FP32 and AMP training from identical YOLO-Master weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and all bytes, preserving element order and dtype."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash a complete named model state including buffers."""
    return hashlib.sha256(
        json.dumps({name: tensor_digest(value) for name, value in sorted(state.items())}).encode()
    ).hexdigest()


def shared_state_trainer(initial_state: dict[str, torch.Tensor]):
    """Restore shared weights after dataset-specific model creation, before optimizer and EMA setup."""

    class SharedStateTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if not initial_state:
                initial_state.update(clone_state(model))
            model.load_state_dict(initial_state, strict=True)
            return model

    return SharedStateTrainer


def batch_signature(batch: dict) -> dict:
    """Identify full ordered detection inputs, including target-to-image associations."""
    return {
        "files": [str(path) for path in batch.get("im_file", ())],
        "tensors": {
            key: tensor_digest(batch[key]) if isinstance(batch.get(key), torch.Tensor) else None
            for key in ("img", "bboxes", "cls", "batch_idx")
        },
    }


def first_batch_mismatch(left: list[str], right: list[str]) -> int | None:
    """Report differing content or the first missing batch, using zero-based indices."""
    return next((i for i, pair in enumerate(zip_longest(left, right)) if pair[0] != pair[1]), None)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model YAML used to create one shared initial state.")
    parser.add_argument("--data", required=True, help="Dataset YAML.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for runs and summary JSON.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-mode", choices=("pure", "fixed", "adaptive"), default="pure")
    parser.add_argument("--rescue", action="store_true")
    parser.add_argument("--rescue-score-floor", type=float, default=0.0)
    parser.add_argument("--rescue-floor-decay-epochs", type=float, default=0.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Reset host and CUDA random generators before constructing each arm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone a model state onto CPU for identical-arm restoration."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def clone_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone trainable parameters onto CPU for update-delta diagnostics."""
    return {name: value.detach().cpu().float().clone() for name, value in model.named_parameters()}


def parameter_delta(initial: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, float | int]:
    """Summarize parameter movement relative to a shared initial state."""
    delta_sq = 0.0
    initial_sq = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        before = initial[name]
        after = parameter.detach().cpu().float()
        delta = after - before
        delta_sq += float(delta.square().sum())
        initial_sq += float(before.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        changed += int(bool(torch.count_nonzero(delta)))
        total += 1
    delta_l2 = math.sqrt(delta_sq)
    initial_l2 = math.sqrt(initial_sq)
    return {
        "parameter_tensors": total,
        "changed_parameter_tensors": changed,
        "parameter_delta_l2": delta_l2,
        "parameter_relative_delta_l2": delta_l2 / max(initial_l2, 1e-12),
        "parameter_max_abs_delta": max_abs,
    }


def last_csv_row(path: Path) -> dict[str, str]:
    """Read the last metrics row when a diagnostic arm emitted one."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def run_arm(args: argparse.Namespace, initial_state: dict[str, torch.Tensor], amp: bool) -> dict:
    """Run one diagnostic arm and collect optimizer/scaler/parameter evidence."""
    seed_everything(args.seed)
    yolo = YOLO(args.model)
    initial_parameters = {}
    arm = "amp" if amp else "fp32"
    trace = {
        "arm": arm,
        "batches": 0,
        "loss_finite": True,
        "loss_items_finite": True,
        "gradient_nonfinite_seen": False,
        "scaler_scales": [],
        "scale_decreases": 0,
        "scale_increases": 0,
        "optimizer_steps": 0,
        "epoch_passes": [],
    }

    def on_train_start(trainer) -> None:
        trainer.final_eval = lambda: None
        trace["initial_state_sha256"] = state_digest(trainer.model.state_dict())
        if trace["initial_state_sha256"] != state_digest(initial_state):
            raise RuntimeError("Training model no longer matches the shared initial state")
        initial_parameters.update(clone_parameters(trainer.model))
        # AMP capability checks may consume random numbers before the dataloader iterator is created.
        # Reset at the shared training boundary so both arms receive identical augmented batches.
        seed_everything(args.seed)
        scale = float(trainer.scaler.get_scale())
        trace["initial_scaler_scale"] = scale
        trace["scaler_scales"].append(scale)

    def on_train_epoch_start(trainer) -> None:
        trace["epoch_passes"].append(
            {
                "epoch": int(trainer.epoch),
                "amp_enabled": bool(trainer.amp),
                "scaler_scale_start": float(trainer.scaler.get_scale()),
                "optimizer_steps_start": int(getattr(trainer, "optimizer_steps", 0)),
                "gradient_nonfinite_seen": False,
                "_batch_signatures": [],
            }
        )

    def on_train_batch_end(trainer) -> None:
        trace["batches"] += 1
        trace["optimizer_steps"] = int(getattr(trainer, "optimizer_steps", 0))
        trace["gradient_nonfinite_seen"] |= bool(getattr(trainer, "_gradient_nonfinite", False))
        current_pass = trace["epoch_passes"][-1]
        current_pass["gradient_nonfinite_seen"] |= bool(getattr(trainer, "_gradient_nonfinite", False))
        diagnostic = getattr(trainer, "_nonfinite_diagnostic", None)
        if diagnostic is not None and diagnostic not in current_pass.setdefault("nonfinite_diagnostics", []):
            current_pass["nonfinite_diagnostics"].append(dict(diagnostic))
        current_pass["optimizer_steps_end"] = int(getattr(trainer, "optimizer_steps", 0))
        current_pass["scaler_scale_end"] = float(trainer.scaler.get_scale())
        loss = getattr(trainer, "loss", None)
        items = getattr(trainer, "loss_items", None)
        if isinstance(loss, torch.Tensor):
            trace["loss_finite"] &= bool(torch.isfinite(loss.detach()).all().item())
        if isinstance(items, torch.Tensor):
            trace["loss_items_finite"] &= bool(torch.isfinite(items.detach()).all().item())
            trace["last_loss_items"] = [float(value) for value in items.detach().float().cpu().reshape(-1)]
        scale = float(trainer.scaler.get_scale())
        previous = trace["scaler_scales"][-1]
        trace["scale_decreases"] += int(scale < previous)
        trace["scale_increases"] += int(scale > previous)
        trace["scaler_scales"].append(scale)

    def on_train_batch_start(trainer) -> None:
        trace["epoch_passes"][-1]["_batch_signatures"].append(batch_signature(trainer.batch))

    yolo.add_callback("on_train_start", on_train_start)
    yolo.add_callback("on_train_epoch_start", on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", on_train_batch_start)
    yolo.add_callback("on_train_batch_end", on_train_batch_end)
    run_name = f"{args.candidate_mode}-{arm}-fraction-{str(args.fraction).replace('.', 'p')}"
    yolo.train(
        trainer=shared_state_trainer(initial_state),
        data=args.data,
        epochs=1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=False,
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        patience=0,
        amp=amp,
        mosaic=0.0,
        plots=False,
        save=False,
        val=False,
        fraction=args.fraction,
        verbose=False,
        project=str(args.output),
        name=run_name,
        exist_ok=True,
        stal_candidate_mode=args.candidate_mode,
        stal_stats=True,
        stal_enabled=False,
        stal_zero_positive_rescue=args.rescue,
        stal_rescue_score_floor=args.rescue_score_floor,
        stal_rescue_floor_decay_epochs=args.rescue_floor_decay_epochs,
    )
    trace["final_scaler_scale"] = float(yolo.trainer.scaler.get_scale())
    if not initial_parameters:
        raise RuntimeError("Training did not reach on_train_start; parameter deltas are unavailable.")
    trace.update(parameter_delta(initial_parameters, yolo.model))
    metrics_path = Path(yolo.trainer.save_dir) / "results.csv"
    trace["results_csv"] = str(metrics_path)
    trace["metrics"] = last_csv_row(metrics_path)
    trace["scaler_scales"] = sorted(set(trace["scaler_scales"]))
    for epoch_pass in trace["epoch_passes"]:
        batch_signatures = epoch_pass.pop("_batch_signatures")
        signature_payloads = [
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode() for signature in batch_signatures
        ]
        epoch_pass["batches"] = len(batch_signatures)
        epoch_pass["batch_signature_sha256"] = hashlib.sha256(b"\n".join(signature_payloads)).hexdigest()
        epoch_pass["batch_signature_items_sha256"] = [
            hashlib.sha256(payload).hexdigest() for payload in signature_payloads
        ]
        epoch_pass["batch_signature_preview"] = batch_signatures[:3]
    primary_pass = trace["epoch_passes"][0]
    trace["batch_signature_sha256"] = primary_pass["batch_signature_sha256"]
    trace["batch_signature_items_sha256"] = primary_pass["batch_signature_items_sha256"]
    trace["batch_signature_preview"] = primary_pass["batch_signature_preview"]
    trace["epoch_pass_count"] = len(trace["epoch_passes"])
    trace["precision_fallback_seen"] = (
        amp and primary_pass["amp_enabled"] and any(not item["amp_enabled"] for item in trace["epoch_passes"][1:])
    )
    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trace


def main() -> int:
    """Run both precision arms and emit a machine-readable diagnostic report."""
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")
    if args.rescue and args.candidate_mode != "adaptive":
        raise ValueError("--rescue requires --candidate-mode adaptive.")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    initial_state = {}
    arms = [run_arm(args, initial_state, amp=False), run_arm(args, initial_state, amp=True)]
    report = {
        "protocol": {
            "model": args.model,
            "data": args.data,
            "device": args.device,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "fraction": args.fraction,
            "seed": args.seed,
            "candidate_mode": args.candidate_mode,
            "rescue": args.rescue,
            "rescue_score_floor": args.rescue_score_floor,
            "rescue_floor_decay_epochs": args.rescue_floor_decay_epochs,
            "shared_initial_state": arms[0]["initial_state_sha256"] == arms[1]["initial_state_sha256"],
        },
        "arms": arms,
        "comparison": {
            "identical_batch_signatures": arms[0]["batch_signature_sha256"] == arms[1]["batch_signature_sha256"],
            "first_mismatched_batch": first_batch_mismatch(
                arms[0]["batch_signature_items_sha256"], arms[1]["batch_signature_items_sha256"]
            ),
            "both_losses_finite": all(arm["loss_finite"] and arm["loss_items_finite"] for arm in arms),
            "any_nonfinite_gradient": any(arm["gradient_nonfinite_seen"] for arm in arms),
            "amp_precision_fallback_seen": arms[1]["precision_fallback_seen"],
        },
    }
    report_path = args.output / "amp-training-diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
