#!/usr/bin/env python3
"""Diagnose persistent zero mAP during VOC detection fine-tuning.

The default mode is read-only and safe to run while training. It inspects the
run configuration, results.csv, and a stable copy of last_healthy.pt. Optional
forward probes, sample prediction, and full validation are disabled by default
because they consume compute resources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import unwrap_model


ADAPTER_TOKENS = ("lora_", "adapter", "modules_to_save", "hada_", "oft_", "boft_", "hra_")
TRAIN_ARG_KEYS = (
    "model", "data", "epochs", "batch", "imgsz", "device", "optimizer",
    "effective_optimizer", "effective_optimizer_lrs", "lr0", "lrf", "warmup_epochs", "nbs", "cos_lr",
    "lora_type", "lora_r", "lora_alpha", "lora_include_head", "lora_use_rslora", "lora_use_dora",
    "lora_backend", "lora_lr_mult", "requested_lora_lr_mult", "lora_layer_decay", "requested_lora_layer_decay",
    "lora_alpha_warmup", "requested_lora_alpha_warmup", "lora_ortho_weight", "requested_lora_ortho_weight",
    "moe_aux_gain", "mixture_aux_budget",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Ultralytics run directory containing args.yaml.")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML override used by optional validation.")
    parser.add_argument("--expected-nc", type=int, default=20, help="Expected number of dataset classes.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for optional probe/validation, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size override for optional inference.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for optional validation.")
    parser.add_argument("--workers", type=int, default=2, help="Workers for optional validation.")
    parser.add_argument("--conf", type=float, default=1e-5, help="Low confidence threshold for optional inference.")
    parser.add_argument("--rows", type=int, default=5, help="Number of recent results.csv rows to print.")
    parser.add_argument("--probe-forward", action="store_true", help="Run a synthetic forward pass for online and EMA.")
    parser.add_argument("--run-val", action="store_true", help="Run full low-confidence validation for online and EMA.")
    parser.add_argument("--sample-image", type=Path, default=None, help="Optional real image for prediction diagnostics.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path for a machine-readable report.")
    parser.add_argument("--strict", action="store_true", help="Exit with status 2 when a critical finding is detected.")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stable_copy(source: Path, destination: Path, retries: int = 8, delay: float = 0.25) -> Path:
    """Copy a checkpoint only when its size and mtime remain stable across the copy."""
    last_error = None
    for _ in range(retries):
        try:
            before = source.stat()
            shutil.copy2(source, destination)
            after = source.stat()
            copied = destination.stat()
            if (
                before.st_size == after.st_size == copied.st_size
                and before.st_mtime_ns == after.st_mtime_ns
            ):
                return destination
        except OSError as exc:
            last_error = exc
        time.sleep(delay)
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"Checkpoint changed while being copied after {retries} attempts{detail}")


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_column(rows: list[dict[str, str]], token: str) -> str | None:
    if not rows:
        return None
    token = token.lower()
    return next((key for key in rows[0] if token in key.lower()), None)


def tensor_is_finite(value: torch.Tensor) -> bool:
    return not value.is_floating_point() or bool(torch.isfinite(value).all().item())


def module_diagnostics(module: nn.Module, expected_nc: int) -> tuple[dict[str, Any], str | None]:
    model = unwrap_model(module)
    state = model.state_dict()
    floating = [value for value in state.values() if isinstance(value, torch.Tensor) and value.is_floating_point()]
    nonfinite_keys = [
        key for key, value in state.items() if isinstance(value, torch.Tensor) and not tensor_is_finite(value)
    ]
    heads = [(name, child) for name, child in model.named_modules() if isinstance(child, Detect)]
    head_name, head = heads[-1] if heads else (None, None)
    names = getattr(model, "names", {})
    names_count = len(names) if isinstance(names, (dict, list, tuple)) else None

    adapter_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(token in name.lower() for token in ADAPTER_TOKENS)
    ]
    trainable_adapter_parameters = [(name, value) for name, value in adapter_parameters if value.requires_grad]
    head_trainable = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad) if head else 0
    head_total = sum(parameter.numel() for parameter in head.parameters()) if head else 0

    cls_biases = []
    if head is not None:
        for name, child in head.named_modules():
            if isinstance(child, nn.Conv2d) and child.out_channels == head.nc and child.bias is not None:
                bias = child.bias.detach().float()
                cls_biases.append(
                    {
                        "name": name,
                        "min": float(bias.min()),
                        "max": float(bias.max()),
                        "mean": float(bias.mean()),
                        "max_sigmoid": float(bias.sigmoid().max()),
                    }
                )

    es_moe_routing = [
        {
            "name": name,
            "use_sparse_inference": bool(getattr(child, "use_sparse_inference", True)),
            "use_top_k": bool(getattr(child, "use_top_k", False)),
            "top_k": int(getattr(child, "top_k", getattr(child, "num_experts", 1))),
            "num_experts": int(getattr(child, "num_experts", len(getattr(child, "experts", ())))),
            "dynamic_threshold": float(getattr(child, "dynamic_threshold", 0.0)),
            "effective_sparse": bool(
                getattr(child, "use_sparse_inference", True)
                and getattr(child, "use_top_k", False)
                and getattr(child, "top_k", getattr(child, "num_experts", 1))
                < getattr(child, "num_experts", 1)
            ),
        }
        for name, child in model.named_modules()
        if isinstance(child, ES_MOE)
    ]

    info = {
        "type": type(model).__name__,
        "finite": not nonfinite_keys,
        "nonfinite_keys": nonfinite_keys[:20],
        "floating_tensor_count": len(floating),
        "names_count": names_count,
        "names": dict(names) if isinstance(names, dict) and len(names) <= 100 else None,
        "head_type": type(head).__name__ if head else None,
        "head_name": head_name,
        "head_nc": getattr(head, "nc", None),
        "head_total_params": head_total,
        "head_trainable_params": head_trainable,
        "adapter_total_params": sum(value.numel() for _, value in adapter_parameters),
        "adapter_trainable_params": sum(value.numel() for _, value in trainable_adapter_parameters),
        "adapter_parameter_tensors": len(adapter_parameters),
        "adapter_trainable_tensors": len(trainable_adapter_parameters),
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "mixture_ema": getattr(model, "_mixture_loss_ema_buf", None).detach().float().tolist()
        if isinstance(getattr(model, "_mixture_loss_ema_buf", None), torch.Tensor)
        else None,
        "classification_biases": cls_biases,
        "es_moe_routing": es_moe_routing,
    }

    critical = None
    if head is None:
        critical = "No Detect head was found in the checkpoint model."
    elif int(head.nc) != expected_nc:
        critical = f"Detection head nc={head.nc}, expected {expected_nc}."
    elif names_count != expected_nc:
        critical = f"Model names count={names_count}, expected {expected_nc}."
    elif nonfinite_keys:
        critical = f"Checkpoint contains non-finite tensors, e.g. {nonfinite_keys[0]}."
    return info, critical


def compare_states(online: nn.Module, ema: nn.Module) -> dict[str, Any]:
    online_state = unwrap_model(online).state_dict()
    ema_state = unwrap_model(ema).state_dict()
    all_diffs, head_diffs, adapter_diffs = [], [], []
    compared = 0
    for key, online_value in online_state.items():
        ema_value = ema_state.get(key)
        if not (
            isinstance(online_value, torch.Tensor)
            and isinstance(ema_value, torch.Tensor)
            and online_value.shape == ema_value.shape
            and online_value.is_floating_point()
        ):
            continue
        difference = float((online_value.float() - ema_value.float()).abs().max())
        all_diffs.append(difference)
        compared += 1
        if any(token in key.lower() for token in ADAPTER_TOKENS):
            adapter_diffs.append(difference)
        if any(token in key.lower() for token in ("cv2", "cv3", "dfl", "detect")):
            head_diffs.append(difference)
    return {
        "compared_tensors": compared,
        "max_difference": max(all_diffs, default=0.0),
        "mean_tensor_max_difference": sum(all_diffs) / max(len(all_diffs), 1),
        "head_max_difference": max(head_diffs, default=0.0),
        "adapter_max_difference": max(adapter_diffs, default=0.0),
        "identical_tensor_count": sum(difference == 0.0 for difference in all_diffs),
    }


def optimizer_diagnostics(optimizer: Any) -> dict[str, Any]:
    if not isinstance(optimizer, dict):
        return {"present": False}
    groups = []
    raw_groups = optimizer.get("param_groups", [])
    for group in raw_groups:
        groups.append(
            {
                "name": group.get("param_group"),
                "lr": group.get("lr"),
                "initial_lr": group.get("initial_lr"),
                "weight_decay": group.get("weight_decay"),
                "parameters": len(group.get("params", [])),
                "use_muon": group.get("use_muon"),
            }
        )
    state_keys = sorted(
        {
            key
            for state in optimizer.get("state", {}).values()
            if isinstance(state, dict)
            for key in state
        }
    )
    if any("use_muon" in group for group in raw_groups):
        effective_type = "MuSGD"
    elif {"exp_avg", "exp_avg_sq"} <= set(state_keys):
        effective_type = "Adam-family"
    elif "momentum_buffer" in state_keys:
        effective_type = "SGD/RMSProp-family"
    else:
        effective_type = "unknown"
    active_groups = [group for group in groups if group["parameters"] > 0]
    active_lrs = [as_float(group["lr"]) for group in active_groups]
    active_lrs = [value for value in active_lrs if value is not None]
    adapter_lrs = [
        as_float(group["lr"])
        for group in active_groups
        if group["name"] == "adapter" and as_float(group["lr"]) is not None
    ]
    return {
        "present": True,
        "effective_type": effective_type,
        "state_entries": len(optimizer.get("state", {})),
        "state_keys": state_keys,
        "active_lr_min": min(active_lrs) if active_lrs else None,
        "active_lr_max": max(active_lrs) if active_lrs else None,
        "active_adapter_lr_min": min(adapter_lrs) if adapter_lrs else None,
        "active_adapter_lr_max": max(adapter_lrs) if adapter_lrs else None,
        "groups": groups,
    }


def first_prediction_tensor(output: Any, expected_nc: int) -> torch.Tensor | None:
    expected_channels = expected_nc + 4
    if isinstance(output, (list, tuple)) and output:
        primary = output[0]
        if isinstance(primary, torch.Tensor) and primary.ndim == 3 and expected_channels in primary.shape[1:]:
            return primary
    candidates = []

    def collect(value: Any) -> None:
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            candidates.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(output)
    if not candidates:
        return None
    exact = [value for value in candidates if value.shape[1] == expected_channels or value.shape[-1] == expected_channels]
    if exact:
        return max(exact, key=lambda value: value.numel())
    valid = [value for value in candidates if value.shape[1] >= expected_channels or value.shape[-1] >= expected_channels]
    return min(valid or candidates, key=lambda value: value.shape[1] if value.shape[1] > 1 else value.shape[-1])


def prediction_scores(prediction: torch.Tensor, expected_nc: int) -> torch.Tensor | None:
    """Extract class scores from decoded predictions regardless of tensor layout."""
    channels = expected_nc + 4
    if prediction.ndim != 3:
        return None
    if prediction.shape[1] == channels:
        return prediction[:, 4 : 4 + expected_nc, :]
    if prediction.shape[-1] == channels:
        return prediction[..., 4 : 4 + expected_nc]
    return None


def forward_probe(module: nn.Module, expected_nc: int, imgsz: int, device: str, conf: float) -> dict[str, Any]:
    model = unwrap_model(module).float().to(device).eval()
    channels = int((getattr(model, "yaml", {}) or {}).get("channels", 3))
    sample = torch.zeros(1, channels, imgsz, imgsz, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(sample)
    prediction = first_prediction_tensor(output, expected_nc)
    elapsed = time.perf_counter() - started
    if prediction is None:
        return {"ok": False, "reason": "No rank-3 prediction tensor found", "seconds": elapsed}
    prediction = prediction.detach().float().cpu()
    scores = prediction_scores(prediction, expected_nc)
    if scores is None:
        return {"ok": False, "reason": f"Unexpected prediction shape {tuple(prediction.shape)}", "seconds": elapsed}
    return {
        "ok": True,
        "shape": tuple(prediction.shape),
        "seconds": elapsed,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "scores_over_conf": int((scores > conf).sum()),
        "scores_over_0_001": int((scores > 0.001).sum()),
    }


def materialize_variant(checkpoint: dict[str, Any], target: str, destination: Path) -> Path:
    model = checkpoint.get(target)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Checkpoint target '{target}' is not a torch module")
    payload = dict(checkpoint)
    payload["model"] = model
    payload["ema"] = None
    payload["optimizer"] = None
    payload["scaler"] = None
    torch.save(payload, destination)
    return destination


def run_runtime_checks(
    checkpoint: dict[str, Any],
    targets: list[str],
    workdir: Path,
    data: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    conf: float,
    run_val: bool,
    sample_image: Path | None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    results = {}
    for target in targets:
        variant = materialize_variant(checkpoint, target, workdir / f"{target}.pt")
        yolo = YOLO(str(variant))
        target_result = {}
        if sample_image is not None:
            predictions = yolo.predict(
                source=str(sample_image), imgsz=imgsz, conf=conf, device=device, save=False, verbose=False
            )
            boxes = predictions[0].boxes
            target_result["sample_prediction"] = {
                "detections": len(boxes),
                "max_confidence": float(boxes.conf.max()) if len(boxes) else 0.0,
                "classes": boxes.cls.int().cpu().tolist() if len(boxes) else [],
            }
        if run_val:
            metrics = yolo.val(
                data=data,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                conf=conf,
                plots=False,
                save_json=False,
                project=str(workdir),
                name=f"val_{target}",
                exist_ok=True,
                verbose=False,
            )
            target_result["validation"] = {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        results[target] = target_result
    return results


def print_section(title: str, value: Any) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    else:
        print(value)


def main() -> int:
    cli = parse_args()
    run_dir = cli.run_dir.expanduser().resolve()
    args_path, results_path = run_dir / "args.yaml", run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    healthy_path, last_path, best_path = (
        weights_dir / "last_healthy.pt",
        weights_dir / "last.pt",
        weights_dir / "best.pt",
    )
    missing = [str(path) for path in (args_path, results_path, healthy_path) if not path.exists()]
    if missing:
        print("Missing required diagnostic files:", *missing, sep="\n  - ")
        return 1

    train_args = YAML.load(args_path)
    rows = read_results(results_path)
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "files": {
            path.name: {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
            for path in (args_path, results_path, healthy_path, last_path, best_path)
        },
        "train_args": {key: train_args.get(key) for key in TRAIN_ARG_KEYS},
        "findings": [],
    }

    metric_columns = {
        "precision": find_column(rows, "metrics/precision"),
        "recall": find_column(rows, "metrics/recall"),
        "map50": find_column(rows, "metrics/map50(b)"),
        "map50_95": find_column(rows, "metrics/map50-95"),
    }
    if metric_columns["map50"] is None:
        metric_columns["map50"] = next(
            (key for key in (rows[0] if rows else {}) if "map50" in key.lower() and "95" not in key.lower()), None
        )
    lr_columns = [key for key in (rows[0] if rows else {}) if key.startswith("lr/")]
    loss_columns = [key for key in (rows[0] if rows else {}) if key.startswith("train/")]
    recent_rows = []
    for row in rows[-max(cli.rows, 1) :]:
        recent_rows.append(
            {
                key: row.get(key)
                for key in ["epoch", *loss_columns, *[value for value in metric_columns.values() if value], *lr_columns]
            }
        )
    map50_values = [as_float(row.get(metric_columns["map50"])) for row in rows] if metric_columns["map50"] else []
    map50_values = [value for value in map50_values if value is not None]
    first_nonzero = next((index + 1 for index, value in enumerate(map50_values) if value > 0), None)
    last_lrs = [as_float(rows[-1].get(key)) for key in lr_columns] if rows else []
    last_lrs = [value for value in last_lrs if value is not None]
    report["results"] = {
        "epochs_recorded": len(rows),
        "metric_columns": metric_columns,
        "loss_columns": loss_columns,
        "lr_columns": lr_columns,
        "first_nonzero_map50_epoch": first_nonzero,
        "last_map50": map50_values[-1] if map50_values else None,
        "all_recorded_map50_zero": bool(map50_values) and all(value == 0 for value in map50_values),
        "last_lr_min": min(last_lrs) if last_lrs else None,
        "last_lr_max": max(last_lrs) if last_lrs else None,
        "recent_rows": recent_rows,
    }
    if len(rows) >= 10 and report["results"]["all_recorded_map50_zero"]:
        report["findings"].append(
            {"severity": "critical", "message": f"mAP50 is exactly zero for all {len(rows)} recorded epochs."}
        )
    warmup_epochs = as_float(train_args.get("warmup_epochs")) or 0.0
    if len(rows) > warmup_epochs + 2 and last_lrs and max(last_lrs) < 1e-5:
        report["findings"].append(
            {"severity": "critical", "message": f"Learning rate remains below 1e-5 after warmup: {last_lrs}."}
        )

    runtime_requested = cli.run_val or cli.sample_image is not None
    data = cli.data or str(train_args.get("data") or "")
    imgsz = int(cli.imgsz or train_args.get("imgsz") or 640)
    if runtime_requested and not data:
        raise ValueError("Optional runtime checks require --data or a valid data entry in args.yaml")
    if runtime_requested and cli.device != "cpu":
        report["findings"].append(
            {
                "severity": "warning",
                "message": "GPU runtime checks were requested. Run them only when enough VRAM is free or training is paused.",
            }
        )

    with tempfile.TemporaryDirectory(prefix="voc-zero-map-") as temporary:
        temp_dir = Path(temporary)
        checkpoint_copy = stable_copy(healthy_path, temp_dir / healthy_path.name)
        checkpoint = torch_load(checkpoint_copy, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
        checkpoint_args = checkpoint.get("train_args")
        if isinstance(checkpoint_args, dict):
            # args.yaml is written before adapter setup; checkpoint metadata is
            # authoritative for effective PEFT and optimizer settings.
            train_args = {**train_args, **checkpoint_args}
            report["train_args"] = {key: train_args.get(key) for key in TRAIN_ARG_KEYS}
        report["checkpoint"] = {
            "epoch_zero_based": checkpoint.get("epoch"),
            "best_fitness": checkpoint.get("best_fitness"),
            "updates": checkpoint.get("updates"),
            "has_online_model": isinstance(checkpoint.get("model"), nn.Module),
            "has_ema": isinstance(checkpoint.get("ema"), nn.Module),
            "optimizer": optimizer_diagnostics(checkpoint.get("optimizer")),
            "scaler_present": checkpoint.get("scaler") is not None,
        }
        optimizer_info = report["checkpoint"]["optimizer"]
        peft_active = int(train_args.get("lora_r") or 0) > 0 or str(train_args.get("lora_type") or "").lower() in {
            "oft",
            "boft",
            "ia3",
            "hra",
        }
        if peft_active and str(train_args.get("optimizer") or "").lower() == "auto" and optimizer_info.get(
            "effective_type"
        ) == "MuSGD":
            report["findings"].append(
                {
                    "severity": "critical",
                    "message": (
                        "optimizer=auto selected MuSGD for active PEFT adapters. The effective adapter LR depends on "
                        "planned epochs and is commonly too large for LoRA/DoRA; use AdamW or the patched auto policy."
                    ),
                }
            )
        adapter_lr_max = optimizer_info.get("active_adapter_lr_max")
        if peft_active and isinstance(adapter_lr_max, (int, float)) and adapter_lr_max > 0.003:
            report["findings"].append(
                {
                    "severity": "critical",
                    "message": f"Active adapter LR is {adapter_lr_max:.6g}, above the 0.003 PEFT safety threshold.",
                }
            )
        inspected = {}
        for target in ("model", "ema"):
            module = checkpoint.get(target)
            if isinstance(module, nn.Module):
                inspected[target], critical = module_diagnostics(module, cli.expected_nc)
                if critical:
                    report["findings"].append({"severity": "critical", "target": target, "message": critical})
        report["models"] = inspected

        all_es_moe = [
            (target, item)
            for target, info in inspected.items()
            for item in info.get("es_moe_routing", [])
        ]
        effective_sparse = [(target, item) for target, item in all_es_moe if item["effective_sparse"]]
        if effective_sparse and len(rows) >= 2 and (report["results"].get("last_map50") or 0.0) < 0.01:
            report["findings"].append(
                {
                    "severity": "critical",
                    "message": (
                        f"Found {len(effective_sparse)} ES_MOE blocks using effective sparse eval while mAP50 remains "
                        "near zero. Compare against dense eval; sparse dispatch must be explicitly requested and "
                        "retained routing weights must be normalized."
                    ),
                }
            )
        legacy_risk = [
            (target, item)
            for target, item in all_es_moe
            if item["use_sparse_inference"] and not item["use_top_k"] and item["dynamic_threshold"] > 0
        ]
        if legacy_risk:
            report["findings"].append(
                {
                    "severity": "warning",
                    "message": (
                        f"Found {len(legacy_risk)} all-expert ES_MOE blocks with a nonzero dynamic threshold. "
                        "Current code keeps these dense, but checkpoints produced before the ES_MOE path fix may have "
                        "validated with threshold-pruned sparse dispatch."
                    ),
                }
            )
        thresholded_sparse = [
            (target, item)
            for target, item in all_es_moe
            if item["effective_sparse"] and item["dynamic_threshold"] > 0
        ]
        if thresholded_sparse:
            report["findings"].append(
                {
                    "severity": "warning",
                    "message": (
                        f"Found {len(thresholded_sparse)} explicit sparse ES_MOE blocks with dynamic pruning. "
                        "Verify retained routing weights are renormalized after threshold pruning."
                    ),
                }
            )

        online = checkpoint.get("model")
        ema = checkpoint.get("ema")
        if isinstance(online, nn.Module):
            online_info = inspected.get("model", {})
            if int(train_args.get("lora_r") or 0) > 0 and online_info.get("adapter_total_params", 0) == 0:
                report["findings"].append(
                    {"severity": "critical", "message": "LoRA is configured but no adapter parameters exist."}
                )
            if online_info.get("head_trainable_params", 0) == 0:
                report["findings"].append(
                    {"severity": "critical", "message": "The online detection head has zero trainable parameters."}
                )
        if isinstance(online, nn.Module) and isinstance(ema, nn.Module):
            report["online_ema_difference"] = compare_states(online, ema)
            if len(rows) >= 2 and report["online_ema_difference"]["max_difference"] == 0.0:
                report["findings"].append(
                    {"severity": "warning", "message": "Online and EMA floating states are exactly identical after training."}
                )

        if cli.probe_forward:
            report["forward_probe"] = {}
            for target in ("model", "ema"):
                module = checkpoint.get(target)
                if isinstance(module, nn.Module):
                    try:
                        report["forward_probe"][target] = forward_probe(
                            module, cli.expected_nc, imgsz, cli.device, cli.conf
                        )
                    except Exception as exc:
                        report["forward_probe"][target] = {
                            "ok": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }

        targets = [target for target in ("model", "ema") if isinstance(checkpoint.get(target), nn.Module)]
        if runtime_requested:
            report["runtime_checks"] = run_runtime_checks(
                checkpoint=checkpoint,
                targets=targets,
                workdir=temp_dir,
                data=data,
                imgsz=imgsz,
                batch=cli.batch,
                workers=cli.workers,
                device=cli.device,
                conf=cli.conf,
                run_val=cli.run_val,
                sample_image=cli.sample_image,
            )
            online_val = report["runtime_checks"].get("model", {}).get("validation", {})
            ema_val = report["runtime_checks"].get("ema", {}).get("validation", {})
            if online_val and ema_val and online_val.get("map50", 0) > 0 and ema_val.get("map50", 0) == 0:
                report["findings"].append(
                    {"severity": "critical", "message": "Online model has nonzero mAP50 but EMA validation is zero."}
                )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    report["findings"].sort(key=lambda item: severity_order.get(item.get("severity", "info"), 9))
    print_section("RUN", report["run_dir"])
    print_section("TRAIN ARGS", report["train_args"])
    print_section("RECENT RESULTS", report["results"])
    print_section("CHECKPOINT", report["checkpoint"])
    print_section("MODEL / EMA", report["models"])
    if "online_ema_difference" in report:
        print_section("ONLINE VS EMA", report["online_ema_difference"])
    if "forward_probe" in report:
        print_section("FORWARD PROBE", report["forward_probe"])
    if "runtime_checks" in report:
        print_section("RUNTIME CHECKS", report["runtime_checks"])
    print_section("FINDINGS", report["findings"] or [{"severity": "info", "message": "No static critical issue found."}])

    if cli.output_json:
        output = cli.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nJSON report: {output}")
    return 2 if cli.strict and any(item.get("severity") == "critical" for item in report["findings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
