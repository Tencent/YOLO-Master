#!/usr/bin/env python3
"""Collect real COCO128/MPS calibration data and refit the PEFT Planner.

The runner is manifest-driven and writes one atomic result record after every
training run, so an interrupted 30+ experiment calibration can be resumed.
Each PEFT result is compared with a Full-SFT baseline trained with the same
model, seed, image size, epoch count, and optimizer settings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

# The legacy OpenAI CLIP package imports ``packaging`` from pkg_resources,
# while modern setuptools no longer re-exports it. Keep the compatibility
# shim local to this experiment runner rather than mutating the environment.
try:
    import packaging as _packaging
    import pkg_resources as _pkg_resources

    if not hasattr(_pkg_resources, "packaging"):
        _pkg_resources.packaging = _packaging
except ImportError:
    pass

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    LOVODataCollector,
    LOVODataPoint,
    LOVOValidator,
    PEFTPlanner,
)
from ultralytics.utils.torch_utils import unwrap_model

assert str(REPO_ROOT) in ultralytics.__file__, (
    f"Expected the current checkout, got ultralytics from {ultralytics.__file__}"
)
SETTINGS["wandb"] = False


DEFAULT_MODELS = (
    "yolo8n=/Users/gatilin/PycharmProjects/YOLO-Master-lora-v260501/yolov8n.pt",
    "yolo11n=/Users/gatilin/PycharmProjects/YOLO-Master-lora-v260501/yolo11n.pt",
    "yolo11s=/Users/gatilin/PycharmProjects/YOLO-Master-lora-v260501/yolo11s.pt",
    "yolo12n=/Users/gatilin/PycharmProjects/YOLO-Master-lora-v260501/yolo12n.pt",
    "yolo12s=/Users/gatilin/PycharmProjects/YOLO-Master-v260130/yolo12s.pt",
    "yolo26n=/Users/gatilin/PycharmProjects/YOLO-Master-optim-training-v260615/yolo26n.pt",
)
RANKS = (4, 8, 16)
RANK_VARIANTS = ("lora", "dora", "loha")
RANKLESS_VARIANTS = ("ia3",)
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: Path
    sha256: str


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_name: str
    weights: str
    weights_sha256: str
    variant: str
    rank: int
    seed: int
    epochs: int
    imgsz: int
    batch: int


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_models(values: list[str]) -> list[ModelSpec]:
    models = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Model must use NAME=/absolute/path.pt syntax: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or not path.is_file():
            raise FileNotFoundError(f"Model weights not found for {name or value}: {path}")
        models.append(ModelSpec(name=name, weights=path, sha256=sha256_file(path)))
    return models


def build_matrix(models: list[ModelSpec], args: argparse.Namespace) -> list[ExperimentSpec]:
    matrix = []
    for model in models:
        common = {
            "model_name": model.name,
            "weights": str(model.weights),
            "weights_sha256": model.sha256,
            "seed": args.seed,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
        }
        matrix.append(ExperimentSpec(experiment_id=f"{model.name}__full", variant="full", rank=0, **common))
        if args.smoke:
            matrix.append(ExperimentSpec(experiment_id=f"{model.name}__lora_r4", variant="lora", rank=4, **common))
            continue
        for variant in RANK_VARIANTS:
            for rank in RANKS:
                matrix.append(
                    ExperimentSpec(
                        experiment_id=f"{model.name}__{variant}_r{rank}", variant=variant, rank=rank, **common
                    )
                )
        for variant in RANKLESS_VARIANTS:
            matrix.append(ExperimentSpec(experiment_id=f"{model.name}__{variant}", variant=variant, rank=1, **common))
    return matrix


def fingerprint_dict(model: torch.nn.Module) -> dict[str, float]:
    return asdict(ArchitectureFingerprint.compute(unwrap_model(model)))


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def adapter_signature(model: torch.nn.Module) -> dict[str, Any]:
    names = [name.lower() for name, _ in model.named_parameters()]
    return {
        "has_lora_a": any("lora_a" in name for name in names),
        "has_lora_b": any("lora_b" in name for name in names),
        "has_dora_magnitude": any("magnitude_vector" in name for name in names),
        "has_loha": any("hada" in name for name in names),
        "has_ia3": any("ia3" in name for name in names),
        "adapter_parameter_tensors": sum(
            any(token in name for token in ("lora_", "hada", "ia3", "magnitude_vector")) for name in names
        ),
    }


def extract_metrics(results: Any) -> dict[str, float]:
    source = getattr(results, "results_dict", None)
    if not source and isinstance(results, dict):
        source = results
    source = source or {}
    metrics = {}
    for key in METRIC_KEYS:
        value = source.get(key)
        if value is not None:
            number = float(value)
            if math.isfinite(number):
                metrics[key] = number
    return metrics


def trainer_metrics(trainer: Any) -> dict[str, float]:
    metrics = dict(getattr(trainer, "metrics", {}) or {})
    fitness = getattr(trainer, "fitness", None)
    if fitness is not None:
        metrics["fitness"] = fitness
    return extract_metrics(metrics)


def peft_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    if spec.variant == "full":
        return {}
    common = {
        "lora_backend": "peft",
        "lora_planner_enabled": False,
        "lora_dropout": 0.05,
        "lora_use_rslora": True,
        "lora_gradient_checkpointing": False,
        "lora_alpha_warmup": 0,
        "lora_layer_decay": 0.0,
        "lora_ortho_weight": 0.0,
        "lora_include_attention": True,
    }
    if spec.variant == "ia3":
        return {**common, "lora_type": "ia3", "lora_r": 0}
    return {
        **common,
        "lora_type": "lora" if spec.variant == "dora" else spec.variant,
        "lora_r": spec.rank,
        "lora_alpha": spec.rank * 2,
        "lora_use_dora": spec.variant == "dora",
    }


def verify_adapter(spec: ExperimentSpec, signature: dict[str, Any], trainable: int, total: int) -> None:
    if spec.variant == "full":
        if trainable <= 0 or trainable < total * 0.8:
            raise RuntimeError(f"Full-SFT exposed only {trainable}/{total} trainable parameters")
        return
    expected = {
        "lora": "has_lora_a",
        "dora": "has_dora_magnitude",
        "loha": "has_loha",
        "ia3": "has_ia3",
    }[spec.variant]
    if not signature.get(expected):
        raise RuntimeError(f"{spec.variant} adapter signature missing: {signature}")
    if trainable <= 0 or trainable >= total:
        raise RuntimeError(f"{spec.variant} trainable parameters are invalid: {trainable}/{total}")


def run_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> dict[str, Any]:
    started = utc_now()
    started_perf = time.perf_counter()
    record = {
        **asdict(spec),
        "status": "running",
        "started_at": started,
        "device_requested": args.device,
        "dataset": str(args.data),
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "error": None,
    }
    model = None
    callback_state: dict[str, Any] = {}
    try:
        model = YOLO(spec.weights)
        record["fingerprint"] = fingerprint_dict(model.model)
        record["architecture_family"] = ArchitectureFingerprint._detect_architecture_family(model.model)
        base_total, base_trainable = count_params(model.model)
        record["params_before"] = {"total": base_total, "trainable": base_trainable}

        def capture_setup(trainer) -> None:
            inner = unwrap_model(trainer.model)
            total, trainable = count_params(inner)
            signature = adapter_signature(inner)
            verify_adapter(spec, signature, trainable, total)
            callback_state.update(
                {
                    "device": str(trainer.device),
                    "params_total": total,
                    "params_trainable": trainable,
                    "trainable_pct": 100.0 * trainable / total,
                    "adapter_signature": signature,
                    "lora_runtime_metadata": getattr(inner, "lora_runtime_metadata", {}) or {},
                }
            )

        def capture_epoch_metrics(trainer) -> None:
            metrics = trainer_metrics(trainer)
            if "metrics/mAP50-95(B)" in metrics:
                callback_state["online_metrics"] = metrics

        def capture_final_metrics(trainer) -> None:
            metrics = trainer_metrics(trainer)
            if "metrics/mAP50-95(B)" in metrics:
                callback_state["final_metrics"] = metrics

        model.add_callback("on_train_start", capture_setup)
        model.add_callback("on_fit_epoch_end", capture_epoch_metrics)
        model.add_callback("on_train_end", capture_final_metrics)
        train_args = {
            "data": str(args.data),
            "epochs": spec.epochs,
            "batch": spec.batch,
            "imgsz": spec.imgsz,
            "device": args.device,
            "workers": args.workers,
            "seed": spec.seed,
            "deterministic": True,
            "patience": 0,
            "plots": False,
            "save": False,
            "verbose": False,
            "project": str(args.project),
            "name": spec.experiment_id,
            "exist_ok": True,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "warmup_epochs": 0.0,
            "close_mosaic": 0,
            "lora_lr_mult": 1.0,
        }
        train_args.update(peft_kwargs(spec))
        results = model.train(**train_args)
        record.update(callback_state)
        if record.get("device") != "mps":
            raise RuntimeError(f"Training did not execute on MPS: {record.get('device')}")
        metrics = callback_state.get("final_metrics") or callback_state.get("online_metrics") or extract_metrics(results)
        if "metrics/mAP50-95(B)" not in metrics:
            raise RuntimeError(f"Final mAP50-95 metric is missing or non-finite: {metrics}")
        record["metrics"] = metrics
        record["save_dir"] = str(model.trainer.save_dir)
        record["status"] = "success"
    except Exception as exc:
        record.update(callback_state)
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        record["finished_at"] = utc_now()
        record["elapsed_sec"] = round(time.perf_counter() - started_perf, 3)
        if args.cleanup_runs:
            shutil.rmtree(args.project / spec.experiment_id / "weights", ignore_errors=True)
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return record


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a result list in {path}")
    return payload


def rank_aware_lovo(points: list[LOVODataPoint]) -> dict[str, Any]:
    predictions = []
    for index, left_out in enumerate(points):
        train = points[:index] + points[index + 1 :]
        planner = PEFTPlanner()
        planner.fit([point.to_tuple() for point in train], ranks=[point.rank for point in train])
        predicted = planner.predict(left_out.fingerprint, left_out.variant, left_out.rank)
        predictions.append(
            {
                "experiment": left_out.notes,
                "variant": left_out.variant,
                "rank": left_out.rank,
                "actual": left_out.delta_mAP,
                "predicted": predicted,
            }
        )
    actual = torch.tensor([item["actual"] for item in predictions], dtype=torch.float64)
    predicted = torch.tensor([item["predicted"] for item in predictions], dtype=torch.float64)
    residual = actual - predicted
    mse = float((residual.square()).mean())
    mae = float(residual.abs().mean())
    total = float(((actual - actual.mean()).square()).sum())
    r2 = 1.0 - float(residual.square().sum()) / total if total > 1e-12 else 0.0
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "r2": r2, "predictions": predictions}


def build_calibration_artifacts(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    successful = {record["experiment_id"]: record for record in results if record.get("status") == "success"}
    collector = LOVODataCollector()
    protocol_counts: dict[str, int] = {}
    selected_protocol = args.calibration_protocol
    for record in results:
        if record.get("status") == "success":
            protocol = json.dumps(
                {key: record.get(key) for key in ("epochs", "imgsz", "batch", "optimizer", "lr0", "amp")},
                sort_keys=True,
            )
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        if record.get("status") != "success" or record["variant"] == "full":
            continue
        if selected_protocol and any(record.get(key) != value for key, value in selected_protocol.items()):
            continue
        baseline = successful.get(f"{record['model_name']}__full")
        if baseline is None:
            continue
        if selected_protocol and any(baseline.get(key) != value for key, value in selected_protocol.items()):
            continue
        metric = record["metrics"]["metrics/mAP50-95(B)"]
        baseline_metric = baseline["metrics"]["metrics/mAP50-95(B)"]
        fingerprint = ArchitectureFingerprint(**record["fingerprint"])
        collector.add(
            LOVODataPoint(
                fingerprint=fingerprint,
                variant=record["variant"],
                rank=max(int(record["rank"]), 1),
                delta_mAP=metric - baseline_metric,
                model_name=record["model_name"],
                dataset="COCO128",
                epochs=record["epochs"],
                timestamp=record["finished_at"],
                notes=record["experiment_id"],
            )
        )

    collector.save(args.lovo_output)
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "matrix_runs": len(results),
        "successful_runs": len(successful),
        "failed_runs": sum(record.get("status") == "failed" for record in results),
        "calibration_samples": len(collector),
        "collector_summary": collector.summary(),
        "calibration_protocol": selected_protocol,
        "successful_protocol_counts": protocol_counts,
    }
    if len(collector) < 5:
        report["status"] = "insufficient_data"
        atomic_json(args.report_output, report)
        return report

    planner = PEFTPlanner()
    planner.fit(collector.to_history(), ranks=collector.to_ranks())
    ordinary_lovo = LOVOValidator().cross_validate(collector.data_points)
    report.update(
        {
            "status": "fitted",
            "coefficients": planner._coeffs,
            "default_coefficients": list(PEFTPlanner.DEFAULT_COEFFS),
            "fit_metadata": planner._calibration_metadata(),
            "rank_aware_lovo": rank_aware_lovo(collector.data_points),
            "lovo": ordinary_lovo.to_dict(),
        }
    )
    coefficient_payload = {
        "generated_at": report["generated_at"],
        "dataset": "COCO128",
        "device": args.device,
        "coefficients": planner._coeffs,
        "feature_order": [
            "intercept",
            "phi_attn",
            "phi_text",
            "phi_dw",
            "variant_xi",
            "phi_depth",
            "phi_width",
            "phi_head",
            "phi_residual",
            "phi_norm",
            "log2_rank",
            "phi_attn_squared",
        ],
        "fit_metadata": report["fit_metadata"],
        "rank_aware_lovo": {key: value for key, value in report["rank_aware_lovo"].items() if key != "predictions"},
        "sample_count": len(collector),
    }
    atomic_json(args.coefficients_output, coefficient_payload)
    atomic_json(args.report_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", help="NAME=/absolute/path.pt; repeatable")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "ultralytics/cfg/datasets/coco128.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs/planner_mps_coco128_calibration")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--lovo-output", type=Path)
    parser.add_argument("--coefficients-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--cleanup-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=("main", "batch4", "all"),
        default="all",
        help="Select comparable runs for fitting; main=batch 8 AdamW, batch4=batch 4 AdamW, all disables filtering",
    )
    parser.add_argument(
        "--merge-results",
        action="append",
        type=Path,
        default=[],
        help="Additional runs.json files to merge before fitting; later records replace duplicate IDs",
    )
    args = parser.parse_args()
    args.project = args.project.expanduser().resolve()
    args.results = (args.results or args.project / "runs.json").expanduser().resolve()
    args.lovo_output = (args.lovo_output or args.project / "lovo_coco128_mps.json").expanduser().resolve()
    args.coefficients_output = (
        args.coefficients_output or args.project / "planner_coefficients_coco128_mps.json"
    ).expanduser().resolve()
    args.report_output = (args.report_output or args.project / "calibration_report.json").expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.merge_results = [path.expanduser().resolve() for path in args.merge_results]
    args.calibration_protocol = {
        "main": {"epochs": 1, "imgsz": 160, "batch": 8, "optimizer": "AdamW", "lr0": 0.001},
        "batch4": {"epochs": 1, "imgsz": 160, "batch": 4, "optimizer": "AdamW", "lr0": 0.001},
        "all": None,
    }[args.protocol]
    return args


def refresh_fingerprints(results: list[dict[str, Any]]) -> None:
    """Refresh fingerprints after a fingerprint implementation change."""
    models: dict[str, torch.nn.Module] = {}
    for record in results:
        if record.get("status") != "success":
            continue
        weights = record.get("weights")
        if not weights or weights in models:
            continue
        try:
            models[weights] = YOLO(weights).model
        except Exception:
            continue
    for record in results:
        model = models.get(record.get("weights"))
        if model is None:
            continue
        record["fingerprint"] = fingerprint_dict(model)
        record["architecture_family"] = ArchitectureFingerprint._detect_architecture_family(model)
    for model in models.values():
        del model
    gc.collect()


def main() -> int:
    args = parse_args()
    if args.device != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("This calibration requires an available Apple MPS device; CPU fallback is not accepted")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    args.project.mkdir(parents=True, exist_ok=True)
    result_files = [args.results, *args.merge_results]
    results = []
    if args.resume or args.fit_only or args.merge_results:
        indexed: dict[str, dict[str, Any]] = {}
        for result_file in result_files:
            for record in load_results(result_file):
                indexed[record["experiment_id"]] = record
        results = list(indexed.values())
    if not args.fit_only:
        models = parse_models(args.models or list(DEFAULT_MODELS))
        matrix = build_matrix(models, args)
        manifest = {
            "generated_at": utc_now(),
            "matrix_size": len(matrix),
            "calibration_sample_target": sum(item.variant != "full" for item in matrix),
            "experiments": [asdict(item) for item in matrix],
        }
        atomic_json(args.project / "manifest.json", manifest)
        indexed = {record["experiment_id"]: record for record in results}
        pending = []
        for spec in matrix:
            prior = indexed.get(spec.experiment_id)
            if prior and prior.get("status") == "success":
                continue
            if prior and prior.get("status") == "failed" and not args.retry_failed:
                continue
            pending.append(spec)
        if args.max_runs > 0:
            pending = pending[: args.max_runs]
        for position, spec in enumerate(pending, start=1):
            print(f"\n[Calibration] {position}/{len(pending)} {spec.experiment_id}", flush=True)
            record = run_experiment(spec, args)
            indexed[spec.experiment_id] = record
            results = [indexed[item.experiment_id] for item in matrix if item.experiment_id in indexed]
            atomic_json(args.results, results)
            print(
                f"[Calibration] {spec.experiment_id}: {record['status']} "
                f"elapsed={record['elapsed_sec']:.1f}s error={record.get('error')}",
                flush=True,
            )
    refresh_fingerprints(results)
    if args.fit_only or args.merge_results:
        atomic_json(args.results, results)
    report = build_calibration_artifacts(results, args)
    print(json.dumps({key: value for key, value in report.items() if key != "rank_aware_lovo"}, indent=2), flush=True)
    return 0 if report.get("failed_runs", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
