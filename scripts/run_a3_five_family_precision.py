#!/usr/bin/env python3
"""Run the ordered A3 five-family precision and routing-sensitivity pipeline.

Ordered ``all`` stages:
0. genuinely trained MoT route drift gate (no random injection)
1. lock five checkpoints/configs/dataset selections
2. export FP32/FP16 and build full/manual INT8
3. unified pre-auto validation
4. per-group INT8 mAP ablation + route-drift correlation
5. build auto fallback
6. final FP32/full INT8/manual/auto comparison (FP16 also reported)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _prime_torch_cuda_runtime() -> None:
    """Load PyTorch's bundled cuDNN before Ultralytics/ONNX dependencies."""
    try:
        import torch
    except ImportError:
        return
    if torch.version.cuda is not None and torch.cuda.is_available():
        # Some cloud images expose an older cuDNN through /usr/local/cuda/lib64.
        # Initializing here prevents a later dependency from loading that copy
        # into the process before PyTorch selects its bundled compatible build.
        torch.backends.cudnn.version()


_prime_torch_cuda_runtime()

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.a3_mot_trained_routing_drift import run as run_trained_mot_drift
from scripts.a3_precision.common import collect_images, file_set_digest, json_dump, sha256_file
from scripts.a3_precision.manifest import FIVE_FAMILIES, load_config, model_lock, verify_lock, write_lock
from scripts.a3_precision.onnx_backend import build_auto_fallback, build_primary_variants
from scripts.a3_precision.report import build_report
from scripts.a3_precision.sensitivity import analyze_sensitivity, write_validation_subset
from scripts.a3_precision.validation import validate_family_variants


def _selected(config, names: list[str] | None):
    selected = set(names or FIVE_FAMILIES)
    return [spec for spec in config.models if spec.enabled and spec.family in selected]


def _lock_checkpoint_hash(lock: dict[str, Any], family: str) -> str:
    row = model_lock(lock, family)
    return next(asset["sha256"] for asset in row["assets"] if asset["role"] == "checkpoint")


def _stage0(config, *, force: bool) -> Path:
    spec = next(model for model in config.models if model.family == "mot")
    settings = config.raw.get("trained_mot_drift", {})
    output_dir = config.output_dir / "stage0_trained_mot_drift"
    summary_path = output_dir / "trained_mot_routing_drift.json"
    if summary_path.is_file() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_images = min(int(settings.get("limit", 548)), config.dataset.validation_samples)
        selected_images = collect_images(
            config.dataset.validation_images,
            limit=expected_images,
            seed=int(config.raw.get("experiment", {}).get("seed", 42)),
        )
        expected_digest = file_set_digest(selected_images, config.dataset.validation_images)
        if (
            summary.get("schema_version") == 2
            and summary.get("status") == "passed"
            and summary.get("model", {}).get("trained_checkpoint_gate") is True
            and summary.get("model", {}).get("sha256") == sha256_file(spec.checkpoint)
            and summary.get("dataset", {}).get("images") == expected_images
            and summary.get("dataset", {}).get("source") == str(config.dataset.validation_images)
            and summary.get("dataset", {}).get("imgsz") == int(config.raw.get("experiment", {}).get("imgsz", 640))
            and summary.get("dataset", {}).get("selection_digest") == expected_digest
            and summary.get("methodology", {}).get("scope") == str(settings.get("scope", "router_only"))
            and summary.get("methodology", {}).get("int8_granularity")
            == ("per-output-channel" if bool(settings.get("per_channel", True)) else "per-tensor")
        ):
            return summary_path
    namespace = SimpleNamespace(
        model=spec.checkpoint,
        source=config.dataset.validation_images,
        limit=min(int(settings.get("limit", 548)), config.dataset.validation_samples),
        imgsz=int(config.raw.get("experiment", {}).get("imgsz", 640)),
        device=str(settings.get("device", config.raw.get("experiment", {}).get("device", "cpu"))),
        seed=int(config.raw.get("experiment", {}).get("seed", 42)),
        scope=str(settings.get("scope", "router_only")),
        per_channel=bool(settings.get("per_channel", True)),
        out=output_dir,
    )
    return run_trained_mot_drift(namespace)


def _get_lock(config, *, refresh: bool) -> dict[str, Any]:
    lock_path = config.output_dir / "experiment.lock.json"
    if refresh or not lock_path.is_file():
        write_lock(config, lock_path)
    return verify_lock(config, lock_path)


def _record_failure(failures: list[dict[str, Any]], family: str, stage: str, exc: Exception) -> None:
    failures.append(
        {
            "family": family,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


def _require_success(results: dict[str, Any], *, context: str) -> dict[str, Any]:
    failed = [name for name, row in results.items() if not isinstance(row, dict) or row.get("status") != "success"]
    if failed:
        raise RuntimeError(f"{context} failed variants: {failed}")
    return results


def _run_family_stage(
    specs,
    stage: str,
    action: Callable[[Any], Any],
    failures: list[dict[str, Any]],
) -> None:
    for spec in specs:
        try:
            action(spec)
        except Exception as exc:
            _record_failure(failures, spec.family, stage, exc)


def run_pipeline(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    specs = _selected(config, args.family)
    failures: list[dict[str, Any]] = []

    if args.stage in {"stage0", "all"}:
        stage0_path = _stage0(config, force=args.force_stage0)
        if args.stage == "stage0":
            print(stage0_path)
            return 0

    if args.stage in {"lock", "all"}:
        lock = _get_lock(config, refresh=args.refresh_lock)
        if args.stage == "all":
            stage0 = json.loads(
                (config.output_dir / "stage0_trained_mot_drift" / "trained_mot_routing_drift.json").read_text(
                    encoding="utf-8"
                )
            )
            if stage0["model"]["sha256"] != _lock_checkpoint_hash(lock, "mot"):
                raise RuntimeError("stage-0 MoT checkpoint hash does not match the five-family lock")
        if args.stage == "lock":
            print(config.output_dir / "experiment.lock.json")
            return 0
    else:
        lock = verify_lock(config)

    validation_images = [Path(path) for path in lock["dataset"]["validation"]["images"]]
    locked_validation_yaml = write_validation_subset(
        config.dataset.data_yaml,
        validation_images,
        config.output_dir / "validation_contract",
    )
    experiment = config.raw.get("experiment", {})
    benchmark_count = min(int(experiment.get("benchmark_images", 16)), len(validation_images))
    benchmark_images = validation_images[:benchmark_count]

    if args.stage in {"export", "all"}:
        _run_family_stage(
            specs,
            "export_and_primary_quantization",
            lambda spec: _require_success(
                build_primary_variants(config, spec, lock), context=f"{spec.family} primary build"
            ),
            failures,
        )
        if args.stage == "export":
            json_dump(config.output_dir / "pipeline_failures.json", failures)
            return 2 if failures else 0

    pre_auto: dict[str, Any] = {}
    if args.stage in {"validate-pre", "sensitivity", "all"}:

        def validate_pre(spec):
            comparison = validate_family_variants(
                config.output_dir / spec.family,
                data_yaml=locked_validation_yaml,
                benchmark_images=benchmark_images,
                settings=experiment,
                include_auto=False,
            )
            pre_auto[spec.family] = comparison
            _require_success(comparison, context=f"{spec.family} pre-auto validation")

        _run_family_stage(specs, "validation_pre_auto", validate_pre, failures)
        if args.stage == "validate-pre":
            json_dump(config.output_dir / "pipeline_failures.json", failures)
            return 2 if failures else 0

    sensitivities: dict[str, Any] = {}
    if args.stage in {"sensitivity", "all"}:

        def sensitivity_stage(spec):
            comparison = pre_auto.get(spec.family)
            if comparison is None:
                comparison_path = config.output_dir / spec.family / "comparison.pre_auto.json"
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            sensitivities[spec.family] = analyze_sensitivity(config, spec, lock, comparison)

        _run_family_stage(specs, "sensitivity", sensitivity_stage, failures)
        if args.stage == "sensitivity":
            json_dump(config.output_dir / "pipeline_failures.json", failures)
            return 2 if failures else 0

    if args.stage in {"auto", "all"}:

        def auto_stage(spec):
            result = sensitivities.get(spec.family)
            if result is None:
                result = json.loads((config.output_dir / spec.family / "sensitivity.json").read_text(encoding="utf-8"))
            _require_success(
                {"auto_fallback": build_auto_fallback(config, spec, lock, result)},
                context=f"{spec.family} auto fallback",
            )

        _run_family_stage(specs, "auto_fallback", auto_stage, failures)
        if args.stage == "auto":
            json_dump(config.output_dir / "pipeline_failures.json", failures)
            return 2 if failures else 0

    if args.stage in {"compare", "all"}:

        def compare_stage(spec):
            comparison = validate_family_variants(
                config.output_dir / spec.family,
                data_yaml=locked_validation_yaml,
                benchmark_images=benchmark_images,
                settings=experiment,
                include_auto=True,
            )
            _require_success(comparison, context=f"{spec.family} final validation")

        _run_family_stage(specs, "final_comparison", compare_stage, failures)
        result = build_report(config, failures)
        json_dump(config.output_dir / "pipeline_failures.json", failures)
        print(config.output_dir / "five_family_precision_report.md")
        return 0 if result["status"] == "success" else 2
    raise ValueError(f"unhandled stage: {args.stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/a3_five_family_precision.yaml"))
    parser.add_argument(
        "--stage",
        choices=("stage0", "lock", "export", "validate-pre", "sensitivity", "auto", "compare", "all"),
        default="all",
    )
    parser.add_argument("--family", action="append", choices=FIVE_FAMILIES, help="resume only selected families")
    parser.add_argument("--refresh-lock", action="store_true", help="explicitly replace an existing asset lock")
    parser.add_argument("--force-stage0", action="store_true", help="rerun trained MoT drift even if evidence matches")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_pipeline(parse_args()))
