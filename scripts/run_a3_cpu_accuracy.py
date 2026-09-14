"""Accuracy-only validation of existing A3 ONNX artifacts on a locked image set.

This script never trains, exports, quantizes, or benchmarks latency. Results are
written incrementally and successful matching runs are resumable. Validation can
run on CPU or through the ONNX Runtime CUDA Execution Provider.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch


VARIANTS = {
    "fp32": "fp32.onnx",
    "fp16": "fp16.onnx",
    "full_int8": "full_int8.onnx",
    "manual_fallback": "manual_fallback.onnx",
}


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def family_comparison(results: dict) -> dict:
    baseline = results.get("fp32", {})
    baseline_accuracy = baseline.get("accuracy", {})
    baseline_map = baseline_accuracy.get("map50_95")
    compared = {}
    for name, row in results.items():
        item = dict(row)
        current_map = row.get("accuracy", {}).get("map50_95")
        if finite_number(baseline_map) and finite_number(current_map):
            item["map50_95_loss"] = float(baseline_map - current_map)
            item["map50_95_loss_percentage_points"] = float((baseline_map - current_map) * 100.0)
        compared[name] = item
    return compared


def write_csv(path: Path, families: dict) -> None:
    fields = (
        "family",
        "variant",
        "status",
        "map50_95",
        "map50",
        "map75",
        "precision",
        "recall",
        "map50_95_loss_percentage_points",
        "elapsed_seconds",
        "size_mb",
        "error_type",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family, variants in families.items():
            for variant, row in variants.items():
                accuracy = row.get("accuracy", {})
                writer.writerow(
                    {
                        "family": family,
                        "variant": variant,
                        "status": row.get("status"),
                        "map50_95": accuracy.get("map50_95"),
                        "map50": accuracy.get("map50"),
                        "map75": accuracy.get("map75"),
                        "precision": accuracy.get("precision"),
                        "recall": accuracy.get("recall"),
                        "map50_95_loss_percentage_points": row.get("map50_95_loss_percentage_points"),
                        "elapsed_seconds": row.get("elapsed_seconds"),
                        "size_mb": row.get("size_mb"),
                        "error_type": row.get("error_type"),
                        "error": row.get("error"),
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", action="append")
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS),
        help="validate only selected variants; repeat for more than one",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics validation device, for example 'cpu' or '0' for CUDA device 0",
    )
    parser.add_argument("--allow-runtime-drift", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")

    if str(args.device).strip().lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    args.output.mkdir(parents=True, exist_ok=True)

    from scripts.a3_precision import manifest as manifest_module
    from scripts.a3_precision.common import sha256_file
    from scripts.a3_precision.sensitivity import write_validation_subset
    from scripts.a3_precision.validation import validate_accuracy

    config = manifest_module.load_config(args.config)
    lock_path = config.output_dir / "experiment.lock.json"
    recorded_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_runtime = recorded_lock["environment"]["contract"]
    current_runtime = manifest_module._runtime_contract()
    runtime_drift = current_runtime != locked_runtime
    if runtime_drift and not args.allow_runtime_drift:
        raise RuntimeError("Runtime differs from export lock; explicitly pass --allow-runtime-drift to record it")

    print("Verifying locked config, weights, datasets, and selections...", flush=True)
    if runtime_drift:
        with patch.object(manifest_module, "_runtime_contract", return_value=locked_runtime):
            lock = manifest_module.verify_lock(config)
    else:
        lock = manifest_module.verify_lock(config)

    selected = [spec for spec in config.models if spec.enabled and (not args.family or spec.family in set(args.family))]
    if not selected or (args.family and set(args.family) - {spec.family for spec in selected}):
        raise ValueError("Requested family is absent or disabled")
    selected_variants = tuple(args.variant or VARIANTS)

    validation_images = [Path(item) for item in lock["dataset"]["validation"]["images"]]
    data_yaml = write_validation_subset(config.dataset.data_yaml, validation_images, args.output / "contract")
    settings = {
        **config.raw.get("experiment", {}),
        "validation_device": str(args.device),
        "validation_batch": 1,
        "workers": args.workers,
    }
    run_contract = {
        "task": "accuracy_only",
        "locked_image_count": len(validation_images),
        "locked_selection_digest": lock["dataset"]["validation"]["selection_digest"],
        "data_yaml_sha256": sha256_file(data_yaml),
        "settings": {
            key: settings[key] for key in ("imgsz", "validation_batch", "validation_device", "workers", "conf", "iou")
        },
        "current_validation_runtime": current_runtime,
    }
    run_contract_sha256 = digest_json(run_contract)
    summary_path = args.output / "accuracy_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"status": "running", "families": {}}
    summary.update(
        claim_boundary=(
            "Accuracy only on the exact locked validation image list; no latency. "
            "Runtime differs from the export lock and both contracts are retained."
            if runtime_drift
            else "Accuracy only on the exact locked validation image list; no latency."
        ),
        source_experiment=str(config.output_dir),
        experiment_lock_sha256=sha256_file(lock_path),
        script_sha256=sha256_file(Path(__file__)),
        runtime_contract_match=not runtime_drift,
        runtime_drift_allowed=bool(runtime_drift and args.allow_runtime_drift),
        locked_export_runtime=locked_runtime,
        current_validation_runtime=current_runtime,
        run_contract=run_contract,
        run_contract_sha256=run_contract_sha256,
        selected_families=[spec.family for spec in selected],
        selected_variants=list(selected_variants),
    )
    save_json(summary_path, summary)

    for spec in selected:
        family = spec.family
        source_dir = config.output_dir / family / "artifacts"
        manifest_path = source_dir / "primary_variants.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["families"].setdefault(family, {})
        print(f"\n========== {family} ==========", flush=True)

        for variant in selected_variants:
            filename = VARIANTS[variant]
            artifact = source_dir / filename
            artifact_sha256 = sha256_file(artifact)
            expected_sha256 = manifest.get(variant, {}).get("artifact", {}).get("sha256")
            if manifest.get(variant, {}).get("status") != "success" or artifact_sha256 != expected_sha256:
                raise RuntimeError(f"Artifact manifest mismatch: {artifact}")

            result_path = args.output / family / f"{variant}.accuracy.json"
            prior = None
            if result_path.is_file():
                prior = json.loads(result_path.read_text(encoding="utf-8"))
            reusable = (
                prior
                and prior.get("status") == "success"
                and prior.get("artifact_sha256") == artifact_sha256
                and prior.get("run_contract_sha256") == run_contract_sha256
            )
            if reusable and not args.retry_failed:
                print(f"  {variant}: reuse completed result", flush=True)
                summary["families"][family][variant] = prior
                continue
            same_contract = (
                prior
                and prior.get("artifact_sha256") == artifact_sha256
                and prior.get("run_contract_sha256") == run_contract_sha256
            )
            if same_contract and prior.get("status") == "failed" and not args.retry_failed:
                print(f"  {variant}: retain failed result (use --retry-failed to retry)", flush=True)
                summary["families"][family][variant] = prior
                continue

            print(f"  {variant}: validating {len(validation_images)} locked images on CPU", flush=True)
            try:
                measured = validate_accuracy(artifact, data_yaml=data_yaml, settings=settings)
                if measured.get("status") == "success":
                    result = {
                        **measured,
                        "artifact": str(artifact),
                        "artifact_sha256": artifact_sha256,
                        "size_bytes": artifact.stat().st_size,
                        "size_mb": artifact.stat().st_size / (1024 * 1024),
                        "run_contract_sha256": run_contract_sha256,
                    }
                else:
                    result = {
                        **measured,
                        "artifact": str(artifact),
                        "artifact_sha256": artifact_sha256,
                        "run_contract_sha256": run_contract_sha256,
                    }
            except Exception as exc:
                result = {
                    "status": "failed",
                    "artifact": str(artifact),
                    "artifact_sha256": artifact_sha256,
                    "run_contract_sha256": run_contract_sha256,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            save_json(result_path, result)
            summary["families"][family][variant] = result
            save_json(summary_path, summary)
            print(f"  {variant}: {result['status']}", flush=True)

        summary["families"][family] = family_comparison(summary["families"][family])
        save_json(summary_path, summary)
        write_csv(args.output / "accuracy_summary.csv", summary["families"])

    expected = len(selected) * len(selected_variants)
    selected_names = {spec.family for spec in selected}
    completed = sum(
        row.get("status") == "success"
        for family, variants in summary["families"].items()
        if family in selected_names
        for variant, row in variants.items()
        if variant in selected_variants
    )
    summary["completed"] = completed
    summary["expected"] = expected
    summary["status"] = "success" if completed == expected else "needs_review"
    save_json(summary_path, summary)
    write_csv(args.output / "accuracy_summary.csv", summary["families"])

    print("\n========== accuracy summary ==========", flush=True)
    for family, variants in summary["families"].items():
        for variant, row in variants.items():
            accuracy = row.get("accuracy", {})
            print(
                f"{family:8s} {variant:18s} status={row.get('status')} "
                f"mAP50-95={accuracy.get('map50_95')} mAP50={accuracy.get('map50')} "
                f"loss_pp={row.get('map50_95_loss_percentage_points')}",
                flush=True,
            )
    print(f"Status: {summary['status']} ({completed}/{expected})", flush=True)
    print(f"JSON: {summary_path}\nCSV: {args.output / 'accuracy_summary.csv'}", flush=True)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
