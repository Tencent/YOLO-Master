#!/usr/bin/env python3
"""Run one Issue #54 Phase 3 seed and export routing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.schema import (  # noqa: E402
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    validate_experiment_manifest,
    with_manifest_checksum,
    write_json,
)
from scripts.issue54.run_phase3_control_seed import load_checkpoint, read_final_metrics  # noqa: E402


MODEL_CONFIGS = {
    "v10": "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml",
    "v10_mot": "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
    "v10_moa": "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
    "v10_moa_mot": "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml",
    "v08": "ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml",
    "v08_mot": "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml",
    "v08_moa": "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml",
    "v08_moa_mot": "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml",
}
ROUTING_MODELS = tuple(model for model in MODEL_CONFIGS if "_mot" in model)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", required=True, choices=ROUTING_MODELS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--routing-split", required=True)
    parser.add_argument(
        "--output-root", type=Path, required=True, help="Root directory for isolated experiment outputs."
    )
    parser.add_argument(
        "--image-manifest", type=Path, required=True, help="Explicit sanitized image manifest for routing export."
    )
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Root directory containing image-manifest relative paths."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--formal", action="store_true", help="Mark the completed run as formal status=passed.")
    parser.add_argument(
        "--routing-only", action="store_true", help="Export routing from an existing training checkpoint."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the complete workflow without training.")
    return parser.parse_args(argv)


def build_training_command(args: argparse.Namespace, project: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/compare_mot_ablation.py"),
        "--train",
        "--models",
        args.model,
        "--project",
        str(project),
        "--data",
        str(args.data.resolve()),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--seed",
        str(args.seed),
        "--deterministic",
    ]
    command.append("--no-amp" if args.no_amp else "--amp")
    return command


def find_checkpoint(project: Path, model: str) -> Path:
    checkpoint = project / model / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint.resolve()


def relative_artifact(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def image_manifest_checksum(path: Path) -> str:
    """Return the image manifest's declared checksum after strict validation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"image manifest is not valid JSON: {path}") from error
    checksum = payload.get("manifest_sha256") if isinstance(payload, dict) else None
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError(f"image manifest missing a valid manifest_sha256: {path}")
    return checksum


def finalize_manifest(payload: dict, *, status: str, failure_reason: str | None) -> dict:
    """Apply a final status change without retaining an invalid prior checksum."""
    updated = dict(payload)
    updated.pop("manifest_sha256", None)
    updated["status"] = status
    updated["failure_reason"] = failure_reason
    return validate_experiment_manifest(with_manifest_checksum(updated))


def make_manifest(
    args: argparse.Namespace,
    output: Path,
    config: Path,
    checkpoint: Path,
    dataset_sha256: str,
    completed_epochs: int,
) -> dict:
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": f"phase3_{args.model}_seed{args.seed}",
        "model_variant": args.model,
        "seed": args.seed,
        "dataset": args.dataset_name,
        "dataset_version": args.dataset_version,
        "dataset_manifest_sha256": dataset_sha256,
        "split": args.routing_split,
        "requested_epochs": args.epochs,
        "epochs": completed_epochs,
        "requested_batch": args.batch,
        "batch": args.batch,
        "effective_batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": "auto",
        "precision_mode": "fp32" if args.no_amp else "amp",
        "checkpoint_path": relative_artifact(checkpoint, output),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": relative_artifact(config, output),
        "config_sha256": sha256_file(config),
        "git_commit": git_commit(),
        "timestamp": timestamp(),
        "status": "diagnostic",
        "failure_reason": None,
    }
    return with_manifest_checksum(validate_experiment_manifest(payload))


def failed_manifest(args: argparse.Namespace, reason: str, dataset_sha256: str | None = None) -> dict:
    """Build a failed manifest without claiming artifacts or completed training values."""
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": f"phase3_{args.model}_seed{args.seed}",
        "model_variant": args.model,
        "seed": args.seed,
        "dataset": args.dataset_name,
        "dataset_version": args.dataset_version,
        "dataset_manifest_sha256": dataset_sha256,
        "split": args.routing_split,
        "requested_epochs": args.epochs,
        "epochs": None,
        "requested_batch": args.batch,
        "batch": None,
        "effective_batch": None,
        "imgsz": args.imgsz,
        "optimizer": "auto",
        "precision_mode": "fp32" if args.no_amp else "amp",
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "config_path": None,
        "config_sha256": None,
        "git_commit": git_commit(),
        "timestamp": timestamp(),
        "status": "failed",
        "failure_reason": reason,
    }
    return with_manifest_checksum(validate_experiment_manifest(payload))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = args.data if args.data.is_absolute() else ROOT / args.data
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    experiment_root = output_root / f"phase3_{args.model}_seed{args.seed}"
    image_manifest = args.image_manifest if args.image_manifest.is_absolute() else ROOT / args.image_manifest
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    dataset_manifest = args.dataset_manifest if args.dataset_manifest.is_absolute() else ROOT / args.dataset_manifest
    args.data = data
    config = ROOT / MODEL_CONFIGS[args.model]
    project = experiment_root / "training"
    output = experiment_root
    checkpoint = project / args.model / "weights" / "best.pt"
    results_path = project / args.model / "results.csv"
    manifest_path = output / "experiment_manifest.json"
    routing_path = output / "routing" / "routing.jsonl"
    train_command = build_training_command(args, project)
    export_command = [
        sys.executable,
        str(ROOT / "scripts/issue54/export_mot_routing.py"),
        "--manifest",
        str(manifest_path),
        "--artifact-root",
        str(output),
        "--image-manifest",
        str(image_manifest),
        "--data-root",
        str(data_root),
        "--output",
        str(routing_path),
    ]
    plan = {
        "dry_run": args.dry_run,
        "model": args.model,
        "seed": args.seed,
        "precision_mode": "fp32" if args.no_amp else "amp",
        "dataset": args.dataset_name,
        "dataset_version": args.dataset_version,
        "routing_split": args.routing_split,
        "formal": args.formal,
        "routing_only": args.routing_only,
        "experiment_root": str(experiment_root),
        "train": train_command,
        "checkpoint": str(checkpoint),
        "experiment_manifest": str(manifest_path),
        "export_mot_routing": export_command,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=True))
    if args.dry_run:
        return 0

    if not args.routing_only and not data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data}")
    if not image_manifest.is_file():
        raise FileNotFoundError(f"image manifest not found: {image_manifest}")
    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {dataset_manifest}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data root not found: {data_root}")
    if not config.is_file():
        raise FileNotFoundError(f"model config not found: {config}")
    image_checksum = image_manifest_checksum(image_manifest)
    if not args.routing_only and output.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output}")

    if args.routing_only:
        if not output.is_dir():
            raise FileNotFoundError(f"experiment directory not found for routing recovery: {output}")
        if routing_path.exists():
            raise FileExistsError(f"refusing to overwrite routing output: {routing_path}")
        if manifest_path.exists():
            recovery_backup = output / "experiment_manifest.json.pre_routing_recovery"
            if recovery_backup.exists():
                raise FileExistsError(f"refusing to overwrite recovery backup: {recovery_backup}")
            shutil.copy2(manifest_path, recovery_backup)
        checkpoint = find_checkpoint(project, args.model)
    else:
        result = subprocess.run(train_command, cwd=ROOT, check=False)
        if result.returncode != 0:
            reason = f"training failed with exit code {result.returncode}"
            output.mkdir(parents=True, exist_ok=True)
            write_json(manifest_path, failed_manifest(args, reason, image_checksum))
            return result.returncode
        checkpoint = find_checkpoint(project, args.model)
        output.mkdir(parents=True, exist_ok=True)

    try:
        load_checkpoint(checkpoint)
        final_metrics = read_final_metrics(results_path, args.epochs)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        write_json(
            manifest_path,
            failed_manifest(args, f"training artifact validation failed: {error}", image_checksum),
            overwrite=manifest_path.exists(),
        )
        return 2

    config_snapshot = output / "config" / config.name
    if not config_snapshot.exists():
        config_snapshot.parent.mkdir(parents=True, exist_ok=True)
        config_snapshot.write_bytes(config.read_bytes())

    manifest = make_manifest(
        args,
        output,
        config_snapshot,
        checkpoint,
        image_checksum,
        final_metrics["completed_epochs"],
    )
    export_manifest = dict(manifest)
    export_manifest["dataset_manifest_sha256"] = sha256_file(image_manifest)
    export_manifest = finalize_manifest(export_manifest, status="passed", failure_reason=None)
    with tempfile.NamedTemporaryFile(prefix="issue54-export-", suffix=".json", delete=False) as handle:
        export_manifest_path = Path(handle.name)
    try:
        write_json(export_manifest_path, export_manifest, overwrite=True)
        export_command[3] = str(export_manifest_path)
        export_result = subprocess.run(export_command, cwd=ROOT, check=False)
    finally:
        export_manifest_path.unlink(missing_ok=True)
    if export_result.returncode != 0:
        failure_manifest = finalize_manifest(
            manifest,
            status="failed",
            failure_reason=f"routing export failed with exit code {export_result.returncode}",
        )
        write_json(manifest_path, failure_manifest, overwrite=manifest_path.exists())
        return export_result.returncode
    final_manifest = finalize_manifest(manifest, status="passed" if args.formal else "diagnostic", failure_reason=None)
    write_json(manifest_path, final_manifest, overwrite=manifest_path.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
