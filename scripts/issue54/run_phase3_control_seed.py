#!/usr/bin/env python3
"""Run and validate one isolated EsMoE or MoA Phase 3 control seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.schema import (  # noqa: E402
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    validate_experiment_manifest,
    with_manifest_checksum,
    write_json,
)

CONFIGS = {
    "v10": "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml",
    "v10_moa": "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
}
METRIC_PREFIXES = ("metrics/", "train/", "val/")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the control-runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--validation-split", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def experiment_root(output_root: Path, model: str, seed: int) -> Path:
    """Return the unique model/seed experiment directory."""
    return output_root / f"phase3_{model}_seed{seed}"


def build_training_command(args: argparse.Namespace, project: Path) -> list[str]:
    """Build the explicit AMP training command without routing export."""
    return [
        sys.executable,
        str(ROOT / "scripts/compare_mot_ablation.py"),
        "--train",
        "--models",
        args.model,
        "--project",
        str(project),
        "--data",
        str(args.data),
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
        "--amp",
    ]


def read_final_metrics(path: Path, requested_epochs: int) -> dict[str, Any]:
    """Validate completion and finite mAP/loss values from the final results row."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("results.csv has no data rows")
    row = {str(key).strip(): value for key, value in rows[-1].items()}
    try:
        completed_epochs = int(float(row["epoch"])) + 1
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("results.csv has no valid final epoch") from error
    if completed_epochs != requested_epochs:
        raise ValueError(f"completed epochs={completed_epochs}, requested={requested_epochs}")
    values: dict[str, float] = {}
    for key, raw in row.items():
        if not key.startswith(METRIC_PREFIXES):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"non-numeric metric or loss: {key}") from error
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric or loss: {key}")
        values[key] = value
    for required in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if required not in values:
            raise ValueError(f"results.csv missing {required}")
    return {"completed_epochs": completed_epochs, "metrics": values}


def load_checkpoint(path: Path) -> None:
    """Require the completed checkpoint to deserialize on CPU."""
    import torch

    torch.load(path, map_location="cpu", weights_only=False)


def signed_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop any stale checksum, sign, then validate the final payload."""
    normalized = dict(payload)
    normalized.pop("manifest_sha256", None)
    return validate_experiment_manifest(with_manifest_checksum(normalized))


def manifest_payload(
    args: argparse.Namespace,
    run_root: Path,
    *,
    status: str,
    failure_reason: str | None,
    checkpoint: Path | None = None,
    config_snapshot: Path | None = None,
    completed_epochs: int | None = None,
) -> dict[str, Any]:
    """Build only fields accepted by the Issue #54 experiment schema."""
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": f"phase3_{args.model}_seed{args.seed}",
        "model_variant": args.model,
        "seed": args.seed,
        "dataset": args.dataset_name,
        "dataset_version": args.dataset_version,
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest) if args.dataset_manifest.is_file() else None,
        "split": args.validation_split,
        "requested_epochs": args.epochs,
        "epochs": completed_epochs,
        "requested_batch": args.batch,
        "batch": args.batch if completed_epochs is not None else None,
        "effective_batch": args.batch if completed_epochs is not None else None,
        "imgsz": args.imgsz,
        "optimizer": "auto",
        "precision_mode": "amp",
        "checkpoint_path": checkpoint.relative_to(run_root).as_posix() if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "config_path": config_snapshot.relative_to(run_root).as_posix() if config_snapshot else None,
        "config_sha256": sha256_file(config_snapshot) if config_snapshot else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timestamp": timestamp(),
        "status": status,
        "failure_reason": failure_reason,
    }


def main(argv: list[str] | None = None) -> int:
    """Run training once and leave a passed/diagnostic/failed audit manifest."""
    args = parse_args(argv)
    args.data = args.data if args.data.is_absolute() else ROOT / args.data
    args.dataset_manifest = (
        args.dataset_manifest if args.dataset_manifest.is_absolute() else ROOT / args.dataset_manifest
    )
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    run_root = experiment_root(output_root, args.model, args.seed)
    project = run_root / "training"
    run_dir = project / args.model
    checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"
    results = run_dir / "results.csv"
    config = ROOT / CONFIGS[args.model]
    command = build_training_command(args, project)
    print(json.dumps({"experiment_root": str(run_root), "amp": True, "train": command}, indent=2))
    if args.dry_run:
        return 0
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {run_root}")
    run_root.mkdir(parents=True)
    manifest_path = run_root / "experiment_manifest.json"
    try:
        for path, label in (
            (args.data, "data YAML"),
            (args.dataset_manifest, "dataset manifest"),
            (config, "model config"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        with (run_root / "runner.log").open("x", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        (run_root / "exitcode").write_text(f"{result.returncode}\n", encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"training failed with exit code {result.returncode}")
        for path, label in ((checkpoint, "best.pt"), (last_checkpoint, "last.pt"), (results, "results.csv")):
            if not path.is_file():
                raise FileNotFoundError(f"required artifact missing: {label}")
        load_checkpoint(checkpoint)
        final = read_final_metrics(results, args.epochs)
        snapshot = run_root / "config" / config.name
        snapshot.parent.mkdir()
        snapshot.write_bytes(config.read_bytes())
        control_metrics = {
            **final,
            "requested_batch": args.batch,
            "actual_batch": args.batch,
            "effective_batch": args.batch,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        write_json(run_root / "control_metrics.json", control_metrics)
        payload = manifest_payload(
            args,
            run_root,
            status="passed" if args.formal else "diagnostic",
            failure_reason=None,
            checkpoint=checkpoint,
            config_snapshot=snapshot,
            completed_epochs=final["completed_epochs"],
        )
        write_json(manifest_path, signed_manifest(payload))
        return 0
    except Exception as error:
        failure = manifest_payload(args, run_root, status="failed", failure_reason=str(error))
        write_json(manifest_path, signed_manifest(failure))
        if not (run_root / "exitcode").exists():
            (run_root / "exitcode").write_text("2\n", encoding="utf-8")
        print(f"[issue54-control] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
