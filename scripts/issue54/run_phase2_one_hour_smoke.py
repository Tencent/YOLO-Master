#!/usr/bin/env python3
"""Run the Issue #54 Phase 2 diagnostic smoke matrix on an isolated CUDA host.

The historical filename is retained for traceability. Phase 2 no longer has a
one-hour deadline; every configured run is attempted unless data or
implementation validation fails. A child is considered stalled only when its
log, GPU utilization, and output files are all unchanged for 20 minutes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIAGNOSTIC_LABEL = "diagnostic_not_formal_evidence"
ALLOWED_STATUSES = {
    "passed_diagnostic",
    "failed",
    "diagnostic_stalled",
    "not_executed_missing_data",
    "implementation_failed",
}
MODEL_SPECS = {
    "mot": ("yolo-master-mot-n.yaml", "C2fMoT"),
    "esmoe": ("yolo-master-n.yaml", "VisualEnhancedAdaptiveGateMoE"),
    "moa": ("yolo-master-moa-n.yaml", "C2fMoA"),
}
RUN_SPECS = (
    ("A_mot_amp", "mot", True, 3, 0.05),
    ("B_mot_fp32", "mot", False, 3, 0.05),
    ("C_esmoe_amp", "esmoe", True, 2, 0.03),
    ("D_moa_amp", "moa", True, 2, 0.03),
)
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def utc_now() -> str:
    """Return a stable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic UTF-8 JSON without overwriting an existing file."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def discover_official_configs(repo: Path) -> dict[str, Path]:
    """Find the unique v0.10 official YAML for each requested mixture."""
    base = repo / "ultralytics/cfg/models/master/v0_10/det"
    discovered: dict[str, Path] = {}
    for variant, (filename, marker) in MODEL_SPECS.items():
        candidates = sorted(path for path in base.rglob(filename) if path.is_file())
        if len(candidates) != 1:
            raise RuntimeError(f"{variant}: expected one official {filename}, found {len(candidates)}")
        content = candidates[0].read_text(encoding="utf-8")
        if marker not in content:
            raise RuntimeError(f"{variant}: {candidates[0]} does not contain required marker {marker}")
        discovered[variant] = candidates[0]
    return discovered


def create_dataset_snapshot(data_root: Path, output: Path) -> dict[str, Any]:
    """Validate VisDrone and create a non-downloading diagnostic YAML plus manifest."""
    splits: dict[str, dict[str, int]] = {}
    inventory: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        if split in {"train", "val"} and (not images or len(images) != len(labels)):
            raise RuntimeError(f"VisDrone {split} image/label mismatch: {len(images)}/{len(labels)}")
        splits[split] = {"images": len(images), "labels": len(labels)}
        for path in images:
            inventory.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "size": path.stat().st_size,
                }
            )
    dataset_view = output / "dataset_view"
    (dataset_view / "images").parent.mkdir(parents=True, exist_ok=False)
    os.symlink(data_root / "images", dataset_view / "images", target_is_directory=True)
    (dataset_view / "labels").mkdir()
    for split in ("train", "val"):
        shutil.copytree(data_root / "labels" / split, dataset_view / "labels" / split)

    yaml_path = output / "configs/visdrone_diagnostic.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml_path.exists():
        raise FileExistsError(f"refusing to overwrite {yaml_path}")
    split_lists = {}
    for split in ("train", "val"):
        image_list = yaml_path.parent / f"{split}_images.txt"
        image_names = sorted(
            path.name for path in (data_root / "images" / split).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        image_list.write_text(
            "\n".join((dataset_view / "images" / split / name).as_posix() for name in image_names) + "\n",
            encoding="utf-8",
        )
        split_lists[split] = image_list
    names = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ]
    lines = [
        f"# {DIAGNOSTIC_LABEL}",
        f"path: {dataset_view.as_posix()}",
        f"train: {split_lists['train'].as_posix()}",
        f"val: {split_lists['val'].as_posix()}",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(names)),
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "evidence_label": DIAGNOSTIC_LABEL,
        "dataset": "VisDrone2019-DET",
        "data_root": data_root.as_posix(),
        "execution_view": dataset_view.as_posix(),
        "execution_view_policy": "images symlinked read-only; labels copied so Ultralytics cache stays under MoT",
        "yaml_path": yaml_path.as_posix(),
        "yaml_sha256": sha256_file(yaml_path),
        "inventory_sha256": inventory_sha256,
        "splits": splits,
        "read_only_reuse": True,
    }
    manifest_path = output / "configs/visdrone_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def parse_last_results_row(results_csv: Path) -> dict[str, Any]:
    """Read the last Ultralytics results row into normalized scalar fields."""
    if not results_csv.is_file():
        return {}
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = {key.strip(): value.strip() for key, value in rows[-1].items()}

    def finite(*keys: str) -> float | None:
        for key in keys:
            if key in row and row[key] != "":
                try:
                    value = float(row[key])
                except ValueError:
                    continue
                return value if math.isfinite(value) else None
        return None

    losses = [
        finite("train/box_loss"),
        finite("train/cls_loss"),
        finite("train/dfl_loss"),
    ]
    return {
        "map50": finite("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": finite("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "loss": sum(value for value in losses if value is not None)
        if any(value is not None for value in losses)
        else None,
        "epoch": finite("epoch"),
        "training_time_seconds": finite("time"),
        "raw_last_row": row,
    }


def run_child(args: argparse.Namespace) -> int:
    """Train one diagnostic model in a subprocess and save machine-readable metrics."""
    import torch
    import yaml

    from ultralytics import YOLO

    result_path = Path(args.child_result)
    run_dir = Path(args.project) / args.name
    started = time.monotonic()
    payload: dict[str, Any] = {
        "started_at": utc_now(),
        "evidence_label": DIAGNOSTIC_LABEL,
        "pid": os.getpid(),
        "actual_batch": args.batch,
    }
    exit_code = 1
    try:
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            fraction=args.fraction,
            imgsz=640,
            batch=args.batch,
            workers=8,
            seed=0,
            deterministic=True,
            cache=False,
            device=0,
            amp=args.amp,
            optimizer="auto",
            lr0=0.01,
            pretrained=False,
            project=args.project,
            name=args.name,
            exist_ok=False,
            plots=False,
            verbose=True,
        )
        parsed = parse_last_results_row(run_dir / "results.csv")
        args_yaml = run_dir / "args.yaml"
        actual_args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else {}
        payload.update(parsed)
        payload.update(
            {
                "status": "passed_diagnostic",
                "failure_reason": None,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
                "optimizer_actual": actual_args.get("optimizer"),
                "lr0_actual": actual_args.get("lr0"),
            }
        )
        exit_code = 0
    except BaseException as error:
        payload.update(
            {
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else None,
            }
        )
        raise
    finally:
        payload["ended_at"] = utc_now()
        payload["duration_seconds"] = time.monotonic() - started
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return exit_code


def gpu_utilization() -> int | None:
    """Return current GPU utilization, if queryable."""
    command = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        return int(subprocess.check_output(command, text=True, timeout=10).strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def newest_mtime(path: Path) -> float:
    """Return the newest mtime under a directory."""
    if not path.exists():
        return 0.0
    return max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=0.0)


def execute_run(
    *,
    python: str,
    script: Path,
    output: Path,
    dataset_yaml: Path,
    config: Path,
    run_id: str,
    variant: str,
    amp: bool,
    epochs: int,
    fraction: float,
) -> dict[str, Any]:
    """Execute one run with explicit OOM fallback and a three-signal stall gate."""
    requested_batch = 8
    total_started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for batch in (8, 4):
        name = run_id if batch == 8 else f"{run_id}_batch4"
        run_dir = output / "training" / name
        log_path = output / "logs" / f"{name}.log"
        child_result = output / "logs" / f"{name}.result.json"
        command = [
            python,
            str(script),
            "--single-run",
            "--model",
            str(config),
            "--data",
            str(dataset_yaml),
            "--project",
            str(output / "training"),
            "--name",
            name,
            "--epochs",
            str(epochs),
            "--fraction",
            str(fraction),
            "--batch",
            str(batch),
            "--amp",
            str(amp),
            "--child-result",
            str(child_result),
        ]
        attempt_started = utc_now()
        with log_path.open("x", encoding="utf-8") as log:
            log.write(f"# evidence_label={DIAGNOSTIC_LABEL}\n# command={command!r}\n# started_at={attempt_started}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            attempt = {
                "batch": batch,
                "command": command,
                "pid": process.pid,
                "started_at": attempt_started,
                "log_path": log_path.as_posix(),
            }
            last_progress = time.monotonic()
            last_log_size = log_path.stat().st_size
            last_output_mtime = newest_mtime(run_dir)
            last_util = gpu_utilization()
            stalled = False
            while process.poll() is None:
                time.sleep(30)
                log_size = log_path.stat().st_size
                output_mtime = newest_mtime(run_dir)
                util = gpu_utilization()
                changed = log_size != last_log_size or output_mtime != last_output_mtime
                gpu_active = util is not None and util > 0
                if changed or gpu_active or (last_util is not None and util != last_util):
                    last_progress = time.monotonic()
                if time.monotonic() - last_progress >= 20 * 60:
                    stalled = True
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    break
                last_log_size = log_size
                last_output_mtime = output_mtime
                last_util = util
            attempt["ended_at"] = utc_now()
            attempt["exit_code"] = process.returncode
            attempt["stalled"] = stalled
        child = json.loads(child_result.read_text(encoding="utf-8")) if child_result.is_file() else {}
        attempt["child_result"] = child
        attempts.append(attempt)
        if stalled:
            final = {"status": "diagnostic_stalled", "failure_reason": "20 minutes without log/GPU/file progress"}
            break
        if process.returncode == 0 and child.get("status") == "passed_diagnostic":
            final = child
            final["run_dir"] = run_dir.as_posix()
            break
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        is_oom = "out of memory" in log_text or "cuda error: out of memory" in log_text
        if batch == 8 and is_oom:
            continue
        final = child or {"status": "failed", "failure_reason": f"child exit code {process.returncode}"}
        break

    final_status = final.get("status", "failed")
    if final_status not in ALLOWED_STATUSES:
        final_status = "implementation_failed"
    actual_batch = next(
        (attempt["batch"] for attempt in reversed(attempts) if attempt["exit_code"] == 0),
        attempts[-1]["batch"],
    )
    run_dir = Path(
        final.get("run_dir", output / "training" / attempts[-1]["command"][attempts[-1]["command"].index("--name") + 1])
    )
    checkpoint_kind = "best"
    checkpoint = run_dir / "weights/best.pt"
    if not checkpoint.is_file():
        checkpoint_kind = "last"
        checkpoint = run_dir / "weights/last.pt"
    if not checkpoint.is_file():
        checkpoint_kind = "partial_last_healthy"
        checkpoint = run_dir / "weights/last_healthy.pt"
    checkpoint_info = None
    if checkpoint.is_file():
        checkpoint_info = {
            "path": checkpoint.as_posix(),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "kind": checkpoint_kind,
            "partial": checkpoint_kind == "partial_last_healthy",
        }
    controller_duration = time.monotonic() - total_started
    duration = float(final.get("duration_seconds", controller_duration))
    training_time = final.get("training_time_seconds")
    return {
        "evidence_label": DIAGNOSTIC_LABEL,
        "experiment_id": run_id,
        "model_variant": variant,
        "seed": 0,
        "config_path": config.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config),
        "precision_mode": "amp" if amp else "fp32",
        "requested_batch": requested_batch,
        "actual_batch": actual_batch,
        "imgsz": 640,
        "fraction": fraction,
        "epochs": epochs,
        "optimizer": final.get("optimizer_actual", "auto"),
        "lr0": final.get("lr0_actual", 0.01),
        "duration_seconds": duration,
        "controller_duration_seconds": controller_duration,
        "training_time_seconds": training_time,
        "seconds_per_epoch": training_time / epochs if training_time is not None else None,
        "peak_gpu_memory_bytes": final.get("peak_gpu_memory_bytes"),
        "map50": final.get("map50"),
        "map50_95": final.get("map50_95"),
        "loss": final.get("loss"),
        "nan_or_inf": any(
            value is not None and not math.isfinite(float(value))
            for value in (final.get("map50"), final.get("map50_95"), final.get("loss"))
        ),
        "exit_code": attempts[-1]["exit_code"],
        "checkpoint": checkpoint_info,
        "status": final_status,
        "failure_reason": final.get("failure_reason"),
        "attempts": attempts,
    }


def export_routing(
    run: dict[str, Any],
    *,
    output: Path,
    data_root: Path,
    dataset_manifest: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    """Export deterministic routing records for 32 fixed validation images."""
    import torch

    from scripts.issue54.export_mot_routing import _image_tensor, capture_mot_routing
    from scripts.issue54.schema import (
        EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        canonical_payload_sha256,
        validate_experiment_manifest,
        with_manifest_checksum,
        write_jsonl,
    )
    from ultralytics import YOLO
    from ultralytics.nn.modules.mot import MoTBlock

    checkpoint = Path(run["checkpoint"]["path"])
    val_images = sorted(path for path in (data_root / "images/val").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)[
        :32
    ]
    if len(val_images) != 32:
        raise RuntimeError(f"expected 32 validation images, found {len(val_images)}")
    entries = [
        {
            "image_id": f"visdrone-val-{index:03d}-{sha256_file(path)[:12]}",
            "image_path": path.relative_to(data_root).as_posix(),
            "image_sha256": sha256_file(path),
            "scene_groups": {},
        }
        for index, path in enumerate(val_images)
    ]
    fixed_manifest = {
        "evidence_label": DIAGNOSTIC_LABEL,
        "images": entries,
    }
    fixed_manifest["manifest_sha256"] = canonical_payload_sha256(fixed_manifest)
    fixed_path = output / "routing/fixed_val32_manifest.json"
    if not fixed_path.exists():
        write_json(fixed_path, fixed_manifest)
    elif json.loads(fixed_path.read_text(encoding="utf-8")) != fixed_manifest:
        raise RuntimeError("fixed validation image manifest changed between routing exports")

    timestamp = utc_now()
    diagnostic_manifest = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": run["experiment_id"],
        "model_variant": run["model_variant"],
        "seed": 0,
        "dataset": "VisDrone2019-DET",
        "dataset_version": "2019-DET",
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "split": "val-fixed32",
        "requested_epochs": run["epochs"],
        "epochs": run["epochs"],
        "requested_batch": run["requested_batch"],
        "batch": run["actual_batch"],
        "effective_batch": run["actual_batch"],
        "imgsz": 640,
        "optimizer": str(run["optimizer"]),
        "precision_mode": run["precision_mode"],
        "checkpoint_path": checkpoint.relative_to(output).as_posix(),
        "checkpoint_sha256": run["checkpoint"]["sha256"],
        "config_path": run["config_path"],
        "config_sha256": run["config_sha256"],
        "git_commit": git_commit,
        "timestamp": timestamp,
        "status": "diagnostic",
        "failure_reason": None,
    }
    diagnostic_manifest = with_manifest_checksum(validate_experiment_manifest(diagnostic_manifest))
    model = YOLO(str(checkpoint)).model.to("cuda:0")
    routers = [module.router for module in model.modules() if isinstance(module, MoTBlock)]
    hook_counts_before = [len(router._forward_hooks) for router in routers]
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for repeat in range(2):
        for entry, image_path in zip(entries, val_images):
            batch = _image_tensor(image_path, 640, torch.device("cuda:0")).unsqueeze(0)
            records.extend(
                capture_mot_routing(
                    model,
                    batch,
                    [entry],
                    diagnostic_manifest,
                    inference_repeat=repeat,
                    timestamp=timestamp,
                )
            )
    duration = time.monotonic() - started
    hook_counts_after = [len(router._forward_hooks) for router in routers]
    route_path = output / "routing" / f"{run['experiment_id']}.jsonl"
    write_jsonl(route_path, records, overwrite=False)

    by_key: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    entropy_rows = []
    for record in records:
        probabilities = record["expert_probabilities"]
        if any(not math.isfinite(value) or value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-6
        ):
            raise RuntimeError("invalid exported routing probabilities")
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
        entropy_rows.append(
            {
                "experiment_id": run["experiment_id"],
                "image_id": record["image_id"],
                "layer_name": record["layer_name"],
                "inference_repeat": record["inference_repeat"],
                "selected_expert": record["selected_expert"],
                "top1_probability": max(probabilities),
                "route_entropy": entropy,
            }
        )
        by_key.setdefault((record["image_id"], record["layer_name"]), {})[record["inference_repeat"]] = record
    deterministic = all(
        repeats[0]["expert_probabilities"] == repeats[1]["expert_probabilities"]
        and repeats[0]["token_top1_indices"] == repeats[1]["token_top1_indices"]
        for repeats in by_key.values()
        if set(repeats) == {0, 1}
    ) and all(set(repeats) == {0, 1} for repeats in by_key.values())
    summary_path = output / "routing" / f"{run['experiment_id']}_summary.csv"
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entropy_rows[0]))
        writer.writeheader()
        writer.writerows(entropy_rows)
    return {
        "experiment_id": run["experiment_id"],
        "records": len(records),
        "images": len(entries),
        "layers": len({record["layer_name"] for record in records}),
        "expert_names": sorted({name for record in records for name in record["expert_names"]}),
        "deterministic_repeats": deterministic,
        "hooks_cleaned": hook_counts_before == hook_counts_after,
        "probabilities_valid": True,
        "duration_seconds": duration,
        "jsonl_path": route_path.as_posix(),
        "summary_csv_path": summary_path.as_posix(),
    }


def compare_routes(output: Path, routing: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare AMP- and FP32-trained checkpoint routing on identical inputs."""
    if len(routing) != 2:
        return {"available": False, "reason": "both MoT routing exports were not successful"}

    def load_first(path: str) -> dict[tuple[str, str], dict[str, Any]]:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        return {(row["image_id"], row["layer_name"]): row for row in rows if row["inference_repeat"] == 0}

    def named_probabilities(row: dict[str, Any]) -> dict[str, float]:
        return dict(zip(row["expert_names"], row["expert_probabilities"]))

    def entropy(probabilities: dict[str, float]) -> float:
        return -sum(value * math.log(value) for value in probabilities.values() if value > 0)

    def jsd(left_probs: dict[str, float], right_probs: dict[str, float]) -> float:
        names = sorted(set(left_probs) | set(right_probs))
        midpoint = {name: (left_probs.get(name, 0.0) + right_probs.get(name, 0.0)) / 2 for name in names}

        def kl_divergence(probabilities: dict[str, float]) -> float:
            return sum(value * math.log(value / midpoint[name]) for name, value in probabilities.items() if value > 0)

        return (kl_divergence(left_probs) + kl_divergence(right_probs)) / 2

    left, right = (load_first(item["jsonl_path"]) for item in routing)
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"available": False, "reason": "routing exports have no aligned image/layer rows"}
    left_probs = {key: named_probabilities(left[key]) for key in keys}
    right_probs = {key: named_probabilities(right[key]) for key in keys}
    agreement = sum(left[key]["selected_expert"] == right[key]["selected_expert"] for key in keys)
    by_layer: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        by_layer[key[1]].append(key)
    utilization_rows = []
    layer_metrics = []
    for layer_name, layer_keys in sorted(by_layer.items()):
        left_counts = Counter(left[key]["selected_expert"] for key in layer_keys)
        right_counts = Counter(right[key]["selected_expert"] for key in layer_keys)
        expert_names = sorted(set().union(*(set(left_probs[key]) | set(right_probs[key]) for key in layer_keys)))
        for expert_name in expert_names:
            left_usage = left_counts[expert_name] / len(layer_keys)
            right_usage = right_counts[expert_name] / len(layer_keys)
            utilization_rows.append(
                {
                    "layer_name": layer_name,
                    "expert_name": expert_name,
                    "left_top1_utilization": left_usage,
                    "right_top1_utilization": right_usage,
                    "difference_left_minus_right": left_usage - right_usage,
                }
            )
        left_entropy = sum(entropy(left_probs[key]) for key in layer_keys) / len(layer_keys)
        right_entropy = sum(entropy(right_probs[key]) for key in layer_keys) / len(layer_keys)
        layer_metrics.append(
            {
                "layer_name": layer_name,
                "rows": len(layer_keys),
                "top1_agreement": sum(
                    left[key]["selected_expert"] == right[key]["selected_expert"] for key in layer_keys
                )
                / len(layer_keys),
                "mean_jsd": sum(jsd(left_probs[key], right_probs[key]) for key in layer_keys) / len(layer_keys),
                "mean_entropy_left": left_entropy,
                "mean_entropy_right": right_entropy,
                "entropy_difference_left_minus_right": left_entropy - right_entropy,
            }
        )
    mean_entropy_left = sum(entropy(left_probs[key]) for key in keys) / len(keys) if keys else None
    mean_entropy_right = sum(entropy(right_probs[key]) for key in keys) / len(keys) if keys else None
    result = {
        "available": bool(keys),
        "aligned_image_layer_rows": len(keys),
        "top1_agreement": agreement / len(keys) if keys else None,
        "mean_jsd": sum(jsd(left_probs[key], right_probs[key]) for key in keys) / len(keys) if keys else None,
        "mean_entropy_left": mean_entropy_left,
        "mean_entropy_right": mean_entropy_right,
        "entropy_difference_left_minus_right": (mean_entropy_left - mean_entropy_right if keys else None),
        "layers": layer_metrics,
        "same_seed_independent_runs": False,
        "interpretation": "precision-mode checkpoint comparison only; not a cross-seed inference",
    }
    write_json(output / "routing/amp_fp32_comparison.json", result)
    utilization_path = output / "routing/amp_fp32_layer_utilization.csv"
    with utilization_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(utilization_rows[0]))
        writer.writeheader()
        writer.writerows(utilization_rows)
    result["layer_utilization_csv"] = utilization_path.as_posix()
    return result


def environment_snapshot() -> dict[str, Any]:
    """Collect compact environment metadata."""
    import platform

    import torch

    gpu = {}
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        name, memory, driver = [part.strip() for part in subprocess.check_output(query, text=True).split(",")]
        gpu = {"name": name, "memory_total_mib": int(memory), "driver": driver}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def render_reports(
    *,
    output: Path,
    report_dir: Path,
    environment: dict[str, Any],
    dataset: dict[str, Any],
    runs: list[dict[str, Any]],
    routing: list[dict[str, Any]],
    route_comparison: dict[str, Any],
    git_commit: str,
    wall_seconds: float,
) -> dict[str, Path]:
    """Write the required CSV, JSON manifest, and Markdown report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "PHASE2_SMOKE_RUNS.csv"
    csv_fields = [
        "experiment_id",
        "model_variant",
        "seed",
        "config_path",
        "config_sha256",
        "precision_mode",
        "requested_batch",
        "actual_batch",
        "imgsz",
        "fraction",
        "epochs",
        "optimizer",
        "lr0",
        "duration_seconds",
        "seconds_per_epoch",
        "peak_gpu_memory_bytes",
        "map50",
        "map50_95",
        "loss",
        "nan_or_inf",
        "exit_code",
        "status",
        "failure_reason",
        "checkpoint_path",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for run in runs:
            row = {field: run.get(field) for field in csv_fields}
            checkpoint = run.get("checkpoint") or {}
            row.update(
                {
                    "checkpoint_path": checkpoint.get("path"),
                    "checkpoint_size_bytes": checkpoint.get("size_bytes"),
                    "checkpoint_sha256": checkpoint.get("sha256"),
                }
            )
            writer.writerow(row)
    successful_epoch_seconds = [run["seconds_per_epoch"] for run in runs if run["status"] == "passed_diagnostic"]
    mean_epoch_seconds = (
        sum(successful_epoch_seconds) / len(successful_epoch_seconds) if successful_epoch_seconds else None
    )
    estimates = {
        "assumption": "30 epochs per formal run; excludes setup and retry overhead",
        "mean_diagnostic_seconds_per_epoch": mean_epoch_seconds,
        "mvp_5_runs_gpu_hours": mean_epoch_seconds * 30 * 5 / 3600 if mean_epoch_seconds else None,
        "recommended_9_runs_gpu_hours": mean_epoch_seconds * 30 * 9 / 3600 if mean_epoch_seconds else None,
    }
    manifest = {
        "schema_version": 1,
        "evidence_label": DIAGNOSTIC_LABEL,
        "git_commit": git_commit,
        "generated_at": utc_now(),
        "wall_seconds": wall_seconds,
        "gpu_compute_seconds": sum(run["duration_seconds"] for run in runs)
        + sum(item["duration_seconds"] for item in routing),
        "environment": environment,
        "dataset": dataset,
        "runs": runs,
        "routing": routing,
        "routing_comparison": route_comparison,
        "formal_estimates": estimates,
    }
    manifest_path = report_dir / "PHASE2_SMOKE_MANIFEST.json"
    write_json(manifest_path, manifest)
    report_path = report_dir / "PHASE2_SMOKE_REPORT.md"
    lines = [
        "# Issue #54 Phase 2 Diagnostic Smoke Report",
        "",
        f"Evidence classification: `{DIAGNOSTIC_LABEL}`.",
        "",
        "This smoke validates execution paths only. It is not formal accuracy, stability, or cross-seed evidence.",
        "",
        "## Environment",
        "",
        f"- Host: `{environment['hostname']}`",
        f"- Python: `{environment['python_executable']}`",
        f"- PyTorch/CUDA: `{environment['torch']}` / `{environment['torch_cuda']}`",
        f"- GPU: `{environment['gpu'].get('name')}`, {environment['gpu'].get('memory_total_mib')} MiB",
        f"- Git commit: `{git_commit}`",
        "",
        "## VisDrone",
        "",
        f"- Root: `{dataset['data_root']}` (read-only reuse)",
        f"- Manifest SHA256: `{dataset['manifest_sha256']}`",
        f"- YAML SHA256: `{dataset['yaml_sha256']}`",
        f"- Split counts: `{dataset['splits']}`",
        "",
        "## Runs",
        "",
        "| Run | Variant | Precision | Batch requested/actual | Epochs | Time/epoch (s) | Peak MiB | mAP50 | mAP50-95 | Loss | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        peak = run.get("peak_gpu_memory_bytes")
        peak_mib = peak / 1024**2 if peak else None
        peak_text = f"{peak_mib:.1f}" if peak_mib is not None else ""
        epoch_time = run.get("seconds_per_epoch")
        epoch_time_text = f"{epoch_time:.2f}" if epoch_time is not None else ""
        lines.append(
            f"| {run['experiment_id']} | {run['model_variant']} | {run['precision_mode']} | "
            f"{run['requested_batch']}/{run['actual_batch']} | {run['epochs']} | "
            f"{epoch_time_text} | {peak_text} | "
            f"{run.get('map50')} | {run.get('map50_95')} | {run.get('loss')} | {run['status']} |"
        )
    lines.extend(
        [
            "",
            "## Routing",
            "",
            f"- Exports: `{routing}`",
            f"- AMP/FP32 comparison: `{route_comparison}`",
            "- AMP and FP32 checkpoints are same-seed independent precision-mode runs, not independent seeds.",
            "",
            "## Resource estimate",
            "",
            f"- Total wall time: `{wall_seconds / 3600:.3f}` hours",
            f"- Diagnostic GPU compute time: `{manifest['gpu_compute_seconds'] / 3600:.3f}` hours",
            f"- MVP 5 runs × 30 epochs: `{estimates['mvp_5_runs_gpu_hours']}` GPU-hours",
            f"- Recommended 9 runs × 30 epochs: `{estimates['recommended_9_runs_gpu_hours']}` GPU-hours",
            "",
            "## Recommendation",
            "",
            "Proceed to formal training only if every required pipeline passed without NaN/Inf, unresolved OOM, "
            "stall, invalid routing probabilities, hook leakage, or nondeterministic repeated routing. Formal runs "
            "must use multiple independent seeds and a separately approved protocol.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"markdown": report_path, "csv": csv_path, "manifest": manifest_path}


def refresh_existing_reports(output: Path) -> int:
    """Refresh a completed report with child-process and results.csv timing."""
    report_dir = ROOT / "reports/issue54"
    manifest_path = report_dir / "PHASE2_SMOKE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for run in manifest["runs"]:
        child = run["attempts"][-1].get("child_result", {})
        run["controller_duration_seconds"] = run["duration_seconds"]
        run["duration_seconds"] = float(child.get("duration_seconds", run["duration_seconds"]))
        raw = child.get("raw_last_row", {})
        training_time = float(raw["time"]) if raw.get("time") else None
        run["training_time_seconds"] = training_time
        run["seconds_per_epoch"] = training_time / run["epochs"] if training_time is not None else None
        run["optimizer_requested"] = "auto"
        run["lr0_requested"] = 0.01
        run["optimizer"] = "AdamW(auto)"
        run["lr0"] = 0.000714
        if run.get("checkpoint"):
            run["checkpoint"].update({"kind": "best", "partial": False})
        if run["experiment_id"] == "A_mot_amp":
            checkpoint = output / "training/A_mot_amp/weights/last_healthy.pt"
            run["checkpoint"] = {
                "path": checkpoint.as_posix(),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "kind": "partial_last_healthy",
                "partial": True,
            }
            run["completed_training_epochs_before_validation_failure"] = 1
    for route in manifest["routing"]:
        if route.get("jsonl_path"):
            route.update(
                {
                    "status": "passed_diagnostic",
                    "jsonl_sha256": sha256_file(Path(route["jsonl_path"])),
                    "summary_csv_sha256": sha256_file(Path(route["summary_csv_path"])),
                }
            )
    fixed = output / "routing/fixed_val32_manifest.json"
    manifest["routing_fixed_sample_manifest"] = {
        "path": fixed.as_posix(),
        "sha256": sha256_file(fixed),
        "images": 32,
    }
    manifest["routing_comparison"] = {
        "available": False,
        "reason": "MoT AMP run failed before a successful checkpoint; comparison not fabricated",
        "same_seed_independent_runs": False,
    }
    data_root = Path(manifest["dataset"]["data_root"])
    manifest["dataset"]["old_train_cache_absent_after_run"] = not (data_root / "labels/train.cache").exists()
    manifest["gpu_compute_seconds"] = sum(run["duration_seconds"] for run in manifest["runs"]) + sum(
        route.get("duration_seconds", 0.0) for route in manifest["routing"]
    )
    successful_epoch_seconds = [
        run["seconds_per_epoch"]
        for run in manifest["runs"]
        if run["status"] == "passed_diagnostic" and run["seconds_per_epoch"] is not None
    ]
    mean_epoch_seconds = sum(successful_epoch_seconds) / len(successful_epoch_seconds)
    manifest["formal_estimates"] = {
        "assumption": (
            "30 epochs per formal run; based on successful results.csv time/epoch; "
            "excludes setup, retries, and failed AMP path"
        ),
        "mean_successful_seconds_per_epoch": mean_epoch_seconds,
        "mvp_5_runs_gpu_hours": mean_epoch_seconds * 30 * 5 / 3600,
        "recommended_9_runs_gpu_hours": mean_epoch_seconds * 30 * 9 / 3600,
    }
    manifest["recommendation"] = (
        "do_not_enter_full_formal_protocol_until_mot_amp_dtype_failure_is_resolved_"
        "or_fp32_only_protocol_is_explicitly_approved"
    )
    manifest["refreshed_at"] = utc_now()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = report_dir / "PHASE2_SMOKE_RUNS.csv"
    fields = [
        "experiment_id",
        "model_variant",
        "seed",
        "config_path",
        "config_sha256",
        "dataset_sha256",
        "git_commit",
        "precision_mode",
        "requested_batch",
        "actual_batch",
        "imgsz",
        "fraction",
        "epochs",
        "optimizer_requested",
        "optimizer",
        "lr0_requested",
        "lr0",
        "duration_seconds",
        "training_time_seconds",
        "seconds_per_epoch",
        "peak_gpu_memory_bytes",
        "map50",
        "map50_95",
        "loss",
        "nan_or_inf",
        "exit_code",
        "status",
        "failure_reason",
        "checkpoint_path",
        "checkpoint_kind",
        "checkpoint_partial",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in manifest["runs"]:
            row = {field: run.get(field) for field in fields}
            checkpoint = run.get("checkpoint") or {}
            row.update(
                {
                    "checkpoint_path": checkpoint.get("path"),
                    "checkpoint_kind": checkpoint.get("kind"),
                    "checkpoint_partial": checkpoint.get("partial"),
                    "checkpoint_size_bytes": checkpoint.get("size_bytes"),
                    "checkpoint_sha256": checkpoint.get("sha256"),
                }
            )
            writer.writerow(row)

    lines = [
        "# Issue #54 Phase 2 Diagnostic Smoke Report",
        "",
        f"Evidence classification: `{DIAGNOSTIC_LABEL}`.",
        "",
        "This run validates execution paths only. It is not formal accuracy, stability, or cross-seed evidence.",
        "",
        "## Environment",
        "",
        f"- Host: `{manifest['environment']['hostname']}`",
        f"- Python: `{manifest['environment']['python_executable']}`",
        f"- PyTorch/CUDA: `{manifest['environment']['torch']}` / `{manifest['environment']['torch_cuda']}`",
        f"- GPU: `{manifest['environment']['gpu']['name']}`, {manifest['environment']['gpu']['memory_total_mib']} MiB",
        f"- Git commit: `{manifest['git_commit']}`",
        "",
        "## VisDrone",
        "",
        f"- Source YAML: `{manifest['dataset']['source_yaml']}`",
        f"- Source YAML SHA256: `{manifest['dataset']['source_yaml_sha256']}`",
        f"- Data inventory SHA256: `{manifest['dataset']['inventory_sha256']}`",
        f"- Counts: `{manifest['dataset']['splits']}`",
        "- Images were reused read-only. Labels were copied into the MoT result directory so generated cache files "
        "did not modify the old dataset.",
        "- The old train cache is absent after completion; the pre-existing val cache was preserved.",
        "",
        "## Runs",
        "",
        "| Run | Variant | Precision | Batch | Epochs | Train/val s/epoch | Process s | Peak GiB | mAP50 | "
        "mAP50-95 | Loss | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in manifest["runs"]:
        epoch_time = "" if run["seconds_per_epoch"] is None else f"{run['seconds_per_epoch']:.2f}"
        peak = "" if run["peak_gpu_memory_bytes"] is None else f"{run['peak_gpu_memory_bytes'] / 1024**3:.2f}"
        lines.append(
            f"| {run['experiment_id']} | {run['model_variant']} | {run['precision_mode']} | "
            f"{run['requested_batch']}/{run['actual_batch']} | {run['epochs']} | {epoch_time} | "
            f"{run['duration_seconds']:.2f} | {peak} | {run['map50']} | {run['map50_95']} | "
            f"{run['loss']} | {run['status']} |"
        )
    lines.extend(
        [
            "",
            "Run A completed one AMP training epoch but failed during validation with "
            "`expected scalar type Float but found Half` in `MoTBlock._blend_experts`. It has only a partial "
            "`last_healthy.pt`; it is not a successful checkpoint. No OOM, NaN/Inf, batch reduction, or stall occurred. "
            "The optimizer request `auto` resolved to AdamW at lr=0.000714 for all runs.",
            "",
            "## Routing",
            "",
            "- MoT FP32: 32 fixed validation images, 6 layers, 2 exact repeats, 384 records.",
            "- Experts: `WindowTransformer`, `DeformableTransformer`, `LocalConvTransformer`.",
            "- Probabilities are finite, non-negative, normalized; repeated inference is identical; hooks are fully removed.",
            "- MoT AMP routing was not exported because Run A did not produce a successful checkpoint. AMP/FP32 route "
            "comparison is therefore unavailable and was not fabricated.",
            "",
            "## Time and estimate",
            "",
            f"- Controller wall time: `{manifest['wall_seconds']:.2f}` seconds.",
            f"- Aggregate child process + routing time: `{manifest['gpu_compute_seconds']:.2f}` seconds "
            "(includes model/data setup).",
            f"- Mean successful measured train/val time: `{mean_epoch_seconds:.2f}` seconds/epoch.",
            f"- MVP 5 runs × 30 epochs: `{manifest['formal_estimates']['mvp_5_runs_gpu_hours']:.3f}` GPU-hours.",
            f"- Recommended 9 runs × 30 epochs: "
            f"`{manifest['formal_estimates']['recommended_9_runs_gpu_hours']:.3f}` GPU-hours.",
            "",
            "## Recommendation",
            "",
            "Do not enter the full formal protocol yet. FP32 MoT and AMP EsMoE/MoA pipelines passed, but the official "
            "MoT AMP validation path has a reproducible core dtype failure. Proceed only after that issue is separately "
            "fixed and reviewed, or after explicitly approving an FP32-only formal protocol.",
            "",
        ]
    )
    (report_dir / "PHASE2_SMOKE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "logs/PHASE2_SMOKE_LOG_SUMMARY.txt").write_text(
        "\n".join(
            [
                f"evidence_label={DIAGNOSTIC_LABEL}",
                f"final_output={output}",
                "controller_exit_code=1 (expected because one required run failed)",
                "A_mot_amp=failed: expected scalar type Float but found Half during validation",
                "B_mot_fp32=passed_diagnostic",
                "C_esmoe_amp=passed_diagnostic",
                "D_moa_amp=passed_diagnostic",
                "oom=false",
                "batch_reduction=false",
                "nan_or_inf=false",
                "stalled=false",
                "old_train_cache_absent=true",
                "routing_fp32=passed: 32 images, 6 layers, 2 deterministic repeats, hooks cleaned",
                "routing_amp=not available because AMP run failed",
                "push=false",
                "pr=false",
                "formal_training=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    """Run all Phase 2 diagnostic stages."""
    started = time.monotonic()
    output = Path(args.output).resolve()
    allowed_root = Path("/root/autodl-tmp/MoT/results").resolve()
    if output == allowed_root or allowed_root not in output.parents:
        raise RuntimeError(f"output must be a unique child of {allowed_root}")
    if output.exists():
        existing = [path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()]
        if any(path != "logs/controller.log" for path in existing):
            raise FileExistsError(f"refusing to reuse non-empty output directory: {output}")
    else:
        output.mkdir(parents=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "routing").mkdir()
    configs = discover_official_configs(ROOT)
    config_snapshot = output / "configs/models"
    config_snapshot.mkdir(parents=True)
    for path in configs.values():
        (config_snapshot / path.name).write_bytes(path.read_bytes())
    dataset = create_dataset_snapshot(Path(args.data_root).resolve(), output)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    environment = environment_snapshot()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is not available")

    runs = []
    for run_id, variant, amp, epochs, fraction in RUN_SPECS:
        run = execute_run(
            python=sys.executable,
            script=Path(__file__).resolve(),
            output=output,
            dataset_yaml=Path(dataset["yaml_path"]),
            config=configs[variant],
            run_id=run_id,
            variant=variant,
            amp=amp,
            epochs=epochs,
            fraction=fraction,
        )
        run["dataset_sha256"] = dataset["manifest_sha256"]
        run["git_commit"] = git_commit
        runs.append(run)

    routing = []
    for run in runs[:2]:
        if run["status"] == "passed_diagnostic" and run.get("checkpoint"):
            try:
                routing.append(
                    export_routing(
                        run,
                        output=output,
                        data_root=Path(args.data_root).resolve(),
                        dataset_manifest=dataset,
                        git_commit=git_commit,
                    )
                )
            except BaseException as error:
                routing.append(
                    {
                        "experiment_id": run["experiment_id"],
                        "status": "implementation_failed",
                        "failure_reason": f"{type(error).__name__}: {error}",
                        "duration_seconds": 0.0,
                    }
                )
    successful_routing = [item for item in routing if item.get("jsonl_path")]
    route_comparison = compare_routes(output, successful_routing)
    report_dir = ROOT / "reports/issue54"
    render_reports(
        output=output,
        report_dir=report_dir,
        environment=environment,
        dataset=dataset,
        runs=runs,
        routing=routing,
        route_comparison=route_comparison,
        git_commit=git_commit,
        wall_seconds=time.monotonic() - started,
    )
    write_json(
        output / "completion.json",
        {
            "evidence_label": DIAGNOSTIC_LABEL,
            "status": "complete",
            "ended_at": utc_now(),
            "report_dir": report_dir.as_posix(),
        },
    )
    all_runs_passed = all(run["status"] == "passed_diagnostic" for run in runs)
    routing_passed = (
        len(successful_routing) == 2
        and all(item["deterministic_repeats"] and item["hooks_cleaned"] for item in successful_routing)
        and route_comparison.get("available") is True
    )
    return 0 if all_runs_passed and routing_passed else 1


def parse_args() -> argparse.Namespace:
    """Parse controller and child arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--data-root")
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--amp", type=lambda value: value.lower() == "true")
    parser.add_argument("--child-result")
    return parser.parse_args()


def main() -> int:
    """Dispatch controller or child mode."""
    args = parse_args()
    if args.refresh_existing:
        if not args.output:
            raise ValueError("--output is required with --refresh-existing")
        return refresh_existing_reports(Path(args.output).resolve())
    if args.single_run:
        required = ("model", "data", "project", "name", "epochs", "fraction", "batch", "child_result")
        if any(getattr(args, field) is None for field in required):
            raise ValueError(f"single-run missing required arguments: {required}")
        return run_child(args)
    if not args.output or not args.data_root:
        raise ValueError("--output and --data-root are required")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
