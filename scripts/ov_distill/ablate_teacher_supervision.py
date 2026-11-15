#!/usr/bin/env python3
"""Compare student training with single-teacher and multi-teacher supervision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.convert import run_convert  # noqa: E402
from scripts.ov_distill.reliability import assert_disjoint_manifests  # noqa: E402
from scripts.ov_distill.split_audit import validate_training_eval_split  # noqa: E402


ARMS = ("single_teacher", "multi_teacher")
METRICS = ("mAP50", "mAP50-95", "precision", "recall")


def _fingerprint(path: Path | None) -> dict[str, Any] | None:
    """Return a content hash and file size for an input artifact."""
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _count_boxes_by_image(grounding_json: Path, image_root: Path) -> dict[str, int]:
    """Count exported boxes by their resolved image path."""
    payload = json.loads(grounding_json.read_text(encoding="utf-8"))
    id_to_path = {}
    for row in payload["images"]:
        image = Path(row["file_name"])
        if not image.is_absolute():
            image = image_root / image
        id_to_path[row["id"]] = str(image.resolve())
    counts = {path: 0 for path in id_to_path.values()}
    for annotation in payload["annotations"]:
        counts[id_to_path[annotation["image_id"]]] += 1
    return counts


def prepare_datasets(
    *,
    out: Path,
    single_cleaned: Path,
    multi_cleaned: Path,
    single_negatives: Path | None,
    multi_negatives: Path | None,
    train_manifest: Path,
    eval_manifest: Path,
    image_root: Path,
    class_split: Path,
) -> dict[str, Any]:
    """Export both arms with a shared split and uniform box weights, then audit them."""
    out = Path(out).resolve()
    train_manifest = Path(train_manifest).resolve()
    eval_manifest = Path(eval_manifest).resolve()
    image_root = Path(image_root).resolve()
    class_split = Path(class_split).resolve()
    single_cleaned, multi_cleaned = Path(single_cleaned).resolve(), Path(multi_cleaned).resolve()
    single_negatives = Path(single_negatives).resolve() if single_negatives else None
    multi_negatives = Path(multi_negatives).resolve() if multi_negatives else None

    for path in (single_cleaned, multi_cleaned, train_manifest, eval_manifest, class_split):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (single_negatives, multi_negatives):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if single_negatives is None and (single_cleaned.parent / "negatives.jsonl").is_file():
        single_negatives = single_cleaned.parent / "negatives.jsonl"
    if multi_negatives is None and (multi_cleaned.parent / "negatives.jsonl").is_file():
        multi_negatives = multi_cleaned.parent / "negatives.jsonl"
    if single_cleaned == multi_cleaned:
        raise ValueError("single-teacher and multi-teacher inputs must be different files")

    train_images, eval_images = assert_disjoint_manifests(
        train_manifest, eval_manifest, first_name="training", second_name="evaluation"
    )
    missing = sorted(path for path in train_images | eval_images if not Path(path).is_file())
    if missing:
        raise FileNotFoundError(f"manifests reference {len(missing)} missing images (examples: {missing[:5]})")
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    report_path = out / "teacher_ablation_manifest.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing ablation report: {report_path}")
    dataset_dirs = [out / data_dir / arm for data_dir in ("data", "data_paired") for arm in ARMS]
    nonempty_dirs = [path for path in dataset_dirs if path.exists() and any(path.iterdir())]
    if nonempty_dirs:
        raise FileExistsError(f"refusing to overwrite existing teacher-ablation datasets: {nonempty_dirs}")
    if (out / "paired_train_manifest.jsonl").exists():
        paired_path = out / "paired_train_manifest.jsonl"
        raise FileExistsError(f"refusing to overwrite an existing paired manifest: {paired_path}")

    sources = {
        "single_teacher": {
            "cleaned": _fingerprint(single_cleaned),
            "negatives": _fingerprint(single_negatives),
        },
        "multi_teacher": {
            "cleaned": _fingerprint(multi_cleaned),
            "negatives": _fingerprint(multi_negatives),
        },
    }
    common = {
        "fmt": "grounding",
        "weight_mode": "uniform",
        "include_manifest": train_manifest,
        "image_root": image_root,
        "class_split": class_split,
    }
    arms: dict[str, dict[str, Any]] = {}
    box_counts_by_arm: dict[str, dict[str, int]] = {}
    for arm, cleaned, negatives in (
        ("single_teacher", single_cleaned, single_negatives),
        ("multi_teacher", multi_cleaned, multi_negatives),
    ):
        dataset_dir = out / "data" / arm
        stats = run_convert(cleaned=cleaned, negatives=negatives, out=dataset_dir, **common)
        grounding_json = dataset_dir / "grounding_train.json"
        if stats.get("images", 0) == 0 or stats.get("boxes", 0) == 0:
            raise ValueError(f"{arm} export has no supervised positive boxes; refusing a meaningless comparison")
        audit = validate_training_eval_split(
            train_manifest=train_manifest,
            eval_manifest=eval_manifest,
            grounding_json=grounding_json,
            image_root=image_root,
            class_split=class_split,
        )
        box_counts = _count_boxes_by_image(grounding_json, image_root)
        box_counts_by_arm[arm] = box_counts
        positive_images = sum(count > 0 for count in box_counts.values())
        arms[arm] = {
            **stats,
            "dataset_json": str(grounding_json.resolve()),
            "positive_images": positive_images,
            "positive_image_coverage": positive_images / len(train_images),
            "all_supervised_image_coverage": stats["images"] / len(train_images),
            "split_audit": audit,
            "source_artifacts": sources[arm],
        }

    single_paths = set(box_counts_by_arm["single_teacher"])
    multi_paths = set(box_counts_by_arm["multi_teacher"])
    shared_paths = single_paths & multi_paths
    if not shared_paths:
        raise ValueError(
            "the two teachers have no overlapping supervised images; paired student comparison is impossible"
        )
    single_shared_boxes = sum(box_counts_by_arm["single_teacher"][path] for path in shared_paths)
    multi_shared_boxes = sum(box_counts_by_arm["multi_teacher"][path] for path in shared_paths)
    arms["single_teacher"]["boxes_on_shared_images"] = single_shared_boxes
    arms["multi_teacher"]["boxes_on_shared_images"] = multi_shared_boxes
    paired = {
        "single_teacher_images": len(single_paths),
        "multi_teacher_images": len(multi_paths),
        "shared_images": len(shared_paths),
        "single_teacher_only_images": len(single_paths - multi_paths),
        "multi_teacher_only_images": len(multi_paths - single_paths),
        "shared_image_fraction_of_training_manifest": len(shared_paths) / len(train_images),
        "single_teacher_boxes_on_shared_images": single_shared_boxes,
        "multi_teacher_boxes_on_shared_images": multi_shared_boxes,
        "shared_image_set_jaccard": len(shared_paths) / len(single_paths | multi_paths),
    }

    paired_manifest = out / "paired_train_manifest.jsonl"
    paired_manifest.parent.mkdir(parents=True, exist_ok=True)
    paired_manifest.write_text(
        "".join(json.dumps({"image": image}) + "\n" for image in sorted(shared_paths)), encoding="utf-8"
    )
    paired_arms: dict[str, dict[str, Any]] = {}
    for arm, cleaned, negatives in (
        ("single_teacher", single_cleaned, single_negatives),
        ("multi_teacher", multi_cleaned, multi_negatives),
    ):
        dataset_dir = out / "data_paired" / arm
        paired_stats = run_convert(
            cleaned=cleaned,
            negatives=negatives,
            out=dataset_dir,
            **{**common, "include_manifest": paired_manifest},
        )
        grounding_json = dataset_dir / "grounding_train.json"
        audit = validate_training_eval_split(
            train_manifest=paired_manifest,
            eval_manifest=eval_manifest,
            grounding_json=grounding_json,
            image_root=image_root,
            class_split=class_split,
        )
        if paired_stats.get("images", 0) != len(shared_paths):
            raise RuntimeError(
                f"{arm} paired export contains {paired_stats.get('images')} images; expected {len(shared_paths)}"
            )
        paired_arms[arm] = {
            **paired_stats,
            "dataset_json": str(grounding_json.resolve()),
            "split_audit": audit,
        }
    return {
        "comparison": "single_teacher_vs_multi_teacher_student_supervision",
        "protocol": {
            "train_manifest": str(train_manifest),
            "eval_manifest": str(eval_manifest),
            "train_manifest_images": len(train_images),
            "eval_manifest_images": len(eval_images),
            "image_root": str(image_root),
            "class_split": str(class_split),
            "train_manifest_sha256": _fingerprint(train_manifest)["sha256"],
            "eval_manifest_sha256": _fingerprint(eval_manifest)["sha256"],
            "class_split_sha256": _fingerprint(class_split)["sha256"],
            "box_weight_mode": "uniform",
            "same_train_eval_split_for_both_arms": True,
        },
        "sources": sources,
        "datasets": arms,
        "paired_datasets": paired_arms,
        "paired_train_manifest": str(paired_manifest.resolve()),
        "paired_image_comparison": paired,
        "training_config": None,
        "training_runs": [],
        "student_metric_delta_multi_minus_single": None,
        "training_launched": False,
    }


def _final_metrics(results_csv: Path) -> dict[str, Any] | None:
    """Read the final metrics row emitted by Ultralytics training."""
    if not results_csv.is_file():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    metrics: dict[str, Any] = {}
    for key, value in rows[-1].items():
        if not key or not value:
            continue
        try:
            metrics[key.strip()] = float(value)
        except ValueError:
            metrics[key.strip()] = value.strip()
    return metrics


def _metric_deltas(runs: list[dict[str, Any]]) -> dict[str, float] | None:
    """Calculate multi-teacher minus single-teacher holdout metric deltas."""
    by_arm = {run.get("arm"): run for run in runs}
    if not all(arm in by_arm and by_arm[arm].get("open_vocabulary_metrics") for arm in ARMS):
        return None
    single = by_arm["single_teacher"]["open_vocabulary_metrics"]
    multi = by_arm["multi_teacher"]["open_vocabulary_metrics"]
    return {metric: round(float(multi[metric]) - float(single[metric]), 6) for metric in METRICS}


def _run_command(command: list[str], label: str) -> None:
    """Run one external training or evaluation process with a contextual failure."""
    print(f"[teacher-ablation] {label}: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}") from exc


def _launch_training(args: argparse.Namespace, report: dict[str, Any], report_path: Path) -> None:
    """Train both student arms from the same initialization, then evaluate unseen classes."""
    required = (args.val_yaml, args.holdout_labels, args.holdout_images)
    if not all(required):
        raise ValueError("--run-training requires --val-yaml, --holdout-labels, and --holdout-images")
    train_script = Path(args.train_s2).resolve()
    eval_script = Path(args.eval_ov_holdout).resolve()
    model_cfg = Path(args.model_cfg)
    if not model_cfg.is_absolute():
        model_cfg = ROOT / model_cfg
    model_cfg = model_cfg.resolve()
    for path in (train_script, eval_script, Path(args.val_yaml).resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not model_cfg.is_file():
        raise FileNotFoundError(model_cfg)
    labels_dir = Path(args.holdout_labels).resolve()
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"holdout ground-truth label directory does not exist: {labels_dir}")
    holdout_images = Path(args.holdout_images).resolve()
    if not holdout_images.is_dir():
        raise FileNotFoundError(holdout_images)
    selected_train_manifest = (
        Path(report["paired_train_manifest"])
        if args.training_view == "paired"
        else Path(report["protocol"]["train_manifest"])
    )
    _, eval_images = assert_disjoint_manifests(
        selected_train_manifest,
        Path(report["protocol"]["eval_manifest"]),
        first_name="training",
        second_name="evaluation",
    )
    wrong_parent = sorted(path for path in eval_images if Path(path).resolve().parent != holdout_images)
    if wrong_parent:
        raise ValueError(
            "every evaluation image must be directly inside --holdout-images "
            f"(examples outside it: {wrong_parent[:5]})"
        )
    init_from = Path(args.init_from).resolve() if args.init_from else None
    if init_from is not None and not init_from.is_file():
        raise FileNotFoundError(init_from)

    prefix = args.name_prefix
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("--name-prefix may contain only letters, digits, dot, underscore, and hyphen")
    run_names = {arm: f"{prefix}_{arm}" for arm in ARMS}
    existing = [ROOT / "runs/ov_s2_train" / name for name in run_names.values()]
    conflicts = [str(path) for path in existing if path.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite existing student runs: {conflicts}")
    eval_dirs = [Path(args.out).resolve() / "eval" / arm for arm in ARMS]
    conflicts = [str(path) for path in eval_dirs if path.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite existing holdout evaluations: {conflicts}")

    for arm in ARMS:
        validate_training_eval_split(
            train_manifest=selected_train_manifest,
            eval_manifest=Path(report["protocol"]["eval_manifest"]),
            grounding_json=Path(
                report["paired_datasets"][arm]["dataset_json"]
                if args.training_view == "paired"
                else report["datasets"][arm]["dataset_json"]
            ),
            image_root=Path(report["protocol"]["image_root"]),
            class_split=Path(report["protocol"]["class_split"]),
            val_yaml=Path(args.val_yaml).resolve(),
        )

    report["training_launched"] = True
    report["training_config"] = {
        "training_view": args.training_view,
        "train_manifest": str(selected_train_manifest.resolve()),
        "train_images": len(
            assert_disjoint_manifests(
                selected_train_manifest,
                Path(report["protocol"]["eval_manifest"]),
                first_name="training",
                second_name="evaluation",
            )[0]
        ),
        "model_cfg": str(model_cfg),
        "initialization": _fingerprint(init_from) if init_from else "same deterministic random initialization",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "lr0": args.lr0,
        "seed": args.seed,
        "val_yaml": str(Path(args.val_yaml).resolve()),
        "holdout_labels": str(labels_dir),
        "holdout_images": str(holdout_images),
        "optimizer": "AdamW",
        "warmup_epochs": 0.0,
        "lrf": 1.0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for arm in ARMS:
        run_name = run_names[arm]
        run_dir = ROOT / "runs/ov_s2_train" / run_name
        run_record: dict[str, Any] = {"arm": arm, "name": run_name, "status": "training"}
        report["training_runs"].append(run_record)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            str(train_script),
            "--json-file",
            report["paired_datasets"][arm]["dataset_json"]
            if args.training_view == "paired"
            else report["datasets"][arm]["dataset_json"],
            "--img-root",
            report["protocol"]["image_root"],
            "--val-yaml",
            str(Path(args.val_yaml).resolve()),
            "--train-manifest",
            str(selected_train_manifest.resolve()),
            "--val-manifest",
            report["protocol"]["eval_manifest"],
            "--class-split",
            report["protocol"]["class_split"],
            "--model-cfg",
            str(model_cfg),
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--lr0",
            str(args.lr0),
            "--seed",
            str(args.seed),
            "--name",
            run_name,
        ]
        if init_from is not None:
            command.extend(["--init-from", str(init_from)])
        try:
            _run_command(command, f"train {arm}")
        except Exception as exc:
            run_record["status"] = "training_failed"
            run_record["error"] = str(exc)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise

        checkpoint = run_dir / "weights" / "best.pt"
        results_csv = run_dir / "results.csv"
        if not checkpoint.is_file():
            run_record["status"] = "training_failed"
            run_record["error"] = f"training finished without expected checkpoint: {checkpoint}"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise FileNotFoundError(checkpoint)
        run_record.update(
            {
                "status": "training_complete",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _fingerprint(checkpoint)["sha256"],
                "results_csv": str(results_csv.resolve()),
                "final_training_metrics": _final_metrics(results_csv),
            }
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        eval_out = Path(args.out).resolve() / "eval" / arm
        eval_command = [
            sys.executable,
            str(eval_script),
            "--weights",
            str(checkpoint.resolve()),
            "--labels",
            str(labels_dir),
            "--images",
            str(holdout_images),
            "--train-manifest",
            str(selected_train_manifest.resolve()),
            "--eval-manifest",
            report["protocol"]["eval_manifest"],
            "--class-split",
            report["protocol"]["class_split"],
            "--out",
            str(eval_out),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
        ]
        try:
            _run_command(eval_command, f"unseen-class evaluation {arm}")
        except Exception as exc:
            run_record["status"] = "evaluation_failed"
            run_record["error"] = str(exc)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        metrics_path = eval_out / "ov_holdout_report.json"
        if not metrics_path.is_file():
            run_record["status"] = "evaluation_failed"
            run_record["error"] = f"holdout evaluator did not write {metrics_path}"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise FileNotFoundError(metrics_path)
        run_record["open_vocabulary_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_record["status"] = "complete"
        report["student_metric_delta_multi_minus_single"] = _metric_deltas(report["training_runs"])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Create matched supervision datasets and optionally train/evaluate both students."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-cleaned", required=True, help="cleaned single-teacher pseudo-box JSONL")
    parser.add_argument("--multi-cleaned", required=True, help="cleaned multi-teacher pseudo-box JSONL")
    parser.add_argument("--single-negatives", default="", help="single-teacher confirmed-negative JSONL")
    parser.add_argument("--multi-negatives", default="", help="multi-teacher confirmed-negative JSONL")
    parser.add_argument("--train-manifest", required=True, help="shared training image manifest for both arms")
    parser.add_argument("--val-manifest", required=True, help="shared held-out evaluation image manifest")
    parser.add_argument("--class-split", required=True, help="shared COCO-80 seen/unseen split JSON")
    parser.add_argument("--image-root", required=True, help="root directory used by GroundingDataset")
    parser.add_argument("--out", required=True, help="new output directory for both datasets and report")
    parser.add_argument("--run-training", action="store_true", help="train both students and evaluate unseen classes")
    parser.add_argument(
        "--training-view",
        choices=("paired", "full"),
        default="paired",
        help="paired trains on images supervised by both teachers; full includes each teacher's complete coverage",
    )
    parser.add_argument("--train-s2", default=str(ROOT / "scripts/ov_distill/train_s2.py"))
    parser.add_argument("--eval-ov-holdout", default=str(ROOT / "scripts/ov_distill/eval_ov_holdout.py"))
    parser.add_argument("--val-yaml", default="", help="YOLO validation YAML that exactly matches --val-manifest")
    parser.add_argument("--holdout-labels", default="", help="full COCO-class-id YOLO ground-truth label directory")
    parser.add_argument("--holdout-images", default="", help="directory directly containing evaluation images")
    parser.add_argument("--model-cfg", default="ultralytics/cfg/models/26/yoloe-26-master-n.yaml")
    parser.add_argument("--init-from", default="", help="shared initialization checkpoint; empty uses same seed")
    parser.add_argument("--name-prefix", default="ov_s2_teacher_ablation", help="unique prefix under runs/ov_s2_train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs, imgsz, and batch must be positive; workers must be non-negative")

    out = Path(args.out).resolve()
    report_path = out / "teacher_ablation_manifest.json"
    report = prepare_datasets(
        out=out,
        single_cleaned=Path(args.single_cleaned),
        multi_cleaned=Path(args.multi_cleaned),
        single_negatives=Path(args.single_negatives) if args.single_negatives else None,
        multi_negatives=Path(args.multi_negatives) if args.multi_negatives else None,
        train_manifest=Path(args.train_manifest),
        eval_manifest=Path(args.val_manifest),
        image_root=Path(args.image_root),
        class_split=Path(args.class_split),
    )
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_training:
        _launch_training(args, report, report_path)
    print(json.dumps({"manifest": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
