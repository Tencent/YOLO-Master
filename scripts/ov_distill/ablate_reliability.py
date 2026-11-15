#!/usr/bin/env python3
"""Prepare and optionally train a calibrated-weight vs uniform-weight ablation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.convert import run_convert


def main() -> None:
    """Create identical train splits with calibrated or unit box weights, then optionally train both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", required=True, help="cleaned pseudo-box JSONL")
    parser.add_argument("--calibration-map", required=True, help="output from calibrate_reliability.py")
    parser.add_argument("--train-manifest", required=True, help="training images, disjoint from calibration")
    parser.add_argument("--class-split", required=True, help="shared seen/unseen COCO-80 split JSON")
    parser.add_argument("--out", required=True, help="ablation output directory")
    parser.add_argument("--image-root", required=True, help="image root used by GroundingDataset")
    parser.add_argument("--negatives", default="", help="confirmed-negative JSONL; defaults beside --cleaned")
    parser.add_argument("--buckets", default="", help="optional comma-separated confidence buckets")
    parser.add_argument("--run-training", action="store_true", help="launch both S2 training runs after conversion")
    parser.add_argument("--name-prefix", default="ov_s2_reliability", help="unique name prefix under runs/ov_s2_train")
    parser.add_argument("--train-s2", default=str(ROOT / "scripts/ov_distill/train_s2.py"))
    parser.add_argument("--val-yaml", default="runs/ov_s2_val1k/val1k.yaml")
    parser.add_argument("--val-manifest", default="", help="independent image manifest represented by --val-yaml")
    parser.add_argument("--init-from", default="", help="same initialization checkpoint for both runs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-val", action="store_true", help="skip validation in both runs")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    common = {
        "cleaned": Path(args.cleaned).resolve(),
        "negatives": Path(args.negatives).resolve() if args.negatives else None,
        "image_root": Path(args.image_root).resolve(),
        "include_manifest": Path(args.train_manifest).resolve(),
        "class_split": Path(args.class_split).resolve(),
        "fmt": "grounding",
        "buckets": {value.strip() for value in args.buckets.split(",") if value.strip()} or None,
    }
    variants = {
        "calibrated": {"weight_mode": "calibrated", "calibration_map": Path(args.calibration_map).resolve()},
        "uniform": {"weight_mode": "uniform", "calibration_map": None},
    }
    stats = {}
    for variant, options in variants.items():
        stats[variant] = run_convert(out=out / "data" / variant, **common, **options)
    for key in ("images", "boxes", "confirmed_negatives", "negative_only_images"):
        if stats["calibrated"].get(key) != stats["uniform"].get(key):
            raise RuntimeError(f"ablation datasets differ in {key}: {stats['calibrated']} vs {stats['uniform']}")
    if stats["calibrated"]["images"] == 0 or stats["calibrated"]["boxes"] == 0:
        raise RuntimeError("training manifest produced no positive training boxes; refusing an empty ablation")

    report = {
        "comparison": "calibrated_reliability_weights_vs_uniform_weights",
        "datasets": stats,
        "class_split": str(Path(args.class_split).resolve()),
        "training_runs": [],
        "training_launched": bool(args.run_training),
    }
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "ablation_manifest.json"

    def save_report() -> None:
        """Persist preparation and each completed run so interrupted ablations remain auditable."""
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_report()
    if args.run_training:
        if not args.val_manifest:
            raise ValueError("--val-manifest is required with --run-training to prevent image leakage")
        train_script = Path(args.train_s2).resolve()
        val_yaml = Path(args.val_yaml)
        if not val_yaml.is_absolute():
            val_yaml = ROOT / val_yaml
        if not args.no_val and not val_yaml.exists():
            raise FileNotFoundError(f"validation yaml does not exist: {val_yaml} (or pass --no-val)")
        run_names = [f"{args.name_prefix}_{variant}" for variant in ("calibrated", "uniform")]
        existing_runs = [ROOT / "runs/ov_s2_train" / name for name in run_names]
        conflicts = [str(path) for path in existing_runs if path.exists()]
        if conflicts:
            raise FileExistsError(f"refusing to overwrite existing ablation runs: {conflicts}")
        for variant in ("calibrated", "uniform"):
            command = [
                sys.executable,
                str(train_script),
                "--json-file",
                str((out / "data" / variant / "grounding_train.json").resolve()),
                "--img-root",
                str(Path(args.image_root).resolve()),
                "--val-yaml",
                str(val_yaml.resolve()),
                "--train-manifest",
                str(Path(args.train_manifest).resolve()),
                "--val-manifest",
                str(Path(args.val_manifest).resolve()),
                "--class-split",
                str(Path(args.class_split).resolve()),
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
                f"{args.name_prefix}_{variant}",
            ]
            if args.init_from:
                command.extend(["--init-from", str(Path(args.init_from).resolve())])
            if args.no_val:
                command.append("--no-val")
            subprocess.run(command, cwd=ROOT, check=True)
            run_name = f"{args.name_prefix}_{variant}"
            result_csv = ROOT / "runs/ov_s2_train" / run_name / "results.csv"
            run_record = {"variant": variant, "name": run_name, "results_csv": str(result_csv)}
            if result_csv.exists():
                with result_csv.open("r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    metrics = {}
                    for key, value in rows[-1].items():
                        if not key or not value:
                            continue
                        try:
                            metrics[key.strip()] = float(value)
                        except ValueError:
                            metrics[key.strip()] = value.strip()
                    run_record["final_metrics"] = metrics
            report["training_runs"].append(run_record)
            save_report()

    print(json.dumps({"manifest": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
