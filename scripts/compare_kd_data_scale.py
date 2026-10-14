"""Compare nested training pools at matched KD update budgets against a frozen baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.suite import collect_environment
from scripts.evaluate_router_choices import RouterChoice, supported_layers
from scripts.evaluate_sparse_checkpoints import load_candidate_state
from scripts.finetune_sparse_detector import train_target, verify_manifest
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


def image_identities(rows):
    """Compare image and annotation identities independently of local storage paths."""
    return {(Path(row["image"]).name, row["image_sha256"], row["label_sha256"]) for row in rows}


def validate_baseline(report, old_manifest, new_manifest, expected):
    """Reject changed settings, nonnested pools or unequal actual sample/update budgets."""
    for key, value in expected.items():
        if report["settings"].get(key) != value:
            raise ValueError(f"baseline setting mismatch: {key}")
    if not report.get("baseline_stable"):
        raise ValueError("baseline run must be complete with stable controls")
    case = next((row for row in report["dev"] if row["name"] == "sparse_kd_long"), None)
    if case is None or case["kd_mode"] != "plain" or len(case["history"]) != 6:
        raise ValueError("requires the frozen six-epoch pure KD baseline")
    if not old_manifest["splits"]["train"] or len(new_manifest["splits"]["train"]) != 2 * len(
        old_manifest["splits"]["train"]
    ):
        raise ValueError("expanded training pool must double the baseline pool")
    old_ids = image_identities(old_manifest["splits"]["train"])
    if not old_ids < image_identities(new_manifest["splits"]["train"]):
        raise ValueError("training pools must be nested with unchanged labels")
    if image_identities(old_manifest["splits"]["dev"]) != image_identities(new_manifest["splits"]["dev"]):
        raise ValueError("development identities must remain unchanged")
    exposure = 3 * len(new_manifest["splits"]["train"])
    if exposure % expected["batch"] or any(row["detection_loss"] != 0 for row in case["history"]):
        raise ValueError("requires full batches and pure KD")
    if (
        sum(row["examples"] for row in case["history"]) != exposure
        or sum(row["optimizer_steps"] for row in case["history"]) != exposure // expected["batch"]
    ):
        raise ValueError("actual update/sample budgets differ")
    if case["training_routing"]["executed_experts_per_sample"] != 2:
        raise ValueError("baseline must actually execute Top-2")
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output, args.dataset = args.output.resolve(), args.dataset.resolve()
    if args.output.exists():
        parser.error("fresh output required")
    if args.baseline.name != "sparse_kd_long.state.pt":
        parser.error("baseline must be the recorded six-epoch sparse_kd_long checkpoint")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    base_hash = sha(args.weights)
    prior_path = args.baseline.parent / "report.json"
    prior = json.loads(prior_path.read_text())
    environment = collect_environment()
    for key in ("python", "torch", "torch_threads", "mps_device"):
        if environment.get(key) != prior["environment"].get(key):
            raise ValueError(f"baseline execution environment changed: {key}")
    for key, name in (("script_sha256", "finetune_sparse_detector.py"), ("loss_script_sha256", "sparse_kd_losses.py")):
        if prior[key] != sha(ROOT / "scripts" / name):
            raise ValueError(f"baseline training implementation changed: {name}")
    recorded = next(
        r for r in json.loads(args.baseline_evidence.read_text())["runs"] if r["settings"]["seed"] == args.seed
    )
    if (
        sha(args.baseline) != recorded["checkpoint_sha256"].get(args.baseline.name)
        or sha(prior_path) != recorded["report_sha256"]
    ):
        raise ValueError("baseline state/report changed since frozen evidence")
    load_candidate_state(args.baseline, base_hash, "model.12")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old_path = Path(prior["settings"]["dataset"]) / "manifest.json"
    old_manifest = json.loads(old_path.read_text())
    if sha(old_path) != prior["manifest_sha256"]:
        raise ValueError("baseline dataset manifest changed")
    verify_manifest(manifest)
    verify_manifest(old_manifest)
    settings = {
        "seed": args.seed,
        "device": args.device,
        "batch": 4,
        "lr": 1e-4,
        "imgsz": 320,
        "layer": "model.12",
        "kd_weight": 1.0,
    }
    baseline = validate_baseline(prior, old_manifest, manifest, settings)
    args.output.mkdir(parents=True)
    report = {
        "settings": vars(args),
        "weights_sha256": base_hash,
        "environment": environment,
        "manifest_sha256": sha(manifest_path),
        "baseline_state_sha256": sha(args.baseline),
        "baseline_report_sha256": sha(prior_path),
        "baseline_evidence_sha256": sha(args.baseline_evidence),
        "baseline_history": baseline["history"],
        "protocol": {
            "epochs": 3,
            **settings,
            "initialization": "original pretrained weights; no warm start",
            "final_comparison": "both predeclared models; no tuning on final scores",
            "training_order": "different nested pools; equal samples and updates, not equal image sequences",
        },
        "source_sha256": {},
    }
    for name in (
        Path(__file__).name,
        "finetune_sparse_detector.py",
        "sparse_kd_losses.py",
        "evaluate_sparse_checkpoints.py",
    ):
        source = ROOT / "scripts" / name
        report["source_sha256"][name] = sha(source)
        (args.output / name).write_bytes(source.read_bytes())

    def save():
        temporary = args.output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
        temporary.replace(args.output / "report.json")

    save()  # Freeze the protocol and baseline provenance before training or scoring.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = YOLO(str(args.weights)).model.to(args.device).float().eval()
    model.args = get_cfg(overrides={"imgsz": 320, "task": "detect", "rect": False})
    teacher = YOLO(str(args.weights)).model.to(args.device).float().eval()
    data = check_det_dataset(str(args.dataset / "dev.yaml"), autodownload=False)
    dataset = build_yolo_dataset(model.args, str(args.dataset / "train.txt"), 4, data, mode="val", rect=False)
    if set(dataset.im_files) != {row["image"] for row in manifest["splits"]["train"]}:
        raise ValueError("training loader identity mismatch")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    with RouterChoice(supported_layers(model)["model.12"], "native_top2") as routing:
        routing.active = True
        history = train_target(model, teacher, "model.12", loader, 3, 1e-4, 1.0, detection_weight=0.0)
        report.update(history=history, training_routing=routing.summary())
    exposure = 3 * len(dataset)
    if (
        sum(row["examples"] for row in history) != exposure
        or sum(row["optimizer_steps"] for row in history) != exposure // 4
    ):
        raise RuntimeError("actual training budget mismatch")
    if (
        report["training_routing"]["router_samples"] != exposure
        or report["training_routing"]["executed_experts_per_sample"] != 2
    ):
        raise RuntimeError("actual sparse execution mismatch")
    candidate = args.output / "sparse_kd_1024.state.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "layer": "model.12",
            "policy": "native_top2",
        },
        candidate,
    )
    report["state_sha256"] = sha(candidate)
    save()
    del model, teacher
    if args.device == "mps":
        torch.mps.empty_cache()
    for split in ("dev", "test"):
        destination = args.output / f"{split}_comparison"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_sparse_checkpoints.py"),
                "--weights",
                str(args.weights.resolve()),
                "--data",
                str(args.dataset / f"{split}.yaml"),
                "--max-images",
                str(len(manifest["splits"][split])),
                "--device",
                args.device,
                "--candidates",
                str(args.baseline.resolve()),
                str(candidate),
                "--output",
                str(destination),
            ],
            check=True,
        )
        comparison = json.loads((destination / "report.json").read_text())
        report[split] = comparison
        if split == "dev":
            candidates = comparison["cases"][2:-1]
            report["selected_from_dev"] = max(candidates, key=lambda c: c["metrics"]["metrics/mAP50-95(B)"])["name"]
        save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
