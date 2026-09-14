"""Prepare, calibrate, and launch the frozen VOC semantic / multi-teacher experiment.

Defaults to a read-only plan. Unit weights in templates are never used for formal training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/d2"
CONFIGS = BASE / "configs/p2_voc"
ARMS = ("siglip_semantic", "multiteacher")
SEEDS = (17, 29, 43)
WEIGHTS = {
    "siglip_semantic": "foundation_semantic_loss_weight",
    "multiteacher": "foundation_router_loss_weight",
}
TARGET_RATIO = 0.10
PROBE_BATCHES = 16


def digest(path):
    """Hash an input file without depending on Git tracking state."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    """Identify templates and the implementation used to select the weights."""
    files = list(CONFIGS.glob("*.yaml")) + [Path(__file__)]
    files += list((ROOT / "ultralytics/nn/foundation").rglob("*.py"))
    files += [
        ROOT / p
        for p in (
            "ultralytics/nn/foundation_distill_model.py",
            "ultralytics/cfg/default.yaml",
            "ultralytics/cfg/__init__.py",
            "ultralytics/engine/trainer.py",
            "ultralytics/nn/modules/head.py",
            "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        )
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def configs():
    """Load templates and reject unintended changes to the shared VOC budget."""
    import yaml

    result = {arm: yaml.safe_load((CONFIGS / f"{arm}.yaml").read_text()) for arm in ARMS}
    ignored = {
        "name",
        "foundation_teacher",
        "foundation_model",
        "foundation_dinov3_model",
        "foundation_loss_weight",
        "foundation_siglip2_model",
        "foundation_router_distill",
        "foundation_router_loss_weight",
        "foundation_router_temperature",
        "foundation_router_teachers",
        "foundation_router_native_state",
        "foundation_semantic_distill",
        "foundation_semantic_loss_weight",
        "foundation_semantic_text_weight",
        "foundation_semantic_image_weight",
        "foundation_semantic_temperature",
    }
    reference = {k: v for k, v in result["siglip_semantic"].items() if k not in ignored}
    for arm, cfg in result.items():
        if {k: v for k, v in cfg.items() if k not in ignored} != reference:
            raise ValueError(f"Unexpected shared-config difference: {arm}")
        if cfg["epochs"] != 400 or cfg["data"] != "VOC.yaml" or cfg["batch"] != 64 or cfg["imgsz"] != 256:
            raise ValueError("Frozen VOC budget changed")
    semantic, multi = result["siglip_semantic"], result["multiteacher"]
    if semantic["foundation_loss_weight"] != 0 or semantic["foundation_router_distill"]:
        raise ValueError("Semantic arm must isolate F13 from dense feature and routing KD")
    if not multi["foundation_router_distill"] or multi["foundation_router_loss_weight"] != 1.0:
        raise ValueError("Multi-teacher template must expose unit-weight routing KD for calibration")
    p1_refs = {
        "siglip_semantic": BASE / "configs/p1_voc/p1_c_siglip2_p4.yaml",
        "multiteacher": BASE / "configs/p1_voc/p1_a_dinov3_p4.yaml",
    }
    for arm, reference_path in p1_refs.items():
        reference_cfg = yaml.safe_load(reference_path.read_text())
        shared = lambda cfg: {
            k: v for k, v in cfg.items() if not k.startswith("foundation_") and k not in {"name", "project"}
        }
        if shared(result[arm]) != shared(reference_cfg):
            raise ValueError(f"{arm} no longer matches the shared P1 VOC budget")
    return result


def check():
    """Validate the CLI schema without downloading teacher weights."""
    from ultralytics.cfg import get_cfg

    for arm, cfg in configs().items():
        get_cfg(overrides=cfg)
        print(f"PASS {arm}")


def calibration(path, arms=None):
    """Reject stale or incomplete calibration before launching a treatment."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run --stage probe on the training machine before formal training")
    data = json.loads(path.read_text())
    if data.get("fingerprint") != fingerprint() or data.get("status") != "passed":
        raise ValueError("Calibration is stale or did not pass; run a new probe in a new directory")
    check_arms = [arm for arm in WEIGHTS if arms is None or arm in arms]
    for arm in check_arms:
        weight = data["weights"].get(arm)
        if not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid calibrated weight for {arm}")
    return data


def command(arm, seed, device, output, weights):
    """Use the installed CLI for supported training, with exact output identities."""
    cli = Path(sys.executable).parent / "yolo"
    cmd = [
        str(cli),
        "train",
        f"cfg={CONFIGS / (arm + '.yaml')}",
        f"seed={seed}",
        f"device={device}",
        f"project={output}",
        f"name={arm}-s{seed}",
        "exist_ok=False",
        "patience=0",
        "save_period=50",
    ]
    if arm in WEIGHTS:
        cmd.append(f"{WEIGHTS[arm]}={weights[arm]}")
    return cmd


def probe(output, device, arms=None, seeds=None):
    """Measure real training-only gradients at initialization, with no optimizer updates."""
    import torch

    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.foundation import foundation_multiteacher_summary
    from ultralytics.utils.torch_utils import init_seeds

    target = output / "calibration.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    output.mkdir(parents=True, exist_ok=True)
    probe_arms = [arm for arm in WEIGHTS if arms is None or arm in arms]
    if not probe_arms:
        raise ValueError(f"No calibration-eligible arms in {arms}; need one of {list(WEIGHTS)}")
    probe_seeds = list(SEEDS) if seeds is None else seeds
    record = {
        "status": "failed",
        "fingerprint": fingerprint(),
        "target_ratio": TARGET_RATIO,
        "optimizer_steps": 0,
        "validation_metrics_used": False,
        "weights": {},
        "arms": {},
    }
    for arm in probe_arms:
        observations = []
        for seed in probe_seeds:
            cfg = configs()[arm].copy()
            # The diagnostic uses real training labels, no validation metrics and deterministic unaugmented views.
            cfg.update(
                device=device,
                seed=seed,
                batch=4,
                workers=0,
                val=False,
                plots=False,
                project=str(output / "probe"),
                name=f"{arm}-s{seed}",
                exist_ok=False,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.0,
            )
            if (output / "probe" / cfg["name"]).exists():
                raise FileExistsError("Probe directory exists; use a fresh output root")
            trainer = DetectionTrainer(overrides=cfg)
            trainer._setup_train()
            model = trainer.model
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            params = [p for p in model.student.parameters() if p.requires_grad]
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            init_seeds(seed, deterministic=True)
            iterator = iter(trainer.train_loader)
            seed_observations = []
            for index in range(PROBE_BATCHES):
                batch = trainer.preprocess_batch(next(iterator))
                preds = model.student(batch["img"])
                task, _ = model.student.loss(batch, preds)
                with torch.no_grad():
                    teacher = model.teacher_manager.encode(batch["img"])
                if arm == "siglip_semantic":
                    added, metrics = model._semantic_kd(batch, preds, teacher)
                    if metrics.get("foundation_semantic_regions", 0) <= 0:
                        raise RuntimeError("No positive semantic regions: semantic branch is not functional")
                else:
                    summary = foundation_multiteacher_summary(teacher)
                    added, metrics = model._routing_kd(summary, batch_size=batch["img"].shape[0])
                    if metrics.get("foundation_router_modules", 0) <= 0:
                        raise RuntimeError("No routed modules: multi-teacher routing KD is not functional")

                def norm(loss, parameters):
                    grads = torch.autograd.grad(loss.sum(), parameters, retain_graph=True, allow_unused=True)
                    squares = [g.detach().double().square().sum() for g in grads if g is not None]
                    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0

                task_norm, added_norm = norm(task, params), norm(added, params)
                if not all(math.isfinite(v) and v > 0 for v in (task_norm, added_norm)):
                    raise RuntimeError("Non-finite or zero gradient in real-model probe")
                labels = hashlib.sha256()
                for key in ("cls", "bboxes", "batch_idx"):
                    labels.update(batch[key].detach().cpu().numpy().tobytes())
                seed_observations.append(
                    {
                        "seed": seed,
                        "batch": index,
                        "task_grad": task_norm,
                        "added_grad": added_norm,
                        "ratio": added_norm / task_norm,
                        "images": {str(p): digest(p) for p in batch["im_file"]},
                        "labels_sha256": labels.hexdigest(),
                    }
                )
                # Native auxiliary losses can update buffers; restore all model state after every observation.
                model.load_state_dict(before)
            assert all(p.grad is None for p in model.parameters())
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
            record["arms"].setdefault(arm, {})[str(seed)] = {
                "teacher": model.checkpoint_metadata(),
                "observations": seed_observations,
            }
            observations.extend(seed_observations)
            model.close()
            del iterator, trainer, model, before, params
        median = statistics.median(row["ratio"] for row in observations)
        weight = TARGET_RATIO / median
        medians = [weight * statistics.median(r["ratio"] for r in observations if r["seed"] == s) for s in probe_seeds]
        if not 1e-6 <= weight <= 100 or not all(0.05 <= value <= 0.20 for value in medians):
            raise RuntimeError(f"{arm}: calibration outside frozen bounds; do not tune using validation mAP")
        record["weights"][arm] = weight
        print(f"{arm}: selected weight={weight:.8g}; per-seed gradient ratios={medians}")
    record["status"] = "passed"
    with target.open("x") as stream:
        json.dump(record, stream, indent=2)
    print(f"Saved {target}")


def main():
    """Run the requested stage; planning is the default and never starts training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "check", "probe", "train"), default="plan")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/detect/d2/p2voc")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    args.output = args.output.resolve()
    configs()
    if args.stage == "check":
        check()
        return
    if args.stage == "probe":
        probe(args.output, args.device, arms=args.arms, seeds=args.seeds)
        return
    weights = {arm: "CALIBRATION_REQUIRED" for arm in WEIGHTS}
    if args.stage == "train":
        check()
        if any(arm in WEIGHTS for arm in args.arms):
            weights = calibration(args.output / "calibration.json", arms=args.arms)["weights"]
    for seed in args.seeds:
        for arm in dict.fromkeys(args.arms):
            cmd = command(arm, seed, args.device, args.output, weights)
            print(shlex.join(cmd), flush=True)
            if args.stage == "plan":
                continue
            run = args.output / f"{arm}-s{seed}"
            marker = args.output / f"{arm}-s{seed}.complete.json"
            if run.exists() or marker.exists():
                raise FileExistsError(f"Refusing overwrite or automatic resume: {run}")
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output / f"{arm}-s{seed}.log").open("x") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            import csv

            with (run / "results.csv").open() as stream:
                rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
            if len(rows) != 400 or int(rows[-1]["epoch"]) != 400:
                raise RuntimeError(f"{run}: incomplete 400-epoch run")
            metric = float(rows[-1]["metrics/mAP50-95(B)"])
            if not math.isfinite(metric):
                raise RuntimeError(f"{run}: non-finite final mAP")
            marker.write_text(json.dumps({"command": cmd, "final_map": metric, "fingerprint": fingerprint()}, indent=2))


if __name__ == "__main__":
    main()
