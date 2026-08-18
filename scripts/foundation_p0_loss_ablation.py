"""Run the P0 isolated Foundation KD-loss ablation (B0/D0/D1/D2) on real COCO.

This runner answers the first research question of YOLO-Master-F v0.1.0-alpha:
whether a DINOv3 dense prior helps the student **per loss type**. Every arm is
paired with a baseline run that shares the same model config, seed, dataset
split, and initialization contract. The runner owns experiment orchestration
and result provenance only; it never turns a missing validation result into an
accuracy claim.

Arms (plan section 17.2)::

    B0  baseline, foundation disabled
    D0  cosine-only KD       (foundation_loss=cosine,    relation_weight=0)
    D1  relational-only KD   (foundation_loss=relational, cosine_weight=0)
    D2  hybrid KD            (foundation_loss=hybrid, both weights=1)

Example dry-run::

    python scripts/foundation_p0_loss_ablation.py \
        --dataset /path/to/coco2017.yaml \
        --teacher-model /path/to/dinov3 \
        --seeds 20260813 --arms baseline,cosine \
        --epochs 1 --fraction 0.0005 --imgsz 128 --batch 2 --device mps \
        --dry-run

Use ``--run-name`` to execute exactly one planned run per invocation, and
``--resume`` to skip runs already present in the output report. ``--scale-ap``
additionally captures COCO scale-bucket APs/APm/APl via pycocotools for detect
runs; when the required files are unavailable the report records an explicit
``null`` result instead of a fabricated number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Make direct ``python scripts/...`` invocation resolve this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.foundation_f15_real_effect_gate import (  # noqa: E402
    TASKS,
    _csv_floats,
    _csv_ints,
    _write_report,
    summarize_run,
)

DEFAULT_MODEL = str(REPO_ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml")
DEFAULT_DATA = "/Users/gatilin/MyWork/datasets/coco2017/coco2017.yaml"
DEFAULT_TEACHER = os.environ.get("YOLO_MASTER_DINOV3_LOCAL", "Tooony133/dinov3-vits16-pretrain-lvd1689m")
SCHEMA_VERSION = 1
BENCHMARK = "p0_foundation_loss_ablation"

#: Loss-isolation contract per arm. Baseline disables Foundation entirely.
ARMS: dict[str, dict[str, Any]] = {
    "baseline": {"foundation": False, "foundation_loss": None, "cosine_weight": 0.0, "relation_weight": 0.0},
    "cosine": {"foundation": True, "foundation_loss": "cosine", "cosine_weight": 1.0, "relation_weight": 0.0},
    "relational": {"foundation": True, "foundation_loss": "relational", "cosine_weight": 0.0, "relation_weight": 1.0},
    "hybrid": {"foundation": True, "foundation_loss": "hybrid", "cosine_weight": 1.0, "relation_weight": 1.0},
}
DEFAULT_ARMS = ["baseline", "cosine", "relational", "hybrid"]

#: COCOeval stat indices → names recorded when --scale-ap is enabled.
COCO_STATS = [
    "AP50_95",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
]


def _csv_strs(value: str) -> list[str]:
    """Parse a non-empty comma-separated list of strings."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _validate_arms(arms: list[str]) -> list[str]:
    """Validate requested arms, requiring baseline for paired deltas."""
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    if any(arm != "baseline" for arm in arms) and "baseline" not in arms:
        raise argparse.ArgumentTypeError("arms must include 'baseline' for paired comparison")
    return arms


def _peak_memory_bytes(device: str) -> int | None:
    """Best-effort peak/allocated device memory; None when the backend cannot report it."""
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
        if device == "mps" and torch.backends.mps.is_available():
            # MPS has no peak counter; driver-allocated memory is the honest best effort.
            return int(torch.mps.driver_allocated_memory())
    except (AttributeError, ImportError, RuntimeError):
        return None
    return None


def capture_scale_bucket_ap(
    checkpoint: str | Path,
    dataset: str,
    *,
    device: str,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
) -> dict[str, Any]:
    """Capture COCO scale-bucket metrics via pycocotools for a trained detect checkpoint.

    Returns a dict with ``available=False`` and a reason whenever any required
    artifact (predictions JSON, GT annotations, pycocotools) is missing. It
    never fabricates metric values.
    """
    result: dict[str, Any] = {"available": False, "reason": None, "checkpoint": str(checkpoint)}
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        result["reason"] = "pycocotools is not installed"
        return result

    import yaml

    from ultralytics import YOLO

    data_yaml = Path(dataset)
    if not data_yaml.is_file():
        result["reason"] = f"dataset yaml not found: {dataset}"
        return result
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(data_cfg.get("path") or data_yaml.parent)
    gt_rel = data_cfg.get("val_instances") or "annotations/instances_val2017.json"
    gt_json = root / gt_rel
    if not gt_json.is_file():
        result["reason"] = f"GT annotation json not found: {gt_json}"
        return result

    model = YOLO(str(checkpoint))
    val_name = f"{name}-scaleap"
    model.val(
        data=dataset,
        split="val",
        save_json=True,
        plots=False,
        verbose=False,
        device=device,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=val_name,
        exist_ok=True,
    )
    # exist_ok=True makes the validator output directory deterministic.
    val_dir = Path(project) / val_name
    candidates = sorted(val_dir.glob("**/*predictions*.json"))
    if not candidates:
        result["reason"] = f"predictions json not found under {val_dir}"
        return result

    eval_result = coco_eval_subset(gt_json, candidates[-1], COCO=COCO, COCOeval=COCOeval)
    result.update(eval_result)
    result["predictions_json"] = str(candidates[-1])
    result["gt_json"] = str(gt_json)
    return result


def coco_eval_subset(gt_json: str | Path, predictions_json: str | Path, *, COCO=None, COCOeval=None) -> dict[str, Any]:
    """Run COCOeval restricted to images present in the predictions file.

    A validation subset must never be scored against GT images it did not
    predict on, so ``imgIds`` is intersected with the prediction set. The
    result records ``evaluated_images`` and ``subset_eval`` so downstream
    consumers can tell a partial evaluation from a full COCO val run.
    """
    if COCO is None or COCOeval is None:
        from pycocotools.coco import COCO as _COCO
        from pycocotools.cocoeval import COCOeval as _COCOeval

        COCO, COCOeval = _COCO, _COCOeval
    with Path(predictions_json).open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    predicted_ids = {int(pred["image_id"]) for pred in predictions}
    coco_gt = COCO(str(gt_json))
    all_ids = set(coco_gt.getImgIds())
    img_ids = sorted(predicted_ids & all_ids)
    if not img_ids:
        return {
            "available": False,
            "reason": "no prediction image_id intersects GT annotations",
            "evaluated_images": 0,
            "subset_eval": True,
        }
    coco_dt = coco_gt.loadRes(str(predictions_json))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = img_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "available": True,
        "reason": None,
        "evaluated_images": len(img_ids),
        "gt_images": len(all_ids),
        "subset_eval": len(img_ids) < len(all_ids),
        "stats": dict(zip(COCO_STATS, (float(value) for value in evaluator.stats))),
    }


def build_run_plan(
    *,
    dataset: str,
    model: str,
    teacher_model: str,
    project: str,
    seeds: list[int],
    foundation_loss_weights: list[float],
    arms: list[str],
    task: str,
    epochs: int,
    fraction: float,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    val: bool,
    scale_ap: bool,
) -> list[dict[str, Any]]:
    """Build deterministic baseline/arm runs in weight-then-seed-then-arm order."""
    plan: list[dict[str, Any]] = []
    for weight in foundation_loss_weights:
        for seed in seeds:
            pair_name = f"s{seed}-w{weight:g}"
            for arm in arms:
                spec_arm = ARMS[arm]
                foundation = bool(spec_arm["foundation"])
                name = f"{arm}-{pair_name}"
                overrides: dict[str, Any] = {
                    "model": model,
                    "data": dataset,
                    "task": task,
                    "mode": "train",
                    "epochs": epochs,
                    "fraction": fraction,
                    "imgsz": imgsz,
                    "batch": batch,
                    "device": device,
                    "workers": workers,
                    "val": val,
                    "seed": seed,
                    "deterministic": True,
                    "pretrained": False,
                    "project": project,
                    "name": name,
                    "exist_ok": True,
                    "save": True,
                    "plots": False,
                    "foundation_enabled": foundation,
                    "foundation_teacher": "dinov3" if foundation else "none",
                    "foundation_model": teacher_model if foundation else None,
                    "foundation_backend": "transformers",
                    "foundation_teacher_dtype": "fp32",
                    "foundation_teacher_device": device,
                    "foundation_target_levels": ["p4"],
                    "foundation_loss": spec_arm["foundation_loss"] if foundation else "hybrid",
                    "foundation_cosine_weight": spec_arm["cosine_weight"],
                    "foundation_relation_weight": spec_arm["relation_weight"],
                    "foundation_relation_mode": "sampled",
                    "foundation_relation_samples": 16,
                    "foundation_loss_weight": weight if foundation else 0.0,
                }
                if task == "multitask":
                    overrides.update(
                        {
                            "foundation_multitask": foundation,
                            "foundation_multitask_tasks": TASKS,
                        }
                    )
                plan.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "teacher_model": teacher_model,
                        "project": project,
                        "seed": seed,
                        "foundation_loss_weight": weight,
                        "arm": arm,
                        "foundation": foundation,
                        "task": task,
                        "epochs": epochs,
                        "fraction": fraction,
                        "imgsz": imgsz,
                        "batch": batch,
                        "device": device,
                        "workers": workers,
                        "val": val,
                        "scale_ap": scale_ap,
                        "name": name,
                        "initialization_contract": {
                            "pretrained": False,
                            "same_model_config": True,
                            "same_seed": True,
                            "same_dataset_split": True,
                        },
                        "overrides": overrides,
                    }
                )
    return plan


def _train_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Train one planned run and return its measured result summary."""
    import torch

    from ultralytics import YOLO

    # Seed model construction explicitly so paired branches share initialization.
    torch.manual_seed(int(spec["seed"]))
    started = time.perf_counter()
    model = YOLO(spec["model"])
    model.train(**spec["overrides"])
    run_dir = Path(str(getattr(getattr(model, "trainer", None), "save_dir", Path(spec["project"]) / spec["name"])))
    summary = summarize_run(run_dir)
    summary.update(
        {
            "elapsed_s": round(time.perf_counter() - started, 4),
            "foundation": spec["foundation"],
            "arm": spec["arm"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "seed": spec["seed"],
            "name": spec["name"],
            "peak_memory_bytes": _peak_memory_bytes(str(spec["device"])),
        }
    )
    if spec.get("scale_ap") and spec.get("task") == "detect" and summary.get("checkpoint"):
        summary["scale_bucket_ap"] = capture_scale_bucket_ap(
            summary["checkpoint"],
            spec["dataset"],
            device=str(spec["device"]),
            imgsz=int(spec["imgsz"]),
            batch=int(spec["batch"]),
            project=str(spec["project"]),
            name=str(spec["name"]),
        )
    return summary


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-arm validation deltas vs the paired baseline without imputing accuracy."""
    grouped: dict[tuple[int, float], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (int(record["seed"]), float(record["foundation_loss_weight"]))
        grouped.setdefault(key, {})[str(record["arm"])] = record
    pairs = []
    for (seed, weight), arms in sorted(grouped.items()):
        baseline = arms.get("baseline")
        baseline_metrics = (baseline or {}).get("validation_metrics", {})
        arm_entries = {}
        for arm, record in sorted(arms.items()):
            if arm == "baseline":
                continue
            metrics = (record or {}).get("validation_metrics", {})
            arm_entries[arm] = {
                "complete": record is not None,
                "validation_metric_deltas_vs_baseline": {
                    key: round(float(metrics[key]) - float(baseline_metrics[key]), 8)
                    for key in sorted(set(baseline_metrics) & set(metrics))
                },
                "scale_bucket_ap_available": bool((record or {}).get("scale_bucket_ap", {}).get("available")),
            }
        pairs.append(
            {
                "seed": seed,
                "foundation_loss_weight": weight,
                "baseline_complete": baseline is not None,
                "arms": arm_entries,
            }
        )
    return {
        "paired_runs": len(pairs),
        "validation_pairs_with_metrics": sum(
            any(entry["validation_metric_deltas_vs_baseline"] for entry in pair["arms"].values()) for pair in pairs
        ),
        "pairs": pairs,
    }


def run_matrix(
    plan: list[dict[str, Any]],
    output: Path,
    *,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    resume: bool = False,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Execute a plan and write progress after every run for interruption safety.

    ``run_name`` restricts execution to one planned run while the report still
    records the full plan, so repeated single-run invocations keep provenance.
    """
    if run_name is not None and not any(spec["name"] == run_name for spec in plan):
        names = ", ".join(spec["name"] for spec in plan)
        raise ValueError(f"--run-name {run_name!r} not in plan. Available: {names}")
    records: list[dict[str, Any]] = []
    if resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("benchmark") != BENCHMARK:
            raise ValueError(f"Cannot resume incompatible report: {output}")
        records = list(previous.get("records") or [])
    completed_names = {record.get("name") for record in records}
    for spec in plan:
        if run_name is not None and spec["name"] != run_name:
            continue
        if spec["name"] in completed_names:
            continue
        result = runner(spec)
        records.append(result)
        completed_names.add(spec["name"])
        _write_report(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": BENCHMARK,
                "real_data": True,
                "accuracy_claim": False,
                "arms": sorted({spec["arm"] for spec in plan}),
                "plan": plan,
                "records": records,
                "completed_runs": len(records),
                "total_runs": len(plan),
                "summary": _aggregate_records(records),
            },
        )
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse P0 loss-ablation runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATA, help="COCO dataset YAML.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Student model YAML.")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER, help="Local or HF DINOv3 model path/id.")
    parser.add_argument("--project", default="runs/detect/p0-loss-ablation", help="Ultralytics project directory.")
    parser.add_argument("--seeds", type=_csv_ints, default=[20260813])
    parser.add_argument("--foundation-loss-weights", type=_csv_floats, default=[0.01])
    parser.add_argument("--arms", type=_csv_strs, default=DEFAULT_ARMS, help=f"Subset of {sorted(ARMS)}.")
    parser.add_argument("--task", choices=["detect", "multitask"], default="detect")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--scale-ap", action="store_true", help="Capture COCO scale-bucket AP/AR via pycocotools.")
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/p0-loss-ablation.json"))
    parser.add_argument("--run-name", default=None, help="Execute exactly one planned run by name.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and do not train.")
    parser.add_argument("--resume", action="store_true", help="Skip completed run names found in --output.")
    parser.add_argument("--val", dest="val", action="store_true", default=True, help="Run validation (default).")
    parser.add_argument("--no-val", dest="val", action="store_false", help="Disable validation for diagnostics only.")
    args = parser.parse_args(argv)
    try:
        args.arms = _validate_arms(args.arms)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.imgsz <= 0 or args.batch <= 0:
        parser.error("--imgsz and --batch must be positive")
    if args.scale_ap and args.task != "detect":
        parser.error("--scale-ap is only supported for --task detect")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point for the P0 loss-ablation runner."""
    args = parse_args(argv)
    plan = build_run_plan(
        dataset=args.dataset,
        model=args.model,
        teacher_model=args.teacher_model,
        project=args.project,
        seeds=args.seeds,
        foundation_loss_weights=args.foundation_loss_weights,
        arms=args.arms,
        task=args.task,
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        val=args.val,
        scale_ap=args.scale_ap,
    )
    if args.run_name:
        selected = [spec for spec in plan if spec["name"] == args.run_name]
        if not selected:
            names = ", ".join(spec["name"] for spec in plan)
            raise SystemExit(f"--run-name {args.run_name!r} not in plan. Available: {names}")
        if args.dry_run:
            plan = selected
    if args.dry_run:
        print(json.dumps({"benchmark": BENCHMARK, "total_runs": len(plan), "plan": plan}, indent=2, ensure_ascii=False))
        return 0
    report = run_matrix(plan, args.output, runner=_train_one, resume=args.resume, run_name=args.run_name)
    print(json.dumps({"completed_runs": report["completed_runs"], "total_runs": report["total_runs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
