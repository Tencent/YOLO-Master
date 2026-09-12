#!/usr/bin/env python3
"""Train A2 Area-Threshold STAL on VisDrone using the default.yaml acceptance protocol."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPRODUCE_DIR = ROOT / "scripts" / "reproduce"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPRODUCE_DIR))

from _reproduce_common import (
    DatasetSpec,
    ModelSpec,
    _make_dense_inference_callback,
    _make_wandb_callbacks,
)

from a2._config import A2_CONFIG
from a2._scale_stats import (
    empty_scale_stats,
    flatten_scale_stats,
    scale_stat_fieldnames,
    update_scale_stats,
)
from ultralytics.utils import DEFAULT_CFG_PATH

DATASET = DatasetSpec(name="VisDrone", data="VisDrone.yaml", project="runs/a2/qa_0831/at_stal")
MODEL_SPECS = {
    "v0.1-N": ModelSpec(
        name="v0.1-N",
        cfg="ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
        uses_esmoe=False,
    ),
    "EsMoE-N": ModelSpec(
        name="EsMoE-N",
        cfg="ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",
        uses_esmoe=True,
    ),
}
STAT_KEYS = (
    "gt_total",
    "eligible_gt",
    "base_candidates",
    "final_candidates",
    "floor_added",
    "pre_assigned",
    "post_assigned",
    "zero_pre",
    "zero_post",
    "eligible_pre_assigned",
    "eligible_post_assigned",
    "conflict_anchors",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the AT-STAL training CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATASET.data, help="Dataset YAML; defaults to built-in VisDrone.yaml.")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if platform.system() == "Windows" else 8,
        help="Data-loader workers (0 is safest on Windows and network filesystems).",
    )
    parser.add_argument("--project", default=DATASET.project)
    parser.add_argument("--name", default="", help="Run name; default includes the selected area threshold.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache",
        nargs="?",
        const="ram",
        default=False,
        help="Cache images with --cache/--cache ram, or use --cache disk.",
    )
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--sparse-eval",
        dest="sparse_eval",
        action="store_true",
        help="Diagnostic only for EsMoE-N; v0.1-N is unaffected.",
    )
    eval_group.add_argument(
        "--no-sparse-eval",
        dest="sparse_eval",
        action="store_false",
        help="Use dense EsMoE validation (v0.1-N is unaffected).",
    )
    parser.set_defaults(sparse_eval=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(
        epochs=A2_CONFIG["epochs"],
        imgsz=A2_CONFIG["imgsz"],
        batch=A2_CONFIG["batch"],
        seed=A2_CONFIG["seed"],
        model=A2_CONFIG["model"],
        patience=A2_CONFIG["patience"],
        stal_area_threshold=A2_CONFIG["area_threshold"],
        stal_min_candidates=A2_CONFIG["min_candidates"],
        wandb_project=A2_CONFIG["wandb_project"],
    )
    return parser


def _install_area_assigner(area_threshold: float, min_candidates: int):
    """Patch the detection loss to construct the configured AT-STAL assigner."""
    import ultralytics.utils.loss as loss_module
    from ultralytics.utils.tal import AreaAwareTaskAlignedAssigner

    original_assigner = loss_module.TaskAlignedAssigner

    class ConfiguredAreaAssigner(AreaAwareTaskAlignedAssigner):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                area_threshold=area_threshold,
                min_candidates=min_candidates,
                **kwargs,
            )

    ConfiguredAreaAssigner.__name__ = "AreaAwareTaskAlignedAssigner"
    loss_module.TaskAlignedAssigner = ConfiguredAreaAssigner

    def restore() -> None:
        loss_module.TaskAlignedAssigner = original_assigner

    return ConfiguredAreaAssigner, restore


def _install_positive_stats(assigner_cls, run_dir: Path, *, resume: bool):
    """Record compact one-to-many assignment statistics for every epoch."""
    state = {
        "phase": "train",
        "last_logged_epoch": None,
        "train_pos_total": 0,
        "train_effective_pos_total": 0,
        "train_zero_weight_pos_total": 0,
        "train_images": 0,
        "train_pos_batches": 0,
        "scale_stats": empty_scale_stats(),
        **{key: 0 for key in STAT_KEYS},
    }
    csv_path = run_dir / "a2_area_stal_positive_stats.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    scale_fields = scale_stat_fieldnames()
    fieldnames = (
        "epoch",
        "train_pos_total",
        "train_effective_pos_total",
        "train_zero_weight_pos_total",
        "train_images",
        "train_pos_batches",
        *STAT_KEYS,
        *scale_fields,
    )
    if not resume or not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fieldnames)
    else:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    state["last_logged_epoch"] = int(row["epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing = tuple(csv.DictReader(handle).fieldnames or ())
        if existing != fieldnames:
            raise RuntimeError("cannot resume: a2_area_stal_positive_stats.csv has an incompatible schema")

    original_forward = assigner_cls.forward

    def forward_with_stats(self, *inputs, **kwargs):
        result = original_forward(self, *inputs, **kwargs)
        stats = getattr(self, "last_area_stats", {})
        if (
            state["phase"] == "train"
            and getattr(self, "n_max_boxes", 0) > 0
            and stats.get("assignment_branch") == "one2many"
        ):
            fg_mask = result[3].detach()
            target_quality = result[2].detach().sum(dim=-1)
            effective = fg_mask & (target_quality > self.eps)
            state["train_pos_total"] += int(fg_mask.sum().item())
            state["train_effective_pos_total"] += int(effective.sum().item())
            state["train_zero_weight_pos_total"] += int((fg_mask & ~effective).sum().item())
            state["train_images"] += int(fg_mask.shape[0])
            state["train_pos_batches"] += 1
            for key in STAT_KEYS:
                state[key] += int(stats.get(key, 0))
            update_scale_stats(
                state["scale_stats"],
                inputs[4],
                inputs[5],
                result,
                pre_assigned=getattr(self, "last_pre_assigned", None),
            )
        return result

    assigner_cls.forward = forward_with_stats

    def on_train_epoch_start(_trainer):
        state["phase"] = "train"
        for key in (
            "train_pos_total",
            "train_effective_pos_total",
            "train_zero_weight_pos_total",
            "train_images",
            "train_pos_batches",
            *STAT_KEYS,
        ):
            state[key] = 0
        state["scale_stats"] = empty_scale_stats()

    def on_val_start(_validator):
        state["phase"] = "val"

    def on_val_end(_validator):
        state["phase"] = "train"

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if state["last_logged_epoch"] == epoch:
            return
        state["last_logged_epoch"] = epoch
        scale_row = flatten_scale_stats(state["scale_stats"])
        row = (
            epoch,
            state["train_pos_total"],
            state["train_effective_pos_total"],
            state["train_zero_weight_pos_total"],
            state["train_images"],
            state["train_pos_batches"],
            *(state[key] for key in STAT_KEYS),
            *scale_row.values(),
        )
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/at_stal_pos_total": state["train_pos_total"],
                        "train/at_stal_effective_pos_total": state["train_effective_pos_total"],
                        "train/at_stal_zero_weight_pos_total": state["train_zero_weight_pos_total"],
                        "train/at_stal_images": state["train_images"],
                        "train/at_stal_pos_batches": state["train_pos_batches"],
                        **{f"train/at_stal_{key}": state[key] for key in STAT_KEYS},
                        **{f"train/at_stal_{key}": value for key, value in scale_row.items()},
                    },
                    step=epoch,
                )
        except Exception:  # noqa: BLE001, S110
            pass
        print(
            f"[A2 AT-STAL] epoch={epoch} pos={state['train_pos_total']} "
            f"effective/zero_weight={state['train_effective_pos_total']}/"
            f"{state['train_zero_weight_pos_total']} eligible_gt={state['eligible_gt']} "
            f"candidates={state['base_candidates']}->{state['final_candidates']} "
            f"assigned={state['pre_assigned']}->{state['post_assigned']} "
            f"zero={state['zero_pre']}->{state['zero_post']} conflicts={state['conflict_anchors']} "
            + " ".join(
                f"{bucket}_avg={scale_row[f'{bucket}_post_avg_pos']:.3f} "
                f"{bucket}_zero={scale_row[f'{bucket}_zero_post_ratio']:.3f}"
                for bucket in ("small", "medium", "large")
            ),
            flush=True,
        )

    callbacks = {
        "on_train_epoch_start": on_train_epoch_start,
        "on_val_start": on_val_start,
        "on_val_end": on_val_end,
        "on_fit_epoch_end": on_fit_epoch_end,
    }

    def restore() -> None:
        assigner_cls.forward = original_forward

    return callbacks, restore


def _write_manifest(path: Path, args: argparse.Namespace, spec: ModelSpec, *, status: str) -> None:
    """Write a reproducibility manifest for one AT-STAL run."""
    git_ref = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    payload = {
        "status": status,
        "topic": "A2",
        "method": "Area-Threshold STAL",
        "git_ref": git_ref,
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "model": spec.name,
        "cfg": spec.cfg,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "pretrained": False,
        "config": str(DEFAULT_CFG_PATH),
        "stal_area_threshold": A2_CONFIG["area_threshold"],
        "stal_min_candidates": A2_CONFIG["min_candidates"],
        "dense_eval": not args.sparse_eval,
        "wandb_project": args.wandb_project,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run one AT-STAL training experiment."""
    args = build_parser().parse_args()
    spec = MODEL_SPECS[args.model]
    area_threshold = float(A2_CONFIG["area_threshold"])
    threshold_name = f"{area_threshold:g}".replace(".", "p")
    if not args.name:
        args.name = f"{spec.name}_AT-STAL_A{threshold_name}"

    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project
    run_dir = project / args.name
    manifest = run_dir / "a2_at_stal_manifest.json"

    print(
        f"[A2 AT-STAL] model={spec.name} cfg={spec.cfg} data={args.data} "
        f"config={DEFAULT_CFG_PATH} area_threshold={area_threshold:g} min_candidates={A2_CONFIG['min_candidates']} "
        f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
        f"dense_eval={not args.sparse_eval} project={project}",
        flush=True,
    )
    if args.dry_run:
        return 0

    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(str(ROOT / spec.cfg), ch=3, nc=10, verbose=False)
        params = sum(parameter.numel() for parameter in model.parameters())
        print(f"[build-ok] {spec.name}: {params:,} parameters", flush=True)
        return 0

    from ultralytics import YOLO

    last_pt = run_dir / "weights" / "last.pt"
    resume = bool(args.resume and last_pt.exists())
    model = YOLO(str(last_pt if resume else ROOT / spec.cfg))

    assigner_cls, restore_assigner = _install_area_assigner(area_threshold, int(A2_CONFIG["min_candidates"]))
    stats_callbacks, restore_stats = _install_positive_stats(assigner_cls, run_dir, resume=resume)
    for event, callback in stats_callbacks.items():
        model.add_callback(event, callback)

    if not args.sparse_eval and spec.uses_esmoe:
        dense_callback = _make_dense_inference_callback()
        model.add_callback("on_pretrain_routine_end", dense_callback)
        model.add_callback("on_train_start", dense_callback)

    if args.wandb and args.wandb_mode != "disabled":
        wandb_dataset = DatasetSpec(name=DATASET.name, data=args.data, project=str(project))
        run_name = f"A2_{spec.name}_AT-STAL_A{threshold_name}"
        for event, callback in _make_wandb_callbacks(
            run_name, wandb_dataset, spec, args, dense_val=not args.sparse_eval
        ).items():
            model.add_callback(event, callback)

    project.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest, args, spec, status="running")
    started = time.time()
    try:
        model.train(
            cfg=str(DEFAULT_CFG_PATH),
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            project=str(project),
            name=args.name,
            exist_ok=True,
            pretrained=False,
            lora_r=0,
            optimizer="auto",
            val=True,
            plots=True,
            cache=args.cache,
            patience=args.patience,
            amp=args.amp,
            resume=resume,
            verbose=False,
        )
    except Exception:
        _write_manifest(manifest, args, spec, status="failed")
        raise
    finally:
        restore_stats()
        restore_assigner()

    _write_manifest(manifest, args, spec, status="completed")
    print(f"[A2 AT-STAL] completed in {(time.time() - started) / 3600:.2f} h", flush=True)
    print(f"[A2 AT-STAL] run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
