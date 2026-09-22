#!/usr/bin/env python3
"""Run the A2 TAL controls on VisDrone using the default.yaml acceptance protocol.

This entry point deliberately keeps the original TAL assigner unchanged. It
is the control run for the later STAL experiment. Dense validation is the
default so that the validation forward matches the ES-MoE training forward;
``--sparse-eval`` is available only for a separate known-issue diagnostic.
"""

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
# Prefer the checked-out YOLO-Master package over any older site-packages copy.
# This matters on shared training servers where an unrelated Ultralytics install
# may otherwise parse the ES_MOE YAML without the repository's mixture registry.
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

DATASET = DatasetSpec(name="VisDrone", data="VisDrone.yaml", project="runs/a2/qa_0831/control")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATASET.data, help="Dataset YAML; defaults to built-in VisDrone.yaml.")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if platform.system() == "Windows" else 8,
        help="Data-loader workers (0 is the safest value on Windows and network filesystems).",
    )
    parser.add_argument("--project", default=DATASET.project)
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--assigner",
        choices=("pure-tal", "fixed-stal"),
        default="fixed-stal",
        help="Assignment control: pure TAL disables the native fixed-stride floor; fixed-stal is the shipped default.",
    )
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
        "--sparse-eval", dest="sparse_eval", action="store_true", help="Diagnostic only for EsMoE-N."
    )
    eval_group.add_argument(
        "--no-sparse-eval", dest="sparse_eval", action="store_false", help="Use dense validation (the A2 default)."
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
        wandb_project=A2_CONFIG["wandb_project"],
    )
    return parser


def _write_manifest(path: Path, args: argparse.Namespace, spec: ModelSpec, *, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "assigner": args.assigner,
        "dense_eval": not args.sparse_eval,
        "wandb_project": args.wandb_project,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _install_assigner(assigner_name: str):
    """Install the pure-TAL control while preserving the shipped fixed-STAL default."""
    if assigner_name == "fixed-stal":
        return lambda: None

    import ultralytics.utils.loss as loss_module
    from ultralytics.utils.tal import TaskAlignedAssigner

    original_assigner = loss_module.TaskAlignedAssigner

    class PureTaskAlignedAssigner(TaskAlignedAssigner):
        """Native TAL candidate geometry without the fixed-stride STAL floor."""

        def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
            lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)
            return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)

    PureTaskAlignedAssigner.__name__ = "PureTaskAlignedAssigner"
    loss_module.TaskAlignedAssigner = PureTaskAlignedAssigner

    def restore() -> None:
        loss_module.TaskAlignedAssigner = original_assigner

    return restore


def _install_positive_stats(run_dir: Path, *, resume: bool) -> tuple[dict, object]:
    """Count foreground assignments without changing the assigner's outputs."""
    from ultralytics.utils.tal import TaskAlignedAssigner

    state = {
        "phase": "train",
        "epoch": 0,
        "pos_total": 0,
        "images": 0,
        "pos_batches": 0,
        "scale_stats": empty_scale_stats(),
    }
    original_forward = TaskAlignedAssigner.forward

    def forward_with_stats(self, *inputs, **kwargs):
        result = original_forward(self, *inputs, **kwargs)
        if state["phase"] == "train" and self.topk2 != 1 and getattr(self, "n_max_boxes", 0) > 0:
            fg_mask = result[3]
            state["pos_total"] += int(fg_mask.detach().sum().item())
            state["images"] += int(fg_mask.shape[0])
            state["pos_batches"] += 1
            update_scale_stats(state["scale_stats"], inputs[4], inputs[5], result)
        return result

    TaskAlignedAssigner.forward = forward_with_stats

    csv_path = run_dir / "a2_positive_stats.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    scale_fields = scale_stat_fieldnames()
    fieldnames = ("epoch", "train_pos_total", "train_images", "train_pos_batches", *scale_fields)
    if not resume or not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fieldnames)
    else:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing = tuple(csv.DictReader(handle).fieldnames or ())
        if existing != fieldnames:
            raise RuntimeError("cannot resume: a2_positive_stats.csv has an incompatible schema")

    def on_train_epoch_start(trainer):
        state["phase"] = "train"
        state["epoch"] = int(getattr(trainer, "epoch", 0)) + 1
        state["pos_total"] = 0
        state["images"] = 0
        state["pos_batches"] = 0
        state["scale_stats"] = empty_scale_stats()

    def on_val_start(_validator):
        state["phase"] = "val"

    def on_val_end(_validator):
        state["phase"] = "train"

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        scale_row = flatten_scale_stats(state["scale_stats"])
        row = (
            epoch,
            state["pos_total"],
            state["images"],
            state["pos_batches"],
            *scale_row.values(),
        )
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/pos_total": state["pos_total"],
                        "train/images": state["images"],
                        "train/pos_batches": state["pos_batches"],
                        **{f"train/{key}": value for key, value in scale_row.items()},
                    },
                    step=epoch,
                )
        except Exception:  # noqa: BLE001, S110
            pass
        print(
            f"[A2 baseline] epoch={epoch} train_pos_total={state['pos_total']} "
            f"train_images={state['images']} train_pos_batches={state['pos_batches']} "
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
    return callbacks, original_forward


def main() -> int:
    args = build_parser().parse_args()
    spec = MODEL_SPECS[args.model]
    if not args.name:
        args.name = f"{spec.name}_fixed-STAL"
    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project
    run_dir = project / args.name
    manifest = run_dir / "a2_baseline_manifest.json"

    print(
        f"[A2 baseline] model={spec.name} cfg={spec.cfg} data={args.data} "
        f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
        f"assigner={args.assigner} dense_eval={not args.sparse_eval} project={project}",
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

    restore_assigner = _install_assigner(args.assigner)
    stats_callbacks, original_assigner_forward = _install_positive_stats(run_dir, resume=resume)
    for event, callback in stats_callbacks.items():
        model.add_callback(event, callback)

    if not args.sparse_eval and spec.uses_esmoe:
        dense_callback = _make_dense_inference_callback()
        model.add_callback("on_pretrain_routine_end", dense_callback)
        model.add_callback("on_train_start", dense_callback)

    if args.wandb and args.wandb_mode != "disabled":
        wandb_dataset = DatasetSpec(name=DATASET.name, data=args.data, project=str(project))
        run_label = "pure-TAL" if args.assigner == "pure-tal" else "fixed-STAL"
        for event, callback in _make_wandb_callbacks(
            f"A2_{spec.name}_{run_label}", wandb_dataset, spec, args, dense_val=not args.sparse_eval
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
        restore_assigner()
        from ultralytics.utils.tal import TaskAlignedAssigner

        TaskAlignedAssigner.forward = original_assigner_forward

    _write_manifest(manifest, args, spec, status="completed")
    print(f"[A2 baseline] completed in {(time.time() - started) / 3600:.2f} h", flush=True)
    print(f"[A2 baseline] run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
